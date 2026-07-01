// driftguard-llm is a long-lived sidecar that performs Driftguard's Anthropic
// calls using the OFFICIAL anthropic-sdk-go. It runs as a SERVER: the Rust core
// starts it once and multiplexes requests over its stdio.
//
// Protocol: newline-delimited JSON. Each request carries an `id`; each response
// echoes that `id`, so Rust can fire many requests concurrently and correlate
// the (out-of-order) responses. The process handles requests in goroutines and
// shares one `anthropic.Client`, which reuses its HTTP/TLS connections — far
// faster than the old one-process-per-call model.
//
//	request : {"id": <n>, "op": "generate|judge_pass|judge_behavior", ...}
//	response: {"id": <n>, "result": {...}}   or   {"id": <n>, "error": "..."}
//
// Models (Driftguard's decisions): generation uses Sonnet 4.6; judging uses
// Opus 4.8 with structured output (output_config.format). Thinking is
// intentionally OFF on the judge — the schema already forces a clean JSON
// verdict, so thinking tokens were wasted spend + latency.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	modelUnderTest = "claude-sonnet-4-6"
	judgeModel     = "claude-opus-4-8"
)

type request struct {
	ID               uint64 `json:"id"`
	Op               string `json:"op"`
	System           string `json:"system"`
	User             string `json:"user"`
	ExpectedBehavior string `json:"expected_behavior"`
	Input            string `json:"input"`
	ActualOutput     string `json:"actual_output"`
	Before           string `json:"before"`
	After            string `json:"after"`
}

type response struct {
	ID     uint64          `json:"id"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY is not set")
		os.Exit(1)
	}

	client := anthropic.NewClient() // one shared client → reused TLS connections
	ctx := context.Background()

	// Serialize writes to stdout (concurrent goroutines share it).
	var writeMu sync.Mutex
	out := bufio.NewWriter(os.Stdout)
	write := func(resp response) {
		b, _ := json.Marshal(resp)
		writeMu.Lock()
		out.Write(b)
		out.WriteByte('\n')
		out.Flush()
		writeMu.Unlock()
	}

	// Large buffer: judge requests carry before/after outputs.
	reader := bufio.NewReaderSize(os.Stdin, 1<<20)
	var wg sync.WaitGroup
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			wg.Add(1)
			go func(raw []byte) {
				defer wg.Done()
				write(handle(ctx, &client, raw))
			}(line)
		}
		if err != nil {
			break // EOF (Rust closed stdin) or read error
		}
	}
	wg.Wait()
}

func handle(ctx context.Context, client *anthropic.Client, raw []byte) response {
	var req request
	if err := json.Unmarshal(raw, &req); err != nil {
		return response{Error: fmt.Sprintf("decoding request: %v", err)}
	}

	switch req.Op {
	case "generate":
		text, err := generate(ctx, client, req.System, req.User)
		if err != nil {
			return response{ID: req.ID, Error: err.Error()}
		}
		result, _ := json.Marshal(map[string]string{"text": text})
		return response{ID: req.ID, Result: result}

	case "judge_pass":
		system := "You are a strict, fair evaluator of AI responses. You are given a test " +
			"input, a description of the expected behavior, and a model's actual response. " +
			"Decide only whether the response satisfies the described expected behavior. Do not " +
			"impose criteria that are not stated. If the response clearly violates or omits a " +
			"required behavior, it fails; otherwise it passes."
		user := fmt.Sprintf("<expected_behavior>%s</expected_behavior>\n<input>%s</input>\n"+
			"<actual_response>%s</actual_response>\n\nDid the response satisfy the expected behavior?",
			req.ExpectedBehavior, req.Input, req.ActualOutput)
		schema := objectSchema(map[string]any{
			"passed":        map[string]any{"type": "boolean"},
			"justification": map[string]any{"type": "string"},
		}, "passed", "justification")
		return judgeResponse(ctx, client, req.ID, system, user, schema)

	case "judge_behavior":
		system := "You are evaluating whether a prompt change altered a model's behavior on a " +
			"specific test case. You are given the test input and the model's response BEFORE and " +
			"AFTER the prompt change. Decide whether the behavior changed substantively — a " +
			"difference a user or downstream system would care about (different decision, different " +
			"content, materially different tone, added or removed information). Ignore trivial " +
			"wording differences that don't change meaning or outcome."
		user := fmt.Sprintf("<input>%s</input>\n<expected_behavior>%s</expected_behavior>\n"+
			"<response_before>%s</response_before>\n<response_after>%s</response_after>\n\n"+
			"Did the behavior change substantively between before and after?",
			req.Input, req.ExpectedBehavior, req.Before, req.After)
		schema := objectSchema(map[string]any{
			"behavior_changed": map[string]any{"type": "boolean"},
			"justification":    map[string]any{"type": "string"},
		}, "behavior_changed", "justification")
		return judgeResponse(ctx, client, req.ID, system, user, schema)

	default:
		return response{ID: req.ID, Error: fmt.Sprintf("unknown op %q", req.Op)}
	}
}

func judgeResponse(ctx context.Context, client *anthropic.Client, id uint64, system, user string, schema map[string]any) response {
	text, err := judge(ctx, client, system, user, schema)
	if err != nil {
		return response{ID: id, Error: err.Error()}
	}
	// `text` is already a JSON object matching the schema; embed it raw.
	return response{ID: id, Result: json.RawMessage(text)}
}

func generate(ctx context.Context, client *anthropic.Client, system, user string) (string, error) {
	msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     modelUnderTest,
		MaxTokens: 1024,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
	})
	if err != nil {
		return "", err
	}
	return textOf(msg), nil
}

// judge runs Opus 4.8 with a forced JSON-schema output. Thinking is OFF: the
// schema already constrains the answer, so thinking would only add tokens.
func judge(ctx context.Context, client *anthropic.Client, system, user string, schema map[string]any) (string, error) {
	msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     judgeModel,
		MaxTokens: 1024,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
		OutputConfig: anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		},
	})
	if err != nil {
		return "", err
	}
	text := strings.TrimSpace(textOf(msg))
	if text == "" {
		return "", fmt.Errorf("judge returned no text (stop_reason=%s)", msg.StopReason)
	}
	return text, nil
}

func textOf(msg *anthropic.Message) string {
	var b strings.Builder
	for _, block := range msg.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func objectSchema(properties map[string]any, required ...string) map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}
