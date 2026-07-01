// driftguard-llm is a thin sidecar that performs Driftguard's Anthropic calls
// using the OFFICIAL anthropic-sdk-go. The Rust core shells out to this binary
// (one process per call), passing a JSON request on stdin and reading a JSON
// response on stdout. This keeps the LLM client on the officially-maintained
// SDK while the rest of Driftguard stays in Rust, coupled only by the JSON
// contract below.
//
// Models (Driftguard's decisions): generation uses the model under test
// (Sonnet 4.6); judging uses Opus 4.8 with adaptive thinking + structured output
// (output_config.format JSON schema). `anthropic.Model` is a string alias, so we
// pass the exact IDs directly and don't depend on per-version model constants.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	modelUnderTest = "claude-sonnet-4-6"
	judgeModel     = "claude-opus-4-8"
)

// request is the JSON contract from the Rust side. `op` selects the operation.
type request struct {
	Op               string `json:"op"`
	System           string `json:"system"`
	User             string `json:"user"`
	ExpectedBehavior string `json:"expected_behavior"`
	Input            string `json:"input"`
	ActualOutput     string `json:"actual_output"`
	Before           string `json:"before"`
	After            string `json:"after"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		return fmt.Errorf("decoding request: %w", err)
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return fmt.Errorf("ANTHROPIC_API_KEY is not set")
	}

	client := anthropic.NewClient() // reads ANTHROPIC_API_KEY from the env
	ctx := context.Background()

	switch req.Op {
	case "generate":
		text, err := generate(ctx, &client, req.System, req.User)
		if err != nil {
			return err
		}
		return emit(map[string]any{"text": text})

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
		text, err := judge(ctx, &client, system, user, schema)
		if err != nil {
			return err
		}
		return emitRaw(text)

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
		text, err := judge(ctx, &client, system, user, schema)
		if err != nil {
			return err
		}
		return emitRaw(text)

	default:
		return fmt.Errorf("unknown op %q", req.Op)
	}
}

// generate runs the model under test (Sonnet): system = prompt being tested.
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

// judge runs Opus 4.8 with adaptive thinking and a forced JSON-schema output.
// The returned string is the raw verdict JSON (already matching the schema).
func judge(ctx context.Context, client *anthropic.Client, system, user string, schema map[string]any) (string, error) {
	msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     judgeModel,
		MaxTokens: 4096,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
		Thinking: anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		},
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

func emit(v map[string]any) error {
	return json.NewEncoder(os.Stdout).Encode(v)
}

// emitRaw writes an already-serialized JSON object straight through.
func emitRaw(jsonText string) error {
	_, err := fmt.Fprintln(os.Stdout, jsonText)
	return err
}
