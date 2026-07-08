package core

// LLM access for the validation pipeline (port of llm.rs + the Go sidecar).
//
// The Rust core shelled out to a long-lived Go sidecar (tools/driftguard-llm)
// and multiplexed newline-delimited JSON over its stdio, because the official
// Anthropic SDK is Go. In the Go rewrite that entire IPC layer disappears:
// the sidecar's request handlers become plain methods on AnthropicLlm, and
// concurrency is ordinary goroutines sharing one client (reused HTTP/TLS
// connections). The judge prompts and schemas below are copied VERBATIM from
// the sidecar — they are part of the validated judge methodology (spot-checked
// against hand labels in Phase 4); do not reword them casually.
//
// Everything downstream depends on the Llm interface, not on Anthropic
// directly, so validation stays testable without a network call or API key.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
)

// PassVerdict is the judge's pass/fail verdict for one eval run.
type PassVerdict struct {
	Passed        bool   `json:"passed"`
	Justification string `json:"justification"`
}

// BehaviorVerdict is the judge's before/after behavior-change verdict
// (Phase 4 ground truth).
type BehaviorVerdict struct {
	BehaviorChanged bool   `json:"behavior_changed"`
	Justification   string `json:"justification"`
}

// Llm is the model boundary for the validation pipeline.
type Llm interface {
	// Generate runs the model under test with the given system prompt.
	Generate(ctx context.Context, system, user string) (string, error)
	// JudgePass asks the judge whether a response satisfies the expected behavior.
	JudgePass(ctx context.Context, expectedBehavior, input, actualOutput string) (PassVerdict, error)
	// JudgeBehavior asks the judge whether behavior changed substantively
	// between the before and after responses.
	JudgeBehavior(ctx context.Context, input, expectedBehavior, before, after string) (BehaviorVerdict, error)
}

// AnthropicLlm implements Llm on the official SDK. Models are env-configurable
// (no rebuild to retune): the model under test should match your production
// app; the judge should be at least as capable. Judging uses structured output
// (output_config.format) with thinking intentionally OFF — the schema already
// forces a clean JSON verdict, so thinking tokens were wasted spend + latency.
type AnthropicLlm struct {
	client         anthropic.Client
	modelUnderTest string
	judgeModel     string
}

// NewAnthropicLlm constructs the client from the environment. Returns a clear
// error (not an opaque 401 later) when ANTHROPIC_API_KEY is missing.
func NewAnthropicLlm() (*AnthropicLlm, error) {
	if strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")) == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY is not set — needed for validate/run-evals/ci")
	}
	return &AnthropicLlm{
		client:         anthropic.NewClient(), // one shared client → reused TLS connections
		modelUnderTest: envOr("DRIFTGUARD_MODEL_UNDER_TEST", "claude-sonnet-4-6"),
		judgeModel:     envOr("DRIFTGUARD_JUDGE_MODEL", "claude-opus-4-8"),
	}, nil
}

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func (a *AnthropicLlm) Generate(ctx context.Context, system, user string) (string, error) {
	msg, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(a.modelUnderTest),
		MaxTokens: 1024,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
	})
	if err != nil {
		return "", err
	}
	return textOf(msg), nil
}

func (a *AnthropicLlm) JudgePass(ctx context.Context, expectedBehavior, input, actualOutput string) (PassVerdict, error) {
	system := "You are a strict, fair evaluator of AI responses. You are given a test " +
		"input, a description of the expected behavior, and a model's actual response. " +
		"Decide only whether the response satisfies the described expected behavior. Do not " +
		"impose criteria that are not stated. If the response clearly violates or omits a " +
		"required behavior, it fails; otherwise it passes."
	user := fmt.Sprintf("<expected_behavior>%s</expected_behavior>\n<input>%s</input>\n"+
		"<actual_response>%s</actual_response>\n\nDid the response satisfy the expected behavior?",
		expectedBehavior, input, actualOutput)
	schema := objectSchema(map[string]any{
		"passed":        map[string]any{"type": "boolean"},
		"justification": map[string]any{"type": "string"},
	}, "passed", "justification")

	var verdict PassVerdict
	err := a.judgeInto(ctx, system, user, schema, &verdict)
	return verdict, err
}

func (a *AnthropicLlm) JudgeBehavior(ctx context.Context, input, expectedBehavior, before, after string) (BehaviorVerdict, error) {
	system := "You are evaluating whether a prompt change altered a model's behavior on a " +
		"specific test case. You are given the test input and the model's response BEFORE and " +
		"AFTER the prompt change. Decide whether the behavior changed substantively — a " +
		"difference a user or downstream system would care about (different decision, different " +
		"content, materially different tone, added or removed information). Ignore trivial " +
		"wording differences that don't change meaning or outcome."
	user := fmt.Sprintf("<input>%s</input>\n<expected_behavior>%s</expected_behavior>\n"+
		"<response_before>%s</response_before>\n<response_after>%s</response_after>\n\n"+
		"Did the behavior change substantively between before and after?",
		input, expectedBehavior, before, after)
	schema := objectSchema(map[string]any{
		"behavior_changed": map[string]any{"type": "boolean"},
		"justification":    map[string]any{"type": "string"},
	}, "behavior_changed", "justification")

	var verdict BehaviorVerdict
	err := a.judgeInto(ctx, system, user, schema, &verdict)
	return verdict, err
}

// judgeInto runs the judge with a forced JSON-schema output and decodes the
// (schema-conforming) text into out.
func (a *AnthropicLlm) judgeInto(ctx context.Context, system, user string, schema map[string]any, out any) error {
	msg, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(a.judgeModel),
		MaxTokens: 1024,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
		OutputConfig: anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		},
	})
	if err != nil {
		return err
	}
	text := strings.TrimSpace(textOf(msg))
	if text == "" {
		return fmt.Errorf("judge returned no text (stop_reason=%s)", msg.StopReason)
	}
	if err := json.Unmarshal([]byte(text), out); err != nil {
		return fmt.Errorf("judge returned non-schema JSON: %w (%q)", err, text)
	}
	return nil
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
