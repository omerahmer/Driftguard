package core

import (
	"time"

	"github.com/google/uuid"
)

// Domain row types, scanned from Postgres by pgx.
//
// JSON tags are snake_case to preserve the exact wire contract the Rust API
// established (dashboard/lib/api.ts depends on these keys). Nullable columns
// are pointers so they marshal as JSON null, matching serde's Option.

type Prompt struct {
	ID               uuid.UUID  `json:"id"`
	Name             string     `json:"name"`
	CurrentVersionID *uuid.UUID `json:"current_version_id"`
	CreatedAt        time.Time  `json:"created_at"`
}

type PromptVersion struct {
	ID              uuid.UUID  `json:"id"`
	PromptID        uuid.UUID  `json:"prompt_id"`
	Content         string     `json:"content"`
	ParentVersionID *uuid.UUID `json:"parent_version_id"`
	// Serialized PromptDiff (JSON) vs. the parent version; nil for the first
	// version of a prompt.
	DiffFromParent *string   `json:"diff_from_parent"`
	CreatedAt      time.Time `json:"created_at"`
}

// EvalCase: a test input plus a natural-language description of what a correct
// response must do. The embedding column (expected_behavior_embedding) is
// intentionally NOT a field here — it's written/read only via raw SQL in the
// selection path, so the typed model stays free of the pgvector dependency.
type EvalCase struct {
	ID               uuid.UUID `json:"id"`
	PromptID         uuid.UUID `json:"prompt_id"`
	Input            string    `json:"input"`
	ExpectedBehavior string    `json:"expected_behavior"`
	CreatedAt        time.Time `json:"created_at"`
}

// EvalRun: the model-under-test's output for one case against one version,
// plus the judge's verdict (nil until judged).
type EvalRun struct {
	ID                 uuid.UUID `json:"id"`
	PromptVersionID    uuid.UUID `json:"prompt_version_id"`
	EvalCaseID         uuid.UUID `json:"eval_case_id"`
	ActualOutput       string    `json:"actual_output"`
	JudgePassed        *bool     `json:"judge_passed"`
	JudgeJustification *string   `json:"judge_justification"`
	CreatedAt          time.Time `json:"created_at"`
}
