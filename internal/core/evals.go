package core

import (
	"context"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Eval-case services. Read paths only for now; bulk creation arrives with the
// fixtures loader port (Phase G5).

// ListEvalCases returns a prompt's eval cases, oldest first.
func ListEvalCases(ctx context.Context, pool *pgxpool.Pool, promptID uuid.UUID) ([]EvalCase, error) {
	rows, err := pool.Query(ctx,
		`SELECT id, prompt_id, input, expected_behavior, created_at
		 FROM eval_cases WHERE prompt_id = $1 ORDER BY created_at ASC`, promptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cases := []EvalCase{}
	for rows.Next() {
		var c EvalCase
		if err := rows.Scan(&c.ID, &c.PromptID, &c.Input, &c.ExpectedBehavior, &c.CreatedAt); err != nil {
			return nil, err
		}
		c.CreatedAt = c.CreatedAt.UTC() // wire contract is UTC (see prompts.go)
		cases = append(cases, c)
	}
	return cases, rows.Err()
}

// ListEvalRuns returns all eval runs recorded for one prompt version (the read
// side for the API).
func ListEvalRuns(ctx context.Context, pool *pgxpool.Pool, versionID uuid.UUID) ([]EvalRun, error) {
	rows, err := pool.Query(ctx,
		`SELECT id, prompt_version_id, eval_case_id, actual_output,
		        judge_passed, judge_justification, created_at
		 FROM eval_runs WHERE prompt_version_id = $1 ORDER BY created_at ASC`, versionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	runs := []EvalRun{}
	for rows.Next() {
		var r EvalRun
		if err := rows.Scan(&r.ID, &r.PromptVersionID, &r.EvalCaseID, &r.ActualOutput,
			&r.JudgePassed, &r.JudgeJustification, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.CreatedAt = r.CreatedAt.UTC() // wire contract is UTC (see prompts.go)
		runs = append(runs, r)
	}
	return runs, rows.Err()
}

// CreateEvalCase adds one eval case to a prompt by name (used by the fixtures
// loader in bulk; the CLI also exposes a one-off `eval add`).
func CreateEvalCase(ctx context.Context, pool *pgxpool.Pool, promptName, input, expectedBehavior string) (*EvalCase, error) {
	prompt, err := GetPromptByName(ctx, pool, promptName)
	if err != nil {
		return nil, err
	}
	if prompt == nil {
		return nil, PromptNotFoundError{Name: promptName}
	}
	return CreateEvalCaseForPrompt(ctx, pool, prompt.ID, input, expectedBehavior)
}

// CreateEvalCaseForPrompt adds one eval case to a prompt by id. The by-id form
// is what the dashboard write path uses (it already has the prompt id from the
// registry); CreateEvalCase is the by-name convenience over it.
func CreateEvalCaseForPrompt(ctx context.Context, pool *pgxpool.Pool, promptID uuid.UUID, input, expectedBehavior string) (*EvalCase, error) {
	var c EvalCase
	err := pool.QueryRow(ctx,
		`INSERT INTO eval_cases (prompt_id, input, expected_behavior)
		 VALUES ($1, $2, $3)
		 RETURNING id, prompt_id, input, expected_behavior, created_at`,
		promptID, input, expectedBehavior).
		Scan(&c.ID, &c.PromptID, &c.Input, &c.ExpectedBehavior, &c.CreatedAt)
	if err != nil {
		return nil, err
	}
	c.CreatedAt = c.CreatedAt.UTC()
	return &c, nil
}
