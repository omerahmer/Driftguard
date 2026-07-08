package core

// Validation pipeline (port of validation.rs): run the eval suite, judge it,
// and label ground truth — the steps that turn fixtures into a measurable
// precision/recall result.
//
// Flow per edit (before = base v1, after = the edit's version):
//  1. run-evals each version once (generate actual_output for every case).
//  2. Judge Question A on every run (did it satisfy expected_behavior?).
//  3. Judge Question B per case (did behavior change before→after?).
//  4. Run similarity selection on the after-version.
//  5. Label selection_records.was_actually_affected with the behavior verdict.
//
// LLM calls are bounded-concurrent via errgroup (the Rust version used
// buffer_unordered over the sidecar; here it's plain goroutines).

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/sync/errgroup"
)

type ValidationSummary struct {
	Prompts           int `json:"prompts"`
	Edits             int `json:"edits"`
	VersionsEvaluated int `json:"versions_evaluated"`
	EvalRuns          int `json:"eval_runs"`
	BehaviorDiffs     int `json:"behavior_diffs"`
}

// RunEvals generates actual_output for every eval case against one version,
// overwriting any prior runs for that version. concurrency calls in flight.
func RunEvals(ctx context.Context, pool *pgxpool.Pool, llm Llm, promptName, versionRef string, concurrency int) (int, error) {
	prompt, err := GetPromptByName(ctx, pool, promptName)
	if err != nil {
		return 0, err
	}
	if prompt == nil {
		return 0, PromptNotFoundError{Name: promptName}
	}
	version, err := ResolveVersionRef(ctx, pool, prompt, versionRef)
	if err != nil {
		return 0, err
	}
	cases, err := ListEvalCases(ctx, pool, prompt.ID)
	if err != nil {
		return 0, err
	}

	if _, err := pool.Exec(ctx,
		`DELETE FROM eval_runs WHERE prompt_version_id = $1`, version.ID); err != nil {
		return 0, err
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(maxInt(concurrency, 1))
	for _, c := range cases {
		g.Go(func() error {
			output, err := llm.Generate(gctx, version.Content, c.Input)
			if err != nil {
				return err
			}
			_, err = pool.Exec(gctx,
				`INSERT INTO eval_runs (prompt_version_id, eval_case_id, actual_output)
				 VALUES ($1, $2, $3)`, version.ID, c.ID, output)
			return err
		})
	}
	if err := g.Wait(); err != nil {
		return 0, err
	}
	return len(cases), nil
}

// ValidateFixtures runs the entire fixture matrix end to end.
func ValidateFixtures(ctx context.Context, pool *pgxpool.Pool, llm Llm, embedder Embedder, fixtures *Fixtures, threshold float64, concurrency int) (*ValidationSummary, error) {
	summary := &ValidationSummary{}

	for _, fp := range fixtures.Prompts {
		prompt, err := GetPromptByName(ctx, pool, fp.Name)
		if err != nil {
			return nil, err
		}
		if prompt == nil {
			return nil, fmt.Errorf("prompt %q is not loaded; run `fixtures load` first", fp.Name)
		}
		cases, err := ListEvalCases(ctx, pool, prompt.ID)
		if err != nil {
			return nil, err
		}
		summary.Prompts++

		// Base version (v1): generate + judge A once.
		if _, err := RunEvals(ctx, pool, llm, fp.Name, "v1", concurrency); err != nil {
			return nil, err
		}
		if err := judgePassForVersion(ctx, pool, llm, prompt, cases, "v1", concurrency); err != nil {
			return nil, err
		}
		summary.VersionsEvaluated++
		summary.EvalRuns += len(cases)

		// Each edit branched off v1 → its version is v{index+2}.
		for index := range fp.Edits {
			afterRef := fmt.Sprintf("v%d", index+2)

			if _, err := RunEvals(ctx, pool, llm, fp.Name, afterRef, concurrency); err != nil {
				return nil, err
			}
			if err := judgePassForVersion(ctx, pool, llm, prompt, cases, afterRef, concurrency); err != nil {
				return nil, err
			}
			if err := behaviorDiffsForEdit(ctx, pool, llm, prompt, cases, "v1", afterRef, concurrency); err != nil {
				return nil, err
			}

			// Selection on the changed version, then label ground truth.
			if _, err := SelectEvals(ctx, pool, embedder, fp.Name, afterRef, threshold); err != nil {
				return nil, err
			}
			if err := labelGroundTruth(ctx, pool, prompt, cases, "v1", afterRef); err != nil {
				return nil, err
			}

			summary.Edits++
			summary.VersionsEvaluated++
			summary.EvalRuns += len(cases)
			summary.BehaviorDiffs += len(cases)
		}
	}

	return summary, nil
}

// judgePassForVersion judges Question A for every eval run of one version.
func judgePassForVersion(ctx context.Context, pool *pgxpool.Pool, llm Llm, prompt *Prompt, cases []EvalCase, versionRef string, concurrency int) error {
	version, err := ResolveVersionRef(ctx, pool, prompt, versionRef)
	if err != nil {
		return err
	}
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(maxInt(concurrency, 1))
	for _, c := range cases {
		g.Go(func() error {
			output, err := fetchOutput(gctx, pool, version.ID, c.ID)
			if err != nil {
				return err
			}
			verdict, err := llm.JudgePass(gctx, c.ExpectedBehavior, c.Input, output)
			if err != nil {
				return err
			}
			_, err = pool.Exec(gctx,
				`UPDATE eval_runs SET judge_passed = $1, judge_justification = $2
				 WHERE prompt_version_id = $3 AND eval_case_id = $4`,
				verdict.Passed, verdict.Justification, version.ID, c.ID)
			return err
		})
	}
	return g.Wait()
}

// behaviorDiffsForEdit judges Question B for an edit, writing one behavior
// diff per case (overwriting any prior diff for the same triple).
func behaviorDiffsForEdit(ctx context.Context, pool *pgxpool.Pool, llm Llm, prompt *Prompt, cases []EvalCase, beforeRef, afterRef string, concurrency int) error {
	before, err := ResolveVersionRef(ctx, pool, prompt, beforeRef)
	if err != nil {
		return err
	}
	after, err := ResolveVersionRef(ctx, pool, prompt, afterRef)
	if err != nil {
		return err
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(maxInt(concurrency, 1))
	for _, c := range cases {
		g.Go(func() error {
			beforeOutput, err := fetchOutput(gctx, pool, before.ID, c.ID)
			if err != nil {
				return err
			}
			afterOutput, err := fetchOutput(gctx, pool, after.ID, c.ID)
			if err != nil {
				return err
			}
			verdict, err := llm.JudgeBehavior(gctx, c.Input, c.ExpectedBehavior, beforeOutput, afterOutput)
			if err != nil {
				return err
			}

			if _, err := pool.Exec(gctx,
				`DELETE FROM behavior_diffs
				 WHERE eval_case_id = $1 AND prompt_version_before_id = $2
				   AND prompt_version_after_id = $3`, c.ID, before.ID, after.ID); err != nil {
				return err
			}
			_, err = pool.Exec(gctx,
				`INSERT INTO behavior_diffs
				     (eval_case_id, prompt_version_before_id, prompt_version_after_id,
				      judge_behavior_changed, judge_justification)
				 VALUES ($1, $2, $3, $4, $5)`,
				c.ID, before.ID, after.ID, verdict.BehaviorChanged, verdict.Justification)
			return err
		})
	}
	return g.Wait()
}

// labelGroundTruth copies each case's behavior-changed verdict onto its
// selection record as ground truth (was_actually_affected).
func labelGroundTruth(ctx context.Context, pool *pgxpool.Pool, prompt *Prompt, cases []EvalCase, beforeRef, afterRef string) error {
	before, err := ResolveVersionRef(ctx, pool, prompt, beforeRef)
	if err != nil {
		return err
	}
	after, err := ResolveVersionRef(ctx, pool, prompt, afterRef)
	if err != nil {
		return err
	}

	for _, c := range cases {
		var changed *bool
		err := pool.QueryRow(ctx,
			`SELECT judge_behavior_changed FROM behavior_diffs
			 WHERE eval_case_id = $1 AND prompt_version_before_id = $2
			   AND prompt_version_after_id = $3`, c.ID, before.ID, after.ID).Scan(&changed)
		if err != nil && err.Error() != "no rows in result set" {
			return err
		}

		if _, err := pool.Exec(ctx,
			`UPDATE selection_records SET was_actually_affected = $1
			 WHERE prompt_version_id = $2 AND eval_case_id = $3`,
			changed, after.ID, c.ID); err != nil {
			return err
		}
	}
	return nil
}

func fetchOutput(ctx context.Context, pool *pgxpool.Pool, versionID, caseID uuid.UUID) (string, error) {
	var output *string
	err := pool.QueryRow(ctx,
		`SELECT actual_output FROM eval_runs
		 WHERE prompt_version_id = $1 AND eval_case_id = $2`, versionID, caseID).Scan(&output)
	if err != nil || output == nil {
		return "", fmt.Errorf("missing eval run (run-evals first)")
	}
	return *output, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ---- judge-validate support (human spot-check of judge verdicts) ----

type DiffSample struct {
	BehaviorDiffID       uuid.UUID `json:"behavior_diff_id"`
	Input                string    `json:"input"`
	ExpectedBehavior     string    `json:"expected_behavior"`
	BeforeOutput         string    `json:"before_output"`
	AfterOutput          string    `json:"after_output"`
	JudgeBehaviorChanged *bool     `json:"judge_behavior_changed"`
	JudgeJustification   *string   `json:"judge_justification"`
}

// SampleUnvalidatedDiffs pulls a random sample of judge verdicts that haven't
// been human-reviewed yet.
func SampleUnvalidatedDiffs(ctx context.Context, pool *pgxpool.Pool, sampleSize int64) ([]DiffSample, error) {
	rows, err := pool.Query(ctx, `
		SELECT bd.id, ec.input, ec.expected_behavior,
		       before_run.actual_output, after_run.actual_output,
		       bd.judge_behavior_changed, bd.judge_justification
		FROM behavior_diffs bd
		JOIN eval_cases ec ON ec.id = bd.eval_case_id
		JOIN eval_runs before_run
		     ON before_run.prompt_version_id = bd.prompt_version_before_id
		    AND before_run.eval_case_id = bd.eval_case_id
		JOIN eval_runs after_run
		     ON after_run.prompt_version_id = bd.prompt_version_after_id
		    AND after_run.eval_case_id = bd.eval_case_id
		WHERE bd.human_agreed IS NULL
		ORDER BY random()
		LIMIT $1`, sampleSize)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	samples := []DiffSample{}
	for rows.Next() {
		var s DiffSample
		if err := rows.Scan(&s.BehaviorDiffID, &s.Input, &s.ExpectedBehavior,
			&s.BeforeOutput, &s.AfterOutput, &s.JudgeBehaviorChanged, &s.JudgeJustification); err != nil {
			return nil, err
		}
		samples = append(samples, s)
	}
	return samples, rows.Err()
}

// SetHumanAgreed records a human's agree/disagree with a judge verdict.
func SetHumanAgreed(ctx context.Context, pool *pgxpool.Pool, behaviorDiffID uuid.UUID, agreed bool) error {
	_, err := pool.Exec(ctx,
		`UPDATE behavior_diffs SET human_agreed = $1 WHERE id = $2`, agreed, behaviorDiffID)
	return err
}

// JudgeAgreement returns (agreed, total reviewed) across all human-reviewed
// verdicts.
func JudgeAgreement(ctx context.Context, pool *pgxpool.Pool) (int64, int64, error) {
	var agreed, total int64
	err := pool.QueryRow(ctx, `
		SELECT count(*) FILTER (WHERE human_agreed),
		       count(*) FILTER (WHERE human_agreed IS NOT NULL)
		FROM behavior_diffs`).Scan(&agreed, &total)
	return agreed, total, err
}
