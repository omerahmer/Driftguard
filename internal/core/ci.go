package core

// CI flow (port of ci.rs): given the BEFORE (base branch) and AFTER (PR)
// versions of a fixtures file, find which prompts changed, predict which eval
// cases each change affects, run ONLY those against the new prompt version,
// judge pass/fail, and produce a report + a PR-comment body.
//
// The git plumbing (extracting the base file) lives in the workflow, not here
// — this module is handed two parsed Fixtures and stays VCS-agnostic.

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/sync/errgroup"
)

type SelectedCase struct {
	EvalCaseID       uuid.UUID `json:"eval_case_id"`
	Input            string    `json:"input"`
	ExpectedBehavior string    `json:"expected_behavior"`
	Similarity       float64   `json:"similarity"`
	Passed           bool      `json:"passed"`
	Justification    string    `json:"justification"`
}

type PromptCiResult struct {
	Name        string         `json:"name"`
	EditSummary string         `json:"edit_summary"`
	TotalCases  int            `json:"total_cases"`
	Reduction   float64        `json:"reduction"`
	Selected    []SelectedCase `json:"selected"`
}

type CiReport struct {
	Threshold     float64          `json:"threshold"`
	Prompts       []PromptCiResult `json:"prompts"`
	TotalCases    int              `json:"total_cases"`
	TotalSelected int              `json:"total_selected"`
	Failed        int              `json:"failed"`
	// Prompts present in the PR but not the base (no baseline to diff against).
	SkippedNew []string `json:"skipped_new"`
	// Changed prompts whose diff has no meaningful changed span (e.g. whitespace).
	SkippedCosmetic []string `json:"skipped_cosmetic"`
	Changed         int      `json:"changed"`
}

// RunCi runs the CI flow over a base→head fixtures pair.
func RunCi(ctx context.Context, pool *pgxpool.Pool, llm Llm, embedder Embedder, base, head *Fixtures, threshold float64, concurrency int) (*CiReport, error) {
	baseContent := make(map[string]string, len(base.Prompts))
	for _, p := range base.Prompts {
		baseContent[p.Name] = p.Content
	}

	report := &CiReport{
		Threshold:       threshold,
		Prompts:         []PromptCiResult{},
		SkippedNew:      []string{},
		SkippedCosmetic: []string{},
	}

	for _, hp := range head.Prompts {
		before, ok := baseContent[hp.Name]
		if !ok {
			report.SkippedNew = append(report.SkippedNew, hp.Name)
			continue
		}
		if before == hp.Content {
			continue // unchanged
		}

		// A content change with no meaningful changed span (e.g. only
		// whitespace) has nothing to select on — skip rather than error.
		diff := ComputeDiff(before, hp.Content)
		if len(diff.ChangedSpans) == 0 {
			report.SkippedCosmetic = append(report.SkippedCosmetic, hp.Name)
			continue
		}
		report.Changed++

		// Load before/after into the (ephemeral) DB. Delete any prior prompt
		// of the same name so re-runs are clean.
		if _, err := pool.Exec(ctx, `DELETE FROM prompts WHERE name = $1`, hp.Name); err != nil {
			return nil, err
		}
		if _, err := CreatePrompt(ctx, pool, hp.Name); err != nil {
			return nil, err
		}
		baseVersion, err := CreateVersion(ctx, pool, hp.Name, before)
		if err != nil {
			return nil, err
		}
		for _, c := range hp.EvalCases {
			if _, err := CreateEvalCase(ctx, pool, hp.Name, c.Input, c.ExpectedBehavior); err != nil {
				return nil, err
			}
		}
		after, err := CreateVersionFromParent(ctx, pool,
			baseVersion.Version.PromptID, baseVersion.Version.ID, hp.Content)
		if err != nil {
			return nil, err
		}

		// Selection runs on the after version ("latest" == current == after).
		outcome, err := SelectEvals(ctx, pool, embedder, hp.Name, "latest", threshold)
		if err != nil {
			return nil, err
		}

		// Run + judge ONLY the selected cases against the new prompt.
		var mu selectedCollector
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(maxInt(concurrency, 1))
		for _, item := range outcome.Items {
			if !item.WasSelected {
				continue
			}
			g.Go(func() error {
				output, err := llm.Generate(gctx, hp.Content, item.Input)
				if err != nil {
					return err
				}
				verdict, err := llm.JudgePass(gctx, item.ExpectedBehavior, item.Input, output)
				if err != nil {
					return err
				}
				if _, err := pool.Exec(gctx,
					`INSERT INTO eval_runs (prompt_version_id, eval_case_id, actual_output,
					     judge_passed, judge_justification)
					 VALUES ($1, $2, $3, $4, $5)`,
					after.ID, item.EvalCaseID, output, verdict.Passed, verdict.Justification); err != nil {
					return err
				}
				mu.add(SelectedCase{
					EvalCaseID:       item.EvalCaseID,
					Input:            item.Input,
					ExpectedBehavior: item.ExpectedBehavior,
					Similarity:       item.Similarity,
					Passed:           verdict.Passed,
					Justification:    verdict.Justification,
				})
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return nil, err
		}
		selected := mu.take()

		for _, c := range selected {
			if !c.Passed {
				report.Failed++
			}
		}
		report.TotalCases += outcome.Total
		report.TotalSelected += len(selected)
		report.Prompts = append(report.Prompts, PromptCiResult{
			Name:        hp.Name,
			EditSummary: fmt.Sprintf("+%d −%d", diff.Stats.Added, diff.Stats.Removed),
			TotalCases:  outcome.Total,
			Reduction:   outcome.Reduction,
			Selected:    selected,
		})
	}

	return report, nil
}

// CommentMarker lets the workflow find and update a single sticky comment
// instead of stacking new ones.
const CommentMarker = "<!-- driftguard-ci -->"

// RenderComment renders a CI report as a Markdown PR comment.
func RenderComment(report *CiReport) string {
	var out strings.Builder
	out.WriteString(CommentMarker)
	out.WriteByte('\n')
	out.WriteString("## 🛡️ Driftguard — affected eval selection\n\n")

	if report.Changed == 0 {
		out.WriteString("No prompt content changes detected in the fixtures file.\n")
		if len(report.SkippedNew) > 0 {
			fmt.Fprintf(&out, "\n_New prompt(s) with no baseline (not evaluated): %s._\n",
				strings.Join(report.SkippedNew, ", "))
		}
		if len(report.SkippedCosmetic) > 0 {
			fmt.Fprintf(&out, "\n_Cosmetic-only change(s), nothing to select: %s._\n",
				strings.Join(report.SkippedCosmetic, ", "))
		}
		return out.String()
	}

	status := "✅ pass"
	if report.Failed > 0 {
		status = "❌ fail"
	}
	fmt.Fprintf(&out, "**%s** — ran **%d** of **%d** eval case(s) (threshold %.2f); **%d** failed.\n\n",
		status, report.TotalSelected, report.TotalCases, report.Threshold, report.Failed)

	for _, p := range report.Prompts {
		fmt.Fprintf(&out, "### `%s`  (%s, %d of %d selected, %.0f%% skipped)\n\n",
			p.Name, p.EditSummary, len(p.Selected), p.TotalCases, p.Reduction*100.0)
		if len(p.Selected) == 0 {
			out.WriteString("_No eval cases selected for this change._\n\n")
			continue
		}
		out.WriteString("| result | sim | eval case |\n|---|---|---|\n")
		for _, c := range p.Selected {
			mark := "✅"
			if !c.Passed {
				mark = "❌"
			}
			fmt.Fprintf(&out, "| %s | %.3f | %s |\n", mark, c.Similarity, truncate(c.ExpectedBehavior, 80))
		}
		out.WriteByte('\n')
	}

	if len(report.SkippedNew) > 0 {
		fmt.Fprintf(&out, "_New prompt(s) with no baseline (not evaluated): %s._\n",
			strings.Join(report.SkippedNew, ", "))
	}
	return out.String()
}

func truncate(text string, max int) string {
	flat := strings.ReplaceAll(text, "\n", " ")
	runes := []rune(flat)
	if len(runes) <= max {
		return flat
	}
	return string(runes[:max-1]) + "…"
}

// selectedCollector gathers results from concurrent goroutines (completion
// order, matching the Rust buffer_unordered semantics).
type selectedCollector struct {
	mu    sync.Mutex
	items []SelectedCase
}

func (c *selectedCollector) add(s SelectedCase) {
	c.mu.Lock()
	c.items = append(c.items, s)
	c.mu.Unlock()
}

func (c *selectedCollector) take() []SelectedCase {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.items == nil {
		return []SelectedCase{}
	}
	return c.items
}
