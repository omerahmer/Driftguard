package core

import (
	"context"
	"math"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Phase 4 scoring: the project's headline result.
//
// For each threshold in a sweep, "similarity_score >= threshold" is the
// selection's prediction, compared against ground truth to report precision,
// recall, and % reduction in eval cases run.
//
// Two ground-truth views per threshold keep the methodology transparent:
//   - behavior — the chosen ground truth: the judge said behavior changed
//     (selection_records.was_actually_affected).
//   - flip — a secondary view: the case's pass/fail verdict flipped between
//     before and after (closer to what a CI gate cares about).
//
// Reduction is ground-truth-independent: the fraction of cases NOT selected.

// DefaultThresholds returns the sweep 0.30..0.90 step 0.05 (13 points) — a
// wider low end than the spec's 0.5 so the curve isn't empty if real cosine
// scores run low.
func DefaultThresholds() []float64 {
	var t []float64
	for x := 0.30; x <= 0.9001; x += 0.05 {
		t = append(t, math.Round(x*100)/100)
	}
	return t
}

// Metrics' JSON keys preserve the Rust wire contract — including "fn_", which
// exists because `fn` is a Rust keyword. The dashboard types depend on it.
type Metrics struct {
	TP        int     `json:"tp"`
	FP        int     `json:"fp"`
	FN        int     `json:"fn_"`
	TN        int     `json:"tn"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
}

type ThresholdRow struct {
	Threshold float64 `json:"threshold"`
	// Fraction of cases NOT selected (same for both ground truths).
	Reduction float64 `json:"reduction"`
	Behavior  Metrics `json:"behavior"`
	Flip      Metrics `json:"flip"`
}

type ScoreReport struct {
	Prompt  *string        `json:"prompt"`
	Samples int            `json:"samples"`
	Rows    []ThresholdRow `json:"rows"`
}

// sample: one labeled selection record — a similarity score plus both
// ground-truth bools.
type sample struct {
	similarity      float64
	behaviorChanged bool
	passFlipped     bool
}

// Score computes the threshold sweep. prompt (nil for all) filters to one prompt.
func Score(ctx context.Context, pool *pgxpool.Pool, prompt *string, thresholds []float64) (*ScoreReport, error) {
	// One row per labeled SelectionRecord. pass_flipped compares the
	// after-version's judge_passed to the before-version's (the after
	// version's parent), via two eval_runs joins.
	rows, err := pool.Query(ctx, `
		SELECT sr.similarity_score AS similarity,
		       sr.was_actually_affected AS behavior_changed,
		       (after_run.judge_passed IS DISTINCT FROM before_run.judge_passed) AS pass_flipped
		FROM selection_records sr
		JOIN prompt_versions pv ON pv.id = sr.prompt_version_id
		JOIN prompts p ON p.id = pv.prompt_id
		LEFT JOIN eval_runs after_run
		     ON after_run.prompt_version_id = sr.prompt_version_id
		    AND after_run.eval_case_id = sr.eval_case_id
		LEFT JOIN eval_runs before_run
		     ON before_run.prompt_version_id = pv.parent_version_id
		    AND before_run.eval_case_id = sr.eval_case_id
		WHERE sr.was_actually_affected IS NOT NULL
		  AND ($1::text IS NULL OR p.name = $1)
	`, prompt)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var samples []sample
	for rows.Next() {
		var similarity float64
		var behaviorChanged bool
		var passFlipped *bool
		if err := rows.Scan(&similarity, &behaviorChanged, &passFlipped); err != nil {
			return nil, err
		}
		samples = append(samples, sample{
			similarity:      similarity,
			behaviorChanged: behaviorChanged,
			// A null flip (e.g. an unjudged run) counts as "not flipped".
			passFlipped: passFlipped != nil && *passFlipped,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	total := len(samples)
	reportRows := make([]ThresholdRow, 0, len(thresholds))

	for _, threshold := range thresholds {
		var behavior, flip counts
		selected := 0

		for _, s := range samples {
			predicted := s.similarity >= threshold
			if predicted {
				selected++
			}
			behavior.add(predicted, s.behaviorChanged)
			flip.add(predicted, s.passFlipped)
		}

		reduction := 0.0
		if total > 0 {
			reduction = 1.0 - float64(selected)/float64(total)
		}

		reportRows = append(reportRows, ThresholdRow{
			Threshold: threshold,
			Reduction: reduction,
			Behavior:  behavior.metrics(),
			Flip:      flip.metrics(),
		})
	}

	return &ScoreReport{Prompt: prompt, Samples: total, Rows: reportRows}, nil
}

type counts struct{ tp, fp, fn, tn int }

func (c *counts) add(predicted, actual bool) {
	switch {
	case predicted && actual:
		c.tp++
	case predicted && !actual:
		c.fp++
	case !predicted && actual:
		c.fn++
	default:
		c.tn++
	}
}

func (c counts) metrics() Metrics {
	// Precision/recall are defined as 1.0 when their denominator is zero
	// (no predictions → vacuously precise; no positives → vacuously complete),
	// a common convention that keeps the sweep table readable.
	precision := 1.0
	if c.tp+c.fp > 0 {
		precision = float64(c.tp) / float64(c.tp+c.fp)
	}
	recall := 1.0
	if c.tp+c.fn > 0 {
		recall = float64(c.tp) / float64(c.tp+c.fn)
	}
	return Metrics{TP: c.tp, FP: c.fp, FN: c.fn, TN: c.tn, Precision: precision, Recall: recall}
}
