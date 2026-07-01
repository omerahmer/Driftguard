//! Phase 4 scoring: the project's headline result.
//!
//! For each threshold in a sweep, we treat "similarity_score >= threshold" as
//! the selection's prediction and compare it to ground truth, reporting
//! precision, recall, and % reduction in eval cases run.
//!
//! We report TWO ground-truth views per threshold so the methodology is
//! transparent:
//!   - `behavior` — the chosen ground truth: the judge said behavior changed
//!     (SelectionRecord.was_actually_affected).
//!   - `flip` — a secondary view: the case's pass/fail verdict flipped between
//!     before and after (closer to what a CI gate cares about).
//!
//! Reduction is ground-truth-independent: the fraction of cases NOT selected.

use serde::Serialize;

use crate::db::Pool;
use crate::error::CoreError;

/// Default sweep: 0.30..=0.90 step 0.05 (13 points) — your decision. Wider low
/// end than the spec's 0.5 so the curve isn't empty if real cosine scores run low.
pub fn default_thresholds() -> Vec<f64> {
    let mut t = Vec::new();
    let mut x = 0.30_f64;
    while x <= 0.9001 {
        t.push((x * 100.0).round() / 100.0);
        x += 0.05;
    }
    t
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdRow {
    pub threshold: f64,
    /// Fraction of cases NOT selected (same for both ground truths).
    pub reduction: f64,
    pub behavior: Metrics,
    pub flip: Metrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreReport {
    pub prompt: Option<String>,
    pub samples: usize,
    pub rows: Vec<ThresholdRow>,
}

/// One labeled selection sample: a similarity score plus both ground-truth bools.
struct Sample {
    similarity: f64,
    behavior_changed: bool,
    pass_flipped: bool,
}

/// Compute the threshold sweep. `prompt` optionally filters to one prompt.
pub async fn score(
    pool: &Pool,
    prompt: Option<&str>,
    thresholds: &[f64],
) -> Result<ScoreReport, CoreError> {
    // Pull one row per labeled SelectionRecord. `pass_flipped` compares the
    // after-version's judge_passed to the before-version's (the after version's
    // parent), via two eval_runs joins. Runtime query (optional filter + the
    // IS DISTINCT FROM boolean are awkward for the compile-time macro).
    let rows = sqlx::query_as::<_, (f64, bool, Option<bool>)>(
        r#"
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
        "#,
    )
    .bind(prompt)
    .fetch_all(pool)
    .await?;

    let samples: Vec<Sample> = rows
        .into_iter()
        .map(|(similarity, behavior_changed, pass_flipped)| Sample {
            similarity,
            behavior_changed,
            // A null flip (e.g. an unjudged run) counts as "not flipped".
            pass_flipped: pass_flipped.unwrap_or(false),
        })
        .collect();

    let total = samples.len();
    let mut report_rows = Vec::with_capacity(thresholds.len());

    for &threshold in thresholds {
        let mut behavior = Counts::default();
        let mut flip = Counts::default();
        let mut selected = 0usize;

        for s in &samples {
            let predicted = s.similarity >= threshold;
            if predicted {
                selected += 1;
            }
            behavior.add(predicted, s.behavior_changed);
            flip.add(predicted, s.pass_flipped);
        }

        let reduction = if total == 0 {
            0.0
        } else {
            1.0 - (selected as f64 / total as f64)
        };

        report_rows.push(ThresholdRow {
            threshold,
            reduction,
            behavior: behavior.metrics(),
            flip: flip.metrics(),
        });
    }

    Ok(ScoreReport {
        prompt: prompt.map(str::to_string),
        samples: total,
        rows: report_rows,
    })
}

#[derive(Default)]
struct Counts {
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
}

impl Counts {
    fn add(&mut self, predicted: bool, actual: bool) {
        match (predicted, actual) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, true) => self.fn_ += 1,
            (false, false) => self.tn += 1,
        }
    }

    fn metrics(self) -> Metrics {
        // Precision/recall are defined as 1.0 when their denominator is zero
        // (no predictions → vacuously precise; no positives → vacuously complete),
        // a common convention that keeps the sweep table readable.
        let precision = if self.tp + self.fp == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        };
        let recall = if self.tp + self.fn_ == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fn_) as f64
        };
        Metrics {
            tp: self.tp,
            fp: self.fp,
            fn_: self.fn_,
            tn: self.tn,
            precision,
            recall,
        }
    }
}
