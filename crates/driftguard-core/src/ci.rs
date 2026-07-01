//! Phase 5 CI flow: given the BEFORE (base branch) and AFTER (PR) versions of a
//! fixtures file, find which prompts changed, predict which eval cases each
//! change affects, run ONLY those against the new prompt version, judge
//! pass/fail, and produce a report + a PR-comment body.
//!
//! This is the "run only the relevant tests" payoff as a PR gate. It's a
//! deliberately lighter flow than the offline research pipeline (`validation`):
//! no before/after behavior-diff, just select → run selected → pass/fail.
//!
//! The git plumbing (extracting the base file) lives in the workflow, not here —
//! this module is handed two parsed `Fixtures` and stays VCS-agnostic.

use std::collections::HashMap;

use futures::stream::{self, StreamExt, TryStreamExt};
use serde::Serialize;
use uuid::Uuid;

use crate::db::Pool;
use crate::embed::Embedder;
use crate::error::CoreError;
use crate::evals::create_eval_case;
use crate::fixtures::Fixtures;
use crate::llm::Llm;
use crate::prompts::{create_prompt, create_version, create_version_from_parent};
use crate::segment::compute_diff;
use crate::selection::select_evals;

#[derive(Debug, Serialize)]
pub struct SelectedCase {
    pub eval_case_id: Uuid,
    pub input: String,
    pub expected_behavior: String,
    pub similarity: f64,
    pub passed: bool,
    pub justification: String,
}

#[derive(Debug, Serialize)]
pub struct PromptCiResult {
    pub name: String,
    pub edit_summary: String,
    pub total_cases: usize,
    pub reduction: f64,
    pub selected: Vec<SelectedCase>,
}

#[derive(Debug, Serialize)]
pub struct CiReport {
    pub threshold: f64,
    pub prompts: Vec<PromptCiResult>,
    pub total_cases: usize,
    pub total_selected: usize,
    pub failed: usize,
    /// Prompts present in the PR but not the base (no baseline to diff against).
    pub skipped_new: Vec<String>,
    /// Changed prompts whose diff has no meaningful changed span (e.g. whitespace).
    pub skipped_cosmetic: Vec<String>,
    pub changed: usize,
}

/// Run the CI flow over a base→head fixtures pair.
pub async fn run_ci<L: Llm, E: Embedder>(
    pool: &Pool,
    llm: &L,
    embedder: &E,
    base: &Fixtures,
    head: &Fixtures,
    threshold: f64,
    concurrency: usize,
) -> Result<CiReport, CoreError> {
    let base_content: HashMap<&str, &str> = base
        .prompts
        .iter()
        .map(|p| (p.name.as_str(), p.content.as_str()))
        .collect();

    let mut report = CiReport {
        threshold,
        prompts: Vec::new(),
        total_cases: 0,
        total_selected: 0,
        failed: 0,
        skipped_new: Vec::new(),
        skipped_cosmetic: Vec::new(),
        changed: 0,
    };

    for hp in &head.prompts {
        let before = match base_content.get(hp.name.as_str()) {
            None => {
                report.skipped_new.push(hp.name.clone());
                continue;
            }
            Some(&b) if b == hp.content => continue, // unchanged
            Some(&b) => b,
        };

        // A content change with no meaningful changed span (e.g. only whitespace)
        // has nothing to select on — skip rather than error.
        let diff = compute_diff(before, &hp.content);
        if diff.changed_spans.is_empty() {
            report.skipped_cosmetic.push(hp.name.clone());
            continue;
        }
        report.changed += 1;

        // Load before/after into the (ephemeral) DB. Delete any prior prompt of
        // the same name so re-runs are clean.
        sqlx::query!("DELETE FROM prompts WHERE name = $1", hp.name)
            .execute(pool)
            .await?;
        create_prompt(pool, &hp.name).await?;
        let base_version = create_version(pool, &hp.name, before).await?;
        for case in &hp.eval_cases {
            create_eval_case(pool, &hp.name, &case.input, &case.expected_behavior).await?;
        }
        let after = create_version_from_parent(
            pool,
            base_version.version.prompt_id,
            base_version.version.id,
            &hp.content,
        )
        .await?;

        // Selection runs on the after version ("latest" == current == after).
        let outcome = select_evals(pool, embedder, &hp.name, "latest", threshold).await?;

        // Run + judge ONLY the selected cases against the new prompt (bounded-concurrent).
        let selected: Vec<SelectedCase> = stream::iter(
            outcome.items.iter().filter(|i| i.was_selected).map(|item| {
                let after = &after;
                async move {
                    let output = llm.generate(&hp.content, &item.input).await?;
                    let verdict = llm
                        .judge_pass(&item.expected_behavior, &item.input, &output)
                        .await?;
                    sqlx::query!(
                        "INSERT INTO eval_runs (prompt_version_id, eval_case_id, actual_output,
                             judge_passed, judge_justification)
                         VALUES ($1, $2, $3, $4, $5)",
                        after.id,
                        item.eval_case_id,
                        output,
                        verdict.passed,
                        verdict.justification
                    )
                    .execute(pool)
                    .await?;
                    Ok::<SelectedCase, CoreError>(SelectedCase {
                        eval_case_id: item.eval_case_id,
                        input: item.input.clone(),
                        expected_behavior: item.expected_behavior.clone(),
                        similarity: item.similarity,
                        passed: verdict.passed,
                        justification: verdict.justification,
                    })
                }
            }),
        )
        .buffer_unordered(concurrency.max(1))
        .try_collect()
        .await?;

        report.failed += selected.iter().filter(|c| !c.passed).count();
        report.total_cases += outcome.total;
        report.total_selected += selected.len();
        report.prompts.push(PromptCiResult {
            name: hp.name.clone(),
            edit_summary: format!("+{} −{}", diff.stats.added, diff.stats.removed),
            total_cases: outcome.total,
            reduction: outcome.reduction,
            selected,
        });
    }

    Ok(report)
}

/// Render a CI report as a Markdown PR comment. The leading marker lets the
/// workflow find and update a single sticky comment instead of stacking new ones.
pub const COMMENT_MARKER: &str = "<!-- driftguard-ci -->";

pub fn render_comment(report: &CiReport) -> String {
    let mut out = String::new();
    out.push_str(COMMENT_MARKER);
    out.push('\n');
    out.push_str("## 🛡️ Driftguard — affected eval selection\n\n");

    if report.changed == 0 {
        out.push_str("No prompt content changes detected in the fixtures file.\n");
        if !report.skipped_new.is_empty() {
            out.push_str(&format!(
                "\n_New prompt(s) with no baseline (not evaluated): {}._\n",
                report.skipped_new.join(", ")
            ));
        }
        if !report.skipped_cosmetic.is_empty() {
            out.push_str(&format!(
                "\n_Cosmetic-only change(s), nothing to select: {}._\n",
                report.skipped_cosmetic.join(", ")
            ));
        }
        return out;
    }

    let status = if report.failed == 0 { "✅ pass" } else { "❌ fail" };
    out.push_str(&format!(
        "**{status}** — ran **{}** of **{}** eval case(s) (threshold {:.2}); **{}** failed.\n\n",
        report.total_selected, report.total_cases, report.threshold, report.failed
    ));

    for p in &report.prompts {
        out.push_str(&format!(
            "### `{}`  ({}, {} of {} selected, {:.0}% skipped)\n\n",
            p.name,
            p.edit_summary,
            p.selected.len(),
            p.total_cases,
            p.reduction * 100.0
        ));
        if p.selected.is_empty() {
            out.push_str("_No eval cases selected for this change._\n\n");
            continue;
        }
        out.push_str("| result | sim | eval case |\n|---|---|---|\n");
        for c in &p.selected {
            let mark = if c.passed { "✅" } else { "❌" };
            out.push_str(&format!(
                "| {} | {:.3} | {} |\n",
                mark,
                c.similarity,
                truncate(&c.expected_behavior, 80)
            ));
        }
        out.push('\n');
    }

    if !report.skipped_new.is_empty() {
        out.push_str(&format!(
            "_New prompt(s) with no baseline (not evaluated): {}._\n",
            report.skipped_new.join(", ")
        ));
    }
    out
}

fn truncate(text: &str, max: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= max {
        flat
    } else {
        let head: String = flat.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}
