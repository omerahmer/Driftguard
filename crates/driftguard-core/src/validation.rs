//! Phase 4 validation pipeline: run the eval suite, judge it, and label ground
//! truth — the steps that turn fixtures into a measurable precision/recall result.
//!
//! Flow per edit (before = base v1, after = the edit's version):
//!   1. `run-evals` each version once (generate actual_output for every case).
//!   2. Judge Question A on every run (did it satisfy expected_behavior?).
//!   3. Judge Question B per case (did behavior change before→after?) → BehaviorDiff.
//!   4. Run similarity selection on the after-version (Phase 3).
//!   5. Label SelectionRecord.was_actually_affected = the case's behavior-changed
//!      verdict (the chosen ground truth).
//!
//! Calls are sequential (your decision): simplest, no rate-limit handling.

use serde::Serialize;
use uuid::Uuid;

use crate::db::Pool;
use crate::embed::Embedder;
use crate::error::CoreError;
use crate::evals::list_eval_cases;
use crate::fixtures::Fixtures;
use crate::llm::Llm;
use crate::models::{EvalCase, Prompt};
use crate::prompts::{get_prompt_by_name, resolve_version_ref};
use crate::selection;

#[derive(Debug, Default, Serialize)]
pub struct ValidationSummary {
    pub prompts: usize,
    pub edits: usize,
    pub versions_evaluated: usize,
    pub eval_runs: usize,
    pub behavior_diffs: usize,
}

/// Generate actual_output for every eval case against one version, overwriting
/// any prior runs for that version (idempotent re-runs — your decision).
pub async fn run_evals<L: Llm>(
    pool: &Pool,
    llm: &L,
    prompt_name: &str,
    version_ref: &str,
) -> Result<usize, CoreError> {
    let prompt = get_prompt_by_name(pool, prompt_name)
        .await?
        .ok_or_else(|| CoreError::PromptNotFound(prompt_name.to_string()))?;
    let version = resolve_version_ref(pool, &prompt, version_ref).await?;
    let cases = list_eval_cases(pool, prompt.id).await?;

    sqlx::query!(
        "DELETE FROM eval_runs WHERE prompt_version_id = $1",
        version.id
    )
    .execute(pool)
    .await?;

    for case in &cases {
        let output = llm.generate(&version.content, &case.input).await?;
        sqlx::query!(
            "INSERT INTO eval_runs (prompt_version_id, eval_case_id, actual_output)
             VALUES ($1, $2, $3)",
            version.id,
            case.id,
            output
        )
        .execute(pool)
        .await?;
    }
    Ok(cases.len())
}

/// Run the entire fixture matrix end to end and return a summary.
pub async fn validate_fixtures<L: Llm, E: Embedder>(
    pool: &Pool,
    llm: &L,
    embedder: &E,
    fixtures: &Fixtures,
    threshold: f64,
) -> Result<ValidationSummary, CoreError> {
    let mut summary = ValidationSummary::default();

    for fp in &fixtures.prompts {
        let prompt = get_prompt_by_name(pool, &fp.name).await?.ok_or_else(|| {
            CoreError::InvalidState(format!(
                "prompt \"{}\" is not loaded; run `fixtures load` first",
                fp.name
            ))
        })?;
        let cases = list_eval_cases(pool, prompt.id).await?;
        summary.prompts += 1;

        // Base version (v1): generate + judge A once.
        run_evals(pool, llm, &fp.name, "v1").await?;
        judge_pass_for_version(pool, llm, &prompt, &cases, "v1").await?;
        summary.versions_evaluated += 1;
        summary.eval_runs += cases.len();

        // Each edit branched off v1 → its version is v{index+2}.
        for (index, _edit) in fp.edits.iter().enumerate() {
            let after_ref = format!("v{}", index + 2);

            run_evals(pool, llm, &fp.name, &after_ref).await?;
            judge_pass_for_version(pool, llm, &prompt, &cases, &after_ref).await?;
            behavior_diffs_for_edit(pool, llm, &prompt, &cases, "v1", &after_ref).await?;

            // Phase 3 selection on the changed version, then label ground truth.
            selection::select_evals(pool, embedder, &fp.name, &after_ref, threshold).await?;
            label_ground_truth(pool, &prompt, &cases, "v1", &after_ref).await?;

            summary.edits += 1;
            summary.versions_evaluated += 1;
            summary.eval_runs += cases.len();
            summary.behavior_diffs += cases.len();
        }
    }

    Ok(summary)
}

/// Judge Question A for every eval run of one version.
async fn judge_pass_for_version<L: Llm>(
    pool: &Pool,
    llm: &L,
    prompt: &Prompt,
    cases: &[EvalCase],
    version_ref: &str,
) -> Result<(), CoreError> {
    let version = resolve_version_ref(pool, prompt, version_ref).await?;
    for case in cases {
        let output = fetch_output(pool, version.id, case.id).await?;
        let verdict = llm
            .judge_pass(&case.expected_behavior, &case.input, &output)
            .await?;
        sqlx::query!(
            "UPDATE eval_runs SET judge_passed = $1, judge_justification = $2
             WHERE prompt_version_id = $3 AND eval_case_id = $4",
            verdict.passed,
            verdict.justification,
            version.id,
            case.id
        )
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Judge Question B for an edit, writing one BehaviorDiff per case (overwriting
/// any prior diff for the same before/after/case triple).
async fn behavior_diffs_for_edit<L: Llm>(
    pool: &Pool,
    llm: &L,
    prompt: &Prompt,
    cases: &[EvalCase],
    before_ref: &str,
    after_ref: &str,
) -> Result<(), CoreError> {
    let before = resolve_version_ref(pool, prompt, before_ref).await?;
    let after = resolve_version_ref(pool, prompt, after_ref).await?;

    for case in cases {
        let before_output = fetch_output(pool, before.id, case.id).await?;
        let after_output = fetch_output(pool, after.id, case.id).await?;
        let verdict = llm
            .judge_behavior(
                &case.input,
                &case.expected_behavior,
                &before_output,
                &after_output,
            )
            .await?;

        sqlx::query!(
            "DELETE FROM behavior_diffs
             WHERE eval_case_id = $1 AND prompt_version_before_id = $2
               AND prompt_version_after_id = $3",
            case.id,
            before.id,
            after.id
        )
        .execute(pool)
        .await?;
        sqlx::query!(
            "INSERT INTO behavior_diffs
                 (eval_case_id, prompt_version_before_id, prompt_version_after_id,
                  judge_behavior_changed, judge_justification)
             VALUES ($1, $2, $3, $4, $5)",
            case.id,
            before.id,
            after.id,
            verdict.behavior_changed,
            verdict.justification
        )
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Copy each case's behavior-changed verdict onto its SelectionRecord as ground
/// truth (was_actually_affected).
async fn label_ground_truth(
    pool: &Pool,
    prompt: &Prompt,
    cases: &[EvalCase],
    before_ref: &str,
    after_ref: &str,
) -> Result<(), CoreError> {
    let before = resolve_version_ref(pool, prompt, before_ref).await?;
    let after = resolve_version_ref(pool, prompt, after_ref).await?;

    for case in cases {
        let changed = sqlx::query_scalar!(
            "SELECT judge_behavior_changed FROM behavior_diffs
             WHERE eval_case_id = $1 AND prompt_version_before_id = $2
               AND prompt_version_after_id = $3",
            case.id,
            before.id,
            after.id
        )
        .fetch_optional(pool)
        .await?
        .flatten();

        sqlx::query!(
            "UPDATE selection_records SET was_actually_affected = $1
             WHERE prompt_version_id = $2 AND eval_case_id = $3",
            changed,
            after.id,
            case.id
        )
        .execute(pool)
        .await?;
    }
    Ok(())
}

async fn fetch_output(pool: &Pool, version_id: Uuid, case_id: Uuid) -> Result<String, CoreError> {
    sqlx::query_scalar!(
        "SELECT actual_output FROM eval_runs
         WHERE prompt_version_id = $1 AND eval_case_id = $2",
        version_id,
        case_id
    )
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| CoreError::InvalidState("missing eval run (run-evals first)".to_string()))
}

// ---- judge-validate support (human spot-check of judge verdicts) ----

#[derive(Debug, Serialize)]
pub struct DiffSample {
    pub behavior_diff_id: Uuid,
    pub input: String,
    pub expected_behavior: String,
    pub before_output: String,
    pub after_output: String,
    pub judge_behavior_changed: Option<bool>,
    pub judge_justification: Option<String>,
}

/// Pull a random sample of judge verdicts that haven't been human-reviewed yet.
pub async fn sample_unvalidated_diffs(
    pool: &Pool,
    sample_size: i64,
) -> Result<Vec<DiffSample>, CoreError> {
    let rows = sqlx::query!(
        r#"SELECT bd.id AS "behavior_diff_id!",
                  ec.input AS "input!",
                  ec.expected_behavior AS "expected_behavior!",
                  before_run.actual_output AS "before_output!",
                  after_run.actual_output AS "after_output!",
                  bd.judge_behavior_changed,
                  bd.judge_justification
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
           LIMIT $1"#,
        sample_size
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| DiffSample {
            behavior_diff_id: r.behavior_diff_id,
            input: r.input,
            expected_behavior: r.expected_behavior,
            before_output: r.before_output,
            after_output: r.after_output,
            judge_behavior_changed: r.judge_behavior_changed,
            judge_justification: r.judge_justification,
        })
        .collect())
}

/// Record a human's agree/disagree with a judge verdict.
pub async fn set_human_agreed(
    pool: &Pool,
    behavior_diff_id: Uuid,
    agreed: bool,
) -> Result<(), CoreError> {
    sqlx::query!(
        "UPDATE behavior_diffs SET human_agreed = $1 WHERE id = $2",
        agreed,
        behavior_diff_id
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// (agreed, total reviewed) across all human-reviewed verdicts.
pub async fn judge_agreement(pool: &Pool) -> Result<(i64, i64), CoreError> {
    let row = sqlx::query!(
        r#"SELECT
             count(*) FILTER (WHERE human_agreed) AS "agreed!",
             count(*) FILTER (WHERE human_agreed IS NOT NULL) AS "total!"
           FROM behavior_diffs"#
    )
    .fetch_one(pool)
    .await?;
    Ok((row.agreed, row.total))
}
