//! Phase 4 end-to-end plumbing test for the validation + scoring pipeline.
//!
//! Run: `cargo run -p driftguard-core --example verify_validation`
//!
//! Like verify_selection, this proves the WHOLE pipeline against the real DB
//! WITHOUT API keys, by swapping in a deterministic mock `Llm` and `Embedder`.
//! It checks the load → run-evals → judge → ground-truth → score path, including
//! that a reordering edit (which produces identical outputs) is labeled
//! "not affected" while a real edit is labeled "affected".
//!
//! This is a plumbing test, not a quality measurement — real numbers come from
//! Voyage + Anthropic via `validate` and `score`.

use anyhow::{Result, ensure};
use driftguard_core::CoreError;
use driftguard_core::db;
use driftguard_core::embed::{Embedder, InputType};
use driftguard_core::fixtures::{Fixtures, FixtureEdit, FixtureEvalCase, FixturePrompt, load_fixtures};
use driftguard_core::llm::{BehaviorVerdict, Llm, PassVerdict};
use driftguard_core::scoring::{default_thresholds, score};
use driftguard_core::validation::validate_fixtures;

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic mock model: generation depends on the *set* of prompt lines
/// (order-independent), so a reordering edit yields identical output (=> no
/// behavior change), while any real edit changes the line set (=> change).
struct MockLlm;

impl MockLlm {
    fn normalized_prompt(system: &str) -> String {
        let mut lines: Vec<&str> = system
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        lines.sort_unstable();
        lines.join("|")
    }
}

impl Llm for MockLlm {
    async fn generate(&self, system: &str, user: &str) -> Result<String, CoreError> {
        Ok(format!("[{}::{}]", fnv(&Self::normalized_prompt(system)), user))
    }

    async fn judge_pass(
        &self,
        _expected: &str,
        _input: &str,
        actual_output: &str,
    ) -> Result<PassVerdict, CoreError> {
        Ok(PassVerdict {
            passed: actual_output.len().is_multiple_of(2),
            justification: "mock".into(),
        })
    }

    async fn judge_behavior(
        &self,
        _input: &str,
        _expected: &str,
        before: &str,
        after: &str,
    ) -> Result<BehaviorVerdict, CoreError> {
        Ok(BehaviorVerdict {
            behavior_changed: before != after,
            justification: "mock".into(),
        })
    }
}

/// Deterministic 1024-dim hash embedder (same idea as verify_selection).
struct MockEmbedder;

impl Embedder for MockEmbedder {
    fn dimension(&self) -> usize {
        1024
    }
    async fn embed(
        &self,
        inputs: &[String],
        _t: InputType,
    ) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(inputs
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 1024];
                for tok in t.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                    if tok.is_empty() {
                        continue;
                    }
                    v[(fnv(tok) % 1024) as usize] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = db::connect(&url).await?;

    let name = format!("_verify_validation_{}", std::process::id());
    let base = "\
You are a test agent.
- Rule one: do the thing.
- Rule two: cite the policy.
- Rule three: be brief.";

    // One real edit (added rule => behavior changes) + one reorder control.
    let added = "\
You are a test agent.
- Rule one: do the thing.
- Rule two: cite the policy.
- Rule three: be brief.
- Rule four: always sign off politely.";
    let reordered = "\
You are a test agent.
- Rule three: be brief.
- Rule one: do the thing.
- Rule two: cite the policy.";

    let fixtures = Fixtures {
        prompts: vec![FixturePrompt {
            name: name.clone(),
            content: base.to_string(),
            eval_cases: vec![
                FixtureEvalCase { input: "case A".into(), expected_behavior: "does the thing".into() },
                FixtureEvalCase { input: "case B".into(), expected_behavior: "cites the policy".into() },
                FixtureEvalCase { input: "case C".into(), expected_behavior: "stays brief".into() },
            ],
            edits: vec![
                FixtureEdit { label: "added".into(), edit_type: "added".into(), content: added.to_string() },
                FixtureEdit { label: "reorder".into(), edit_type: "reorder".into(), content: reordered.to_string() },
            ],
        }],
    };

    load_fixtures(&pool, &fixtures, true).await?;
    let summary = validate_fixtures(&pool, &MockLlm, &MockEmbedder, &fixtures, 0.5, 4).await?;
    println!(
        "validated: {} prompt(s), {} edit(s), {} eval run(s), {} behavior diff(s)",
        summary.prompts, summary.edits, summary.eval_runs, summary.behavior_diffs
    );

    // Ground-truth labeling: 3 cases x 2 edits = 6 selection records.
    // The added edit -> 3 affected; the reorder control -> 0 affected.
    let affected: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM selection_records sr
         JOIN prompt_versions pv ON pv.id = sr.prompt_version_id
         JOIN prompts p ON p.id = pv.prompt_id
         WHERE p.name = $1 AND sr.was_actually_affected = true",
    )
    .bind(&name)
    .fetch_one(&pool)
    .await?;
    let not_affected: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM selection_records sr
         JOIN prompt_versions pv ON pv.id = sr.prompt_version_id
         JOIN prompts p ON p.id = pv.prompt_id
         WHERE p.name = $1 AND sr.was_actually_affected = false",
    )
    .bind(&name)
    .fetch_one(&pool)
    .await?;
    ensure!(affected == 3, "expected 3 affected cases (real edit), got {affected}");
    ensure!(
        not_affected == 3,
        "expected 3 not-affected cases (reorder control), got {not_affected}"
    );

    // Scoring runs over the labeled records.
    let report = score(&pool, Some(&name), &default_thresholds()).await?;
    ensure!(report.samples == 6, "expected 6 labeled samples, got {}", report.samples);
    let first = &report.rows[0];
    println!(
        "\nscore @ threshold {:.2}: behavior P {:.2} / R {:.2}, reduction {:.0}%  (samples={})",
        first.threshold,
        first.behavior.precision,
        first.behavior.recall,
        first.reduction * 100.0,
        report.samples
    );

    sqlx::query("DELETE FROM prompts WHERE name = $1")
        .bind(&name)
        .execute(&pool)
        .await?;

    println!("\n✅ validation + scoring pipeline verified end-to-end (ground truth labeled, scored, cleaned up).");
    Ok(())
}
