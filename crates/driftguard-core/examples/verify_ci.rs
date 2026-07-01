//! Phase 5 offline plumbing test for the CI flow.
//!
//! Run: `cargo run -p driftguard-core --example verify_ci`
//!
//! Exercises `ci::run_ci` over a base→head fixtures pair with mock providers
//! (no API keys): a changed prompt is detected, its eval cases are selected,
//! run, and judged, and a failing case makes the report fail. Threshold 0.0 so
//! selection is deterministic (all cases selected) for the plumbing check.

use anyhow::{Result, ensure};
use driftguard_core::CoreError;
use driftguard_core::ci::{render_comment, run_ci};
use driftguard_core::db;
use driftguard_core::embed::{Embedder, InputType};
use driftguard_core::fixtures::{Fixtures, FixtureEvalCase, FixturePrompt};
use driftguard_core::llm::{BehaviorVerdict, Llm, PassVerdict};

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct MockLlm;

impl Llm for MockLlm {
    async fn generate(&self, _system: &str, user: &str) -> Result<String, CoreError> {
        Ok(format!("response to: {user}"))
    }
    async fn judge_pass(
        &self,
        _expected: &str,
        input: &str,
        _actual_output: &str,
    ) -> Result<PassVerdict, CoreError> {
        // Deterministic: any case whose input mentions "fail" fails.
        let passed = !input.to_lowercase().contains("fail");
        Ok(PassVerdict {
            passed,
            justification: "mock".into(),
        })
    }
    async fn judge_behavior(
        &self,
        _input: &str,
        _expected: &str,
        _before: &str,
        _after: &str,
    ) -> Result<BehaviorVerdict, CoreError> {
        Ok(BehaviorVerdict {
            behavior_changed: true,
            justification: "mock".into(),
        })
    }
}

struct MockEmbedder;

impl Embedder for MockEmbedder {
    fn dimension(&self) -> usize {
        1024
    }
    async fn embed(&self, inputs: &[String], _t: InputType) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(inputs
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 1024];
                for tok in t.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                    if !tok.is_empty() {
                        v[(fnv(tok) % 1024) as usize] += 1.0;
                    }
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

    let name = format!("_verify_ci_{}", std::process::id());
    let cases = vec![
        FixtureEvalCase { input: "greet the user".into(), expected_behavior: "greets".into() },
        FixtureEvalCase { input: "please fail this one".into(), expected_behavior: "handles it".into() },
        FixtureEvalCase { input: "cite the policy".into(), expected_behavior: "cites policy".into() },
    ];

    let base = Fixtures {
        prompts: vec![FixturePrompt {
            name: name.clone(),
            content: "You are a bot. Be brief.".into(),
            eval_cases: cases.clone(),
            edits: vec![],
        }],
    };
    // Head changes the prompt content.
    let head = Fixtures {
        prompts: vec![FixturePrompt {
            name: name.clone(),
            content: "You are a bot. Be brief and always cite the policy ID.".into(),
            eval_cases: cases,
            edits: vec![],
        }],
    };

    // Threshold 0.0 => all cases selected (deterministic plumbing check).
    let report = run_ci(&pool, &MockLlm, &MockEmbedder, &base, &head, 0.0, 4).await?;

    println!("{}", render_comment(&report));

    ensure!(report.changed == 1, "expected 1 changed prompt, got {}", report.changed);
    ensure!(report.total_selected == 3, "expected 3 selected, got {}", report.total_selected);
    ensure!(report.failed == 1, "expected 1 failed (the 'fail' case), got {}", report.failed);
    let comment = render_comment(&report);
    ensure!(comment.contains("❌ fail"), "comment should mark overall failure");

    sqlx::query("DELETE FROM prompts WHERE name = $1")
        .bind(&name)
        .execute(&pool)
        .await?;

    println!("\n✅ CI flow verified end-to-end (change detected, selected, judged, gated, cleaned up).");
    Ok(())
}
