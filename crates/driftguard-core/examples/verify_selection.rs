//! Phase 3 end-to-end plumbing test for similarity selection.
//!
//! Run: `cargo run -p driftguard-core --example verify_selection`
//!
//! This exercises the WHOLE selection path against the real database —
//! embedding storage in the pgvector columns, the cosine query, and
//! SelectionRecord persistence — WITHOUT needing a Voyage API key. It does that
//! by swapping in a deterministic local `Embedder` (a bag-of-words hash). That's
//! exactly why the selection code is generic over `Embedder`: the plumbing is
//! verifiable offline.
//!
//! NOTE: this is a *plumbing* test, not a quality measurement. The hash embedder
//! is not a real semantic model — real precision/recall (Phase 4) uses Voyage.

use anyhow::{Result, ensure};
use driftguard_core::CoreError;
use driftguard_core::db;
use driftguard_core::embed::{Embedder, InputType};
use driftguard_core::evals::create_eval_case;
use driftguard_core::prompts::{create_prompt, create_version};
use driftguard_core::selection::select_evals;

/// Deterministic, network-free embedder: hashes each token into one of 1024
/// buckets and L2-normalizes. Identical text → identical vector (cosine 1);
/// texts sharing tokens score higher than unrelated ones. Enough to prove the
/// pipeline ranks and persists correctly.
struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn dimension(&self) -> usize {
        1024
    }

    async fn embed(
        &self,
        inputs: &[String],
        _input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(inputs.iter().map(|t| hash_embed(t)).collect())
    }
}

fn hash_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; 1024];
    for token in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
    {
        // FNV-1a hash → bucket index.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in token.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        v[(h % 1024) as usize] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = db::connect(&url).await?;

    // Unique name so repeated runs don't collide; cleaned up at the end.
    let name = format!("_verify_selection_{}", std::process::id());
    let embedder = HashEmbedder;

    // Build a prompt with a parent->child diff (v2's changed span is what we embed).
    create_prompt(&pool, &name).await?;
    create_version(
        &pool,
        &name,
        "You are a support agent. Always cite the policy ID. Be concise.",
    )
    .await?;
    create_version(
        &pool,
        &name,
        "You are a support agent. Always cite the policy ID and link. Be concise.",
    )
    .await?;
    // Changed span here is: "Always cite the policy ID and link."

    // Eval cases: one identical to the changed span (must rank ~1.0), some
    // related, one unrelated.
    for (input, expected) in [
        ("q1", "Always cite the policy ID and link."),
        ("q2", "Response cites the relevant policy ID."),
        ("q3", "Response includes a link to the policy document."),
        ("q4", "Response greets the user warmly with a friendly tone."),
    ] {
        create_eval_case(&pool, &name, input, expected).await?;
    }

    let outcome = select_evals(&pool, &embedder, &name, "latest", 0.5).await?;

    println!(
        "select-evals {} (threshold {:.2})\n",
        outcome.prompt, outcome.threshold
    );
    for item in &outcome.items {
        let marker = if item.was_selected { "●" } else { "○" };
        println!("  {marker} {:.4}  {}", item.similarity, item.expected_behavior);
    }
    println!(
        "\nSelected {} of {} ({:.0}% reduction).",
        outcome.selected,
        outcome.total,
        outcome.reduction * 100.0
    );

    // Assertions: the identical span ranks first at ~1.0 and is selected; the
    // unrelated greeting case ranks last and is not.
    ensure!(outcome.total == 4, "expected 4 eval cases");
    let top = &outcome.items[0];
    ensure!(
        top.expected_behavior == "Always cite the policy ID and link." && top.similarity > 0.99,
        "identical changed-span case should rank first at ~1.0"
    );
    let bottom = outcome.items.last().unwrap();
    ensure!(
        bottom.expected_behavior.contains("greets") && !bottom.was_selected,
        "unrelated greeting case should rank last and not be selected"
    );

    // Confirm SelectionRecords were persisted.
    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM selection_records sr
             JOIN prompt_versions pv ON pv.id = sr.prompt_version_id
             JOIN prompts p ON p.id = pv.prompt_id WHERE p.name = $1")
            .bind(&name)
            .fetch_one(&pool)
            .await?;
    ensure!(count == 4, "expected 4 selection records, got {count}");

    // Clean up.
    sqlx::query("DELETE FROM prompts WHERE name = $1")
        .bind(&name)
        .execute(&pool)
        .await?;

    println!("\n✅ similarity selection pipeline verified end-to-end (records persisted + cleaned up).");
    Ok(())
}
