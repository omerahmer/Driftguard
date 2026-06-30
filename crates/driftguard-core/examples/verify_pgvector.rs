//! Phase 1 end-to-end smoke test for pgvector, exercised through sqlx.
//!
//! Run with: `cargo run -p driftguard-core --example verify_pgvector`
//!
//! It proves three things against the real, migrated database:
//!   1. the `vector` extension is installed and reports its version,
//!   2. pgvector's cosine-distance operator `<=>` works and orders correctly,
//!   3. the migrated schema is reachable (counts rows in `prompts`).
//!
//! We intentionally compute similarity over inline `VALUES` rows cast to
//! `vector` rather than a real table. Two reasons: (a) the production vector
//! *dimension* is dictated by the embedding model, a Phase 3 open decision, so
//! we don't want to bake a dimension into anything committed yet; (b) a TEMP
//! table would be invisible across pooled connections, so a self-contained
//! single statement is the robust way to smoke-test through a pool.

use anyhow::{Context, Result, bail};
use driftguard_core::db;

// `anyhow` is application-level error handling. This example is a *binary-like*
// consumer: it doesn't need to match on error variants, it just wants to
// `?`-propagate failures and attach human context (`.context(...)`) for the
// person reading the terminal. Library code in driftguard-core uses `thiserror`
// instead (introduced in Phase 2) so its callers CAN match on specific errors.
#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL").context("DATABASE_URL is not set")?;
    let pool = db::connect(&url).await.context("connecting to Postgres")?;

    // Idempotent; also created by migration 0001, but harmless to assert here.
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await
        .context("enabling pgvector extension")?;

    let version: String =
        sqlx::query_scalar("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            .fetch_one(&pool)
            .await
            .context("reading pgvector version")?;
    println!("pgvector extension version: {version}");

    // `<=>` is cosine *distance*; cosine *similarity* = 1 - distance. Ordering by
    // distance ascending puts the most similar row first.
    let rows: Vec<(String, f64)> = sqlx::query_as(
        "SELECT label, 1 - (embedding <=> $1::vector) AS cosine_similarity
         FROM (VALUES
             ('identical',  '[1,0,0]'::vector),
             ('orthogonal', '[0,1,0]'::vector),
             ('opposite',   '[-1,0,0]'::vector),
             ('similar',    '[0.9,0.1,0]'::vector)
         ) AS t(label, embedding)
         ORDER BY embedding <=> $1::vector ASC",
    )
    .bind("[1,0,0]")
    .fetch_all(&pool)
    .await
    .context("running cosine similarity query")?;

    println!("\nCosine similarity to [1,0,0]:");
    for (label, sim) in &rows {
        println!("  {label:<11} {sim:.4}");
    }

    let get = |name: &str| {
        rows.iter()
            .find(|(l, _)| l == name)
            .map(|(_, s)| *s)
            .expect("label present")
    };
    let ok = (get("identical") - 1.0).abs() < 1e-6
        && (get("opposite") + 1.0).abs() < 1e-6
        && get("orthogonal").abs() < 1e-6
        && get("similar") > 0.98;
    if !ok {
        bail!("cosine similarity results did not match expectations");
    }

    // Confirm migrations actually ran and the schema is reachable.
    let prompt_count: i64 = sqlx::query_scalar("SELECT count(*) FROM prompts")
        .fetch_one(&pool)
        .await
        .context("querying prompts table (did migrations run?)")?;
    println!("\nschema reachable: prompts table has {prompt_count} row(s).");

    println!("\n✅ pgvector cosine similarity verified end-to-end via sqlx.");
    Ok(())
}
