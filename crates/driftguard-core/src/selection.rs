//! Similarity-based eval selection.
//!
//! Given a prompt version's changed span, embed it, embed each eval case's
//! expected_behavior, and rank eval cases by cosine similarity (computed by
//! pgvector). Cases at or above the threshold are "selected" — these are the
//! evals CI would run for that change. Every case + score is persisted as a
//! `SelectionRecord` so Phase 4 can score precision/recall against ground truth.
//!
//! The pgvector columns are written/read through raw SQL here (binding the
//! embedding as a `::vector` text literal). Everything *outside* the vector path
//! still uses sqlx's compile-time-checked macros; the vector type isn't cleanly
//! introspectable by those macros, so the few vector statements use runtime
//! queries — a deliberate, localized trade-off.

use serde::Serialize;
use sqlx::Row;
use uuid::Uuid;

use crate::db::Pool;
use crate::embed::{Embedder, InputType, to_pgvector_literal};
use crate::error::CoreError;
use crate::evals::list_eval_cases;
use crate::models::PromptVersion;
use crate::prompts::{get_prompt_by_name, resolve_version_ref};
use crate::segment::PromptDiff;

#[derive(Debug, Serialize)]
pub struct SelectionItem {
    pub eval_case_id: Uuid,
    pub input: String,
    pub expected_behavior: String,
    pub similarity: f64,
    pub was_selected: bool,
}

#[derive(Debug, Serialize)]
pub struct SelectionOutcome {
    pub prompt: String,
    pub version_id: Uuid,
    pub threshold: f64,
    pub total: usize,
    pub selected: usize,
    /// Fraction of eval cases NOT selected — the "% reduction in evals run".
    pub reduction: f64,
    pub items: Vec<SelectionItem>,
}

/// Run similarity selection for a prompt version and persist the records.
pub async fn select_evals<E: Embedder>(
    pool: &Pool,
    embedder: &E,
    prompt_name: &str,
    version_ref: &str,
    threshold: f64,
) -> Result<SelectionOutcome, CoreError> {
    let prompt = get_prompt_by_name(pool, prompt_name)
        .await?
        .ok_or_else(|| CoreError::PromptNotFound(prompt_name.to_string()))?;
    let version = resolve_version_ref(pool, &prompt, version_ref).await?;

    // Make sure both sides are embedded (lazy, cached in the DB).
    ensure_version_embedding(pool, embedder, &version).await?;

    let cases = list_eval_cases(pool, prompt.id).await?;
    if cases.is_empty() {
        return Err(CoreError::InvalidState(format!(
            "prompt \"{}\" has no eval cases; add some with `eval add`",
            prompt.name
        )));
    }
    ensure_eval_embeddings(pool, embedder, prompt.id).await?;

    // Cosine similarity via pgvector's `<=>` (distance); similarity = 1 - distance.
    // The version embedding is read from the joined row, so we don't round-trip
    // it back out of the DB into Rust just to bind it again.
    let rows = sqlx::query(
        r#"
        SELECT ec.id            AS id,
               ec.input         AS input,
               ec.expected_behavior AS expected_behavior,
               1 - (ec.expected_behavior_embedding <=> pv.changed_span_embedding) AS similarity
        FROM eval_cases ec
        CROSS JOIN prompt_versions pv
        WHERE pv.id = $1
          AND ec.prompt_id = $2
          AND ec.expected_behavior_embedding IS NOT NULL
        ORDER BY similarity DESC
        "#,
    )
    .bind(version.id)
    .bind(prompt.id)
    .fetch_all(pool)
    .await?;

    let mut items = Vec::with_capacity(rows.len());
    for row in rows {
        let similarity: f64 = row.try_get("similarity")?;
        items.push(SelectionItem {
            eval_case_id: row.try_get("id")?,
            input: row.try_get("input")?,
            expected_behavior: row.try_get("expected_behavior")?,
            similarity,
            was_selected: similarity >= threshold,
        });
    }

    persist_selection_records(pool, version.id, &items).await?;

    let total = items.len();
    let selected = items.iter().filter(|i| i.was_selected).count();
    let reduction = if total == 0 {
        0.0
    } else {
        1.0 - (selected as f64 / total as f64)
    };

    Ok(SelectionOutcome {
        prompt: prompt.name,
        version_id: version.id,
        threshold,
        total,
        selected,
        reduction,
        items,
    })
}

/// Embed a version's changed span if it isn't embedded yet.
async fn ensure_version_embedding<E: Embedder>(
    pool: &Pool,
    embedder: &E,
    version: &PromptVersion,
) -> Result<(), CoreError> {
    let existing: Option<String> =
        sqlx::query_scalar("SELECT changed_span_embedding::text FROM prompt_versions WHERE id = $1")
            .bind(version.id)
            .fetch_one(pool)
            .await?;
    if existing.is_some() {
        return Ok(());
    }

    let diff_json = version.diff_from_parent.as_deref().ok_or_else(|| {
        CoreError::InvalidState(format!(
            "version {} has no parent diff, so there is no changed span to embed. \
             select-evals needs a version created from a parent.",
            version.id
        ))
    })?;
    let diff: PromptDiff = serde_json::from_str(diff_json)?;
    if diff.changed_spans.is_empty() {
        return Err(CoreError::InvalidState(
            "this version's changed span is empty (content is identical to its parent)".to_string(),
        ));
    }

    // Concatenate the added-side spans into one text and embed as the query side.
    let text = diff.changed_spans.join("\n");
    let vectors = embedder
        .embed(std::slice::from_ref(&text), InputType::Query)
        .await?;
    let vector = vectors
        .into_iter()
        .next()
        .ok_or_else(|| CoreError::Embedding("embedder returned no vectors".to_string()))?;

    sqlx::query("UPDATE prompt_versions SET changed_span_embedding = $1::vector WHERE id = $2")
        .bind(to_pgvector_literal(&vector))
        .bind(version.id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Embed any eval cases for this prompt that don't have an embedding yet, in one
/// batched API call.
async fn ensure_eval_embeddings<E: Embedder>(
    pool: &Pool,
    embedder: &E,
    prompt_id: Uuid,
) -> Result<(), CoreError> {
    let rows = sqlx::query(
        "SELECT id, expected_behavior FROM eval_cases
         WHERE prompt_id = $1 AND expected_behavior_embedding IS NULL",
    )
    .bind(prompt_id)
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        return Ok(());
    }

    let mut ids = Vec::with_capacity(rows.len());
    let mut texts = Vec::with_capacity(rows.len());
    for row in &rows {
        ids.push(row.try_get::<Uuid, _>("id")?);
        texts.push(row.try_get::<String, _>("expected_behavior")?);
    }

    let vectors = embedder.embed(&texts, InputType::Document).await?;

    let mut tx = pool.begin().await?;
    for (id, vector) in ids.iter().zip(vectors.iter()) {
        sqlx::query(
            "UPDATE eval_cases SET expected_behavior_embedding = $1::vector WHERE id = $2",
        )
        .bind(to_pgvector_literal(vector))
        .bind(id)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Replace this version's selection records with the latest run (idempotent —
/// re-running `select-evals` overwrites rather than accumulating duplicates).
async fn persist_selection_records(
    pool: &Pool,
    version_id: Uuid,
    items: &[SelectionItem],
) -> Result<(), CoreError> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM selection_records WHERE prompt_version_id = $1")
        .bind(version_id)
        .execute(&mut *tx)
        .await?;
    for item in items {
        sqlx::query(
            "INSERT INTO selection_records
                 (prompt_version_id, eval_case_id, similarity_score, was_selected)
             VALUES ($1, $2, $3, $4)",
        )
        .bind(version_id)
        .bind(item.eval_case_id)
        .bind(item.similarity)
        .bind(item.was_selected)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}
