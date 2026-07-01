//! Prompt versioning services.
//!
//! Plain async functions over a `&PgPool` — no clap, no println, no exit codes.
//! The CLI (and the Phase 6 axum API) call these and own their own I/O. Keeping
//! the boundary here is what lets later delivery layers reuse the logic.
//!
//! All queries use sqlx's `query!`/`query_as!` macros, which validate the SQL
//! and the Rust<->Postgres type mapping *at compile time* against the live
//! schema (or the committed `.sqlx/` offline cache). A typo or a column type
//! change becomes a build error rather than a runtime panic.

use serde::Serialize;
use sqlx::PgPool;
use sqlx::error::DatabaseError;
use uuid::Uuid;

use crate::error::CoreError;
use crate::models::{Prompt, PromptVersion};
use crate::segment::{PromptDiff, compute_diff};

/// Result of adding a version, with everything the CLI needs to report.
#[derive(Debug, Serialize)]
pub struct CreateVersionOutcome {
    pub version: PromptVersion,
    pub parent_version_id: Option<Uuid>,
    pub diff: Option<PromptDiff>,
    /// True when the new content is identical to its parent (empty changed span).
    pub unchanged: bool,
}

pub async fn get_prompt_by_name(
    pool: &PgPool,
    name: &str,
) -> Result<Option<Prompt>, CoreError> {
    let prompt = sqlx::query_as!(
        Prompt,
        r#"SELECT id, name, current_version_id, created_at
           FROM prompts WHERE name = $1"#,
        name
    )
    .fetch_optional(pool)
    .await?;
    Ok(prompt)
}

pub async fn list_prompts(pool: &PgPool) -> Result<Vec<Prompt>, CoreError> {
    let rows = sqlx::query_as!(
        Prompt,
        r#"SELECT id, name, current_version_id, created_at
           FROM prompts ORDER BY created_at ASC"#
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Versions of a prompt, oldest first (the ordinal v1, v2… is the index + 1).
pub async fn list_versions(
    pool: &PgPool,
    prompt_id: Uuid,
) -> Result<Vec<PromptVersion>, CoreError> {
    let rows = sqlx::query_as!(
        PromptVersion,
        r#"SELECT id, prompt_id, content, parent_version_id, diff_from_parent, created_at
           FROM prompt_versions WHERE prompt_id = $1 ORDER BY created_at ASC"#,
        prompt_id
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_version_by_id(
    pool: &PgPool,
    id: Uuid,
) -> Result<Option<PromptVersion>, CoreError> {
    let version = sqlx::query_as!(
        PromptVersion,
        r#"SELECT id, prompt_id, content, parent_version_id, diff_from_parent, created_at
           FROM prompt_versions WHERE id = $1"#,
        id
    )
    .fetch_optional(pool)
    .await?;
    Ok(version)
}

/// Create an empty prompt (no versions yet). Per the spec, `prompt create` only
/// registers the name; the first `prompt version` becomes v1.
pub async fn create_prompt(pool: &PgPool, name: &str) -> Result<Prompt, CoreError> {
    let result = sqlx::query_as!(
        Prompt,
        r#"INSERT INTO prompts (name) VALUES ($1)
           RETURNING id, name, current_version_id, created_at"#,
        name
    )
    .fetch_one(pool)
    .await;

    match result {
        Ok(prompt) => Ok(prompt),
        // 23505 = unique_violation on prompts.name. Translate the opaque DB error
        // into a typed, user-facing variant the CLI can render nicely.
        Err(sqlx::Error::Database(db)) if is_unique_violation(db.as_ref()) => {
            Err(CoreError::DuplicateName(name.to_string()))
        }
        Err(other) => Err(other.into()),
    }
}

/// Add a new version of an existing prompt: compute the diff vs. the current
/// version, persist the version, and advance `current_version_id` — atomically.
pub async fn create_version(
    pool: &PgPool,
    name: &str,
    content: &str,
) -> Result<CreateVersionOutcome, CoreError> {
    let prompt = get_prompt_by_name(pool, name)
        .await?
        .ok_or_else(|| CoreError::PromptNotFound(name.to_string()))?;

    let parent_version_id = prompt.current_version_id;

    // Compute the diff against the parent's content (if there is a parent).
    let diff = match parent_version_id {
        Some(parent_id) => {
            let parent = get_version_by_id(pool, parent_id).await?.ok_or_else(|| {
                CoreError::VersionNotFound(format!("parent version {parent_id} is missing"))
            })?;
            Some(compute_diff(&parent.content, content))
        }
        None => None,
    };

    let unchanged = diff
        .as_ref()
        .is_some_and(|d| d.changed_spans.is_empty() && d.removed_spans.is_empty());

    // Serialize once; `?` converts serde_json::Error via CoreError's #[from].
    let diff_json = match &diff {
        Some(d) => Some(serde_json::to_string(d)?),
        None => None,
    };

    // Transaction: the version insert and the current-pointer update must either
    // both land or neither. If we crashed between them, the prompt would point at
    // a stale version. `begin()` borrows the pool; the tx is committed explicitly.
    let mut tx = pool.begin().await?;

    let version = sqlx::query_as!(
        PromptVersion,
        r#"INSERT INTO prompt_versions (prompt_id, content, parent_version_id, diff_from_parent)
           VALUES ($1, $2, $3, $4)
           RETURNING id, prompt_id, content, parent_version_id, diff_from_parent, created_at"#,
        prompt.id,
        content,
        parent_version_id,
        diff_json
    )
    .fetch_one(&mut *tx)
    .await?;

    sqlx::query!(
        r#"UPDATE prompts SET current_version_id = $1 WHERE id = $2"#,
        version.id,
        prompt.id
    )
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(CreateVersionOutcome {
        version,
        parent_version_id,
        diff,
        unchanged,
    })
}

/// Create a version whose parent is an EXPLICIT version (not necessarily the
/// current head). Used by the fixtures loader, where every edit branches off the
/// same base v1 so each edit's diff/changed-span is measured against the base.
/// Advances `current_version_id` to the new version (harmless for fixtures,
/// which address versions explicitly).
pub async fn create_version_from_parent(
    pool: &PgPool,
    prompt_id: Uuid,
    parent_version_id: Uuid,
    content: &str,
) -> Result<PromptVersion, CoreError> {
    let parent = get_version_by_id(pool, parent_version_id)
        .await?
        .ok_or_else(|| {
            CoreError::VersionNotFound(format!("parent version {parent_version_id} not found"))
        })?;
    let diff = compute_diff(&parent.content, content);
    let diff_json = serde_json::to_string(&diff)?;

    let mut tx = pool.begin().await?;
    let version = sqlx::query_as!(
        PromptVersion,
        r#"INSERT INTO prompt_versions (prompt_id, content, parent_version_id, diff_from_parent)
           VALUES ($1, $2, $3, $4)
           RETURNING id, prompt_id, content, parent_version_id, diff_from_parent, created_at"#,
        prompt_id,
        content,
        Some(parent_version_id),
        Some(diff_json)
    )
    .fetch_one(&mut *tx)
    .await?;
    sqlx::query!(
        r#"UPDATE prompts SET current_version_id = $1 WHERE id = $2"#,
        version.id,
        prompt_id
    )
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(version)
}

/// Resolve a human-friendly version reference within a prompt.
/// Accepts: `latest`/`current`, `parent`, `vN` (1-based ordinal), or a full UUID
/// or unambiguous UUID prefix. Scoped to the prompt so short refs are reusable.
pub async fn resolve_version_ref(
    pool: &PgPool,
    prompt: &Prompt,
    reference: &str,
) -> Result<PromptVersion, CoreError> {
    let versions = list_versions(pool, prompt.id).await?;
    if versions.is_empty() {
        return Err(CoreError::VersionNotFound(format!(
            "prompt \"{}\" has no versions",
            prompt.name
        )));
    }

    let normalized = reference.trim().to_lowercase();

    if normalized == "latest" || normalized == "current" {
        return versions
            .iter()
            .find(|v| Some(v.id) == prompt.current_version_id)
            .cloned()
            .ok_or_else(|| CoreError::VersionNotFound("prompt has no current version".into()));
    }

    if normalized == "parent" {
        let current = versions.iter().find(|v| Some(v.id) == prompt.current_version_id);
        return current
            .and_then(|c| c.parent_version_id)
            .and_then(|pid| versions.iter().find(|v| v.id == pid).cloned())
            .ok_or_else(|| CoreError::VersionNotFound("current version has no parent".into()));
    }

    if let Some(n) = normalized
        .strip_prefix('v')
        .and_then(|rest| rest.parse::<usize>().ok())
    {
        return versions
            .get(n.wrapping_sub(1))
            .cloned()
            .ok_or_else(|| {
                CoreError::VersionNotFound(format!(
                    "prompt \"{}\" has no version v{n} (it has {})",
                    prompt.name,
                    versions.len()
                ))
            });
    }

    // UUID or prefix match, scoped to this prompt.
    let matches: Vec<&PromptVersion> = versions
        .iter()
        .filter(|v| v.id.to_string().starts_with(&normalized))
        .collect();
    match matches.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(CoreError::VersionNotFound(format!(
            "could not resolve version \"{reference}\" (use latest, parent, vN, or an id)"
        ))),
        _ => Err(CoreError::AmbiguousVersionRef(reference.to_string())),
    }
}

fn is_unique_violation(db: &dyn DatabaseError) -> bool {
    db.code().as_deref() == Some("23505")
}
