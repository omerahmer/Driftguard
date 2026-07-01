//! Fixtures: realistic prompts + eval cases + a batch of prompt edits, loaded
//! from a hand-editable TOML file. This is the input whose realism makes the
//! Phase 4 precision/recall result meaningful.
//!
//! Loading model: each prompt becomes a base version (v1) holding `content`;
//! each edit becomes a NEW version branched off v1 (via
//! `create_version_from_parent`), so every edit's changed span is measured
//! against the same base. An edit's `content` is the FULL post-edit prompt (not
//! a patch) — simplest and unambiguous.

use serde::Deserialize;

use crate::db::Pool;
use crate::error::CoreError;
use crate::evals::create_eval_case;
use crate::prompts::{create_prompt, create_version, create_version_from_parent, get_prompt_by_name};

#[derive(Debug, Clone, Deserialize)]
pub struct Fixtures {
    pub prompts: Vec<FixturePrompt>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixturePrompt {
    pub name: String,
    pub content: String,
    #[serde(default)]
    pub eval_cases: Vec<FixtureEvalCase>,
    #[serde(default)]
    pub edits: Vec<FixtureEdit>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureEvalCase {
    pub input: String,
    pub expected_behavior: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureEdit {
    /// Human label, e.g. "added-constraint".
    pub label: String,
    /// Edit category for reporting (wording / added / removed / tone / reorder).
    pub edit_type: String,
    /// The FULL prompt content after applying this edit to the base.
    pub content: String,
}

/// Parse a fixtures TOML file.
pub fn parse_fixtures(text: &str) -> Result<Fixtures, CoreError> {
    toml::from_str(text).map_err(|e| CoreError::Fixtures(e.to_string()))
}

#[derive(Debug, Default)]
pub struct LoadSummary {
    pub prompts: usize,
    pub eval_cases: usize,
    pub versions: usize,
    pub skipped_existing: usize,
}

/// Load fixtures into the DB. Prompts that already exist are skipped (so this is
/// safe to call from `validate`); use `force` to delete and recreate them for a
/// clean reload.
pub async fn load_fixtures(
    pool: &Pool,
    fixtures: &Fixtures,
    force: bool,
) -> Result<LoadSummary, CoreError> {
    let mut summary = LoadSummary::default();

    for prompt in &fixtures.prompts {
        if let Some(existing) = get_prompt_by_name(pool, &prompt.name).await? {
            if force {
                // Cascade-deletes versions, eval cases, runs, selections, diffs.
                sqlx::query!("DELETE FROM prompts WHERE id = $1", existing.id)
                    .execute(pool)
                    .await?;
            } else {
                summary.skipped_existing += 1;
                continue;
            }
        }

        // Base version (v1).
        create_prompt(pool, &prompt.name).await?;
        let base = create_version(pool, &prompt.name, &prompt.content).await?;
        summary.prompts += 1;
        summary.versions += 1;

        for case in &prompt.eval_cases {
            create_eval_case(pool, &prompt.name, &case.input, &case.expected_behavior).await?;
            summary.eval_cases += 1;
        }

        // Each edit branches off the base version.
        for edit in &prompt.edits {
            create_version_from_parent(
                pool,
                base.version.prompt_id,
                base.version.id,
                &edit.content,
            )
            .await?;
            summary.versions += 1;
        }
    }

    Ok(summary)
}
