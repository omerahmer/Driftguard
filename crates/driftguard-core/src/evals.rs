//! Eval-case services.
//!
//! Like `prompts`, these are plain async functions over `&PgPool`. Phase 4's
//! fixtures loader will create eval cases in bulk through `create_eval_case`;
//! the CLI also exposes a one-off `eval add` for testing.

use sqlx::PgPool;
use uuid::Uuid;

use crate::error::CoreError;
use crate::models::{EvalCase, EvalRun};
use crate::prompts::get_prompt_by_name;

pub async fn create_eval_case(
    pool: &PgPool,
    prompt_name: &str,
    input: &str,
    expected_behavior: &str,
) -> Result<EvalCase, CoreError> {
    let prompt = get_prompt_by_name(pool, prompt_name)
        .await?
        .ok_or_else(|| CoreError::PromptNotFound(prompt_name.to_string()))?;

    let case = sqlx::query_as!(
        EvalCase,
        r#"INSERT INTO eval_cases (prompt_id, input, expected_behavior)
           VALUES ($1, $2, $3)
           RETURNING id, prompt_id, input, expected_behavior, created_at"#,
        prompt.id,
        input,
        expected_behavior
    )
    .fetch_one(pool)
    .await?;
    Ok(case)
}

pub async fn list_eval_cases(
    pool: &PgPool,
    prompt_id: Uuid,
) -> Result<Vec<EvalCase>, CoreError> {
    let rows = sqlx::query_as!(
        EvalCase,
        r#"SELECT id, prompt_id, input, expected_behavior, created_at
           FROM eval_cases WHERE prompt_id = $1 ORDER BY created_at ASC"#,
        prompt_id
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// All eval runs recorded for one prompt version (the read side for the API).
pub async fn list_eval_runs(
    pool: &PgPool,
    version_id: Uuid,
) -> Result<Vec<EvalRun>, CoreError> {
    let rows = sqlx::query_as!(
        EvalRun,
        r#"SELECT id, prompt_version_id, eval_case_id, actual_output,
                  judge_passed, judge_justification, created_at
           FROM eval_runs WHERE prompt_version_id = $1 ORDER BY created_at ASC"#,
        version_id
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}
