//! Domain row types, mapped from Postgres by sqlx's `query_as!`.
//!
//! These derive `Serialize` (not `Deserialize`) because the CLI's `--json` mode
//! and the Phase 6 API only ever emit them outward. Field order and types match
//! the SELECT lists in `prompts.rs` exactly — `query_as!` checks that mapping at
//! compile time, so a schema/column drift becomes a build error, not a runtime
//! surprise.

use chrono::{DateTime, Utc};
use serde::Serialize;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    pub id: Uuid,
    pub name: String,
    pub current_version_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptVersion {
    pub id: Uuid,
    pub prompt_id: Uuid,
    pub content: String,
    pub parent_version_id: Option<Uuid>,
    /// Serialized `PromptDiff` (JSON) vs. the parent version; `None` for the
    /// first version of a prompt.
    pub diff_from_parent: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// An eval case: a test input plus a natural-language description of what a
/// correct response must do. The embedding column (`expected_behavior_embedding`)
/// is intentionally NOT a field here — it's written/read only via raw SQL in the
/// selection path, so the typed model stays free of the pgvector dependency.
#[derive(Debug, Clone, Serialize)]
pub struct EvalCase {
    pub id: Uuid,
    pub prompt_id: Uuid,
    pub input: String,
    pub expected_behavior: String,
    pub created_at: DateTime<Utc>,
}

/// A single eval run: the model-under-test's output for one case against one
/// version, plus the judge's verdict (nullable until judged).
#[derive(Debug, Clone, Serialize)]
pub struct EvalRun {
    pub id: Uuid,
    pub prompt_version_id: Uuid,
    pub eval_case_id: Uuid,
    pub actual_output: String,
    pub judge_passed: Option<bool>,
    pub judge_justification: Option<String>,
    pub created_at: DateTime<Utc>,
}
