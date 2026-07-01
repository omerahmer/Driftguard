//! Library-level error type for driftguard-core.
//!
//! We use `thiserror` here, in contrast to `anyhow` at the binary edge. The
//! distinction (it comes up a lot in Rust):
//!
//! - `thiserror` defines a concrete, *matchable* enum. core is a library, and
//!   its callers (the CLI today, the axum API later) need to tell failures
//!   apart — render "duplicate name" as a friendly message, map "not found" to a
//!   specific exit code, retry on a DB outage. Each variant is a named contract.
//!   `#[from]` generates `From` impls so `?` converts underlying errors for free.
//!
//! - `anyhow` (used in the CLI) is for the application edge, which does NOT match
//!   on errors — it just attaches human context and prints. Returning `anyhow`
//!   from a library would erase the type information callers depend on, so core
//!   never does.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("a prompt named \"{0}\" already exists")]
    DuplicateName(String),

    #[error("no prompt named \"{0}\"")]
    PromptNotFound(String),

    /// Carries a fully-formed human message (the various ways a version ref can
    /// fail to resolve don't each need their own variant yet).
    #[error("{0}")]
    VersionNotFound(String),

    #[error("version reference \"{0}\" is ambiguous")]
    AmbiguousVersionRef(String),

    /// Missing/invalid configuration (e.g. an unset API key env var).
    #[error("{0}")]
    Config(String),

    /// The operation can't proceed given the current data (e.g. a version with
    /// no changed span to embed). Carries a fully-formed human message.
    #[error("{0}")]
    InvalidState(String),

    /// The embedding provider returned an error response.
    #[error("embedding provider error: {0}")]
    Embedding(String),

    /// The LLM (Anthropic) API returned an error or an unusable response.
    #[error("Anthropic API error: {0}")]
    Api(String),

    /// Failed to read or parse the fixtures file.
    #[error("fixtures error: {0}")]
    Fixtures(String),

    #[error(transparent)]
    Db(#[from] sqlx::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Transport-level failure talking to the embedding provider.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}
