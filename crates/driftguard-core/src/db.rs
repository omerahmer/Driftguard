//! Database access for driftguard-core.

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

// Re-exported so delivery layers (CLI, future axum API) can name the pool type
// without taking a direct dependency on sqlx. The persistence library stays an
// implementation detail of core.
pub use sqlx::PgPool as Pool;

/// Open a connection pool to Postgres.
///
/// Returns `sqlx::Error` directly rather than a bespoke error type. At this
/// layer the only failure mode is "couldn't connect / bad URL", and sqlx's own
/// error already describes that precisely — wrapping it would add a variant
/// that carries no extra meaning. We introduce a `thiserror`-based core error
/// type in Phase 2, once core has several *distinct* failure modes (parse,
/// not-found, conflict, …) that callers genuinely need to tell apart.
///
/// We hand back an owned `PgPool` (which is internally `Arc`-shared and cheap to
/// `clone`) instead of borrowing one, so callers — a short-lived CLI command or
/// a long-lived axum handler — each own a handle and decide their own lifetime.
pub async fn connect(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
}
