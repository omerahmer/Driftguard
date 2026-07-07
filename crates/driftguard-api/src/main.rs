//! driftguard-api — read-only HTTP API for the dashboard.
//!
//! Every handler calls a `driftguard-core` read function (the same ones the CLI
//! uses) and serializes the result. There is deliberately no domain logic and no
//! write path here — writes go through the CLI/Action, and this is a thin second
//! consumer of core. Precision/recall is computed live per request (derived
//! state), not read from a stored column.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use uuid::Uuid;

use driftguard_core::CoreError;
use driftguard_core::db::{self, Pool};
use driftguard_core::models::{EvalRun, Prompt, PromptVersion};
use driftguard_core::scoring::{self, ScoreReport};
use driftguard_core::{evals, prompts};

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL is not set");
    let pool = db::connect(&url).await.expect("failed to connect to Postgres");

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/prompts", get(list_prompts))
        .route("/prompts/{id}/versions", get(list_versions))
        .route("/versions/{id}/runs", get(list_runs))
        .route("/score", get(score))
        // Permissive CORS is fine for a local read-only dashboard; the Next.js
        // dev proxy also makes it same-origin. Tighten if ever deployed.
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(pool);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    println!("driftguard-api listening on http://{addr}");
    axum::serve(listener, app).await.expect("server error");
}

async fn list_prompts(State(pool): State<Pool>) -> Result<Json<Vec<Prompt>>, ApiError> {
    Ok(Json(prompts::list_prompts(&pool).await?))
}

async fn list_versions(
    State(pool): State<Pool>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<PromptVersion>>, ApiError> {
    // Each version carries `diff_from_parent` (the PromptDiff JSON) — that's the
    // diff viewer's data, so no separate diff endpoint is needed.
    Ok(Json(prompts::list_versions(&pool, id).await?))
}

async fn list_runs(
    State(pool): State<Pool>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<EvalRun>>, ApiError> {
    Ok(Json(evals::list_eval_runs(&pool, id).await?))
}

#[derive(Deserialize)]
struct ScoreQuery {
    prompt: Option<String>,
}

async fn score(
    State(pool): State<Pool>,
    Query(q): Query<ScoreQuery>,
) -> Result<Json<ScoreReport>, ApiError> {
    let thresholds = scoring::default_thresholds();
    let report = scoring::score(&pool, q.prompt.as_deref(), &thresholds).await?;
    Ok(Json(report))
}

/// Wraps a `CoreError` so it can become an HTTP response. Not-found variants map
/// to 404; everything else is a 500.
struct ApiError(CoreError);

impl From<CoreError> for ApiError {
    fn from(err: CoreError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            CoreError::PromptNotFound(_) | CoreError::VersionNotFound(_) => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(serde_json::json!({ "error": self.0.to_string() }))).into_response()
    }
}
