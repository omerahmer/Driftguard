//! Embedding providers.
//!
//! All selection logic depends on the [`Embedder`] trait, not on Voyage
//! directly. That boundary is deliberate and interview-defensible: it keeps the
//! similarity/selection code testable without a network call or an API key (an
//! examples/tests `Embedder` can return deterministic vectors), and it means
//! swapping the provider later is a one-impl change rather than a rewrite.
//!
//! Why Voyage at all: Anthropic does not offer a first-party embeddings API and
//! recommends Voyage AI as its embeddings partner. The Anthropic key is still
//! used — for the Phase 4 LLM judge — just not here.

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Voyage's asymmetric retrieval hint. Embedding the prompt's changed span as a
/// `Query` and each eval case's expected_behavior as a `Document` matches the
/// retrieval framing ("which described behaviors does this change retrieve?")
/// and tends to improve relevance over embedding both sides identically.
#[derive(Debug, Clone, Copy)]
pub enum InputType {
    Query,
    Document,
}

impl InputType {
    fn as_str(self) -> &'static str {
        match self {
            InputType::Query => "query",
            InputType::Document => "document",
        }
    }
}

/// A source of text embeddings.
///
/// `async fn` in a trait is stable (Rust 1.75+). We consume `Embedder` via
/// generics (`impl Embedder` / `<E: Embedder>`) rather than `dyn`, which both
/// sidesteps the dyn-compatibility caveats of async-fn-in-trait and monomorphizes
/// to zero-cost calls. The CLI has exactly one impl to choose from today, so
/// generics cost us nothing.
// We only ever call `Embedder` through generics and await it inline (never as
// `dyn`, never spawned onto another task), so we don't need to name `Send`/`Sync`
// bounds on the returned future — the `async_fn_in_trait` lint's concern doesn't
// apply to our usage. Suppress it rather than desugar to `impl Future + Send`.
#[allow(async_fn_in_trait)]
pub trait Embedder {
    /// Output dimension — must match the pgvector column dimension.
    fn dimension(&self) -> usize;

    /// Embed a batch of texts, returning one vector per input in input order.
    async fn embed(
        &self,
        inputs: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, CoreError>;
}

/// Voyage AI embeddings (`voyage-3.5-lite`, 1024-dim).
pub struct VoyageEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: usize,
}

impl VoyageEmbedder {
    pub const DEFAULT_MODEL: &'static str = "voyage-3.5-lite";
    pub const DEFAULT_DIMENSION: usize = 1024;
    const ENDPOINT: &'static str = "https://api.voyageai.com/v1/embeddings";

    /// Construct from `VOYAGE_API_KEY`. Returns a `Config` error (not a panic) so
    /// the CLI can print a clean message when the key is missing.
    pub fn from_env() -> Result<Self, CoreError> {
        // Treat an unset OR empty value as missing — .env ships with an empty
        // `VOYAGE_API_KEY=`, and an empty key would otherwise reach Voyage and
        // come back as an opaque 401 instead of a clear "set your key" message.
        let api_key = std::env::var("VOYAGE_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| {
                CoreError::Config(
                    "VOYAGE_API_KEY is not set — needed for embeddings in `select-evals`. \
                     Get a key at voyageai.com and add it to .env."
                        .to_string(),
                )
            })?;
        // Model is configurable via VOYAGE_MODEL (default voyage-3.5-lite). The
        // DIMENSION is intentionally NOT env-configurable: it must match the
        // pgvector column (vector(1024) in migration 0002), so changing it
        // requires a migration. If you pick a model that doesn't support a 1024
        // output_dimension, Voyage returns a clear error.
        let model = std::env::var("VOYAGE_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            dimension: Self::DEFAULT_DIMENSION,
        })
    }
}

#[derive(Serialize)]
struct VoyageRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbeddingItem>,
}

#[derive(Deserialize)]
struct VoyageEmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

impl Embedder for VoyageEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(
        &self,
        inputs: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, CoreError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let body = VoyageRequest {
            input: inputs,
            model: &self.model,
            input_type: input_type.as_str(),
            output_dimension: self.dimension,
        };

        let response = self
            .client
            .post(Self::ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            return Err(CoreError::Embedding(format!(
                "Voyage API returned {status}: {detail}"
            )));
        }

        let mut parsed: VoyageResponse = response.json().await?;
        // The API documents in-order results, but sort by index defensively so a
        // reordering never silently misaligns embeddings with their inputs.
        parsed.data.sort_by_key(|item| item.index);
        Ok(parsed.data.into_iter().map(|item| item.embedding).collect())
    }
}

/// Serialize a vector to pgvector's text input format, e.g. `[0.1,0.2,0.3]`.
///
/// We bind embeddings as this text literal and cast with `$1::vector` in SQL,
/// rather than pulling in a pgvector Rust binding crate. That avoids coupling a
/// third dependency's version to our sqlx version, and keeps the wire format
/// something we fully control and can debug by eye.
pub fn to_pgvector_literal(vector: &[f32]) -> String {
    let mut out = String::with_capacity(vector.len() * 8 + 2);
    out.push('[');
    for (i, value) in vector.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_pgvector_literal() {
        assert_eq!(to_pgvector_literal(&[]), "[]");
        assert_eq!(to_pgvector_literal(&[1.0, 0.0, -1.0]), "[1,0,-1]");
        assert_eq!(to_pgvector_literal(&[0.5, 0.25]), "[0.5,0.25]");
    }
}
