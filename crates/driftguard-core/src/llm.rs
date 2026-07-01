//! LLM access for Phase 4: generation (the model under test) and judging.
//!
//! As with `Embedder`, the validation/scoring pipeline depends on the [`Llm`]
//! trait, not on any concrete client — so the whole pipeline is verifiable
//! offline with a mock impl (no API key, deterministic), and the provider is
//! swappable.
//!
//! The real impl is [`GoLlm`], which shells out to the `driftguard-llm` Go
//! sidecar (one process per call, JSON over stdio). The Go program uses the
//! OFFICIAL `anthropic-sdk-go`, so the LLM client rides a maintained SDK while
//! the rest of Driftguard stays in Rust. Generation uses Sonnet 4.6; judging
//! uses Opus 4.8 with adaptive thinking + structured output — all configured in
//! the Go program (`tools/driftguard-llm/main.go`), so model IDs live in one place.

use std::process::Stdio;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::error::CoreError;

/// Verdict for judge Question A: did the response satisfy `expected_behavior`?
#[derive(Debug, Clone, Deserialize)]
pub struct PassVerdict {
    pub passed: bool,
    pub justification: String,
}

/// Verdict for judge Question B: did behavior change substantively before→after?
#[derive(Debug, Clone, Deserialize)]
pub struct BehaviorVerdict {
    pub behavior_changed: bool,
    pub justification: String,
}

// We only call `Llm` through generics and await inline — same rationale as
// `Embedder` for suppressing the async-fn-in-trait lint.
#[allow(async_fn_in_trait)]
pub trait Llm {
    /// Run the model under test: `system` is the prompt being tested, `user` is
    /// the eval case input. Returns the response text (the eval's actual_output).
    async fn generate(&self, system: &str, user: &str) -> Result<String, CoreError>;

    /// Judge Question A.
    async fn judge_pass(
        &self,
        expected_behavior: &str,
        input: &str,
        actual_output: &str,
    ) -> Result<PassVerdict, CoreError>;

    /// Judge Question B.
    async fn judge_behavior(
        &self,
        input: &str,
        expected_behavior: &str,
        before: &str,
        after: &str,
    ) -> Result<BehaviorVerdict, CoreError>;
}

/// Talks to Anthropic via the `driftguard-llm` Go sidecar (official SDK).
pub struct GoLlm {
    binary: String,
}

impl GoLlm {
    /// Resolve the sidecar binary from `DRIFTGUARD_LLM_BIN` (default
    /// `driftguard-llm`, i.e. expected on PATH). Also fails fast with a friendly
    /// message if `ANTHROPIC_API_KEY` is unset (the sidecar inherits our env).
    pub fn from_env() -> Result<Self, CoreError> {
        std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| {
                CoreError::Config(
                    "ANTHROPIC_API_KEY is not set — needed for run-evals/validate. Add it to .env."
                        .to_string(),
                )
            })?;
        let binary =
            std::env::var("DRIFTGUARD_LLM_BIN").unwrap_or_else(|_| "driftguard-llm".to_string());
        Ok(Self { binary })
    }

    /// Spawn the sidecar, write the request JSON to its stdin, and return its
    /// stdout. The payloads are small, so writing-then-reading can't deadlock.
    async fn call(&self, request: Value) -> Result<String, CoreError> {
        let mut child = Command::new(&self.binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                CoreError::Config(format!(
                    "could not spawn LLM sidecar \"{}\" ({e}). Build it \
                     (`go build -o tools/driftguard-llm/driftguard-llm ./tools/driftguard-llm`) \
                     and set DRIFTGUARD_LLM_BIN.",
                    self.binary
                ))
            })?;

        let payload = serde_json::to_vec(&request)?;
        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| CoreError::Api("sidecar stdin unavailable".to_string()))?;
            stdin
                .write_all(&payload)
                .await
                .map_err(|e| CoreError::Api(format!("writing to sidecar: {e}")))?;
            // stdin dropped here → EOF, so the sidecar stops reading.
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| CoreError::Api(format!("waiting on sidecar: {e}")))?;

        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(CoreError::Api(format!("sidecar failed: {detail}")));
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

impl Llm for GoLlm {
    async fn generate(&self, system: &str, user: &str) -> Result<String, CoreError> {
        let out = self
            .call(json!({ "op": "generate", "system": system, "user": user }))
            .await?;
        let value: Value = serde_json::from_str(&out)
            .map_err(|e| CoreError::Api(format!("parsing sidecar output ({e}): {out}")))?;
        value
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| CoreError::Api(format!("sidecar generate: no text in {out}")))
    }

    async fn judge_pass(
        &self,
        expected_behavior: &str,
        input: &str,
        actual_output: &str,
    ) -> Result<PassVerdict, CoreError> {
        let out = self
            .call(json!({
                "op": "judge_pass",
                "expected_behavior": expected_behavior,
                "input": input,
                "actual_output": actual_output,
            }))
            .await?;
        serde_json::from_str(&out)
            .map_err(|e| CoreError::Api(format!("parsing pass verdict ({e}): {out}")))
    }

    async fn judge_behavior(
        &self,
        input: &str,
        expected_behavior: &str,
        before: &str,
        after: &str,
    ) -> Result<BehaviorVerdict, CoreError> {
        let out = self
            .call(json!({
                "op": "judge_behavior",
                "input": input,
                "expected_behavior": expected_behavior,
                "before": before,
                "after": after,
            }))
            .await?;
        serde_json::from_str(&out)
            .map_err(|e| CoreError::Api(format!("parsing behavior verdict ({e}): {out}")))
    }
}
