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

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{Mutex as AsyncMutex, oneshot};

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

type Pending = std::sync::Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// Talks to Anthropic via the `driftguard-llm` Go sidecar (official SDK).
///
/// The sidecar is a LONG-LIVED process started once; we multiplex requests over
/// its stdio. Each request carries an `id`; a background reader task correlates
/// each id-tagged response back to the caller's oneshot channel. This lets the
/// pipeline fire many requests concurrently (the sidecar handles them in
/// goroutines, reusing one HTTP/TLS client) instead of paying process-spawn +
/// connection setup per call.
///
/// `GoLlm` is `Sync`, so `&self` can be shared across concurrent tasks.
pub struct GoLlm {
    stdin: AsyncMutex<ChildStdin>,
    pending: Pending,
    next_id: AtomicU64,
}

impl GoLlm {
    /// Spawn the sidecar and wire up the reader task. Must be called from within
    /// a tokio runtime (it spawns background tasks). Fails fast with a friendly
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

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                CoreError::Config(format!(
                    "could not spawn LLM sidecar \"{binary}\" ({e}). Build it (`make sidecar`) \
                     and set DRIFTGUARD_LLM_BIN."
                ))
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CoreError::Api("sidecar stdin unavailable".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CoreError::Api("sidecar stdout unavailable".to_string()))?;
        let stderr = child.stderr.take();
        // We intentionally drop the `Child` handle: keeping stdin open keeps the
        // server running; when this `GoLlm` drops, stdin closes → the server sees
        // EOF and exits gracefully. (Not storing Child keeps GoLlm `Sync`.)

        let pending: Pending = std::sync::Arc::new(Mutex::new(HashMap::new()));

        // Reader task: match id-tagged responses to their pending oneshot.
        {
            let pending = pending.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let Ok(value) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    let Some(id) = value.get("id").and_then(Value::as_u64) else {
                        continue;
                    };
                    let sender = pending.lock().unwrap().remove(&id);
                    if let Some(tx) = sender {
                        let result = match value.get("error").and_then(Value::as_str) {
                            Some(err) => Err(err.to_string()),
                            None => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
                        };
                        let _ = tx.send(result);
                    }
                }
                // stdout closed: fail any still-pending requests.
                for (_, tx) in pending.lock().unwrap().drain() {
                    let _ = tx.send(Err("sidecar stdout closed".to_string()));
                }
            });
        }

        // Drain stderr so the pipe never blocks; surface lines for debugging.
        if let Some(stderr) = stderr {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("[driftguard-llm] {line}");
                }
            });
        }

        Ok(Self {
            stdin: AsyncMutex::new(stdin),
            pending,
            next_id: AtomicU64::new(1),
        })
    }

    /// Send one request and await its correlated response.
    async fn call(&self, mut request: Value) -> Result<Value, CoreError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        request["id"] = json!(id);

        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);

        let mut line = serde_json::to_vec(&request)?;
        line.push(b'\n');
        {
            let mut stdin = self.stdin.lock().await;
            if let Err(e) = stdin.write_all(&line).await {
                self.pending.lock().unwrap().remove(&id);
                return Err(CoreError::Api(format!("writing to sidecar: {e}")));
            }
            let _ = stdin.flush().await;
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => Err(CoreError::Api(format!("sidecar: {e}"))),
            Err(_) => Err(CoreError::Api("sidecar dropped the request".to_string())),
        }
    }
}

impl Llm for GoLlm {
    async fn generate(&self, system: &str, user: &str) -> Result<String, CoreError> {
        let result = self
            .call(json!({ "op": "generate", "system": system, "user": user }))
            .await?;
        result
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| CoreError::Api(format!("sidecar generate: no text in {result}")))
    }

    async fn judge_pass(
        &self,
        expected_behavior: &str,
        input: &str,
        actual_output: &str,
    ) -> Result<PassVerdict, CoreError> {
        let result = self
            .call(json!({
                "op": "judge_pass",
                "expected_behavior": expected_behavior,
                "input": input,
                "actual_output": actual_output,
            }))
            .await?;
        serde_json::from_value(result)
            .map_err(|e| CoreError::Api(format!("parsing pass verdict: {e}")))
    }

    async fn judge_behavior(
        &self,
        input: &str,
        expected_behavior: &str,
        before: &str,
        after: &str,
    ) -> Result<BehaviorVerdict, CoreError> {
        let result = self
            .call(json!({
                "op": "judge_behavior",
                "input": input,
                "expected_behavior": expected_behavior,
                "before": before,
                "after": after,
            }))
            .await?;
        serde_json::from_value(result)
            .map_err(|e| CoreError::Api(format!("parsing behavior verdict: {e}")))
    }
}
