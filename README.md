# Driftguard

Version LLM prompts like schema migrations, and predict which eval cases a given
prompt change affects — so CI runs only the relevant tests instead of the whole
suite or nothing. The project's core contribution is a **measured
precision/recall result** for that selection, not just a working CLI.

## Stack

- Language: Rust (stable, edition 2024)
- CLI: `clap` v4 (derive)
- Async: `tokio`
- DB: PostgreSQL + pgvector via `sqlx` (async, compile-time-checked queries)
- Embeddings: Voyage AI via `reqwest` (Phase 3)
- LLM (generate + judge): a **long-lived Go sidecar** (`tools/driftguard-llm`)
  using the official `anthropic-sdk-go` (no official Rust SDK). Rust starts it
  once and **multiplexes id-tagged requests** over its stdio, running many in
  flight at once (`--concurrency`, default 6) against one reused HTTP client.
  The judge (Opus 4.8) uses schema-forced JSON with thinking off; the SDK
  handles 429/5xx retries. All behind the `Llm` trait.
- Errors: `anyhow` (app-level) + `thiserror` (library-level)

## Workspace layout

```
crates/driftguard-core   all business logic (diff, embed, select, judge, score)
crates/driftguard-cli    thin clap binary `driftguard` that calls into core
tools/driftguard-llm     Go sidecar for Anthropic calls (official anthropic-sdk-go)
migrations/              sqlx migrations
fixtures/                Phase 4 fixtures (prompts + eval cases + edits)
crates/driftguard-api    axum JSON API — added in Phase 6 only
```

The core/cli split is deliberate: core holds no delivery concerns, so the CLI
now and the axum API later both call the same functions.

## Local setup

Requirements: Rust 1.91+, Go 1.25+ (for the LLM sidecar), Docker, `sqlx-cli`
(`cargo install sqlx-cli@^0.8 --no-default-features --features native-tls,postgres`).

```bash
cp .env.example .env                 # DATABASE_URL / VOYAGE_API_KEY / ANTHROPIC_API_KEY
docker compose up -d                 # Postgres + pgvector on host port 5433
sqlx migrate run                     # apply migrations/ to the DB
make build                           # build the Rust workspace + the Go sidecar
make verify                          # offline pipeline checks (no API keys needed)
make test                            # unit tests
```

`make sidecar` alone builds just the Go binary; `.env`'s `DRIFTGUARD_LLM_BIN`
points the CLI at it. The Anthropic-touching commands (`run-evals`, `validate`)
shell out to that sidecar.

**Compile-time-checked queries.** sqlx validates every query against the schema
at build time. The committed `.sqlx/` cache lets the workspace build offline
(`SQLX_OFFLINE=true cargo build`) without a live DB — e.g. in CI. After changing
any SQL, regenerate it with `cargo sqlx prepare --workspace` (needs the DB up)
and commit the result.

## CLI usage (Phase 2)

The binary is `driftguard`. `--json` is a global flag (accepted after any
subcommand) that switches to machine-readable output for CI and the dashboard.

```bash
# Create an (empty) prompt; the first `version` becomes v1
driftguard prompt create --name support-agent

# Add a version — content from a file path, `-`, or piped stdin
driftguard prompt version --prompt support-agent --content ./prompt.txt
echo "You are a support agent. Be concise." | driftguard prompt version --prompt support-agent

# Diff two versions (refs: latest, parent, vN, or a version id/prefix)
driftguard prompt diff --prompt support-agent --from v1 --to latest
driftguard prompt diff --prompt support-agent --from v1 --to latest --json

# Inspect
driftguard prompt list
driftguard prompt history --prompt support-agent
```

**Diff granularity:** structure-aware (`crates/driftguard-core/src/segment.rs`).
Prose splits into sentences; headings and list items stay per-line; fenced code
stays atomic. Each diff's added-side `changed_spans` is what Phase 3 embeds.

## Eval selection (Phase 3)

```bash
# Add eval cases to a prompt
driftguard eval add --prompt support-agent \
  --input "Customer asks for a refund" \
  --expected "Response cites the policy ID and never invents a refund."
driftguard eval list --prompt support-agent

# Predict which eval cases a version's change affects (default threshold 0.5)
driftguard select-evals --prompt support-agent --version latest --threshold 0.5
driftguard select-evals --prompt support-agent --json
```

`select-evals` embeds the version's changed span and each eval case's
`expected_behavior`, ranks by pgvector cosine similarity, and writes a
`SelectionRecord` per case (Phase 4 scores precision/recall against these).

- **Embeddings:** Voyage AI `voyage-3.5-lite` (1024-dim) — Anthropic has no
  embeddings API, so `VOYAGE_API_KEY` is required for `select-evals`. The
  provider sits behind an `Embedder` trait (`crates/driftguard-core/src/embed.rs`),
  so the selection path is verifiable offline:
  `cargo run -p driftguard-core --example verify_selection` (no key needed).
- **Default threshold:** 0.5 (favors recall). Phase 4 sweeps the full range.

## Validation pipeline (Phase 4 — the core result)

The headline result: how well similarity-based selection predicts which eval
cases a prompt change actually affects.

```bash
# 1. Load the co-drafted fixtures (prompts + eval cases + edits)
driftguard fixtures load --file fixtures/driftguard-fixtures.toml

# 2. Run the full pipeline: run-evals (model under test) + LLM judge + ground truth
#    Needs VOYAGE_API_KEY and ANTHROPIC_API_KEY. Sequential; makes many calls.
driftguard validate --fixtures fixtures/driftguard-fixtures.toml

# 3. (optional) Spot-check judge verdicts to report an agreement rate
driftguard judge-validate --sample-size 20

# 4. The result: precision / recall / % reduction across a threshold sweep
driftguard score
driftguard score --json   # for the Phase 6 dashboard chart
```

- **Model under test:** `claude-sonnet-4-6` · **Judge:** `claude-opus-4-8`
  (`output_config.format` JSON-schema structured output). Both behind an `Llm`
  trait (`crates/driftguard-core/src/llm.rs`).
- **Ground truth** (`was_actually_affected`): the judge says behavior changed
  substantively (any user-meaningful difference). `score` *also* reports a
  pass/fail-flip view for comparison.
- **Threshold sweep:** 0.30–0.90 step 0.05.
- Verify the whole pipeline offline (no keys):
  `cargo run -p driftguard-core --example verify_validation`.

`run-evals` and `score` also exist as standalone building blocks.

## CI gate (Phase 5)

`.github/workflows/driftguard.yml` runs on PRs that change
`fixtures/driftguard-fixtures.toml`. It diffs the base vs PR version of each
prompt, predicts which eval cases the change affects, runs **only those** against
the new prompt version, judges pass/fail, and posts a sticky PR comment. The
check fails on a selected-eval failure unless the repo variable
`DRIFTGUARD_FAIL_ON_EVAL_FAILURE` is `false`.

- Repo secrets: `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`.
- Repo variables (optional): `DRIFTGUARD_THRESHOLD` (default 0.5),
  `DRIFTGUARD_FAIL_ON_EVAL_FAILURE` (default `true`).
- The `driftguard ci --base <file> --head <file>` command is the building block
  (emits the Markdown comment, or `--json`; exit code is the gate).

```bash
# Reproduce the CI decision locally against two fixtures snapshots:
driftguard ci --base /tmp/base.toml --head fixtures/driftguard-fixtures.toml
```

## Build phases

1. **Cargo workspace + DB schema + migrations** ✅
2. **CLI scaffolding + prompt versioning/diffing** ✅
3. **Embedding + similarity-based eval selection** ✅
4. **Ground-truth validation pipeline (precision/recall)** ✅
5. **GitHub Action integration** ✅ (current)
6. axum API + React dashboard
