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
- HTTP: `reqwest` (Anthropic API, from Phase 3)
- Errors: `anyhow` (app-level) + `thiserror` (library-level)

## Workspace layout

```
crates/driftguard-core   all business logic (diff, embed, select, judge, score)
crates/driftguard-cli    thin clap binary `driftguard` that calls into core
migrations/              sqlx migrations
crates/driftguard-api    axum JSON API — added in Phase 6 only
```

The core/cli split is deliberate: core holds no delivery concerns, so the CLI
now and the axum API later both call the same functions.

## Local setup (Phase 1)

Requirements: Rust 1.91+, Docker, `sqlx-cli` (`cargo install sqlx-cli@^0.8
--no-default-features --features native-tls,postgres`).

```bash
cp .env.example .env                 # adjust DATABASE_URL / ANTHROPIC_API_KEY
docker compose up -d                 # Postgres + pgvector on host port 5433
sqlx migrate run                     # apply migrations/ to the DB
cargo build                          # build the workspace
cargo run -p driftguard-core --example verify_pgvector   # pgvector smoke test
```

## Build phases

1. **Cargo workspace + DB schema + migrations** ✅ (current)
2. CLI scaffolding + prompt versioning/diffing
3. Embedding + similarity-based eval selection
4. Ground-truth validation pipeline (precision/recall — the core result)
5. GitHub Action integration
6. axum API + React dashboard
