//! driftguard — the CLI binary.
//!
//! Phase 1 placeholder. The real clap-based subcommand tree
//! (`prompt create`, `prompt version`, `prompt diff`) plus the global `--json`
//! flag are built in Phase 2, where this file becomes a thin layer that parses
//! args, calls into driftguard-core, formats output, and sets the exit code.
//! Kept minimal here so the workspace compiles and the `driftguard` binary name
//! is wired from the start.

fn main() {
    eprintln!(
        "driftguard CLI — subcommands arrive in Phase 2.\n\
         Phase 1: set up the DB with `docker compose up -d` and `sqlx migrate run`,\n\
         then verify pgvector with `cargo run -p driftguard-core --example verify_pgvector`."
    );
}
