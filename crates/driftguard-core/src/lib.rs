//! driftguard-core — all Driftguard business logic.
//!
//! This crate is delivery-agnostic: no clap, no axum, no terminal formatting.
//! It exposes plain functions and types that the CLI (and the Phase 6 axum API)
//! call into. Phase 1 contains only the database connection helper; diffing,
//! embedding, similarity selection, the judge pipeline, and scoring arrive in
//! later phases.

pub mod db;
