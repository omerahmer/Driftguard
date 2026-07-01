//! driftguard-core — all Driftguard business logic.
//!
//! This crate is delivery-agnostic: no clap, no axum, no terminal formatting.
//! It exposes plain functions and types that the CLI (and the Phase 6 axum API)
//! call into. Phase 2 adds structure-aware prompt diffing ([`segment`]) and the
//! prompt-versioning services ([`prompts`]); embedding, the judge pipeline, and
//! scoring arrive in later phases.

pub mod ci;
pub mod db;
pub mod embed;
pub mod error;
pub mod evals;
pub mod fixtures;
pub mod llm;
pub mod models;
pub mod prompts;
pub mod scoring;
pub mod segment;
pub mod selection;
pub mod validation;

pub use error::CoreError;
