// Package core is the Go port of driftguard-core: prompt versioning, diffing,
// eval selection, validation, and scoring. Delivery layers (the CLI and the
// HTTP API) call these functions and own their own I/O — the same "one source
// of truth, thin consumers" boundary the Rust workspace had.
package core

import "fmt"

// Typed errors mirror the Rust CoreError enum: callers (CLI, API) need to tell
// failures apart — render "duplicate name" nicely, map "not found" to 404 —
// so each is a matchable type for errors.As, not a bare fmt.Errorf.

// PromptNotFoundError: no prompt with the given name.
type PromptNotFoundError struct{ Name string }

func (e PromptNotFoundError) Error() string { return fmt.Sprintf("no prompt named %q", e.Name) }

// VersionNotFoundError carries a fully-formed human message (the various ways
// a version ref can fail to resolve don't each need their own type).
type VersionNotFoundError struct{ Msg string }

func (e VersionNotFoundError) Error() string { return e.Msg }

// DuplicateNameError: unique violation on prompts.name.
type DuplicateNameError struct{ Name string }

func (e DuplicateNameError) Error() string {
	return fmt.Sprintf("a prompt named %q already exists", e.Name)
}

// AmbiguousVersionRefError: a UUID prefix matched more than one version.
type AmbiguousVersionRefError struct{ Ref string }

func (e AmbiguousVersionRefError) Error() string {
	return fmt.Sprintf("version reference %q is ambiguous", e.Ref)
}
