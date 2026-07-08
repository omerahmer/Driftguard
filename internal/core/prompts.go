package core

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Prompt versioning services: plain functions over a *pgxpool.Pool — no cobra,
// no printing, no exit codes. The CLI and the HTTP API call these and own
// their own I/O.
//
// Read paths only for now; the write path (create version + structural diff)
// lands with the diff engine port (Phase G3).

const promptCols = `id, name, current_version_id, created_at`
const versionCols = `id, prompt_id, content, parent_version_id, diff_from_parent, created_at`

func scanPrompt(row pgx.Row) (Prompt, error) {
	var p Prompt
	err := row.Scan(&p.ID, &p.Name, &p.CurrentVersionID, &p.CreatedAt)
	// pgx scans timestamptz in the session's local zone; the wire contract
	// (established by chrono's DateTime<Utc>) is UTC.
	p.CreatedAt = p.CreatedAt.UTC()
	return p, err
}

func scanVersion(row pgx.Row) (PromptVersion, error) {
	var v PromptVersion
	err := row.Scan(&v.ID, &v.PromptID, &v.Content, &v.ParentVersionID, &v.DiffFromParent, &v.CreatedAt)
	v.CreatedAt = v.CreatedAt.UTC()
	return v, err
}

// GetPromptByName returns the prompt or nil if no prompt has that name.
func GetPromptByName(ctx context.Context, pool *pgxpool.Pool, name string) (*Prompt, error) {
	row := pool.QueryRow(ctx,
		`SELECT `+promptCols+` FROM prompts WHERE name = $1`, name)
	p, err := scanPrompt(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

func ListPrompts(ctx context.Context, pool *pgxpool.Pool) ([]Prompt, error) {
	rows, err := pool.Query(ctx,
		`SELECT `+promptCols+` FROM prompts ORDER BY created_at ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Initialized (not nil) so an empty table serializes as JSON [] — the
	// contract the Rust Vec established.
	prompts := []Prompt{}
	for rows.Next() {
		p, err := scanPrompt(rows)
		if err != nil {
			return nil, err
		}
		prompts = append(prompts, p)
	}
	return prompts, rows.Err()
}

// ListVersions returns a prompt's versions, oldest first (the ordinal v1, v2…
// is the index + 1).
func ListVersions(ctx context.Context, pool *pgxpool.Pool, promptID uuid.UUID) ([]PromptVersion, error) {
	rows, err := pool.Query(ctx,
		`SELECT `+versionCols+` FROM prompt_versions WHERE prompt_id = $1 ORDER BY created_at ASC`,
		promptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	versions := []PromptVersion{}
	for rows.Next() {
		v, err := scanVersion(rows)
		if err != nil {
			return nil, err
		}
		versions = append(versions, v)
	}
	return versions, rows.Err()
}

// GetVersionByID returns the version or nil if the id doesn't exist.
func GetVersionByID(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (*PromptVersion, error) {
	row := pool.QueryRow(ctx,
		`SELECT `+versionCols+` FROM prompt_versions WHERE id = $1`, id)
	v, err := scanVersion(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &v, nil
}

// ── Write path ──────────────────────────────────────────────────────────────

// CreateVersionOutcome is the result of adding a version, with everything the
// CLI needs to report.
type CreateVersionOutcome struct {
	Version         PromptVersion `json:"version"`
	ParentVersionID *uuid.UUID    `json:"parent_version_id"`
	Diff            *PromptDiff   `json:"diff"`
	// Unchanged is true when the new content is identical to its parent
	// (empty changed span).
	Unchanged bool `json:"unchanged"`
}

// CreatePrompt creates an empty prompt (no versions yet). Per the spec,
// `prompt create` only registers the name; the first `prompt version`
// becomes v1.
func CreatePrompt(ctx context.Context, pool *pgxpool.Pool, name string) (*Prompt, error) {
	row := pool.QueryRow(ctx,
		`INSERT INTO prompts (name) VALUES ($1) RETURNING `+promptCols, name)
	p, err := scanPrompt(row)
	if err != nil {
		// 23505 = unique_violation on prompts.name. Translate the opaque DB
		// error into a typed variant the CLI can render nicely.
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, DuplicateNameError{Name: name}
		}
		return nil, err
	}
	return &p, nil
}

// CreateVersion adds a new version of an existing prompt: computes the diff
// vs. the current version, persists the version, and advances
// current_version_id — atomically.
func CreateVersion(ctx context.Context, pool *pgxpool.Pool, name, content string) (*CreateVersionOutcome, error) {
	prompt, err := GetPromptByName(ctx, pool, name)
	if err != nil {
		return nil, err
	}
	if prompt == nil {
		return nil, PromptNotFoundError{Name: name}
	}

	parentVersionID := prompt.CurrentVersionID

	// Compute the diff against the parent's content (if there is a parent).
	var diff *PromptDiff
	if parentVersionID != nil {
		parent, err := GetVersionByID(ctx, pool, *parentVersionID)
		if err != nil {
			return nil, err
		}
		if parent == nil {
			return nil, VersionNotFoundError{Msg: fmt.Sprintf("parent version %s is missing", parentVersionID)}
		}
		d := ComputeDiff(parent.Content, content)
		diff = &d
	}

	unchanged := diff != nil && len(diff.ChangedSpans) == 0 && len(diff.RemovedSpans) == 0

	version, err := insertVersion(ctx, pool, prompt.ID, content, parentVersionID, diff)
	if err != nil {
		return nil, err
	}
	return &CreateVersionOutcome{
		Version:         *version,
		ParentVersionID: parentVersionID,
		Diff:            diff,
		Unchanged:       unchanged,
	}, nil
}

// CreateVersionFromParent creates a version whose parent is an EXPLICIT
// version (not necessarily the current head). Used by the fixtures loader,
// where every edit branches off the same base v1 so each edit's diff/changed
// span is measured against the base. Advances current_version_id to the new
// version (harmless for fixtures, which address versions explicitly).
func CreateVersionFromParent(ctx context.Context, pool *pgxpool.Pool, promptID, parentVersionID uuid.UUID, content string) (*PromptVersion, error) {
	parent, err := GetVersionByID(ctx, pool, parentVersionID)
	if err != nil {
		return nil, err
	}
	if parent == nil {
		return nil, VersionNotFoundError{Msg: fmt.Sprintf("parent version %s not found", parentVersionID)}
	}
	diff := ComputeDiff(parent.Content, content)
	return insertVersion(ctx, pool, promptID, content, &parentVersionID, &diff)
}

// insertVersion runs the shared transaction: the version insert and the
// current-pointer update must either both land or neither — if we crashed
// between them, the prompt would point at a stale version.
func insertVersion(ctx context.Context, pool *pgxpool.Pool, promptID uuid.UUID, content string, parentVersionID *uuid.UUID, diff *PromptDiff) (*PromptVersion, error) {
	var diffJSON *string
	if diff != nil {
		// Serialized like serde_json: compact, no HTML escaping — keeps
		// diff_from_parent byte-compatible with rows the Rust CLI wrote.
		s, err := marshalCompact(diff)
		if err != nil {
			return nil, err
		}
		diffJSON = &s
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	row := tx.QueryRow(ctx,
		`INSERT INTO prompt_versions (prompt_id, content, parent_version_id, diff_from_parent)
		 VALUES ($1, $2, $3, $4)
		 RETURNING `+versionCols,
		promptID, content, parentVersionID, diffJSON)
	version, err := scanVersion(row)
	if err != nil {
		return nil, err
	}

	if _, err := tx.Exec(ctx,
		`UPDATE prompts SET current_version_id = $1 WHERE id = $2`,
		version.ID, promptID); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &version, nil
}

// ResolveVersionRef resolves a human-friendly version reference within a
// prompt. Accepts: latest/current, parent, vN (1-based ordinal), or a full
// UUID or unambiguous UUID prefix. Scoped to the prompt so short refs are
// reusable.
func ResolveVersionRef(ctx context.Context, pool *pgxpool.Pool, prompt *Prompt, reference string) (*PromptVersion, error) {
	versions, err := ListVersions(ctx, pool, prompt.ID)
	if err != nil {
		return nil, err
	}
	if len(versions) == 0 {
		return nil, VersionNotFoundError{Msg: fmt.Sprintf("prompt %q has no versions", prompt.Name)}
	}

	normalized := strings.ToLower(strings.TrimSpace(reference))

	if normalized == "latest" || normalized == "current" {
		for _, v := range versions {
			if prompt.CurrentVersionID != nil && v.ID == *prompt.CurrentVersionID {
				return &v, nil
			}
		}
		return nil, VersionNotFoundError{Msg: "prompt has no current version"}
	}

	if normalized == "parent" {
		var current *PromptVersion
		for i, v := range versions {
			if prompt.CurrentVersionID != nil && v.ID == *prompt.CurrentVersionID {
				current = &versions[i]
				break
			}
		}
		if current != nil && current.ParentVersionID != nil {
			for _, v := range versions {
				if v.ID == *current.ParentVersionID {
					return &v, nil
				}
			}
		}
		return nil, VersionNotFoundError{Msg: "current version has no parent"}
	}

	if rest, ok := strings.CutPrefix(normalized, "v"); ok {
		if n, err := strconv.Atoi(rest); err == nil && n >= 0 {
			if n >= 1 && n <= len(versions) {
				return &versions[n-1], nil
			}
			return nil, VersionNotFoundError{Msg: fmt.Sprintf(
				"prompt %q has no version v%d (it has %d)", prompt.Name, n, len(versions))}
		}
	}

	// UUID or prefix match, scoped to this prompt.
	var matches []*PromptVersion
	for i := range versions {
		if strings.HasPrefix(versions[i].ID.String(), normalized) {
			matches = append(matches, &versions[i])
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return nil, VersionNotFoundError{Msg: fmt.Sprintf(
			"could not resolve version %q (use latest, parent, vN, or an id)", reference)}
	default:
		return nil, AmbiguousVersionRefError{Ref: reference}
	}
}
