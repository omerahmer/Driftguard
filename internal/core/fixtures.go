package core

// Fixtures loading (port of fixtures.rs). Each prompt has a base content
// (becomes v1), eval cases, and edits; every edit's content is the FULL
// post-edit prompt, loaded as a new version branched off v1 so its changed
// span is measured against the base.

import (
	"context"
	"fmt"

	"github.com/BurntSushi/toml"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Fixtures struct {
	Prompts []FixturePrompt `toml:"prompts"`
}

type FixturePrompt struct {
	Name      string            `toml:"name"`
	Content   string            `toml:"content"`
	EvalCases []FixtureEvalCase `toml:"eval_cases"`
	Edits     []FixtureEdit     `toml:"edits"`
}

type FixtureEvalCase struct {
	Input            string `toml:"input"`
	ExpectedBehavior string `toml:"expected_behavior"`
}

type FixtureEdit struct {
	// Human label, e.g. "added-constraint".
	Label string `toml:"label"`
	// Edit category for reporting (wording / added / removed / tone / reorder).
	EditType string `toml:"edit_type"`
	// The FULL prompt content after applying this edit to the base.
	Content string `toml:"content"`
}

// ParseFixtures parses a fixtures TOML document.
func ParseFixtures(text string) (*Fixtures, error) {
	var f Fixtures
	if _, err := toml.Decode(text, &f); err != nil {
		return nil, fmt.Errorf("fixtures error: %w", err)
	}
	return &f, nil
}

type LoadSummary struct {
	Prompts         int `json:"prompts"`
	EvalCases       int `json:"eval_cases"`
	Versions        int `json:"versions"`
	SkippedExisting int `json:"skipped_existing"`
}

// LoadFixtures loads fixtures into the DB. Prompts that already exist are
// skipped (so this is safe to call from validate); force deletes and recreates
// them for a clean reload.
func LoadFixtures(ctx context.Context, pool *pgxpool.Pool, fixtures *Fixtures, force bool) (*LoadSummary, error) {
	summary := &LoadSummary{}

	for _, fp := range fixtures.Prompts {
		existing, err := GetPromptByName(ctx, pool, fp.Name)
		if err != nil {
			return nil, err
		}
		if existing != nil {
			if force {
				// Cascade-deletes versions, eval cases, runs, selections, diffs.
				if _, err := pool.Exec(ctx, `DELETE FROM prompts WHERE id = $1`, existing.ID); err != nil {
					return nil, err
				}
			} else {
				summary.SkippedExisting++
				continue
			}
		}

		// Base version (v1).
		if _, err := CreatePrompt(ctx, pool, fp.Name); err != nil {
			return nil, err
		}
		base, err := CreateVersion(ctx, pool, fp.Name, fp.Content)
		if err != nil {
			return nil, err
		}
		summary.Prompts++
		summary.Versions++

		for _, c := range fp.EvalCases {
			if _, err := CreateEvalCase(ctx, pool, fp.Name, c.Input, c.ExpectedBehavior); err != nil {
				return nil, err
			}
			summary.EvalCases++
		}

		// Each edit branches off the base version.
		for _, edit := range fp.Edits {
			if _, err := CreateVersionFromParent(ctx, pool, base.Version.PromptID, base.Version.ID, edit.Content); err != nil {
				return nil, err
			}
			summary.Versions++
		}
	}

	return summary, nil
}
