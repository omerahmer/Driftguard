package core

// Similarity-based eval selection (port of selection.rs).
//
// Given a prompt version's changed span, embed it, embed each eval case's
// expected_behavior, and rank eval cases by cosine similarity (computed by
// pgvector). Cases at or above the threshold are "selected" — these are the
// evals CI would run for that change. Every case + score is persisted as a
// selection record so scoring can measure precision/recall against ground
// truth. Embeddings are bound as `$1::vector` text literals (see embed.go).

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type SelectionItem struct {
	EvalCaseID       uuid.UUID `json:"eval_case_id"`
	Input            string    `json:"input"`
	ExpectedBehavior string    `json:"expected_behavior"`
	Similarity       float64   `json:"similarity"`
	WasSelected      bool      `json:"was_selected"`
}

type SelectionOutcome struct {
	Prompt    string    `json:"prompt"`
	VersionID uuid.UUID `json:"version_id"`
	Threshold float64   `json:"threshold"`
	Total     int       `json:"total"`
	Selected  int       `json:"selected"`
	// Fraction of eval cases NOT selected — the "% reduction in evals run".
	Reduction float64         `json:"reduction"`
	Items     []SelectionItem `json:"items"`
}

// SelectEvals runs similarity selection for a prompt version and persists the
// records.
func SelectEvals(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, promptName, versionRef string, threshold float64) (*SelectionOutcome, error) {
	prompt, err := GetPromptByName(ctx, pool, promptName)
	if err != nil {
		return nil, err
	}
	if prompt == nil {
		return nil, PromptNotFoundError{Name: promptName}
	}
	version, err := ResolveVersionRef(ctx, pool, prompt, versionRef)
	if err != nil {
		return nil, err
	}

	// Make sure both sides are embedded (lazy, cached in the DB).
	if err := ensureVersionEmbedding(ctx, pool, embedder, version); err != nil {
		return nil, err
	}

	cases, err := ListEvalCases(ctx, pool, prompt.ID)
	if err != nil {
		return nil, err
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("prompt %q has no eval cases; add some with `eval add`", prompt.Name)
	}
	if err := ensureEvalEmbeddings(ctx, pool, embedder, prompt.ID); err != nil {
		return nil, err
	}

	// Cosine similarity via pgvector's `<=>` (distance); similarity = 1 - distance.
	rows, err := pool.Query(ctx, `
		SELECT ec.id, ec.input, ec.expected_behavior,
		       1 - (ec.expected_behavior_embedding <=> pv.changed_span_embedding) AS similarity
		FROM eval_cases ec
		CROSS JOIN prompt_versions pv
		WHERE pv.id = $1
		  AND ec.prompt_id = $2
		  AND ec.expected_behavior_embedding IS NOT NULL
		ORDER BY similarity DESC`, version.ID, prompt.ID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	items := []SelectionItem{}
	for rows.Next() {
		var it SelectionItem
		if err := rows.Scan(&it.EvalCaseID, &it.Input, &it.ExpectedBehavior, &it.Similarity); err != nil {
			return nil, err
		}
		it.WasSelected = it.Similarity >= threshold
		items = append(items, it)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if err := persistSelectionRecords(ctx, pool, version.ID, items); err != nil {
		return nil, err
	}

	total := len(items)
	selected := 0
	for _, it := range items {
		if it.WasSelected {
			selected++
		}
	}
	reduction := 0.0
	if total > 0 {
		reduction = 1.0 - float64(selected)/float64(total)
	}

	return &SelectionOutcome{
		Prompt:    prompt.Name,
		VersionID: version.ID,
		Threshold: threshold,
		Total:     total,
		Selected:  selected,
		Reduction: reduction,
		Items:     items,
	}, nil
}

// ensureVersionEmbedding embeds a version's changed span if it isn't embedded
// yet.
func ensureVersionEmbedding(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, version *PromptVersion) error {
	var existing *string
	if err := pool.QueryRow(ctx,
		`SELECT changed_span_embedding::text FROM prompt_versions WHERE id = $1`,
		version.ID).Scan(&existing); err != nil {
		return err
	}
	if existing != nil {
		return nil
	}

	if version.DiffFromParent == nil {
		return fmt.Errorf("version %s has no parent diff, so there is no changed span to embed. "+
			"select-evals needs a version created from a parent.", version.ID)
	}
	var diff PromptDiff
	if err := json.Unmarshal([]byte(*version.DiffFromParent), &diff); err != nil {
		return err
	}
	if len(diff.ChangedSpans) == 0 {
		return fmt.Errorf("this version's changed span is empty (content is identical to its parent)")
	}

	// Concatenate the added-side spans into one text and embed as the query side.
	text := strings.Join(diff.ChangedSpans, "\n")
	vectors, err := embedder.Embed(ctx, []string{text}, InputQuery)
	if err != nil {
		return err
	}
	if len(vectors) == 0 {
		return fmt.Errorf("embedding provider error: embedder returned no vectors")
	}

	_, err = pool.Exec(ctx,
		`UPDATE prompt_versions SET changed_span_embedding = $1::vector WHERE id = $2`,
		ToPgvectorLiteral(vectors[0]), version.ID)
	return err
}

// ensureEvalEmbeddings embeds any eval cases for this prompt that don't have an
// embedding yet, in one batched API call.
func ensureEvalEmbeddings(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, promptID uuid.UUID) error {
	rows, err := pool.Query(ctx,
		`SELECT id, expected_behavior FROM eval_cases
		 WHERE prompt_id = $1 AND expected_behavior_embedding IS NULL`, promptID)
	if err != nil {
		return err
	}
	defer rows.Close()

	var ids []uuid.UUID
	var texts []string
	for rows.Next() {
		var id uuid.UUID
		var text string
		if err := rows.Scan(&id, &text); err != nil {
			return err
		}
		ids = append(ids, id)
		texts = append(texts, text)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}

	vectors, err := embedder.Embed(ctx, texts, InputDocument)
	if err != nil {
		return err
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	for i, id := range ids {
		if _, err := tx.Exec(ctx,
			`UPDATE eval_cases SET expected_behavior_embedding = $1::vector WHERE id = $2`,
			ToPgvectorLiteral(vectors[i]), id); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

// persistSelectionRecords replaces this version's selection records with the
// latest run (idempotent — re-running select-evals overwrites rather than
// accumulating duplicates).
func persistSelectionRecords(ctx context.Context, pool *pgxpool.Pool, versionID uuid.UUID, items []SelectionItem) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx,
		`DELETE FROM selection_records WHERE prompt_version_id = $1`, versionID); err != nil {
		return err
	}
	for _, it := range items {
		if _, err := tx.Exec(ctx,
			`INSERT INTO selection_records
			     (prompt_version_id, eval_case_id, similarity_score, was_selected)
			 VALUES ($1, $2, $3, $4)`,
			versionID, it.EvalCaseID, it.Similarity, it.WasSelected); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}
