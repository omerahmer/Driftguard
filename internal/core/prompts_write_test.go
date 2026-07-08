package core

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// Round-trip test of the write path against the live DB (skipped without
// DATABASE_URL). Creates a throwaway prompt, adds two versions, exercises
// ResolveVersionRef and the duplicate-name error, and deletes everything it
// created.
func TestWritePathRoundTrip(t *testing.T) {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		t.Skip("DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := Connect(ctx, url)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	name := fmt.Sprintf("go-write-test-%d", time.Now().UnixNano())
	prompt, err := CreatePrompt(ctx, pool, name)
	if err != nil {
		t.Fatalf("CreatePrompt: %v", err)
	}
	// defer, NOT t.Cleanup: Cleanup callbacks run after deferred calls, so the
	// pool (closed by the defer above) would already be gone and the deletes
	// would silently fail, leaving test rows behind.
	defer func() {
		// Clear the current pointer first (FK to prompt_versions), then cascade
		// by hand: versions, then the prompt.
		for _, q := range []string{
			`UPDATE prompts SET current_version_id = NULL WHERE id = $1`,
			`DELETE FROM prompt_versions WHERE prompt_id = $1`,
			`DELETE FROM prompts WHERE id = $1`,
		} {
			if _, err := pool.Exec(ctx, q, prompt.ID); err != nil {
				t.Errorf("cleanup %q: %v", q, err)
			}
		}
	}()

	if _, err := CreatePrompt(ctx, pool, name); err == nil {
		t.Fatal("duplicate CreatePrompt: want DuplicateNameError, got nil")
	} else if _, ok := err.(DuplicateNameError); !ok {
		t.Fatalf("duplicate CreatePrompt: want DuplicateNameError, got %T (%v)", err, err)
	}

	v1Content := "You are a test agent. Cite the policy ID. Be concise."
	out1, err := CreateVersion(ctx, pool, name, v1Content)
	if err != nil {
		t.Fatalf("CreateVersion v1: %v", err)
	}
	if out1.Diff != nil || out1.ParentVersionID != nil || out1.Unchanged {
		t.Fatalf("v1: want no parent/diff, got %+v", out1)
	}

	v2Content := "You are a test agent. Cite the policy ID and link. Be concise."
	out2, err := CreateVersion(ctx, pool, name, v2Content)
	if err != nil {
		t.Fatalf("CreateVersion v2: %v", err)
	}
	if out2.Diff == nil || out2.ParentVersionID == nil || *out2.ParentVersionID != out1.Version.ID {
		t.Fatalf("v2: want diff + parent=v1, got %+v", out2)
	}
	wantSpan := "Cite the policy ID and link."
	if len(out2.Diff.ChangedSpans) != 1 || out2.Diff.ChangedSpans[0] != wantSpan {
		t.Fatalf("v2 changed_spans: got %v, want [%q]", out2.Diff.ChangedSpans, wantSpan)
	}

	// The stored diff round-trips byte-identically through our serializer.
	stored, err := GetVersionByID(ctx, pool, out2.Version.ID)
	if err != nil || stored == nil || stored.DiffFromParent == nil {
		t.Fatalf("read back v2: %v / %+v", err, stored)
	}
	reser, err := marshalCompact(ComputeDiff(v1Content, v2Content))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if *stored.DiffFromParent != reser {
		t.Fatalf("stored diff != recomputed\nstored: %s\nrecomputed: %s", *stored.DiffFromParent, reser)
	}

	// Identical content → unchanged=true.
	out3, err := CreateVersion(ctx, pool, name, v2Content)
	if err != nil {
		t.Fatalf("CreateVersion v3: %v", err)
	}
	if !out3.Unchanged {
		t.Fatal("v3 identical content: want unchanged=true")
	}

	// ResolveVersionRef: latest, parent, vN, prefix, ambiguity behavior.
	fresh, err := GetPromptByName(ctx, pool, name)
	if err != nil || fresh == nil {
		t.Fatalf("refetch prompt: %v", err)
	}
	for ref, wantID := range map[string]interface{}{
		"latest": out3.Version.ID, "current": out3.Version.ID,
		"v1": out1.Version.ID, "v2": out2.Version.ID,
		"parent":                     out2.Version.ID,
		out1.Version.ID.String()[:8]: out1.Version.ID,
		"  V2  ":                     out2.Version.ID, // trimmed + lowercased
	} {
		got, err := ResolveVersionRef(ctx, pool, fresh, ref)
		if err != nil {
			t.Errorf("resolve %q: %v", ref, err)
			continue
		}
		if got.ID != wantID {
			t.Errorf("resolve %q: got %s, want %s", ref, got.ID, wantID)
		}
	}
	if _, err := ResolveVersionRef(ctx, pool, fresh, "v99"); err == nil {
		t.Error("resolve v99: want error")
	}
}
