package core

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

// Golden vectors generated from the Rust implementation:
//   cargo run -p driftguard-core --example gen_segment_goldens \
//     > internal/core/testdata/segment_goldens.json
// They pin the Go port to the exact segmentation, sentence-splitting, and
// serialized-diff output that produced every diff_from_parent in the DB.

type goldens struct {
	Segmentation []struct {
		Input    string `json:"input"`
		Segments []struct {
			Kind string `json:"kind"`
			Text string `json:"text"`
		} `json:"segments"`
	} `json:"segmentation"`
	Sentences []struct {
		Input     string   `json:"input"`
		Sentences []string `json:"sentences"`
	} `json:"sentences"`
	Diffs []struct {
		Before   string `json:"before"`
		After    string `json:"after"`
		DiffJSON string `json:"diff_json"`
	} `json:"diffs"`
}

func loadGoldens(t *testing.T) goldens {
	t.Helper()
	raw, err := os.ReadFile("testdata/segment_goldens.json")
	if err != nil {
		t.Fatalf("read goldens: %v", err)
	}
	var g goldens
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse goldens: %v", err)
	}
	return g
}

// marshalDiff serializes the way persisted diffs are written (see json.go).
func marshalDiff(d PromptDiff) (string, error) {
	return marshalCompact(d)
}

func TestSegmentationGoldens(t *testing.T) {
	for i, c := range loadGoldens(t).Segmentation {
		got := SegmentPrompt(c.Input)
		if len(got) != len(c.Segments) {
			t.Errorf("case %d: got %d segments, want %d\ninput: %q", i, len(got), len(c.Segments), c.Input)
			continue
		}
		for k, seg := range got {
			if string(seg.Kind) != c.Segments[k].Kind || seg.Text != c.Segments[k].Text {
				t.Errorf("case %d seg %d: got (%s, %q), want (%s, %q)",
					i, k, seg.Kind, seg.Text, c.Segments[k].Kind, c.Segments[k].Text)
			}
		}
	}
}

func TestSentenceGoldens(t *testing.T) {
	for i, c := range loadGoldens(t).Sentences {
		got := SplitSentences(c.Input)
		if len(got) != len(c.Sentences) {
			t.Errorf("case %d: got %v, want %v", i, got, c.Sentences)
			continue
		}
		for k := range got {
			if got[k] != c.Sentences[k] {
				t.Errorf("case %d sentence %d: got %q, want %q", i, k, got[k], c.Sentences[k])
			}
		}
	}
}

// TestDiffGoldens asserts BYTE equality of the serialized diff against what
// serde_json produced — the same bytes Rust persisted into diff_from_parent.
func TestDiffGoldens(t *testing.T) {
	for i, c := range loadGoldens(t).Diffs {
		got, err := marshalDiff(ComputeDiff(c.Before, c.After))
		if err != nil {
			t.Fatalf("case %d: marshal: %v", i, err)
		}
		if got != c.DiffJSON {
			t.Errorf("case %d: serialized diff mismatch\nbefore: %.80q\nafter:  %.80q\ngot:  %s\nwant: %s",
				i, c.Before, c.After, got, c.DiffJSON)
		}
	}
}

// TestDiffAgainstStoredDB recomputes the diff for every version in the live DB
// and compares against the Rust-written diff_from_parent column. Runs only
// when DATABASE_URL is set (skipped in bare CI).
func TestDiffAgainstStoredDB(t *testing.T) {
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

	rows, err := pool.Query(ctx, `
		SELECT child.id, parent.content, child.content, child.diff_from_parent
		FROM prompt_versions child
		JOIN prompt_versions parent ON parent.id = child.parent_version_id
		WHERE child.diff_from_parent IS NOT NULL`)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()

	checked := 0
	for rows.Next() {
		var id, parentContent, content, stored string
		if err := rows.Scan(&id, &parentContent, &content, &stored); err != nil {
			t.Fatalf("scan: %v", err)
		}
		got, err := marshalDiff(ComputeDiff(parentContent, content))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if got != stored {
			t.Errorf("version %s: recomputed diff != stored diff\ngot:    %s\nstored: %s", id, got, stored)
		}
		checked++
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	if checked == 0 {
		t.Skip("no stored diffs to compare")
	}
	t.Logf("byte-compared %d stored diffs", checked)
}
