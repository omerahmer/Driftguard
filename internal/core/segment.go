package core

// Structure-aware prompt segmentation + diffing (port of segment.rs).
//
// Why this design (chosen over plain line-level or plain sentence-level):
// prompts vary wildly — terse one-liners, long prose paragraphs, Markdown-ish
// bulleted constraint lists, headings, embedded code/JSON examples. A pure line
// diff makes a one-word edit look like a whole-paragraph change (coarse span →
// noisy embedding). A pure sentence diff mangles lists and code.
//
// So we segment hierarchically and adaptively:
//   - fenced code blocks stay atomic (never sentence-split),
//   - headings and list items are one segment per line,
//   - prose paragraphs (possibly hard-wrapped) are joined then sentence-split.
//
// It's deterministic and free (no model calls): the "changed span" produced
// here is the independent variable the selection phase embeds and the scoring
// phase measures precision/recall against.
//
// The diff itself is an LCS at segment granularity with delete-before-insert
// tie-breaking — the same op stream shape the Rust `similar` Myers produced
// (verified against golden vectors in segment_test.go and against every
// Rust-written diff_from_parent in the DB).

import (
	"strings"
	"unicode"
)

type SegmentKind string

const (
	SegmentCode     SegmentKind = "code"
	SegmentHeading  SegmentKind = "heading"
	SegmentListItem SegmentKind = "list-item"
	SegmentSentence SegmentKind = "sentence"
)

type Segment struct {
	Kind SegmentKind
	Text string
}

type DiffOpType string

const (
	OpEqual   DiffOpType = "equal"
	OpAdded   DiffOpType = "added"
	OpRemoved DiffOpType = "removed"
)

type DiffOp struct {
	Type DiffOpType `json:"type"`
	Text string     `json:"text"`
}

type DiffStats struct {
	Added     int `json:"added"`
	Removed   int `json:"removed"`
	Unchanged int `json:"unchanged"`
}

// PromptDiff's field order matches the Rust struct so serialized JSON is
// key-order identical to what Rust wrote into prompt_versions.diff_from_parent.
type PromptDiff struct {
	// Identifier for the segmentation strategy, stored alongside the diff so a
	// future change in granularity is detectable in old records.
	Granularity string `json:"granularity"`
	// Full ordered op list — enough to render the diff without recomputing it.
	Ops []DiffOp `json:"ops"`
	// After-side text of added segments: exactly what selection embeds as the
	// "changed span" of the edit.
	ChangedSpans []string `json:"changed_spans"`
	// Before-side text of removed segments.
	RemovedSpans []string  `json:"removed_spans"`
	Stats        DiffStats `json:"stats"`
}

// ComputeDiff diffs two prompt contents at segment granularity. A reworded
// segment surfaces as a removed (old) + added (new) pair — we don't detect
// in-place edits, because for selection the new text is what we embed and the
// old text is what we lost, and both are captured.
func ComputeDiff(before, after string) PromptDiff {
	beforeSegs := segmentTexts(before)
	afterSegs := segmentTexts(after)

	ops := diffSegments(beforeSegs, afterSegs)

	changed := []string{}
	removed := []string{}
	unchanged := 0
	for _, op := range ops {
		switch op.Type {
		case OpAdded:
			changed = append(changed, op.Text)
		case OpRemoved:
			removed = append(removed, op.Text)
		default:
			unchanged++
		}
	}

	return PromptDiff{
		Granularity:  "structure-aware-v1",
		Ops:          ops,
		ChangedSpans: changed,
		RemovedSpans: removed,
		Stats:        DiffStats{Added: len(changed), Removed: len(removed), Unchanged: unchanged},
	}
}

func segmentTexts(content string) []string {
	segs := SegmentPrompt(content)
	texts := make([]string, len(segs))
	for i, s := range segs {
		texts[i] = s.Text
	}
	return texts
}

// diffSegments runs the ported similar pipeline (myers.go) and flattens the
// range ops into per-segment ops, exactly as the Rust code flattened
// `similar`'s op list: Equal per segment, and each changed block as its
// removals (old side) followed by its additions (new side) — the ordering
// similar's Replace hook guarantees.
func diffSegments(before, after []string) []DiffOp {
	ops := []DiffOp{}
	var pendingDel, pendingIns []int

	flush := func() {
		for _, i := range pendingDel {
			ops = append(ops, DiffOp{Type: OpRemoved, Text: before[i]})
		}
		for _, j := range pendingIns {
			ops = append(ops, DiffOp{Type: OpAdded, Text: after[j]})
		}
		pendingDel = pendingDel[:0]
		pendingIns = pendingIns[:0]
	}

	for _, op := range myersDiff(before, after) {
		switch op.tag {
		case tagEqual:
			flush()
			for k := 0; k < op.newLen; k++ {
				ops = append(ops, DiffOp{Type: OpEqual, Text: after[op.newIndex+k]})
			}
		case tagDelete:
			for k := 0; k < op.oldLen; k++ {
				pendingDel = append(pendingDel, op.oldIndex+k)
			}
		case tagInsert:
			for k := 0; k < op.newLen; k++ {
				pendingIns = append(pendingIns, op.newIndex+k)
			}
		}
	}
	flush()
	return ops
}

// SegmentPrompt segments a prompt into comparable units per the strategy in
// the package docs.
func SegmentPrompt(content string) []Segment {
	var segments []Segment
	var paragraph []string
	var codeBuffer []string
	inCode := false

	for _, line := range strings.Split(content, "\n") {
		if isFence(line) {
			if inCode {
				codeBuffer = append(codeBuffer, line)
				segments, codeBuffer = flushCode(segments, codeBuffer)
				inCode = false
			} else {
				segments, paragraph = flushParagraph(segments, paragraph)
				inCode = true
				codeBuffer = append(codeBuffer, line)
			}
			continue
		}
		if inCode {
			codeBuffer = append(codeBuffer, line)
			continue
		}
		if strings.TrimSpace(line) == "" {
			segments, paragraph = flushParagraph(segments, paragraph)
			continue
		}
		if isHeading(line) {
			segments, paragraph = flushParagraph(segments, paragraph)
			segments = append(segments, Segment{Kind: SegmentHeading, Text: strings.TrimSpace(line)})
			continue
		}
		if isListItem(line) {
			segments, paragraph = flushParagraph(segments, paragraph)
			segments = append(segments, Segment{Kind: SegmentListItem, Text: strings.TrimSpace(line)})
			continue
		}
		paragraph = append(paragraph, line)
	}

	segments, _ = flushParagraph(segments, paragraph)
	segments, _ = flushCode(segments, codeBuffer) // handles an unterminated fence

	return segments
}

func flushParagraph(segments []Segment, paragraph []string) ([]Segment, []string) {
	if len(paragraph) == 0 {
		return segments, paragraph[:0]
	}
	// Join hard-wrapped lines, collapse whitespace, then sentence-split.
	normalized := strings.Join(strings.Fields(strings.Join(paragraph, " ")), " ")
	for _, sentence := range SplitSentences(normalized) {
		segments = append(segments, Segment{Kind: SegmentSentence, Text: sentence})
	}
	return segments, paragraph[:0]
}

func flushCode(segments []Segment, codeBuffer []string) ([]Segment, []string) {
	if len(codeBuffer) == 0 {
		return segments, codeBuffer[:0]
	}
	segments = append(segments, Segment{
		Kind: SegmentCode,
		Text: strings.TrimSpace(strings.Join(codeBuffer, "\n")),
	})
	return segments, codeBuffer[:0]
}

func isFence(line string) bool {
	t := strings.TrimLeftFunc(line, unicode.IsSpace)
	return strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~")
}

func isHeading(line string) bool {
	t := strings.TrimLeftFunc(line, unicode.IsSpace)
	hashes := 0
	for hashes < len(t) && t[hashes] == '#' {
		hashes++
	}
	if hashes == 0 || hashes > 6 {
		return false
	}
	rest := t[hashes:]
	if rest == "" || (rest[0] != ' ' && rest[0] != '\t') {
		return false
	}
	return strings.TrimSpace(rest) != ""
}

func isListItem(line string) bool {
	t := strings.TrimLeftFunc(line, unicode.IsSpace)
	// Unordered: -, *, + followed by whitespace and content.
	if len(t) > 0 && (t[0] == '-' || t[0] == '*' || t[0] == '+') {
		rest := t[1:]
		return len(rest) > 0 && (rest[0] == ' ' || rest[0] == '\t') && strings.TrimSpace(rest) != ""
	}
	// Ordered: digits then '.' or ')' then whitespace and content.
	digits := 0
	for digits < len(t) && t[digits] >= '0' && t[digits] <= '9' {
		digits++
	}
	if digits > 0 && digits < len(t) && (t[digits] == '.' || t[digits] == ')') {
		rest := t[digits+1:]
		return len(rest) > 0 && (rest[0] == ' ' || rest[0] == '\t') && strings.TrimSpace(rest) != ""
	}
	return false
}

// SplitSentences is a heuristic sentence splitter. Deterministic,
// dependency-free. Splits on `.`/`!`/`?` when whitespace follows and the next
// non-space char looks like a new sentence (or the string ends), while
// guarding a short abbreviation list.
func SplitSentences(text string) []string {
	chars := []rune(text)
	n := len(chars)
	var sentences []string
	start, i := 0, 0

	for i < n {
		ch := chars[i]
		if ch == '.' || ch == '!' || ch == '?' {
			// Absorb runs of terminators and trailing closing quotes/brackets.
			j := i + 1
			for j < n && isTerminatorTail(chars[j]) {
				j++
			}

			atEnd := j >= n || allWhitespace(chars[j:])
			hasSpaceAfter := j < n && unicode.IsSpace(chars[j])
			var nextNonSpace rune
			hasNext := false
			for _, c := range chars[j:] {
				if !unicode.IsSpace(c) {
					nextNonSpace, hasNext = c, true
					break
				}
			}
			startsNew := hasSpaceAfter && hasNext && startsNewSentence(nextNonSpace)

			candidate := string(chars[start:j])
			if (atEnd || startsNew) && !endsWithAbbrev(candidate) {
				if trimmed := strings.TrimSpace(candidate); trimmed != "" {
					sentences = append(sentences, trimmed)
				}
				start = j
				i = j
				continue
			}
		}
		i++
	}

	if tail := strings.TrimSpace(string(chars[start:])); tail != "" {
		sentences = append(sentences, tail)
	}
	return sentences
}

func isTerminatorTail(c rune) bool {
	switch c {
	case '.', '!', '?', '"', '\'', ')', ']':
		return true
	}
	return false
}

func allWhitespace(chars []rune) bool {
	for _, c := range chars {
		if !unicode.IsSpace(c) {
			return false
		}
	}
	return true
}

func startsNewSentence(c rune) bool {
	if c >= 'A' && c <= 'Z' {
		return true
	}
	if c >= '0' && c <= '9' {
		return true
	}
	switch c {
	case '"', '\'', '(', '[', '-':
		return true
	}
	return false
}

// endsWithAbbrev reports whether candidate ends with a known non-terminal
// abbreviation (e.g. "e.g.", "etc."). Small and explicit on purpose — this is
// a heuristic, not full NLP, and we'd rather under-split than mishandle
// exotic cases.
func endsWithAbbrev(candidate string) bool {
	abbrevs := []string{
		"e.g", "i.e", "etc", "vs", "cf", "al", "approx", "dr", "mr", "mrs", "ms", "prof", "no",
		"fig", "inc", "ltd", "co", "st",
	}
	trimmed := strings.TrimRight(candidate, " \t\n\r")
	if !strings.HasSuffix(trimmed, ".") {
		return false
	}
	lower := strings.ToLower(trimmed)
	withoutDot := lower[:len(lower)-1] // last char is '.', one byte
	for _, ab := range abbrevs {
		if !strings.HasSuffix(withoutDot, ab) {
			continue
		}
		// Word-boundary check: the char before the abbrev must not be
		// alphanumeric (so "etc." matches but "xetc." doesn't).
		start := len(withoutDot) - len(ab)
		if start == 0 || !isASCIIAlphanumeric(withoutDot[start-1]) {
			return true
		}
	}
	return false
}

func isASCIIAlphanumeric(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}
