//! Structure-aware prompt segmentation + diffing.
//!
//! Why this design (chosen over plain line-level or plain sentence-level):
//! prompts vary wildly — terse one-liners, long prose paragraphs, Markdown-ish
//! bulleted constraint lists, headings, embedded code/JSON examples. A pure line
//! diff makes a one-word edit look like a whole-paragraph change (coarse span ->
//! noisy Phase 3 embedding). A pure sentence diff mangles lists and code (it
//! splits a bulleted constraint or a JSON example on every period).
//!
//! So we segment hierarchically and adaptively:
//!   - fenced code blocks stay atomic (never sentence-split),
//!   - headings and list items are one segment per line (structure carries
//!     meaning; an edited bullet should localize to that bullet),
//!   - prose paragraphs (possibly hard-wrapped across lines) are joined and then
//!     split into sentences.
//!
//! It's deterministic and free (no model calls). That matters: the "changed
//! span" produced here is the independent variable Phase 3 embeds and Phase 4
//! measures precision/recall against. Keeping the span definition cheap and
//! reproducible avoids polluting that measurement with model nondeterminism.

use serde::{Deserialize, Serialize};
use similar::{Algorithm, DiffOp as SimilarOp, capture_diff_slices};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentKind {
    Code,
    Heading,
    ListItem,
    Sentence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub kind: SegmentKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffOpType {
    Equal,
    Added,
    Removed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffOp {
    // `type` is a Rust keyword, so the field is `kind` but serializes as "type".
    #[serde(rename = "type")]
    pub kind: DiffOpType,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub added: usize,
    pub removed: usize,
    pub unchanged: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDiff {
    /// Identifier for the segmentation strategy, stored alongside the diff so a
    /// future change in granularity is detectable in old records.
    pub granularity: String,
    /// Full ordered op list — enough to render the diff without recomputing it.
    pub ops: Vec<DiffOp>,
    /// After-side text of added segments: exactly what Phase 3 embeds as the
    /// "changed span" of the edit.
    pub changed_spans: Vec<String>,
    /// Before-side text of removed segments: useful in Phase 4 for reasoning
    /// about what behavior was taken away.
    pub removed_spans: Vec<String>,
    pub stats: DiffStats,
}

/// Diff two prompt contents at segment granularity. A reworded segment surfaces
/// as a removed (old) + added (new) pair — we don't detect in-place edits,
/// because for selection the new text is what we embed and the old text is what
/// we lost, and both are captured.
pub fn compute_diff(before: &str, after: &str) -> PromptDiff {
    let before_segs: Vec<String> = segment_prompt(before).into_iter().map(|s| s.text).collect();
    let after_segs: Vec<String> = segment_prompt(after).into_iter().map(|s| s.text).collect();

    let ops = diff_segments(&before_segs, &after_segs);

    let changed_spans: Vec<String> = ops
        .iter()
        .filter(|o| o.kind == DiffOpType::Added)
        .map(|o| o.text.clone())
        .collect();
    let removed_spans: Vec<String> = ops
        .iter()
        .filter(|o| o.kind == DiffOpType::Removed)
        .map(|o| o.text.clone())
        .collect();
    let unchanged = ops.iter().filter(|o| o.kind == DiffOpType::Equal).count();

    PromptDiff {
        granularity: "structure-aware-v1".to_string(),
        stats: DiffStats {
            added: changed_spans.len(),
            removed: removed_spans.len(),
            unchanged,
        },
        changed_spans,
        removed_spans,
        ops,
    }
}

/// Map `similar`'s index-range diff ops back into a flat, per-segment op list.
fn diff_segments(before: &[String], after: &[String]) -> Vec<DiffOp> {
    let mut ops = Vec::new();
    for op in capture_diff_slices(Algorithm::Myers, before, after) {
        match op {
            SimilarOp::Equal {
                new_index, len, ..
            } => {
                for k in 0..len {
                    ops.push(DiffOp {
                        kind: DiffOpType::Equal,
                        text: after[new_index + k].clone(),
                    });
                }
            }
            SimilarOp::Delete {
                old_index, old_len, ..
            } => {
                for k in 0..old_len {
                    ops.push(DiffOp {
                        kind: DiffOpType::Removed,
                        text: before[old_index + k].clone(),
                    });
                }
            }
            SimilarOp::Insert {
                new_index, new_len, ..
            } => {
                for k in 0..new_len {
                    ops.push(DiffOp {
                        kind: DiffOpType::Added,
                        text: after[new_index + k].clone(),
                    });
                }
            }
            SimilarOp::Replace {
                old_index,
                old_len,
                new_index,
                new_len,
            } => {
                // A reworded segment: emit the old text as removed, new as added.
                for k in 0..old_len {
                    ops.push(DiffOp {
                        kind: DiffOpType::Removed,
                        text: before[old_index + k].clone(),
                    });
                }
                for k in 0..new_len {
                    ops.push(DiffOp {
                        kind: DiffOpType::Added,
                        text: after[new_index + k].clone(),
                    });
                }
            }
        }
    }
    ops
}

/// Segment a prompt into comparable units per the strategy in the module docs.
pub fn segment_prompt(content: &str) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut paragraph: Vec<&str> = Vec::new();
    let mut code_buffer: Vec<&str> = Vec::new();
    let mut in_code = false;

    for line in content.split('\n') {
        if is_fence(line) {
            if in_code {
                code_buffer.push(line);
                flush_code(&mut segments, &mut code_buffer);
                in_code = false;
            } else {
                flush_paragraph(&mut segments, &mut paragraph);
                in_code = true;
                code_buffer.push(line);
            }
            continue;
        }
        if in_code {
            code_buffer.push(line);
            continue;
        }
        if line.trim().is_empty() {
            flush_paragraph(&mut segments, &mut paragraph);
            continue;
        }
        if is_heading(line) {
            flush_paragraph(&mut segments, &mut paragraph);
            segments.push(Segment {
                kind: SegmentKind::Heading,
                text: line.trim().to_string(),
            });
            continue;
        }
        if is_list_item(line) {
            flush_paragraph(&mut segments, &mut paragraph);
            segments.push(Segment {
                kind: SegmentKind::ListItem,
                text: line.trim().to_string(),
            });
            continue;
        }
        paragraph.push(line);
    }

    flush_paragraph(&mut segments, &mut paragraph);
    flush_code(&mut segments, &mut code_buffer); // handles an unterminated fence

    segments
}

fn flush_paragraph(segments: &mut Vec<Segment>, paragraph: &mut Vec<&str>) {
    if paragraph.is_empty() {
        return;
    }
    // Join hard-wrapped lines, collapse whitespace, then sentence-split.
    let joined = paragraph.join(" ");
    let normalized = joined.split_whitespace().collect::<Vec<_>>().join(" ");
    for sentence in split_sentences(&normalized) {
        segments.push(Segment {
            kind: SegmentKind::Sentence,
            text: sentence,
        });
    }
    paragraph.clear();
}

fn flush_code(segments: &mut Vec<Segment>, code_buffer: &mut Vec<&str>) {
    if code_buffer.is_empty() {
        return;
    }
    segments.push(Segment {
        kind: SegmentKind::Code,
        text: code_buffer.join("\n").trim().to_string(),
    });
    code_buffer.clear();
}

fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

fn is_heading(line: &str) -> bool {
    let t = line.trim_start();
    let hashes = t.bytes().take_while(|b| *b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return false;
    }
    let rest = &t[hashes..]; // '#' is one byte, so `hashes` is a valid byte offset
    matches!(rest.chars().next(), Some(' ') | Some('\t')) && !rest.trim().is_empty()
}

fn is_list_item(line: &str) -> bool {
    let t = line.trim_start();
    // Unordered: -, *, + followed by whitespace and content.
    if let Some(rest) = t.strip_prefix(['-', '*', '+']) {
        return matches!(rest.chars().next(), Some(' ') | Some('\t')) && !rest.trim().is_empty();
    }
    // Ordered: digits then '.' or ')' then whitespace and content.
    let digits = t.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 {
        let after = &t[digits..];
        if let Some(rest) = after.strip_prefix(['.', ')']) {
            return matches!(rest.chars().next(), Some(' ') | Some('\t')) && !rest.trim().is_empty();
        }
    }
    false
}

/// Heuristic sentence splitter. Deterministic, dependency-free. Splits on
/// `.`/`!`/`?` when whitespace follows and the next non-space char looks like a
/// new sentence (or the string ends), while guarding a short abbreviation list.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < n {
        let ch = chars[i];
        if ch == '.' || ch == '!' || ch == '?' {
            // Absorb runs of terminators and trailing closing quotes/brackets.
            let mut j = i + 1;
            while j < n && matches!(chars[j], '.' | '!' | '?' | '"' | '\'' | ')' | ']') {
                j += 1;
            }

            let at_end = j >= n || chars[j..].iter().all(|c| c.is_whitespace());
            let has_space_after = j < n && chars[j].is_whitespace();
            let next_non_space = chars[j..].iter().find(|c| !c.is_whitespace());
            let starts_new = has_space_after
                && matches!(next_non_space, Some(c)
                    if c.is_ascii_uppercase()
                        || c.is_ascii_digit()
                        || matches!(c, '"' | '\'' | '(' | '[' | '-'));

            let candidate: String = chars[start..j].iter().collect();
            if (at_end || starts_new) && !ends_with_abbrev(&candidate) {
                let trimmed = candidate.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }

    let tail: String = chars[start..].iter().collect();
    let tail = tail.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences
}

/// True if `candidate` ends with a known non-terminal abbreviation (e.g. "e.g.",
/// "etc."). Small and explicit on purpose — this is a heuristic, not full NLP,
/// and we'd rather under-split than mishandle exotic cases.
fn ends_with_abbrev(candidate: &str) -> bool {
    const ABBREVS: &[&str] = &[
        "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "dr", "mr", "mrs", "ms", "prof", "no",
        "fig", "inc", "ltd", "co", "st",
    ];
    let trimmed = candidate.trim_end();
    if !trimmed.ends_with('.') {
        return false;
    }
    let lower = trimmed.to_lowercase();
    let without_dot = &lower[..lower.len() - 1]; // last char is '.', one byte
    for ab in ABBREVS {
        // strip_suffix (not manual byte slicing): an arbitrary byte offset can
        // land inside a multibyte char and panic (e.g. an em-dash near the end).
        if let Some(prefix) = without_dot.strip_suffix(ab) {
            // Word-boundary check: the char before the abbrev must not be
            // alphanumeric (so "etc." matches but "xetc." doesn't).
            let boundary_ok = prefix
                .as_bytes()
                .last()
                .is_none_or(|b| !b.is_ascii_alphanumeric());
            if boundary_ok {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(s: &str) -> Vec<SegmentKind> {
        segment_prompt(s).into_iter().map(|seg| seg.kind).collect()
    }

    #[test]
    fn splits_prose_but_respects_abbreviations() {
        assert_eq!(
            split_sentences("Be concise. Cite the policy ID."),
            vec!["Be concise.", "Cite the policy ID."]
        );
        assert_eq!(
            split_sentences("Use a tool, e.g. search, when needed."),
            vec!["Use a tool, e.g. search, when needed."]
        );
    }

    #[test]
    fn keeps_lists_and_headings_per_line_prose_per_sentence() {
        let prompt = "# System\nYou are a support agent. Always be concise.\n\n- Cite the policy ID\n- Never invent refunds";
        assert_eq!(
            kinds(prompt),
            vec![
                SegmentKind::Heading,
                SegmentKind::Sentence,
                SegmentKind::Sentence,
                SegmentKind::ListItem,
                SegmentKind::ListItem,
            ]
        );
    }

    #[test]
    fn keeps_fenced_code_atomic() {
        let prompt = "Return JSON like:\n```\n{ \"a\": 1. \"b\": 2 }\n```";
        let segs = segment_prompt(prompt);
        let code = segs.iter().find(|s| s.kind == SegmentKind::Code).unwrap();
        assert!(code.text.contains("{ \"a\": 1. \"b\": 2 }"));
    }

    #[test]
    fn localizes_a_one_word_edit_to_a_single_span() {
        let before = "You are a support agent. Always cite the policy ID. Be concise.";
        let after = "You are a support agent. Always cite the policy ID and link. Be concise.";
        let diff = compute_diff(before, after);
        assert_eq!(diff.stats.unchanged, 2);
        assert_eq!(diff.changed_spans, vec!["Always cite the policy ID and link."]);
        assert_eq!(diff.removed_spans, vec!["Always cite the policy ID."]);
    }

    #[test]
    fn identical_content_yields_empty_changed_span() {
        let diff = compute_diff("Same prompt.", "Same prompt.");
        assert!(diff.changed_spans.is_empty());
        assert!(diff.removed_spans.is_empty());
    }
}
