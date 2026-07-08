//! Golden-vector generator for the Go port of `segment.rs`.
//!
//! Emits JSON test vectors (segmentation, sentence-splitting, and full diffs)
//! from THIS implementation, so the Go port can assert byte-level parity.
//! Diff pairs cover every fixtures edit vs. its base prompt — the exact inputs
//! whose diffs are persisted in prompt_versions.diff_from_parent — plus
//! synthetic adversarial cases (unicode, abbreviations, code fences, reorders).
//!
//! Run: cargo run -p driftguard-core --example gen_segment_goldens \
//!        > internal/core/testdata/segment_goldens.json

use driftguard_core::fixtures::parse_fixtures;
use driftguard_core::segment::{compute_diff, segment_prompt, split_sentences};
use serde_json::json;

fn main() {
    let fixtures_text = std::fs::read_to_string("fixtures/driftguard-fixtures.toml")
        .expect("run from the repo root");
    let fixtures = parse_fixtures(&fixtures_text).expect("fixtures parse");

    let synthetic_texts: Vec<String> = vec![
        "You are a support agent. Always cite the policy ID. Be concise.".into(),
        "Use a tool, e.g. search, when needed. Then answer.".into(),
        "See fig. 3 for details. Mr. Smith agrees. Answer carefully!".into(),
        "# System\nYou are a support agent. Always be concise.\n\n- Cite the policy ID\n- Never invent refunds".into(),
        "Return JSON like:\n```\n{ \"a\": 1. \"b\": 2 }\n```\nThen stop.".into(),
        "Unicode test: café résumé. Über alles? Naïve — yes.".into(),
        "1. First step\n2) Second step\n\nProse line one\nwrapped onto line two. Done.".into(),
        "~~~\nunterminated fence. still code\n".into(),
        "".into(),
        "   \n\n  ".into(),
        "No terminator at all".into(),
        "Ends with abbrev etc.".into(),
        "Quote test. \"He said no.\" Then left. (Really.) [Yes.]".into(),
    ];

    // Diff pairs: every fixtures edit vs. its base + identity + synthetics.
    let mut pairs: Vec<(String, String)> = Vec::new();
    for p in &fixtures.prompts {
        pairs.push((p.content.clone(), p.content.clone()));
        for e in &p.edits {
            pairs.push((p.content.clone(), e.content.clone()));
        }
    }
    pairs.push((
        "You are a support agent. Always cite the policy ID. Be concise.".into(),
        "You are a support agent. Always cite the policy ID and link. Be concise.".into(),
    ));
    pairs.push((synthetic_texts[3].clone(), synthetic_texts[4].clone()));
    pairs.push(("".into(), "New prompt. Fresh start.".into()));
    pairs.push(("Old prompt. Goodbye.".into(), "".into()));
    // Reorder: ambiguous LCS — the tie-break case worth pinning.
    pairs.push((
        "- Rule A\n- Rule B\n- Rule C".into(),
        "- Rule C\n- Rule A\n- Rule B".into(),
    ));

    let mut segment_texts: Vec<String> = synthetic_texts.clone();
    for p in &fixtures.prompts {
        segment_texts.push(p.content.clone());
        for e in &p.edits {
            segment_texts.push(e.content.clone());
        }
    }

    let segmentation: Vec<_> = segment_texts
        .iter()
        .map(|t| {
            json!({
                "input": t,
                "segments": segment_prompt(t).into_iter().map(|s| json!({
                    "kind": s.kind,
                    "text": s.text,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let sentences: Vec<_> = synthetic_texts
        .iter()
        .map(|t| json!({ "input": t, "sentences": split_sentences(t) }))
        .collect();

    let diffs: Vec<_> = pairs
        .iter()
        .map(|(before, after)| {
            let diff = compute_diff(before, after);
            json!({
                "before": before,
                "after": after,
                // Byte-exact JSON string as serde_json wrote it (what the DB stores).
                "diff_json": serde_json::to_string(&diff).unwrap(),
            })
        })
        .collect();

    let out = json!({
        "segmentation": segmentation,
        "sentences": sentences,
        "diffs": diffs,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
