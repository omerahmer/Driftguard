"use client";

import { useEffect, useState } from "react";
import { api, EvalCase } from "@/lib/api";

// Lists a prompt's eval cases and lets the user author a new one. This is the
// dashboard's only write path — eval cases have no home in the codebase, so
// authoring them here (rather than via the CLI's fiddly --input/--expected
// flags) is the scoped write we added. Prompts still come from the CLI/scan.
export function EvalCases({ promptId }: { promptId: string }) {
  const [cases, setCases] = useState<EvalCase[]>([]);
  const [input, setInput] = useState("");
  const [expected, setExpected] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reload() {
    api.evalCases(promptId).then(setCases).catch(() => setCases([]));
  }

  useEffect(reload, [promptId]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !expected.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await api.addEvalCase(promptId, input.trim(), expected.trim());
      setInput("");
      setExpected("");
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to add eval case");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      {cases.length === 0 ? (
        <p className="text-sm text-gray-500">No eval cases yet.</p>
      ) : (
        <ul className="space-y-2">
          {cases.map((c) => (
            <li key={c.id} className="rounded border border-gray-100 bg-gray-50 p-2 text-sm">
              <p className="text-gray-800">{c.input}</p>
              <p className="mt-1 text-xs text-gray-500">expects: {c.expected_behavior}</p>
            </li>
          ))}
        </ul>
      )}

      <form onSubmit={submit} className="space-y-2 border-t pt-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Test input (e.g. “I want a refund now”)"
          className="w-full rounded border px-2 py-1 text-sm"
        />
        <textarea
          value={expected}
          onChange={(e) => setExpected(e.target.value)}
          placeholder="Expected behavior (e.g. “Does not promise a refund; offers to escalate.”)"
          rows={2}
          className="w-full rounded border px-2 py-1 text-sm"
        />
        {error && <p className="text-xs text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={saving || !input.trim() || !expected.trim()}
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
        >
          {saving ? "Adding…" : "Add eval case"}
        </button>
      </form>
    </div>
  );
}
