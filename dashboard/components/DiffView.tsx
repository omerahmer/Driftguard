import { PromptDiff } from "@/lib/api";

// Renders a structure-aware diff straight from the `ops` array the API serves —
// no delta/difftastic, no unified-diff parsing. equal/added/removed segments,
// colored git-style.
export function DiffView({ diff }: { diff: PromptDiff | null }) {
  if (!diff) {
    return <p className="text-sm text-gray-500">First version — nothing to diff against.</p>;
  }
  return (
    <div>
      <div className="mb-2 text-xs text-gray-500">
        +{diff.stats.added} −{diff.stats.removed} · {diff.stats.unchanged} unchanged
      </div>
      <div className="overflow-hidden rounded border font-mono text-xs">
        {diff.ops.map((op, i) => {
          const cls =
            op.type === "added"
              ? "bg-green-50 text-green-800"
              : op.type === "removed"
                ? "bg-red-50 text-red-800"
                : "text-gray-600";
          const mark = op.type === "added" ? "+ " : op.type === "removed" ? "- " : "  ";
          return (
            <div key={i} className={`whitespace-pre-wrap px-2 py-0.5 ${cls}`}>
              {mark}
              {op.text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
