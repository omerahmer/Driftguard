"use client";

import { useEffect, useState } from "react";
import { api, BehaviorDiff, JudgeAudit } from "@/lib/api";

// Hand-labeling view: the gold ground-truth surface. For each behavior diff you
// judge INDEPENDENTLY whether behavior changed (the judge's verdict is hidden
// until you pick, so you don't anchor to it), and the header shows the judge's
// running audit against your labels.
export function LabelingView() {
  const [diffs, setDiffs] = useState<BehaviorDiff[]>([]);
  const [audit, setAudit] = useState<JudgeAudit | null>(null);
  const [loaded, setLoaded] = useState(false);

  function reload() {
    api.behaviorDiffs().then(setDiffs).catch(() => setDiffs([]));
    api.judgeAudit().then(setAudit).catch(() => setAudit(null));
  }

  useEffect(() => {
    reload();
    setLoaded(true);
  }, []);

  async function label(id: string, changed: boolean) {
    await api.label(id, changed);
    reload();
  }

  if (loaded && diffs.length === 0) {
    return (
      <p className="text-sm text-gray-500">
        No behavior diffs to label yet — run <code className="rounded bg-gray-100 px-1">validate</code> first
        to generate before/after outputs.
      </p>
    );
  }

  const labeled = diffs.filter((d) => d.human_behavior_changed !== null).length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500">
          {labeled} / {diffs.length} labeled
        </span>
        {audit && audit.labeled > 0 && (
          <span className="text-gray-600">
            judge vs your labels: <b>{(audit.agreement * 100).toFixed(0)}%</b> agree · P{" "}
            {audit.precision.toFixed(2)} · R {audit.recall.toFixed(2)}{" "}
            <span className="text-gray-400">
              (tp{audit.tp} fp{audit.fp} fn{audit.fn_} tn{audit.tn})
            </span>
          </span>
        )}
      </div>

      <ul className="space-y-3">
        {diffs.map((d) => (
          <li key={d.behavior_diff_id} className="rounded-lg border p-3">
            <p className="text-sm font-medium text-gray-800">{d.input}</p>
            <p className="mb-2 text-xs text-gray-500">expects: {d.expected_behavior}</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <pre className="whitespace-pre-wrap rounded bg-gray-50 p-2 text-gray-700">
                {d.before_output}
              </pre>
              <pre className="whitespace-pre-wrap rounded bg-gray-50 p-2 text-gray-700">
                {d.after_output}
              </pre>
            </div>

            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-gray-500">Did behavior change?</span>
              <LabelButton
                active={d.human_behavior_changed === true}
                onClick={() => label(d.behavior_diff_id, true)}
              >
                Changed
              </LabelButton>
              <LabelButton
                active={d.human_behavior_changed === false}
                onClick={() => label(d.behavior_diff_id, false)}
              >
                No change
              </LabelButton>

              {/* Judge verdict revealed only after the human has labeled. */}
              {d.human_behavior_changed !== null && d.judge_behavior_changed !== null && (
                <span
                  className={`ml-auto text-xs ${
                    d.judge_behavior_changed === d.human_behavior_changed
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  judge said {d.judge_behavior_changed ? "changed" : "no change"}
                  {d.judge_behavior_changed === d.human_behavior_changed ? " ✓" : " ✗"}
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function LabelButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-2 py-0.5 text-xs ${
        active ? "bg-blue-600 text-white" : "border text-gray-700 hover:bg-gray-100"
      }`}
    >
      {children}
    </button>
  );
}
