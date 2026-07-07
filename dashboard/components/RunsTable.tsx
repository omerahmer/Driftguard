"use client";

import { useEffect, useState } from "react";
import { api, EvalRun } from "@/lib/api";

export function RunsTable({ versionId }: { versionId: string }) {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);

  useEffect(() => {
    let active = true;
    setRuns(null);
    api
      .runs(versionId)
      .then((r) => active && setRuns(r))
      .catch(() => active && setRuns([]));
    return () => {
      active = false;
    };
  }, [versionId]);

  if (runs === null) return <p className="text-sm text-gray-500">Loading runs…</p>;
  if (runs.length === 0) {
    return (
      <p className="text-sm text-gray-500">
        No eval runs for this version yet. Run <code className="rounded bg-gray-100 px-1">driftguard run-evals</code> or{" "}
        <code className="rounded bg-gray-100 px-1">validate</code>.
      </p>
    );
  }

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-xs uppercase text-gray-400">
          <th className="py-1 pr-2">result</th>
          <th className="py-1 pr-2">actual output</th>
          <th className="py-1">judge justification</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => (
          <tr key={r.id} className="border-b align-top">
            <td className="py-1 pr-2">
              {r.judge_passed === null ? "—" : r.judge_passed ? "✅" : "❌"}
            </td>
            <td className="max-w-md py-1 pr-2 text-gray-700">{truncate(r.actual_output, 160)}</td>
            <td className="py-1 text-gray-500">{truncate(r.judge_justification ?? "", 160)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function truncate(s: string, n: number) {
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length <= n ? flat : flat.slice(0, n - 1) + "…";
}
