"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ScoreReport } from "@/lib/api";

// The precision/recall curve is DERIVED state — computed live by the API from
// raw eval runs, not stored. The chart just renders what it's given.
export function ScoreChart({ promptName }: { promptName: string }) {
  const [report, setReport] = useState<ScoreReport | null>(null);
  // Which "actually affected" label the curve is scored against.
  const [groundTruth, setGroundTruth] = useState<"judge" | "human">("judge");

  useEffect(() => {
    let active = true;
    setReport(null);
    api
      .score(promptName, groundTruth)
      .then((r) => active && setReport(r))
      .catch(() => active && setReport(null));
    return () => {
      active = false;
    };
  }, [promptName, groundTruth]);

  const toggle = (
    <div className="mb-2 flex gap-1 text-xs">
      {(["judge", "human"] as const).map((g) => (
        <button
          key={g}
          onClick={() => setGroundTruth(g)}
          className={`rounded px-2 py-0.5 ${
            groundTruth === g ? "bg-blue-600 text-white" : "border text-gray-600 hover:bg-gray-100"
          }`}
        >
          {g === "judge" ? "judge truth" : "hand labels"}
        </button>
      ))}
    </div>
  );

  if (!report)
    return (
      <div>
        {toggle}
        <p className="text-sm text-gray-500">Loading…</p>
      </div>
    );
  if (report.samples === 0) {
    return (
      <div>
        {toggle}
        <p className="text-sm text-gray-500">
          {groundTruth === "human"
            ? "No hand labels yet — label some behavior diffs below."
            : "No labeled selections yet — run "}
          {groundTruth === "judge" && (
            <code className="rounded bg-gray-100 px-1">driftguard validate</code>
          )}
          {groundTruth === "judge" && " to populate the curve."}
        </p>
      </div>
    );
  }

  const data = report.rows.map((r) => ({
    threshold: r.threshold,
    precision: r.behavior.precision,
    recall: r.behavior.recall,
    reduction: r.reduction,
  }));

  return (
    <div>
      {toggle}
      <div className="mb-2 text-xs text-gray-500">
        {report.samples} selection records · {report.ground_truth} ground truth
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="threshold" tickFormatter={(t) => t.toFixed(2)} fontSize={11} />
          <YAxis domain={[0, 1]} fontSize={11} />
          <Tooltip formatter={(v: number) => v.toFixed(3)} />
          <Legend />
          <Line type="monotone" dataKey="precision" stroke="#2563eb" dot={false} />
          <Line type="monotone" dataKey="recall" stroke="#16a34a" dot={false} />
          <Line type="monotone" dataKey="reduction" stroke="#9ca3af" strokeDasharray="4 4" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
