// Typed client for the driftguard-api read endpoints. All requests go through
// /api (Next rewrites to the axum server), so no CORS/absolute URLs here.

export interface Prompt {
  id: string;
  name: string;
  current_version_id: string | null;
  created_at: string;
}

export interface PromptVersion {
  id: string;
  prompt_id: string;
  content: string;
  parent_version_id: string | null;
  diff_from_parent: string | null; // JSON string of PromptDiff, or null for v1
  created_at: string;
}

export interface DiffOp {
  type: "equal" | "added" | "removed";
  text: string;
}

export interface PromptDiff {
  granularity: string;
  ops: DiffOp[];
  changed_spans: string[];
  removed_spans: string[];
  stats: { added: number; removed: number; unchanged: number };
}

export interface EvalCase {
  id: string;
  prompt_id: string;
  input: string;
  expected_behavior: string;
  created_at: string;
}

export interface EvalRun {
  id: string;
  prompt_version_id: string;
  eval_case_id: string;
  actual_output: string;
  judge_passed: boolean | null;
  judge_justification: string | null;
  created_at: string;
}

export interface Metrics {
  tp: number;
  fp: number;
  fn_: number;
  tn: number;
  precision: number;
  recall: number;
}

export interface ThresholdRow {
  threshold: number;
  reduction: number;
  behavior: Metrics;
  flip: Metrics;
}

export interface ScoreReport {
  prompt: string | null;
  samples: number;
  rows: ThresholdRow[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // The API returns { error } on failure — surface it if present.
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.error ?? `${res.status} ${res.statusText} for ${path}`);
  }
  return res.json();
}

export const api = {
  prompts: () => get<Prompt[]>("/prompts"),
  versions: (promptId: string) => get<PromptVersion[]>(`/prompts/${promptId}/versions`),
  evalCases: (promptId: string) => get<EvalCase[]>(`/prompts/${promptId}/eval-cases`),
  addEvalCase: (promptId: string, input: string, expected_behavior: string) =>
    post<EvalCase>(`/prompts/${promptId}/eval-cases`, { input, expected_behavior }),
  runs: (versionId: string) => get<EvalRun[]>(`/versions/${versionId}/runs`),
  score: (promptName: string) =>
    get<ScoreReport>(`/score?prompt=${encodeURIComponent(promptName)}`),
};

/** Parse a version's stored diff (null for the first version). */
export function parseDiff(v: PromptVersion): PromptDiff | null {
  if (!v.diff_from_parent) return null;
  try {
    return JSON.parse(v.diff_from_parent) as PromptDiff;
  } catch {
    return null;
  }
}
