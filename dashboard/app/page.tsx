"use client";

import { useEffect, useState } from "react";
import { api, parseDiff, Prompt, PromptVersion } from "@/lib/api";
import { DiffView } from "@/components/DiffView";
import { RunsTable } from "@/components/RunsTable";
import { ScoreChart } from "@/components/ScoreChart";
import { EvalCases } from "@/components/EvalCases";

export default function Home() {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [versions, setVersions] = useState<PromptVersion[]>([]);
  const [version, setVersion] = useState<PromptVersion | null>(null);

  useEffect(() => {
    api.prompts().then(setPrompts).catch(() => setPrompts([]));
  }, []);

  useEffect(() => {
    if (!prompt) {
      setVersions([]);
      setVersion(null);
      return;
    }
    api
      .versions(prompt.id)
      .then((v) => {
        setVersions(v);
        setVersion(v.length ? v[v.length - 1] : null); // default to latest
      })
      .catch(() => setVersions([]));
  }, [prompt]);

  const versionOrdinal = (v: PromptVersion) => versions.findIndex((x) => x.id === v.id) + 1;

  return (
    <main className="flex min-h-screen">
      <aside className="w-72 shrink-0 space-y-6 border-r bg-white p-4">
        <h1 className="text-lg font-semibold">🛡️ Driftguard</h1>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase text-gray-400">Prompts</p>
          <ul className="space-y-0.5">
            {prompts.map((p) => (
              <li key={p.id}>
                <button
                  onClick={() => setPrompt(p)}
                  className={`w-full rounded px-2 py-1 text-left text-sm ${
                    prompt?.id === p.id ? "bg-blue-100 text-blue-800" : "hover:bg-gray-100"
                  }`}
                >
                  {p.name}
                </button>
              </li>
            ))}
          </ul>
        </div>

        {prompt && (
          <div>
            <p className="mb-1 text-xs font-semibold uppercase text-gray-400">Versions</p>
            <ul className="space-y-0.5">
              {versions.map((v) => (
                <li key={v.id}>
                  <button
                    onClick={() => setVersion(v)}
                    className={`flex w-full items-center justify-between rounded px-2 py-1 text-left text-sm ${
                      version?.id === v.id ? "bg-blue-100 text-blue-800" : "hover:bg-gray-100"
                    }`}
                  >
                    <span>v{versionOrdinal(v)}</span>
                    {v.id === prompt.current_version_id && (
                      <span className="text-[10px] uppercase text-gray-400">current</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </aside>

      <section className="flex-1 space-y-6 p-6">
        {!prompt && <p className="text-gray-500">Select a prompt to inspect its history.</p>}

        {prompt && (
          <>
            <h2 className="text-xl font-semibold">{prompt.name}</h2>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title={version ? `Diff — v${versionOrdinal(version)} vs parent` : "Diff"}>
                <DiffView diff={version ? parseDiff(version) : null} />
              </Card>
              <Card title="Precision / recall vs threshold">
                <ScoreChart promptName={prompt.name} />
              </Card>
              <Card title="Eval cases">
                <EvalCases promptId={prompt.id} />
              </Card>
            </div>

            {version && (
              <Card title={`Eval runs — v${versionOrdinal(version)}`}>
                <RunsTable versionId={version.id} />
              </Card>
            )}
          </>
        )}
      </section>
    </main>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">{title}</h3>
      {children}
    </div>
  );
}
