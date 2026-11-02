import { useEffect, useRef, useState } from "react";
import { coreRequest, type CoreConnection } from "../api/client";

interface Progress {
  source_id: string;
  status: string;
  step: string | null;
  waiting_reason: string | null;
  failure_code: string | null;
  completed_chunks: number;
  total_chunks: number;
  job_id?: string | null;
  can_cancel?: boolean;
  can_resume?: boolean;
}

const STEPS: Record<string, string> = {
  "plan-search": "Organizing searchable text",
  "embed-search": "Preparing meaning-based search",
  "build-search": "Finishing word and meaning search",
  "publish-search": "Checking searchable readiness",
};

export function SearchProgress({ connection, sourceId }: { connection: CoreConnection; sourceId: string }) {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const epoch = useRef(0);
  const revision = useRef(0);
  const acting = useRef(false);
  useEffect(() => {
    const current = ++epoch.current;
    let stopped = false;
    let loading = false;
    setProgress(null);
    setError(null);
    setBusy(false);
    acting.current = false;
    async function refresh() {
      if (stopped || loading || acting.current) return;
      loading = true;
      const read = ++revision.current;
      try {
        const result = await coreRequest<Progress>(connection, `/sources/${encodeURIComponent(sourceId)}/search-progress`);
        if (!stopped && current === epoch.current && read === revision.current && result.source_id === sourceId) setProgress(result);
      } catch { /* The existing source/import view reports unavailable core errors. */ }
      finally { loading = false; }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => { stopped = true; ++epoch.current; window.clearInterval(timer); };
  }, [connection, sourceId]);

  async function action(kind: "cancel" | "resume") {
    if (!progress?.job_id || acting.current) return;
    const current = epoch.current;
    ++revision.current;
    acting.current = true;
    setBusy(true);
    setError(null);
    try {
      await coreRequest(connection, `/sources/${encodeURIComponent(sourceId)}/search/${kind}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: progress.job_id, ...(kind === "resume" ? { request_key: crypto.randomUUID() } : {}) }),
      });
      const result = await coreRequest<Progress>(connection, `/sources/${encodeURIComponent(sourceId)}/search-progress`);
      if (current === epoch.current && result.source_id === sourceId) setProgress(result);
    } catch (cause) {
      if (current === epoch.current) setError(cause instanceof Error ? cause.message : "Search preparation could not be updated.");
    } finally {
      if (current === epoch.current) { acting.current = false; setBusy(false); }
    }
  }
  if (!progress || progress.status === "waiting_for_extraction") return null;
  return <section aria-label="Search preparation" role="status">
    <h3>Search preparation</h3>
    <p>{progress.waiting_reason === "model_not_installed" ? "Waiting for model setup. Use Local search in the notebook panel to download it."
      : progress.status === "cancelled" ? "Search preparation cancelled"
      : progress.status === "failed" ? "Search preparation failed"
      : progress.status === "completed" ? "Search data ready"
      : progress.step ? (STEPS[progress.step] ?? progress.step) : progress.status}</p>
    {progress.total_chunks > 0 ? <>
      <progress value={progress.completed_chunks} max={progress.total_chunks} aria-label="Search pieces prepared" />
      <p>{progress.completed_chunks} of {progress.total_chunks} search pieces prepared</p>
    </> : null}
    {progress.failure_code ? <p>{progress.failure_code.replaceAll("_", " ")}</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    {progress.can_cancel ? <button type="button" disabled={busy} onClick={() => void action("cancel")}>Cancel search preparation</button> : null}
    {progress.can_resume ? <button type="button" disabled={busy} onClick={() => void action("resume")}>Resume search preparation</button> : null}
  </section>;
}
