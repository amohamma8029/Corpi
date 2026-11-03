import { useEffect, useRef, useState } from "react";
import { coreRequest, type CoreConnection, type SourceSearchProgress } from "../api/client";
import { modelName } from "./modelNames";

const STEPS: Record<string, string> = {
  "plan-search": "Organizing searchable text",
  "embed-search": "Preparing meaning-based search",
  "build-search": "Finishing word and meaning search",
  "publish-search": "Checking searchable readiness",
};

const REASONS: Record<string, string> = {
  "review reopened": "Review Again reopened the text review. Confirm the current review to prepare search again.",
  "model selection changed": "The notebook model changed. This preparation belongs to the previous selection.",
  "search input changed": "The PDF, text review or notebook model changed. This preparation is no longer current.",
  "user cancelled PDF processing": "PDF processing was cancelled. Retry processing from the saved PDF to start again.",
  "search preparation cancelled": "You cancelled this search preparation. Resume is available if its inputs and saved allowance still permit it.",
};

export function SearchProgress({ connection, sourceId }: { connection: CoreConnection; sourceId: string }) {
  const [progress, setProgress] = useState<SourceSearchProgress | null>(null);
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
        const result = await coreRequest<SourceSearchProgress>(connection, `/sources/${encodeURIComponent(sourceId)}/search-progress`);
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
      const result = await coreRequest<SourceSearchProgress>(connection, `/sources/${encodeURIComponent(sourceId)}/search-progress`);
      if (current === epoch.current && result.source_id === sourceId) setProgress(result);
    } catch (cause) {
      if (current === epoch.current) setError(cause instanceof Error ? cause.message : "Search preparation could not be updated.");
    } finally {
      if (current === epoch.current) { acting.current = false; setBusy(false); }
    }
  }
  if (!progress || progress.status === "waiting_for_extraction") return null;
  const reason = progress.waiting_reason ?? progress.failure_reason;
  const finishing = ["queued", "running"].includes(progress.status)
    && progress.total_chunks > 0 && progress.completed_chunks >= progress.total_chunks;
  return <section aria-label="Search preparation" role="status">
    <h3>Search preparation</h3>
    {progress.filename ? <p dir="auto">Saved PDF: {progress.filename} · {progress.pdf_role === "pending" ? "Pending replacement" : progress.pdf_role === "current" ? "Current saved version" : "Previous saved version"}</p> : null}
    {progress.profile_id ? <small>Search model: {modelName(progress.profile_id)}</small> : null}
    <p>{progress.status === "cancelled" ? "Search preparation cancelled"
      : progress.status === "failed" ? "Search preparation failed"
      : progress.status === "completed" ? "Search data ready"
      : progress.waiting_reason === "model_not_installed" ? "Waiting for model setup. Use Local search in the notebook panel to download it."
      : progress.step ? (STEPS[progress.step] ?? progress.step) : progress.status}</p>
    {progress.total_chunks > 0 ? <>
      <progress value={progress.completed_chunks} max={progress.total_chunks} aria-label="Text pieces embedded" />
      <p>{progress.completed_chunks} of {progress.total_chunks} text pieces embedded</p>
      {finishing ? <p>All text pieces are embedded. Search is not ready yet: finishing validation and publication.</p> : null}
    </> : null}
    {reason && reason !== "model_not_installed" ? <p>{REASONS[reason] ?? reason.replaceAll("_", " ")}</p> : null}
    {progress.failure_code ? <p>{progress.failure_code.replaceAll("_", " ")}</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    {progress.can_cancel ? <button type="button" disabled={busy} onClick={() => void action("cancel")}>Cancel search preparation</button> : null}
    {progress.can_resume ? <button type="button" disabled={busy} onClick={() => void action("resume")}>Resume search preparation</button> : null}
  </section>;
}
