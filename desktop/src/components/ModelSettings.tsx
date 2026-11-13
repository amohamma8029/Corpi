import { useEffect, useRef, useState } from "react";
import { modelName } from "./modelNames";

import {
  cancelModelSetup, cancelNotebookModel, chooseNotebookModel, getNotebookModel,
  listModels, setupModel, type CoreConnection, type NotebookModel, type SupportedModel,
} from "../api/client";

const NAMES: Record<string, string> = {
  "bekko-a8m-quint8-avx2": "English, French and Arabic search",
  "minilm-en": "Fast English search",
  "multilingual-e5": "Multilingual search",
};

function bytes(value: number) { return `${(value / 1_000_000).toFixed(1)} MB`; }

export function ModelSettings({ connection, notebookId }: { connection: CoreConnection; notebookId: string }) {
  const [models, setModels] = useState<SupportedModel[]>([]);
  const [notebook, setNotebook] = useState<NotebookModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const epoch = useRef(0);
  const revision = useRef(0);
  const acting = useRef(false);

  useEffect(() => {
    const current = ++epoch.current;
    let stopped = false;
    let loading = false;
    setNotebook(null);
    setModels([]);
    setError(null);
    setBusy(false);
    acting.current = false;
    async function refresh() {
      if (loading || stopped || acting.current || !notebookId) return;
      loading = true;
      const read = ++revision.current;
      try {
        const [available, selection] = await Promise.all([
          listModels(connection), getNotebookModel(connection, notebookId),
        ]);
        if (!stopped && epoch.current === current && read === revision.current && selection.notebook_id === notebookId) {
          setModels(available);
          setNotebook(selection);
        }
      } catch (cause) {
        if (!stopped && epoch.current === current && read === revision.current) setError(cause instanceof Error ? cause.message : String(cause));
      } finally { loading = false; }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => { stopped = true; ++epoch.current; window.clearInterval(timer); };
  }, [connection, notebookId]);

  async function action(operation: () => Promise<unknown>) {
    if (acting.current) return;
    const current = epoch.current;
    ++revision.current;
    acting.current = true;
    setBusy(true);
    setError(null);
    try {
      await operation();
      const [available, selection] = await Promise.all([
        listModels(connection), getNotebookModel(connection, notebookId),
      ]);
      if (epoch.current === current && selection.notebook_id === notebookId) { setModels(available); setNotebook(selection); }
    } catch (cause) {
      if (epoch.current === current) setError(cause instanceof Error ? cause.message : String(cause));
    } finally { if (epoch.current === current) { acting.current = false; setBusy(false); } }
  }

  if (!notebookId) return null;
  const failed = notebook?.failed_sources ?? 0;
  const cancelled = notebook?.cancelled_sources ?? 0;
  const blocked = failed > 0 || cancelled > 0;
  return (
    <section aria-label="Local search models" className="model-settings" data-active-profile={notebook?.active_profile_id} data-pending-profile={notebook?.pending_profile_id ?? undefined}>
      <h3>Local search</h3>
      <p>Models prepare search on your device. Originals stay accessible during setup.</p>
      <p>Large sources may take up to twenty minutes to prepare. Search in the same language as your documents.</p>
      {error ? <p role="alert">{error}</p> : null}
      {notebook ? <>
        <label htmlFor="search-model">Notebook model</label>
        <select id="search-model" value={notebook.pending_profile_id ?? notebook.active_profile_id}
          disabled={busy} onChange={(event) => {
            const profile = event.currentTarget.value;
            void action(() => chooseNotebookModel(connection, notebook, profile));
          }}>
          {models.map((model) => <option key={model.profile_id} value={model.profile_id}>{NAMES[model.profile_id] ?? model.model}</option>)}
        </select>
        <p>Active: {modelName(notebook.active_profile_id)} · {NAMES[notebook.active_profile_id] ?? notebook.active_profile_id}</p>
        {notebook.pending_profile_id ? <div role={blocked ? "alert" : "status"}>
          <p>Requested: {modelName(notebook.pending_profile_id)} · {NAMES[notebook.pending_profile_id] ?? notebook.pending_profile_id}</p>
          <p>{blocked
            ? `Model change could not finish: ${failed} failed, ${cancelled} cancelled out of ${notebook.ready_sources} ready sources. Your current model remains active. Cancel the model change to keep it.`
            : `Rebuilding ${notebook.rebuilt_sources} of ${notebook.ready_sources} ready sources. Current search stays usable until the switch finishes.`}</p>
          <button type="button" disabled={busy} onClick={() => void action(() => cancelNotebookModel(connection, notebook))}>Cancel model change</button>
        </div> : null}
      </> : <p role="status">Loading model settings…</p>}
      <ul aria-label="Supported search models">
        {models.map((model) => <li key={model.profile_id}>
          <strong>{NAMES[model.profile_id] ?? model.model}</strong>
          <p>{model.language_scope} · {bytes(model.download_bytes)} download</p>
          <p role="status">{model.state === "downloading" ? `Downloading: ${bytes(model.received_bytes)} received` : model.state}</p>
          {model.state === "downloading" ? <progress aria-label={`${NAMES[model.profile_id]} download`} /> : null}
          {model.failure_code ? <p>Setup: {model.failure_code.replaceAll("_", " ")}. Retry is available after cleanup finishes.</p> : null}
          {model.can_retry ? <button type="button" disabled={busy} onClick={() => void action(() => setupModel(connection, model))}>
            {model.state === "missing" ? "Download" : "Retry setup"} {NAMES[model.profile_id] ?? model.profile_id}
          </button> : null}
          {model.can_cancel ? <button type="button" disabled={busy} onClick={() => void action(() => cancelModelSetup(connection, model))}>Cancel download</button> : null}
        </li>)}
      </ul>
      <small>Changing the model rebuilds search data. Downloads start only when you choose Download.</small>
    </section>
  );
}
