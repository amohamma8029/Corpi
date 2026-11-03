import { useEffect, useRef, useState } from "react";
import { modelName } from "./modelNames";

import {
  getNotebookSearchState, searchNotebook, type CoreConnection, type NotebookSearchResponse,
  type NotebookSearchResult,
} from "../api/client";

export function NotebookSearch({ connection, notebookId, invalidationKey = 0, onOpen }: {
  connection: CoreConnection;
  notebookId: string;
  invalidationKey?: number;
  onOpen: (result: NotebookSearchResult, page: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"meaning" | "words">("meaning");
  const [response, setResponse] = useState<NotebookSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const request = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const epoch = useRef(0);
  const revision = useRef<string | null>(null);
  const stateRead = useRef(0);
  const searching = useRef(false);
  const searched = useRef(false);

  useEffect(() => {
    ++request.current;
    controller.current?.abort();
    setQuery("");
    setResponse(null);
    setError(null);
    setBusy(false);
    setNotice(null);
    searched.current = false;
    return () => { ++request.current; controller.current?.abort(); };
  }, [connection, notebookId]);

  useEffect(() => {
    const current = ++epoch.current;
    let stopped = false;
    let loading = false;
    const stateController = new AbortController();
    function clear(message: string) {
      ++request.current;
      controller.current?.abort();
      searching.current = false;
      setResponse(null);
      setBusy(false);
      if (searched.current) setNotice(message);
    }
    clear("Search data changed. Search again for current results.");
    revision.current = null;
    async function refresh() {
      if (stopped || loading || searching.current || !notebookId) return;
      loading = true;
      const read = ++stateRead.current;
      try {
        const state = await getNotebookSearchState(connection, notebookId, stateController.signal);
        if (stopped || current !== epoch.current || read !== stateRead.current) return;
        if (state.notebook_id !== notebookId) throw new Error("Search state belongs to another notebook.");
        if (revision.current !== null && state.revision !== revision.current) {
          clear("Search data changed. Search again for current results.");
        }
        revision.current = state.revision;
      } catch {
        if (!stopped && current === epoch.current && read === stateRead.current) {
          clear("Current search data could not be checked. Search again when the connection is available.");
          revision.current = null;
        }
      } finally { loading = false; }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => { stopped = true; ++epoch.current; stateController.abort(); window.clearInterval(timer); };
  }, [connection, notebookId, invalidationKey]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || !notebookId) return;
    const current = ++request.current;
    const currentEpoch = epoch.current;
    ++stateRead.current;
    searching.current = true;
    searched.current = true;
    controller.current?.abort();
    controller.current = new AbortController();
    setBusy(true);
    setError(null);
    setResponse(null);
    setNotice(null);
    try {
      const result = await searchNotebook(connection, notebookId, query, mode, controller.current.signal);
      if (current !== request.current) return;
      if (result.notebook_id !== notebookId) throw new Error("The search response belongs to another notebook.");
      const state = await getNotebookSearchState(connection, notebookId, controller.current.signal);
      if (current !== request.current || currentEpoch !== epoch.current) return;
      if (state.notebook_id !== notebookId) throw new Error("Search state belongs to another notebook.");
      revision.current = state.revision;
      if (result.revision !== state.revision) {
        setNotice("Search data changed. Search again for current results.");
        return;
      }
      setResponse(result);
    } catch (cause) {
      if (current === request.current && !(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Local search could not finish.");
      }
    } finally { if (current === request.current) { searching.current = false; setBusy(false); } }
  }

  if (!notebookId) return null;
  return <section className="notebook-search" aria-label="Notebook search">
    <h2>Search this notebook</h2>
    <p>Search ready documents. Documents still preparing are not included.</p>
    <form onSubmit={(event) => void submit(event)}>
      <label htmlFor="notebook-search-query">Find a passage</label>
      <div className="notebook-search__controls">
        <input id="notebook-search-query" type="search" dir="auto" required maxLength={8192}
          value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
        <label className="visually-hidden" htmlFor="notebook-search-mode">Search mode</label>
        <select id="notebook-search-mode" value={mode}
          onChange={(event) => setMode(event.currentTarget.value as "meaning" | "words")}>
          <option value="meaning">Meaning</option>
          <option value="words">Words</option>
        </select>
        <button type="submit" disabled={busy || !query.trim()}>{busy ? "Searching…" : "Search"}</button>
      </div>
    </form>
    {busy ? <p role="status">Searching locally…</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    {notice ? <p role="status">{notice}</p> : null}
    {response ? <>
      <p role="status">{response.ready_sources === 0
        ? "No sources are ready yet. Finish model setup and search preparation first."
        : response.results.length === 0 ? "No matches. Try different words or switch search mode."
        : `${response.results.length} results from ${response.ready_sources} ready sources.`}</p>
      <ol className="search-results" aria-label="Search results">
        {response.results.map((result) => <li key={`${result.source_version_id}:${result.chunk_id}`}>
          <h3 dir="auto">{result.filename ?? result.source_name}</h3>
          {result.filename && result.filename !== result.source_name ? <small dir="auto">Notebook source: {result.source_name}</small> : null}
          {result.filename ? <p dir="auto">Saved PDF: {result.filename} · {result.pdf_role === "current" ? "Current saved version" : result.pdf_role === "previous" ? "Previous saved version" : "Saved version"}</p> : null}
          <small>Search model: {modelName(result.profile_id ?? result.manifest_id)}</small>
          <p className="search-result__text" dir="auto">{result.text}</p>
          {result.import_id ? <div className="search-result__citations">
            {result.citations.filter((citation, index, citations) => citation.page_number !== null
              && citations.findIndex((other) => other.page_number === citation.page_number) === index).map((citation) => (
              <button type="button" key={`${citation.passage_id}:${citation.ordinal}`}
                data-passage-id={citation.passage_id}
                aria-label={`Open ${result.filename ?? result.source_name}, page ${citation.page_number}`}
                onClick={(event) => { event.currentTarget.focus(); onOpen(result, citation.page_number!); }}>
                Open PDF · page {citation.page_number}
              </button>
            ))}
          </div> : <small>No saved PDF is available for this result.</small>}
        </li>)}
      </ol>
    </> : null}
  </section>;
}
