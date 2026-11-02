import { useEffect, useRef, useState } from "react";

import {
  searchNotebook, type CoreConnection, type NotebookSearchResponse,
  type NotebookSearchResult,
} from "../api/client";

export function NotebookSearch({ connection, notebookId, onOpen }: {
  connection: CoreConnection;
  notebookId: string;
  onOpen: (result: NotebookSearchResult, page: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"meaning" | "words">("meaning");
  const [response, setResponse] = useState<NotebookSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const request = useRef(0);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    ++request.current;
    controller.current?.abort();
    setQuery("");
    setResponse(null);
    setError(null);
    setBusy(false);
    return () => { ++request.current; controller.current?.abort(); };
  }, [connection, notebookId]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || !notebookId) return;
    const current = ++request.current;
    controller.current?.abort();
    controller.current = new AbortController();
    setBusy(true);
    setError(null);
    setResponse(null);
    try {
      const result = await searchNotebook(connection, notebookId, query, mode, controller.current.signal);
      if (current !== request.current) return;
      if (result.notebook_id !== notebookId) throw new Error("The search response belongs to another notebook.");
      setResponse(result);
    } catch (cause) {
      if (current === request.current && !(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Local search could not finish.");
      }
    } finally { if (current === request.current) setBusy(false); }
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
    {response ? <>
      <p role="status">{response.ready_sources === 0
        ? "No sources are ready yet. Finish model setup and search preparation first."
        : response.results.length === 0 ? "No matches. Try different words or switch search mode."
        : `${response.results.length} results from ${response.ready_sources} ready sources.`}</p>
      <ol className="search-results" aria-label="Search results">
        {response.results.map((result) => <li key={`${result.source_version_id}:${result.chunk_id}`}>
          <h3 dir="auto">{result.source_name}</h3>
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
