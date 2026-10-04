import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import {
  cancelPdfAcquisition,
  cancelPdfProcessing,
  CoreApiError,
  CoreConnection,
  CoreHealth,
  createNotebook,
  createPdfAcquisition,
  decideExtractionReview,
  ExtractionReport,
  ExtractionReview,
  getExtraction,
  getPdfAcquisition,
  listNotebookSources,
  listNotebooks,
  listPdfAcquisitions,
  listPdfImports,
  Notebook,
  NotebookSource,
  PdfAcquisition,
  PdfImport,
  reopenExtractionReview,
  retryPdfAcquisition,
  retryPdfProcessing,
  StorageReport,
  storageReport,
  trashPdfImport,
  trashSource,
  uploadPdf,
  coreHealthWithRetry,
} from "./api/client";

const PdfViewer = lazy(() =>
  import("./components/PdfViewer").then((module) => ({ default: module.PdfViewer })),
);

type CoreStatus = "connecting" | "ok" | "error";
type ImportIntent = "add" | "replace";

const STATUS_TEXT: Record<CoreStatus, string> = {
  connecting: "Connecting to core…",
  ok: "Core ready",
  error: "Core unavailable",
};

export function App() {
  const [connection, setConnection] = useState<CoreConnection | null>(null);
  const [status, setStatus] = useState<CoreStatus>("connecting");
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [selectedNotebookId, setSelectedNotebookId] = useState("");
  const [notebookTitle, setNotebookTitle] = useState("");
  const [sources, setSources] = useState<NotebookSource[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState("");
  const [imports, setImports] = useState<PdfImport[]>([]);
  const [storage, setStorage] = useState<StorageReport | null>(null);
  const [acquisition, setAcquisition] = useState<PdfAcquisition | null>(null);
  const [savedAcquisitions, setSavedAcquisitions] = useState<PdfAcquisition[]>([]);
  const [extraction, setExtraction] = useState<ExtractionReport | null>(null);
  const [uploading, setUploading] = useState(false);
  const [viewerImport, setViewerImport] = useState<PdfImport | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const selectionSourceRef = useRef("");
  const nextIntentRef = useRef<{ intent: ImportIntent; sourceId?: string; acquisitionId?: string }>({ intent: "add" });
  const uploadControllerRef = useRef<AbortController | null>(null);
  const acquisitionIdRef = useRef("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const conn = await invoke<CoreConnection>("core_connection");
        if (cancelled) return;
        const health: CoreHealth = await coreHealthWithRetry(conn);
        if (cancelled) return;
        setConnection(conn);
        setVersion(health.version);
        setStatus("ok");
      } catch (cause) {
        if (!cancelled) {
          setStatus("error");
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      }
    })();
    return () => {
      cancelled = true;
      uploadControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!connection) return;
    let cancelled = false;
    void Promise.all([listNotebooks(connection), storageReport(connection)])
      .then(([items, report]) => {
        if (cancelled) return;
        setNotebooks(items);
        setStorage(report);
        setSelectedNotebookId((selected) => selected || items[0]?.id || "");
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(messageFor(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [connection]);

  useEffect(() => {
    setAcquisition(null);
    if (!connection || !selectedNotebookId) {
      setSources([]);
      setSavedAcquisitions([]);
      setSelectedSourceId("");
      return;
    }
    let cancelled = false;
    setSavedAcquisitions([]);
    void Promise.all([
      listNotebookSources(connection, selectedNotebookId),
      listPdfAcquisitions(connection, selectedNotebookId),
    ])
      .then(([items, attempts]) => {
        if (cancelled) return;
        setSources(items);
        setSavedAcquisitions(attempts);
        setSelectedSourceId((selected) =>
          items.some((item) => item.id === selected) ? selected : items[0]?.id || "",
        );
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(messageFor(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [connection, selectedNotebookId]);

  useEffect(() => {
    if (!connection || !selectedSourceId) {
      setImports([]);
      return;
    }
    let cancelled = false;
    void listPdfImports(connection, selectedSourceId)
      .then((items) => {
        if (!cancelled) setImports(items);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(messageFor(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [connection, selectedSourceId, acquisition?.updated_at]);

  useEffect(() => {
    if (!connection || !acquisition || !uploading) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getPdfAcquisition(connection, acquisition.id)
        .then((latest) => { if (!cancelled) rememberAcquisition(latest); })
        .catch(() => undefined);
    }, 180);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [acquisition?.id, connection, uploading]);

  useEffect(() => {
    if (selectionSourceRef.current !== selectedSourceId) {
      selectionSourceRef.current = selectedSourceId;
      setExtraction(null);
    }
    if (!connection || !selectedSourceId) {
      return;
    }
    let cancelled = false;
    void getExtraction(connection, selectedSourceId)
      .then((report) => {
        if (!cancelled && selectionSourceRef.current === selectedSourceId) setExtraction(report);
      })
      .catch(() => {
        if (!cancelled && selectionSourceRef.current === selectedSourceId) setExtraction(null);
      });
    return () => {
      cancelled = true;
    };
  }, [connection, selectedSourceId, imports]);

  const processingActive = imports.some((item) => !item.trashed && item.state === "pending");
  useEffect(() => {
    if (!connection || !selectedSourceId || !(processingActive || uploading)) return;
    const timer = window.setInterval(() => {
      void listPdfImports(connection, selectedSourceId)
        .then((items) => setImports(items))
        .catch(() => undefined);
      void getExtraction(connection, selectedSourceId)
        .then((report) => {
          if (selectionSourceRef.current === selectedSourceId) setExtraction(report);
        })
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [connection, selectedSourceId, processingActive, uploading]);
  const selectedSource = sources.find((source) => source.id === selectedSourceId) ?? null;
  const visibleAcquisitions = [
    ...savedAcquisitions.filter((item) => item.id !== acquisition?.id),
    ...(acquisition ? [acquisition] : []),
  ].filter((item) => item.notebook_id === selectedNotebookId && item.state !== "completed" && item.state !== "cancelled");

  function rememberAcquisition(item: PdfAcquisition) {
    setAcquisition(item);
    setSavedAcquisitions((items) => [
      ...items.filter((saved) => saved.id !== item.id),
      ...(item.state === "completed" || item.state === "cancelled" ? [] : [item]),
    ]);
  }

  async function refreshNotebook(notebookId: string, preferredSourceId?: string) {
    if (!connection) return;
    const [sourceItems, report, attempts] = await Promise.all([
      listNotebookSources(connection, notebookId),
      storageReport(connection),
      listPdfAcquisitions(connection, notebookId),
    ]);
    setSources(sourceItems);
    const nextSelected =
      preferredSourceId && sourceItems.some((item) => item.id === preferredSourceId)
        ? preferredSourceId
        : sourceItems.some((item) => item.id === selectedSourceId)
          ? selectedSourceId
          : sourceItems[0]?.id || "";
    setSelectedSourceId(nextSelected);
    setStorage(report);
    setSavedAcquisitions(attempts);
    if (nextSelected) {
      void listPdfImports(connection, nextSelected)
        .then((items) => {
          if (selectionSourceRef.current === nextSelected) setImports(items);
        })
        .catch(() => undefined);
      refreshExtractionFor(nextSelected);
    }
  }

  async function submitNotebook(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!connection || !notebookTitle.trim()) return;
    try {
      const created = await createNotebook(connection, notebookTitle.trim());
      const items = await listNotebooks(connection);
      setNotebooks(items);
      setSelectedNotebookId(created.id);
      setNotebookTitle("");
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  function refreshExtractionFor(sourceId: string) {
    if (!connection) return;
    void getExtraction(connection, sourceId)
      .then((report) => {
        if (selectionSourceRef.current === sourceId) setExtraction(report);
      })
      .catch(() => {
        if (selectionSourceRef.current === sourceId) setExtraction(null);
      });
  }

  async function decideReview(decision: "confirmed" | "declined") {
    if (!connection || !extraction?.extraction_id || !extraction.review) return;
    const targetId = extraction.extraction_id;
    try {
      const review: ExtractionReview = await decideExtractionReview(connection, targetId, {
        decision,
        generation: extraction.review.generation,
      });
      setExtraction((current) =>
        current?.extraction_id === targetId ? { ...current, review } : current,
      );
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  async function reviewAgain() {
    if (!connection || !extraction?.extraction_id) return;
    const targetId = extraction.extraction_id;
    try {
      const review: ExtractionReview = await reopenExtractionReview(connection, targetId);
      setExtraction((current) =>
        current?.extraction_id === targetId ? { ...current, review } : current,
      );
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  function choosePdf(intent: ImportIntent, sourceId?: string, acquisitionId?: string) {
    if (!selectedNotebookId || uploading) return;
    nextIntentRef.current = { intent, sourceId, acquisitionId };
    fileInputRef.current?.click();
  }

  async function acceptPdf(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file || !connection || !selectedNotebookId) return;
    await importPdf(file, nextIntentRef.current.intent, nextIntentRef.current.sourceId, nextIntentRef.current.acquisitionId);
  }

  async function importPdf(file: File, intent: ImportIntent, targetSourceId?: string, resumeId?: string) {
    if (!connection || uploading) return;
    setUploading(true);
    setError(null);
    const controller = new AbortController();
    uploadControllerRef.current = controller;
    acquisitionIdRef.current = resumeId ?? "";
    try {
      const action = resumeId ? await getPdfAcquisition(connection, resumeId) : await createPdfAcquisition(connection, selectedNotebookId, {
        retry_key: crypto.randomUUID(),
        filename: file.name || "import.pdf",
        intent,
        ...(targetSourceId ? { target_source_id: targetSourceId } : {}),
      });
      if (action.notebook_id !== selectedNotebookId) throw new Error("This import belongs to another notebook.");
      rememberAcquisition(action);
      acquisitionIdRef.current = action.id;
      const saved = await uploadPdf(connection, action.id, file, controller.signal);
      rememberAcquisition(saved);
      if (saved.source_id) setSelectedSourceId(saved.source_id);
      await refreshNotebook(selectedNotebookId, saved.source_id ?? undefined);
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
      if (acquisitionIdRef.current) {
        void getPdfAcquisition(connection, acquisitionIdRef.current)
          .then(async (latest) => {
            rememberAcquisition(latest);
            if (latest.state === "completed") {
              if (latest.source_id) setSelectedSourceId(latest.source_id);
              await refreshNotebook(selectedNotebookId,latest.source_id ?? undefined);
            }
          })
          .catch(() => undefined);
      }
    } finally {
      uploadControllerRef.current = null;
      setUploading(false);
    }
  }

  async function retryImport(item: PdfAcquisition) {
    if (!connection || uploading || !item.can_retry) return;
    rememberAcquisition(item);
    setUploading(true);
    setError(null);
    try {
      const saved = await retryPdfAcquisition(connection, item.id);
      rememberAcquisition(saved);
      if (saved.source_id) setSelectedSourceId(saved.source_id);
      if (selectedNotebookId) await refreshNotebook(selectedNotebookId);
    } catch (cause) {
      setError(messageFor(cause));
      rememberAcquisition(await getPdfAcquisition(connection, item.id).catch(() => item));
    } finally {
      setUploading(false);
    }
  }

  async function cancelImport(item: PdfAcquisition) {
    if (!connection || !item.can_cancel) return;
    if (item.id === acquisitionIdRef.current) uploadControllerRef.current?.abort();
    try {
      rememberAcquisition(await cancelPdfAcquisition(connection, item.id));
      if (item.id === acquisitionIdRef.current) setUploading(false);
      if (selectedNotebookId) await refreshNotebook(selectedNotebookId);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  async function removeEarlierImport(item: PdfImport) {
    if (!connection || !selectedSourceId || item.is_current || item.is_pending || item.trashed) return;
    try {
      await trashPdfImport(connection, selectedSourceId, item.id);
      setImports(await listPdfImports(connection, selectedSourceId));
      refreshExtractionFor(selectedSourceId);
      setStorage(await storageReport(connection));
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  async function retryProcessing(item: PdfImport) {
    if (!connection) return;
    try {
      await retryPdfProcessing(connection, item.source_id, item.id);
      setImports(await listPdfImports(connection, item.source_id));
      refreshExtractionFor(item.source_id);
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  async function cancelProcessing(item: PdfImport) {
    if (!connection) return;
    try {
      await cancelPdfProcessing(connection, item.source_id, item.id);
      setImports(await listPdfImports(connection, item.source_id));
      refreshExtractionFor(item.source_id);
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  async function removeSource(sourceId: string) {
    if (!connection || !selectedNotebookId) return;
    try {
      await trashSource(connection, sourceId);
      setViewerImport(null);
      await refreshNotebook(selectedNotebookId);
      setError(null);
    } catch (cause) {
      setError(messageFor(cause));
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>Corpi</h1>
          <p id="core-status" role="status">
            {STATUS_TEXT[status]}{version ? ` (v${version})` : ""}
          </p>
        </div>
        <div className="storage-summary" aria-label="Original PDF storage">
          <strong>{storage ? formatBytes(storage.original_bytes) : "—"} in originals</strong>
          {storage ? <span>{storage.original_file_count} physical PDF files</span> : null}
          {storage?.bytes_pending_removal ? <small>{formatBytes(storage.bytes_pending_removal)} waiting for removal</small> : null}
          {storage?.retained_staging_bytes ? <small>{formatBytes(storage.retained_staging_bytes)} retained staging</small> : null}
          <small>Shared PDFs count once. Database and search data aren't included.</small>
        </div>
      </header>

      {error ? <p id="core-error" role="alert">{error}</p> : null}

      {!connection ? null : (
        <div className="workspace">
          <aside className="notebook-panel" aria-label="Notebooks">
            <h2>Notebooks</h2>
            <label htmlFor="notebook-select">Open notebook</label>
            <select
              id="notebook-select"
              value={selectedNotebookId}
              disabled={uploading}
              onChange={(event) => setSelectedNotebookId(event.currentTarget.value)}
            >
              <option value="">Choose a notebook</option>
              {notebooks.map((notebook) => (
                <option key={notebook.id} value={notebook.id}>{notebook.title}</option>
              ))}
            </select>
            <form onSubmit={submitNotebook} className="create-notebook">
              <label htmlFor="new-notebook-title">New notebook</label>
              <input
                id="new-notebook-title"
                value={notebookTitle}
                onChange={(event) => setNotebookTitle(event.currentTarget.value)}
                maxLength={200}
                required
              />
              <button id="create-notebook" type="submit" disabled={uploading}>Create notebook</button>
            </form>
            {selectedNotebookId ? (
              <>
                <button
                  type="button"
                  id="add-pdf"
                  className="primary-action"
                  onClick={() => choosePdf("add")}
                  disabled={uploading}
                >
                  Add PDF
                </button>
                <small className="import-hint">
                  One PDF at a time · limit {formatBytes(storage?.max_upload_bytes ?? 25_000_000)}
                </small>
              </>
            ) : null}
            <input
              ref={fileInputRef}
              className="visually-hidden"
              data-testid="pdf-file-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={acceptPdf}
            />
            <ul className="source-list" aria-label="Sources">
              {sources.map((source) => (
                <li key={source.id}>
                  <button
                    type="button"
                    className={source.id === selectedSourceId ? "source-list__selected" : ""}
                    onClick={() => setSelectedSourceId(source.id)}
                  >
                    <span>{source.display_name}</span>
                    {source.pending_import_id ? <small>Replacement pending</small> : null}
                  </button>
                </li>
              ))}
            </ul>
          </aside>

          <section className="source-panel" aria-label="Notebook source">
            {selectedSource ? (
              <>
                <div className="source-heading">
                  <div>
                    <h2>{selectedSource.display_name}</h2>
                    <p>PDF files and processing state</p>
                  </div>
                  <button
                    type="button"
                    id="replace-pdf"
                    onClick={() => choosePdf("replace", selectedSource.id)}
                    disabled={uploading || !selectedSource.current_import_id}
                  >
                    Replace PDF
                  </button>
                  <button type="button" onClick={() => void removeSource(selectedSource.id)}>
                    Move source to trash
                  </button>
                </div>
                <h3>Files</h3>
                {imports.length ? (
                  <ul className="import-list" aria-label="PDF file history">
                    {imports.map((item) => (
                      <li key={item.id} data-testid={`pdf-import-${item.id}`}>
                        <div>
                          <strong>{item.filename}</strong>
                          <span>{formatBytes(item.size_bytes)} · {new Date(item.created_at).toLocaleString()}</span>
                          <small>{importStateLabel(item, selectedSource)}</small>
                        </div>
                        <button
                          type="button"
                          onClick={(event) => { event.currentTarget.focus(); setViewerImport(item); }}
                          disabled={item.trashed}
                        >
                          Open
                        </button>
                        {!item.is_current && !item.is_pending && !item.trashed ? (
                          <button type="button" onClick={() => void removeEarlierImport(item)}>
                            Remove earlier import
                          </button>
                        ) : null}
                        {item.state === "failed" && !item.trashed ? (
                          <button type="button" onClick={() => void retryProcessing(item)}>
                            Retry processing
                          </button>
                        ) : null}
                        {item.state === "pending" && !item.trashed ? (
                          <button type="button" onClick={() => void cancelProcessing(item)}>
                            Cancel processing
                          </button>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : <p>No PDFs imported yet.</p>}
                {extraction ? (
                  <section className="extraction-report" aria-label="Text extraction">
                    <h3>Text extraction</h3>
                    {extraction.failure_code ? (
                      <p role="alert">
                        {extraction.failure_code === "no_usable_text"
                          ? "Extraction failed: no usable text could be read from this PDF."
                          : extraction.failure_code === "page_limit_exceeded"
                            ? "Extraction failed: the document is over the page limit."
                            : `Extraction failed: ${extraction.failure_code}`}
                      </p>
                    ) : null}
                    {extraction.outcome ? <p>Outcome: {extraction.outcome}</p> : null}
                    {extraction.pages.length ? (
                      <ul className="page-report" aria-label="Page report">
                        {extraction.pages.map((page) => (
                          <li key={page.page_number}>
                            Page {page.page_number} — {page.classification} (
                            {page.extracted_chars} chars
                            {page.image_area_ratio > 0
                              ? `, ${Math.round(page.image_area_ratio * 100)}% images`
                              : ""}
                            )
                          </li>
                        ))}
                      </ul>
                    ) : null}
                    {extraction.review && extraction.extraction_id ? (
                      extraction.review.state === "pending" ? (
                        <div>
                          <button type="button" onClick={() => void decideReview("confirmed")}>
                            Confirm
                          </button>
                          <button type="button" onClick={() => void decideReview("declined")}>
                            Decline
                          </button>
                        </div>
                      ) : (
                        <div>
                          <p>Review: {extraction.review.state}</p>
                          <button type="button" onClick={() => void reviewAgain()}>
                            Review Again
                          </button>
                        </div>
                      )
                    ) : null}
                  </section>
                ) : null}
              </>
            ) : (
              <div className="empty-state">
                <h2>Start with a notebook</h2>
                <p>Create one, then add a PDF. Corpi keeps the saved original available while text processing runs.</p>
              </div>
            )}

            {visibleAcquisitions.map((item) => {
              const activeUpload = uploading && item.id === acquisition?.id;
              return (
                <section key={item.id} className="acquisition-status" aria-label={`PDF import status: ${item.filename}`} aria-live="polite">
                  <h3>{item.filename}</h3>
                  <p>{acquisitionStatus(item, activeUpload)}</p>
                  {activeUpload ? (
                    <progress
                      max={Math.max(item.expected_size_bytes ?? 1, 1)}
                      value={item.received_size_bytes}
                      aria-label="PDF upload progress"
                    />
                  ) : null}
                  {item.can_retry ? (
                    <button type="button" disabled={uploading} onClick={() => void retryImport(item)}>Retry from saved PDF</button>
                  ) : null}
                  {item.requires_upload ? (
                    <button type="button" disabled={uploading} onClick={() => choosePdf(item.intent === "replace" ? "replace" : "add", item.target_source_id ?? undefined, item.id)}>Select PDF again</button>
                  ) : null}
                  {item.can_cancel ? (
                    <button type="button" disabled={uploading && !activeUpload} onClick={() => void cancelImport(item)}>Cancel import</button>
                  ) : null}
                </section>
              );
            })}
          </section>
        </div>
      )}

      {viewerImport && connection && storage ? (
        <Suspense fallback={<p role="status">Loading PDF viewer…</p>}>
          <PdfViewer
            connection={connection}
            sourceId={viewerImport.source_id}
            importId={viewerImport.id}
            filename={viewerImport.filename}
            renderedPagePixelLimit={storage.rendered_page_pixel_limit}
            onClose={() => setViewerImport(null)}
          />
        </Suspense>
      ) : null}
    </main>
  );
}

function messageFor(cause: unknown): string {
  if (cause instanceof CoreApiError) return cause.message;
  return cause instanceof Error ? cause.message : String(cause);
}

function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${Math.round(bytes / 1_000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function importStateLabel(item: PdfImport, source: NotebookSource): string {
  if (item.trashed) return "In trash; retained for 30 days";
  if (item.is_pending) return "Pending replacement";
  if (item.is_current && item.state === "ready") return "Current PDF · ready for questions";
  if (item.is_current && item.state === "failed") {
    return `Current PDF · processing failed${item.failure_code ? ` (${item.failure_code})` : ""}; original is available`;
  }
  if (item.is_current) return "Current PDF · waiting for text processing";
  if (source.current_import_id !== item.id) {
    if (item.state === "failed") {
      return `Earlier import · processing failed${item.failure_code ? ` (${item.failure_code})` : ""}`;
    }
    return "Earlier import";
  }
  return item.state;
}

function acquisitionStatus(item: PdfAcquisition, uploading: boolean): string {
  if (uploading) {
    return item.expected_size_bytes
      ? `Saving PDF: ${formatBytes(item.received_size_bytes)} of ${formatBytes(item.expected_size_bytes)}`
      : `Saving PDF: ${formatBytes(item.received_size_bytes)} received`;
  }
  if (item.state === "waiting_cleanup") return "The previous copy is still being removed. Retry after cleanup finishes.";
  if (item.state === "staged" || item.state === "finalizing") return "PDF saved; completing the import…";
  if (item.state === "failed") return `Import stopped${item.failure_code ? ` (${item.failure_code})` : ""}. ${item.requires_upload ? "Select the PDF again; no complete file was saved." : "The original file remains available if it was saved."}`;
  if (item.state === "cancelled") return "Import cancelled.";
  if (item.requires_upload) return "Select the PDF again; the interrupted upload did not save a complete file.";
  if (item.outcome_kind === "duplicate") return "This PDF is already in the notebook. Open the existing source.";
  if (item.outcome_kind === "unchanged") return "The selected PDF matches the current file; no replacement was needed.";
  return "PDF saved. Text extraction will become available when M1.5 processing is connected.";
}
