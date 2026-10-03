import { useEffect, useRef, useState } from "react";
import {
  getDocument,
  PDFWorker,
  type PDFDocumentProxy,
  type RenderTask,
} from "pdfjs-dist/legacy/build/pdf.mjs";
import PdfJsWorker from "pdfjs-dist/legacy/build/pdf.worker.mjs?worker";

import { CoreConnection, coreBinary } from "../api/client";

interface PdfViewerProps {
  connection: CoreConnection;
  sourceId: string;
  importId: string;
  filename: string;
  renderedPagePixelLimit: number;
  onClose: () => void;
}

export function PdfViewer({
  connection,
  sourceId,
  importId,
  filename,
  renderedPagePixelLimit,
  onClose,
}: PdfViewerProps) {
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [renderedPage, setRenderedPage] = useState<{
    document: PDFDocumentProxy;
    pageNumber: number;
    zoom: number;
  } | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const renderRef = useRef<RenderTask | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const opener = window.document.activeElement;
    dialog?.showModal();
    return () => {
      dialog?.close();
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, []);

  useEffect(() => {
    let current = true;
    // React clears refs before passive unmount cleanup; keep the owned canvas.
    const canvas = canvasRef.current;
    const controller = new AbortController();
    let loadingTask: ReturnType<typeof getDocument> | undefined;
    let pdfWorker: PDFWorker | undefined;
    let nativeWorker: Worker | undefined;

    setDocument(null);
    setRenderedPage(null);
    setPageNumber(1);
    setLoading(true);
    setLoadError(null);
    setRenderError(null);

    async function loadPdf() {
      try {
        const path = `/sources/${encodeURIComponent(sourceId)}/imports/${encodeURIComponent(importId)}/pdf`;
        const bytes = await coreBinary(connection, path, controller.signal);
        if (!current) return;

        nativeWorker = new PdfJsWorker();
        pdfWorker = PDFWorker.create({ port: nativeWorker });
        const publicAssets = new URL("pdfjs/", window.location.href).toString();
        loadingTask = getDocument({
          data: bytes,
          worker: pdfWorker,
          enableXfa: false,
          cMapUrl: new URL("cmaps/", publicAssets).toString(),
          standardFontDataUrl: new URL("standard_fonts/", publicAssets).toString(),
          wasmUrl: new URL("wasm/", publicAssets).toString(),
          iccUrl: new URL("iccs/", publicAssets).toString(),
          useWorkerFetch: true,
        });
        const loaded = await loadingTask.promise;
        if (current) setDocument(loaded);
      } catch (cause) {
        await loadingTask?.destroy().catch(() => undefined);
        pdfWorker?.destroy();
        nativeWorker?.terminate();
        if (current && !(cause instanceof DOMException && cause.name === "AbortError")) {
          const message = cause instanceof Error ? cause.message : String(cause);
          setLoadError(`Couldn't open ${filename}: ${message}`);
        }
      } finally {
        if (current) setLoading(false);
      }
    }

    void loadPdf();
    return () => {
      current = false;
      controller.abort();
      renderRef.current?.cancel();
      if (canvas) {
        canvas.width = 0;
        canvas.height = 0;
      }
      void (async () => {
        await loadingTask?.destroy().catch(() => undefined);
        pdfWorker?.destroy();
        nativeWorker?.terminate();
      })();
    };
  }, [connection, filename, importId, sourceId]);

  useEffect(() => {
    if (!document) return;
    const activeDocument = document;
    setRenderedPage(null);
    setRenderError(null);
    let cancelled = false;
    let renderTask: RenderTask | undefined;

    async function renderPage() {
      try {
        const page = await activeDocument.getPage(pageNumber);
        if (cancelled) return;
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d", { alpha: false });
        if (!canvas || !context) throw new Error("This webview couldn't allocate a PDF canvas.");

        const deviceScale = window.devicePixelRatio || 1;
        const cssViewport = page.getViewport({ scale: zoom });
        const maxScale = Math.sqrt(
          renderedPagePixelLimit / (cssViewport.width * cssViewport.height),
        );
        const outputScale = Math.min(deviceScale, maxScale * 0.999);
        const viewport = page.getViewport({ scale: zoom * outputScale });
        canvas.width = Math.max(1, Math.ceil(viewport.width));
        canvas.height = Math.max(1, Math.ceil(viewport.height));
        canvas.style.width = `${viewport.width / outputScale}px`;
        canvas.style.height = `${viewport.height / outputScale}px`;
        renderTask = page.render({ canvas, viewport });
        renderRef.current = renderTask;
        await renderTask.promise;
        if (!cancelled) {
          setRenderError(null);
          setRenderedPage({ document: activeDocument, pageNumber, zoom });
        }
      } catch (cause) {
        if (!cancelled && !(cause instanceof Error && cause.name === "RenderingCancelledException")) {
          setRenderError(cause instanceof Error ? cause.message : "The selected page could not be rendered.");
        }
      }
    }

    void renderPage();
    return () => {
      cancelled = true;
      renderTask?.cancel();
      renderRef.current = null;
    };
  }, [document, pageNumber, renderedPagePixelLimit, zoom]);

  const pageCount = document?.numPages ?? 0;

  return (
    <dialog
      ref={dialogRef}
      className="pdf-viewer"
      aria-label="PDF viewer"
      aria-modal="true"
      data-testid="pdf-viewer"
      onCancel={(event) => { event.preventDefault(); onClose(); }}
    >
      <header className="pdf-viewer__toolbar">
        <div>
          <h2>{filename}</h2>
          <p>{loading ? "Opening saved PDF…" : pageCount ? `${pageCount} pages` : ""}</p>
        </div>
        <div className="pdf-viewer__controls">
          <button type="button" onClick={() => setPageNumber((page) => Math.max(1, page - 1))} disabled={!document || pageNumber <= 1}>
            Previous page
          </button>
          <label>
            Page
            <input
              aria-label="Page number"
              type="number"
              min={1}
              max={Math.max(1, pageCount)}
              value={pageNumber}
              onChange={(event) => {
                const value = Number(event.currentTarget.value);
                if (Number.isInteger(value) && value >= 1 && value <= pageCount) setPageNumber(value);
              }}
              disabled={!document}
            />
            of {pageCount || "…"}
          </label>
          <button type="button" onClick={() => setPageNumber((page) => Math.min(pageCount, page + 1))} disabled={!document || pageNumber >= pageCount}>
            Next page
          </button>
          <button type="button" onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))} disabled={!document || zoom <= 0.5}>
            Zoom out
          </button>
          <span aria-live="polite">{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => setZoom((value) => Math.min(3, value + 0.1))} disabled={!document || zoom >= 3}>
            Zoom in
          </button>
          <button type="button" autoFocus onClick={onClose}>Close PDF</button>
        </div>
      </header>
      <div className="pdf-viewer__content">
        {loadError ? (
          <div className="pdf-viewer__message">
            <h3>The saved PDF could not be opened</h3>
            <p role="alert">{loadError}</p>
            <p>The saved original is retained. Close the viewer to return to Files.</p>
          </div>
        ) : null}
        {loading ? <p className="pdf-viewer__message" role="status">Verifying and opening the saved copy…</p> : null}
        {renderError ? <p className="pdf-viewer__message" role="alert">{renderError}</p> : null}
        <div
          className="pdf-viewer__page"
          hidden={loading || loadError !== null}
          aria-label={`Page ${pageNumber}`}
          aria-busy={document !== null && (renderedPage?.document !== document || renderedPage?.pageNumber !== pageNumber || renderedPage?.zoom !== zoom)}
        >
          <canvas ref={canvasRef} aria-label={`${filename}, page ${pageNumber}`} />
        </div>
      </div>
    </dialog>
  );
}
