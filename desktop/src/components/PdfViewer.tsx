import { useEffect, useRef, useState } from "react";
import {
  getDocument,
  PDFWorker,
  type PDFDocumentProxy,
  type RenderTask,
} from "pdfjs-dist/legacy/build/pdf.mjs";
import PdfJsWorker from "pdfjs-dist/legacy/build/pdf.worker.mjs?worker";

import { CoreConnection, coreBinary } from "../api/client";
import { pageLayout, visibleTiles, type Tile } from "./pdfTiles";

interface PdfViewerProps {
  connection: CoreConnection;
  sourceId: string;
  importId: string;
  filename: string;
  renderedPagePixelLimit: number;
  initialPage?: number;
  onClose: () => void;
}

export function PdfViewer({
  connection,
  sourceId,
  importId,
  filename,
  renderedPagePixelLimit,
  initialPage = 1,
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
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const renderRef = useRef<RenderTask | null>(null);
  const renderQueue = useRef<Promise<void>>(Promise.resolve());
  const workerRelease = useRef<Promise<void>>(Promise.resolve());
  const documentOwner = useRef<PDFDocumentProxy | null>(null);

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
    const controller = new AbortController();
    let loadingTask: ReturnType<typeof getDocument> | undefined;
    let pdfWorker: PDFWorker | undefined;
    let nativeWorker: Worker | undefined;
    let release: Promise<void> | undefined;

    function releaseDocument() {
      // A cancelled task must settle before PDF.js destroys its transport/worker.
      const drain = renderQueue.current;
      return release ??= (async () => {
        await drain;
        await loadingTask?.destroy().catch(() => undefined);
        pdfWorker?.destroy();
        nativeWorker?.terminate();
      })();
    }

    documentOwner.current = null;
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
        // The same mounted dialog can receive a new saved import/citation.
        // Finish the previous document's worker release before creating its successor.
        await workerRelease.current;
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
        if (current) {
          if (!Number.isInteger(initialPage) || initialPage < 1 || initialPage > loaded.numPages) {
            throw new Error("The saved citation page is outside this PDF.");
          }
          setPageNumber(initialPage);
          documentOwner.current = loaded;
          setDocument(loaded);
        }
      } catch (cause) {
        await releaseDocument();
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
      documentOwner.current = null;
      controller.abort();
      renderRef.current?.cancel();
      workerRelease.current = releaseDocument();
    };
  }, [connection, filename, importId, initialPage, sourceId]);

  useEffect(() => {
    if (!document || documentOwner.current !== document) return;
    const activeDocument = document;
    const surface = surfaceRef.current!;
    const scroller = scrollRef.current!;
    setRenderedPage(null);
    setRenderError(null);
    let stopped = false;
    let failed = false;
    let frame: number | undefined;
    let queued = false;
    let activeTile: string | undefined;
    let task: RenderTask | undefined;
    let desired: Tile[] = [];
    const entries = new Map<string, { canvas: HTMLCanvasElement; ready: boolean }>();
    const current = () => !stopped && documentOwner.current === activeDocument;

    function releaseTile(id: string) {
      const entry = entries.get(id);
      if (!entry) return;
      if (activeTile === id) task?.cancel();
      // Removing a DOM node alone does not free its backing storage.
      entry.canvas.width = entry.canvas.height = 0;
      entry.canvas.remove();
      entries.delete(id);
    }

    async function preparePage() {
      try {
        const page = await activeDocument.getPage(pageNumber);
        if (!current()) return;
        const cssViewport = page.getViewport({ scale: zoom });
        const layout = pageLayout(cssViewport.width, cssViewport.height,
          window.devicePixelRatio || 1, renderedPagePixelLimit);
        const viewport = page.getViewport({ scale: zoom * layout.outputScale });
        surface.style.width = `${layout.cssWidth}px`;
        surface.style.height = `${layout.cssHeight}px`;
        Object.assign(surface.dataset, { pageWidth: String(layout.width), pageHeight: String(layout.height),
          pageNumber: String(pageNumber), zoom: String(zoom), outputScale: String(layout.outputScale) });

        function pump() {
          if (queued || !current() || failed) return;
          queued = true;
          // The queue also spans page/zoom changes: cancellation cannot overlap a
          // new render with the previous PDF.js task that is still settling.
          renderQueue.current = renderQueue.current.then(async () => {
            while (current() && !failed) {
              const tile = desired.find((item) => !entries.get(item.id)?.ready);
              if (!tile) break;
              let entry = entries.get(tile.id);
              if (!entry) {
                const canvas = window.document.createElement("canvas");
                canvas.width = tile.width;
                canvas.height = tile.height;
                canvas.dataset.tile = tile.id;
                canvas.setAttribute("aria-label", `${filename}, page ${pageNumber}, tile ${tile.id}`);
                Object.assign(canvas.style, { position: "absolute", display: "block", boxShadow: "none",
                  left: `${tile.x * layout.cssWidth / layout.width}px`,
                  top: `${tile.y * layout.cssHeight / layout.height}px`,
                  width: `${tile.width * layout.cssWidth / layout.width}px`,
                  height: `${tile.height * layout.cssHeight / layout.height}px` });
                if (!canvas.getContext("2d", { alpha: false, willReadFrequently: true })) {
                  throw new Error("This webview couldn't allocate a PDF tile.");
                }
                entry = { canvas, ready: false };
                entries.set(tile.id, entry);
                surface.append(entry.canvas);
              }
              activeTile = tile.id;
              task = page.render({ canvas: entry.canvas, viewport,
                transform: [1, 0, 0, 1, -tile.x, -tile.y] });
              renderRef.current = task;
              try {
                await task.promise;
                if (current() && entries.get(tile.id) === entry) {
                  entry.ready = true;
                  entry.canvas.dataset.ready = "true";
                }
              } catch (cause) {
                if (current() && entries.get(tile.id) === entry) throw cause;
              } finally {
                if (renderRef.current === task) renderRef.current = null;
                task = undefined;
                activeTile = undefined;
              }
            }
            if (current() && !failed) setRenderedPage({ document: activeDocument, pageNumber, zoom });
          }).catch((cause: unknown) => {
            if (current()) {
              failed = true;
              setRenderError(cause instanceof Error ? cause.message : "The selected page could not be rendered.");
            }
          }).finally(() => { queued = false; });
        }

        function updateVisible() {
          if (!current() || failed) return;
          const region = scroller.getBoundingClientRect();
          const bounds = surface.getBoundingClientRect();
          const next = visibleTiles(layout, { left: region.left + scroller.clientLeft - bounds.left,
            top: region.top + scroller.clientTop - bounds.top,
            width: scroller.clientWidth, height: scroller.clientHeight });
          if (desired.map((tile) => tile.id).join() === next.map((tile) => tile.id).join()) return;
          desired = next;
          const keep = new Set(next.map((tile) => tile.id));
          for (const id of entries.keys()) if (!keep.has(id)) releaseTile(id);
          setRenderedPage(null);
          pump();
        }
        refresh = () => {
          if (frame !== undefined || !current()) return;
          frame = requestAnimationFrame(() => { frame = undefined; updateVisible(); });
        };
        updateVisible();
      } catch (cause) {
        if (current()) {
          failed = true;
          setRenderError(cause instanceof Error ? cause.message : "The selected page could not be rendered.");
        }
      }
    }

    let refresh = () => {};
    const schedule = () => refresh();
    scroller.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    const observer = new ResizeObserver(schedule);
    observer.observe(scroller);
    void preparePage();
    return () => {
      stopped = true;
      if (frame !== undefined) cancelAnimationFrame(frame);
      scroller.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      observer.disconnect();
      task?.cancel();
      for (const id of entries.keys()) releaseTile(id);
      for (const key of ["pageWidth", "pageHeight", "pageNumber", "zoom", "outputScale"]) delete surface.dataset[key];
    };
  }, [document, filename, pageNumber, renderedPagePixelLimit, zoom]);

  const pageCount = document?.numPages ?? 0;

  return (
    <dialog
      ref={dialogRef}
      className="pdf-viewer"
      aria-label="PDF viewer"
      aria-modal="true"
      data-testid="pdf-viewer"
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onKeyDown={(event) => {
        if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return;
        // Keep the toolbar cycle consistent when a native webview would focus
        // the dialog/container between the last and first enabled controls.
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled)",
        ));
        if (!controls.length) return;
        const index = controls.findIndex((control) => control === window.document.activeElement);
        const next = index < 0 ? (event.shiftKey ? controls.length - 1 : 0)
          : (index + (event.shiftKey ? -1 : 1) + controls.length) % controls.length;
        event.preventDefault();
        controls[next].focus();
      }}
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
          ref={scrollRef}
          className="pdf-viewer__page"
          hidden={loading || loadError !== null}
          aria-label={`Page ${pageNumber}`}
          aria-busy={document !== null && (renderedPage?.document !== document || renderedPage?.pageNumber !== pageNumber || renderedPage?.zoom !== zoom)}
        >
          <div ref={surfaceRef} data-testid="pdf-tile-surface"
            style={{ position: "relative", margin: "0 auto", textAlign: "left" }} />
        </div>
      </div>
    </dialog>
  );
}
