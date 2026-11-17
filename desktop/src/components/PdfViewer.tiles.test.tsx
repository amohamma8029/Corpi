import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { PdfViewer } from "./PdfViewer";
import { coreBinary } from "../api/client";

const control = vi.hoisted(() => ({
  active: 0, maximum: 0, workers: 0, maximumWorkers: 0, started: 0, failPage: 0, holdNext: false,
  calls: [] as { page: number; canvas: HTMLCanvasElement; transform: number[]; viewport: { width: number; height: number } }[],
  pending: [] as { finish: () => void; cancel: ReturnType<typeof vi.fn> }[],
  events: [] as string[], observers: [] as (() => void)[],
}));
vi.mock("../api/client", () => ({ coreBinary: vi.fn(async () => new Uint8Array([37, 80, 68, 70])) }));
vi.mock("pdfjs-dist/legacy/build/pdf.worker.mjs?worker", () => ({
  default: class {
    constructor() { control.workers++; control.started++; control.maximumWorkers = Math.max(control.maximumWorkers, control.workers); }
    terminate() { control.workers--; control.events.push("terminate"); }
  },
}));
vi.mock("pdfjs-dist/legacy/build/pdf.mjs", () => ({
  PDFWorker: { create: () => ({ destroy() {} }) },
  getDocument: () => ({
    destroy: async () => { control.events.push("destroy"); },
    promise: Promise.resolve({ numPages: 3, getPage: async (page: number) => ({
      getViewport: ({ scale }: { scale: number }) => ({ width: 1700 * scale, height: 1700 * scale }),
      render: (options: (typeof control.calls)[number]) => {
        control.active++;
        control.maximum = Math.max(control.maximum, control.active);
        control.calls.push({ ...options, page });
        let promise: Promise<void>;
        const cancel = vi.fn(() => { control.events.push("cancel"); });
        if (control.holdNext) {
          control.holdNext = false;
          promise = new Promise<void>((finish) => { control.pending.push({ finish, cancel }); });
        } else promise = control.failPage === page ? Promise.reject(new Error("Selected page failed")) : Promise.resolve();
        return { cancel, promise: promise.finally(() => { control.active--; control.events.push("settled"); }) };
      },
    }) }),
  }),
}));

const connection = { base_url: "http://127.0.0.1:9", token: "test-token" };
const props = { connection, sourceId: "saved-source", importId: "original-import", filename: "original.pdf",
  renderedPagePixelLimit: 20_000_000, onClose() {} };
const modalDescriptors = ["showModal", "close"].map((name) => Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, name));
const surface = () => screen.getByTestId("pdf-tile-surface");
const scroller = () => surface().parentElement!;
async function ready() {
  await waitFor(() => expect(surface().dataset.pageWidth).toBeDefined());
  await waitFor(() => expect(scroller().getAttribute("aria-busy")).toBe("false"));
  expect(surface().querySelectorAll("canvas[data-ready='true']").length).toBeGreaterThan(0);
}

beforeEach(() => {
  Object.assign(control, { active: 0, maximum: 0, workers: 0, maximumWorkers: 0, started: 0, failPage: 0, holdNext: false,
    calls: [], pending: [], events: [], observers: [] });
  vi.mocked(coreBinary).mockReset().mockResolvedValue(new Uint8Array([37, 80, 68, 70]));
  Object.defineProperties(HTMLDialogElement.prototype, {
    showModal: { configurable: true, value: function (this: HTMLDialogElement) { this.open = true; } },
    close: { configurable: true, value: function (this: HTMLDialogElement) { this.open = false; } },
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as CanvasRenderingContext2D);
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("pdf-viewer__page") ? 600 : 0;
  });
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("pdf-viewer__page") ? 400 : 0;
  });
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    const host = this.dataset.testid === "pdf-tile-surface" ? this.parentElement : null;
    return { left: -(host?.scrollLeft ?? 0), top: -(host?.scrollTop ?? 0), width: 600, height: 400,
      right: 600, bottom: 400, x: 0, y: 0, toJSON() {} };
  });
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { control.observers.push(callback); }
    observe() {}
    disconnect() { control.events.push("disconnect"); }
  });
  vi.stubGlobal("requestAnimationFrame", (callback: () => void) => window.setTimeout(callback, 0));
  vi.stubGlobal("cancelAnimationFrame", (id: number) => window.clearTimeout(id));
});
afterEach(async () => {
  cleanup();
  await act(async () => { for (const item of control.pending) item.finish(); });
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  ["showModal", "close"].forEach((name, index) => {
    if (modalDescriptors[index]) Object.defineProperty(HTMLDialogElement.prototype, name, modalDescriptors[index]!);
    else Reflect.deleteProperty(HTMLDialogElement.prototype, name);
  });
});

it("paints only bounded visible tiles, in global coordinates, through one serial worker", async () => {
  render(<PdfViewer {...props} />);
  await ready();
  expect(control.maximum).toBe(1);
  expect(control.workers).toBe(1);
  expect(control.started).toBe(1);
  expect(control.calls.every(({ canvas }) => canvas.width <= 512 && canvas.height <= 512)).toBe(true);
  expect(control.calls.some(({ transform }) => transform.join() === "1,0,0,1,-512,0")).toBe(true);
  expect(control.calls.every(({ viewport }) => viewport.width === 1700 * window.devicePixelRatio)).toBe(true);
});

it("evicts and zeros off-screen backing while retaining still-visible canvas identities", async () => {
  render(<PdfViewer {...props} />);
  await ready();
  const first = surface().querySelector<HTMLCanvasElement>("[data-tile='0:0']")!;
  const retained = surface().querySelector<HTMLCanvasElement>("[data-tile='512:0']")!;
  scroller().scrollLeft = 550;
  fireEvent.scroll(scroller());
  await waitFor(() => expect(first.width).toBe(0));
  await ready();
  expect(first.height).toBe(0);
  expect(first.isConnected).toBe(false);
  expect(surface().querySelector("[data-tile='512:0']")).toBe(retained);
  expect(surface().querySelector("[data-tile='1024:0']")).not.toBeNull();
});

it("waits for cancelled zoom work to settle and rejects its late completion", async () => {
  control.holdNext = true;
  render(<PdfViewer {...props} />);
  await waitFor(() => expect(control.pending).toHaveLength(1));
  const old = control.calls[0].canvas;
  fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
  await waitFor(() => expect(control.pending[0].cancel).toHaveBeenCalled());
  expect(control.calls).toHaveLength(1);
  expect(old.width).toBe(0);
  await act(async () => { control.pending[0].finish(); });
  await ready();
  expect(old.dataset.ready).toBeUndefined();
  expect(surface().dataset.zoom).toBe("1.1");
  expect(control.maximum).toBe(1);
});

it("releases every canvas and settles rendering before destroying the document and worker", async () => {
  control.holdNext = true;
  const view = render(<PdfViewer {...props} />);
  await waitFor(() => expect(control.pending).toHaveLength(1));
  const old = control.calls[0].canvas;
  view.unmount();
  expect(old.width).toBe(0);
  expect(old.height).toBe(0);
  expect(control.events).not.toContain("destroy");
  await act(async () => { control.pending[0].finish(); });
  await waitFor(() => expect(control.workers).toBe(0));
  expect(control.events.indexOf("settled")).toBeLessThan(control.events.indexOf("destroy"));
  expect(control.events.indexOf("destroy")).toBeLessThan(control.events.indexOf("terminate"));
});

it("opens the exact saved citation page and rejects an out-of-range citation", async () => {
  const view = render(<PdfViewer {...props} initialPage={2} />);
  await ready();
  expect(control.calls.every(({ page }) => page === 2)).toBe(true);
  expect(coreBinary).toHaveBeenCalledWith(connection, "/sources/saved-source/imports/original-import/pdf", expect.any(AbortSignal));
  view.rerender(<PdfViewer {...props} initialPage={4} />);
  expect((await screen.findByRole("alert")).textContent).toContain("outside this PDF");
  expect(surface().querySelector("canvas")).toBeNull();
});

it("shows a genuine tile failure and recovers on another page without hiding toolbar/focus controls", async () => {
  control.failPage = 1;
  render(<PdfViewer {...props} />);
  expect((await screen.findByRole("alert")).textContent).toContain("Selected page failed");
  const close = screen.getByRole("button", { name: "Close PDF" });
  close.focus();
  fireEvent.keyDown(close, { key: "Tab" });
  expect(document.activeElement).toBe(screen.getByRole("spinbutton", { name: "Page number" }));
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await ready();
  expect(screen.queryByRole("alert")).toBeNull();
  expect(surface().dataset.pageNumber).toBe("2");
});

it("aborts a pending file request and never creates a worker for its late bytes", async () => {
  let finish!: (bytes: Uint8Array<ArrayBuffer>) => void;
  vi.mocked(coreBinary).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
  const view = render(<PdfViewer {...props} />);
  const signal = vi.mocked(coreBinary).mock.calls[0][2]!;
  view.unmount();
  expect(signal.aborted).toBe(true);
  await act(async () => { finish(new Uint8Array([37, 80, 68, 70])); });
  expect(control.started).toBe(0);
});

it("keeps the original near-20-MP resolution at high zoom and releases old zoom tiles", async () => {
  render(<PdfViewer {...props} />);
  await ready();
  const old = [...surface().querySelectorAll("canvas")];
  for (let index = 0; index < 17; index++) fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
  await ready();
  expect(Number(surface().dataset.pageWidth) * Number(surface().dataset.pageHeight)).toBeGreaterThanOrEqual(19_000_000);
  expect(Number(surface().dataset.pageWidth) * Number(surface().dataset.pageHeight)).toBeLessThanOrEqual(20_000_000);
  expect(old.every((canvas) => canvas.width === 0 && canvas.height === 0)).toBe(true);
  expect(control.maximum).toBe(1);
});

it("refreshes newly visible regions after resize without retaining removed tiles", async () => {
  render(<PdfViewer {...props} />);
  await ready();
  const old = surface().querySelector<HTMLCanvasElement>("[data-tile='0:0']")!;
  scroller().scrollTop = 600;
  act(() => { for (const callback of control.observers) callback(); });
  await waitFor(() => expect(old.width).toBe(0));
  await ready();
  expect(surface().querySelector("[data-tile='0:512']")).not.toBeNull();
});

it("waits for the old saved import's worker to stop before creating its successor", async () => {
  control.holdNext = true;
  const view = render(<PdfViewer {...props} />);
  await waitFor(() => expect(control.pending).toHaveLength(1));
  view.rerender(<PdfViewer {...props} importId="new-saved-import" />);
  await waitFor(() => expect(coreBinary).toHaveBeenCalledTimes(2));
  expect(control.started).toBe(1);
  await act(async () => { control.pending[0].finish(); });
  await ready();
  expect(control.started).toBe(2);
  expect(control.maximumWorkers).toBe(1);
});
