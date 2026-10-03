import { $, browser, expect } from "@wdio/globals";
import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { pdfFixtureBytes, type PdfFixture } from "../pdf-fixtures";

interface ViewerSnapshot {
  filename: string;
  page: number;
  zoom: number;
  width: number;
  height: number;
  rendering: boolean;
  pixels: number[][];
  activeWorkers: number;
  startedWorkers: number;
  terminatedWorkers: number;
  clearedCanvas: boolean;
  errors: string[];
  cspViolations: string[];
}

declare global {
  interface Window {
    __corpiE2E: {
      activeWorkers: Set<Worker>;
      startedWorkers: number;
      terminatedWorkers: number;
      canvas: HTMLCanvasElement | null;
      errors: string[];
      cspViolations: string[];
      fetchHeld: boolean;
      fetchAborted: boolean;
      renderHeld: boolean;
      releaseWork?: () => void;
    };
  }
}

const results = process.env.CORPI_E2E_RESULTS ?? path.resolve("e2e-results");
const textPdf: PdfFixture = { filename: "native-text.pdf", kind: "text" };
const rasterPdf: PdfFixture = { filename: "native-raster.pdf", kind: "raster" };
const invalidPdf: PdfFixture = { filename: `native-invalid-${"x".repeat(180)}.pdf`, kind: "invalid" };
const evidence: Record<string, unknown> = { fixtures: [], observations: [] };
const firstPageColor = [51, 128, 204];
const secondPageColor = [204, 51, 77];
const rasterColor = [204, 136, 85];

function record(name: string, value: unknown) {
  (evidence.observations as unknown[]).push({ name, value });
  mkdirSync(results, { recursive: true });
  writeFileSync(path.join(results, "viewer.json"), `${JSON.stringify(evidence, null, 2)}\n`);
}

async function viewerSnapshot(): Promise<ViewerSnapshot> {
  return browser.execute(() => {
    const viewer = document.querySelector("[data-testid='pdf-viewer']");
    const canvas = viewer?.querySelector<HTMLCanvasElement>("canvas");
    const context = canvas?.getContext("2d");
    const observation = window.__corpiE2E;
    if (canvas) observation.canvas = canvas;
    const pixels = canvas && context && canvas.width && canvas.height
      ? [[0.1, 0.1], [0.5, 0.1], [0.9, 0.9]].map(([x, y]) =>
          Array.from(context.getImageData(Math.floor(canvas.width * x), Math.floor(canvas.height * y), 1, 1).data))
      : [];
    return {
      filename: viewer?.querySelector("h2")?.textContent ?? "",
      page: Number(viewer?.querySelector<HTMLInputElement>("[aria-label='Page number']")?.value ?? 0),
      zoom: Number(viewer?.querySelector("[aria-live='polite']")?.textContent?.replace("%", "") ?? 0),
      width: canvas?.width ?? 0,
      height: canvas?.height ?? 0,
      rendering: viewer?.querySelector(".pdf-viewer__page")?.getAttribute("aria-busy") !== "false",
      pixels,
      activeWorkers: observation.activeWorkers.size,
      startedWorkers: observation.startedWorkers,
      terminatedWorkers: observation.terminatedWorkers,
      clearedCanvas: observation.canvas?.width === 0 && observation.canvas?.height === 0,
      errors: Array.from(viewer?.querySelectorAll("[role='alert']") ?? [], (node) => node.textContent ?? ""),
      cspViolations: observation.cspViolations,
    };
  });
}

function matchesPage(snapshot: ViewerSnapshot, color: number[]): boolean {
  return !snapshot.rendering && snapshot.width > 0 && snapshot.height > 0 && snapshot.errors.length === 0
    && snapshot.pixels.length === 3
    && snapshot.pixels.every((pixel) => pixel[3] === 255
      && color.every((channel, index) => Math.abs(channel - pixel[index]) <= 3));
}

async function waitForPage(color: number[], page = 1): Promise<ViewerSnapshot> {
  let snapshot: ViewerSnapshot;
  await browser.waitUntil(async () => {
    snapshot = await viewerSnapshot();
    if (snapshot.errors.length) throw new Error(`PDF renderer error: ${snapshot.errors.join("; ")}`);
    return snapshot.page === page && snapshot.activeWorkers === 1 && matchesPage(snapshot, color);
  }, { timeout: 30000, timeoutMsg: `The native webview did not paint the expected content of page ${page}` });
  return snapshot!;
}

async function importPdf(fixture: PdfFixture) {
  await $("#add-pdf").waitForEnabled();
  // The descriptor is small even for the near-limit PDF. All bytes go through the UI upload.
  const size = await browser.execute(`
    const bytes = (${pdfFixtureBytes.toString()})(arguments[0]);
    const input = document.querySelector('[data-testid="pdf-file-input"]');
    if (!input) throw new Error('PDF input is missing');
    const transfer = new DataTransfer();
    transfer.items.add(new File([bytes], arguments[0].filename, { type: 'application/pdf' }));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return bytes.byteLength;
  `, fixture) as number;
  const bytes = pdfFixtureBytes(fixture);
  await expect(size).toBe(bytes.byteLength);
  (evidence.fixtures as unknown[]).push({ ...fixture, bytes: size, sha256: createHash("sha256").update(bytes).digest("hex") });
  record("uploaded-fixture", { filename: fixture.filename, bytes: size });
  await browser.waitUntil(async () => (await $(".source-list").getText()).includes(fixture.filename), {
    timeout: 60000, timeoutMsg: `${fixture.filename} was not saved as a notebook source`,
  });
}

async function openPdf(filename: string) {
  const buttons = await $$(".source-list li button");
  for (const button of buttons) {
    if ((await button.getText()).includes(filename)) {
      await button.click();
      break;
    }
  }
  await browser.waitUntil(async () => (await $(".import-list").getText()).includes(filename), {
    timeout: 15000, timeoutMsg: `${filename} was not listed in saved import history`,
  });
  await $(".import-list").$("button=Open").click();
  await $("[data-testid='pdf-viewer']").waitForDisplayed();
  await $("[data-testid='pdf-viewer'] h2").waitForDisplayed();
  // WebKitGTK's rendered-text endpoint reports empty text for this clipped
  // heading, even when the retained screenshot visibly shows the filename.
  await browser.waitUntil(async () => (await viewerSnapshot()).filename === filename, {
    timeout: 15000, timeoutMsg: `The visible PDF viewer did not identify ${filename}`,
  });
  await browser.waitUntil(() => browser.execute(() => {
    const viewer = document.querySelector("[data-testid='pdf-viewer']");
    return viewer?.matches(":modal") && viewer.contains(document.activeElement);
  }), { timeout: 5000, timeoutMsg: "Opening the PDF did not move focus into a modal viewer" });
}

async function closePdf(viaCancel = false) {
  if (viaCancel && process.env.CORPI_E2E_PROVIDER === "embedded") {
    // This provider dispatches untrusted KeyboardEvents, which cannot trigger
    // native Escape dismissal. Exercise the browser's actual cancel API instead.
    await browser.execute(() => {
      document.querySelector<HTMLDialogElement>("[data-testid='pdf-viewer']")!.requestClose();
    });
  } else if (viaCancel) await browser.keys("Escape");
  else await $("[data-testid='pdf-viewer']").$("button=Close PDF").click();
  await browser.waitUntil(async () => {
    const snapshot = await viewerSnapshot();
    return !snapshot.filename && snapshot.activeWorkers === 0 && snapshot.clearedCanvas;
  }, { timeout: 15000, timeoutMsg: "Closing the PDF did not release its worker and canvas" });
  record("closed", await viewerSnapshot());
  await browser.waitUntil(() => browser.execute(() => {
    const focused = document.activeElement;
    return focused?.closest(".import-list") !== null && focused?.textContent === "Open";
  }), { timeout: 5000, timeoutMsg: "Closing the viewer did not restore focus to its Open button" });
}

describe("Corpi native notebook and PDF workflow", () => {
  it("launches the real shell and core in the native webview", async () => {
    try {
      await browser.waitUntil(async () => {
        for (const handle of await browser.getWindowHandles()) {
          await browser.switchToWindow(handle);
          if (await $("#core-status").isExisting()) return true;
        }
        return false;
      }, { timeout: 30000, timeoutMsg: "No app webview displayed the core status" });
    } catch (cause) {
      const page = await browser.execute(() => ({
        href: location.href,
        title: document.title,
        readyState: document.readyState,
        root: document.querySelector("#root")?.innerHTML ?? null,
        body: document.body?.innerText ?? null,
        resources: performance.getEntriesByType("resource").map((entry) => entry.name),
      }));
      record("startup-failure", page);
      record("startup-window-handles", await browser.getWindowHandles());
      throw new Error(`Native app startup failed: ${JSON.stringify(page)}`, { cause });
    }
    await expect($("#core-status")).toHaveText(/Core ready/);
    const platform = await browser.execute(() => {
      const observation: Window["__corpiE2E"] = {
        activeWorkers: new Set(), startedWorkers: 0, terminatedWorkers: 0,
        canvas: null, errors: [], cspViolations: [],
        fetchHeld: false, fetchAborted: false, renderHeld: false,
      };
      window.__corpiE2E = observation;
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(url: string | URL, options?: WorkerOptions) {
          super(url, options);
          observation.activeWorkers.add(this);
          observation.startedWorkers += 1;
          this.addEventListener("error", () => observation.errors.push("Native worker error"));
        }
        terminate() {
          if (observation.activeWorkers.delete(this)) observation.terminatedWorkers += 1;
          super.terminate();
        }
      };
      document.addEventListener("securitypolicyviolation", (event) => {
        observation.cspViolations.push(event.violatedDirective);
      });
      window.addEventListener("error", (event) => observation.errors.push(event.message));
      window.addEventListener("unhandledrejection", () => observation.errors.push("Unhandled rejection"));
      return { userAgent: navigator.userAgent, origin: location.origin, devicePixelRatio: devicePixelRatio };
    });
    record("native-platform", { ...platform, capabilities: browser.capabilities });
    record("app-webview", { selected: await browser.getWindowHandle(), handles: await browser.getWindowHandles() });
    await $("#new-notebook-title").setValue(`Native viewer ${randomUUID()}`);
    await $("#create-notebook").click();
    await expect($("#notebook-select")).toHaveValue(/.+/);
  });

  it("paints distinct PDF pages, navigates, zooms, and releases the native worker", async () => {
    await importPdf(textPdf);
    await openPdf(textPdf.filename);
    const first = await waitForPage(firstPageColor);
    const inertBackground = await browser.execute(() => {
      document.querySelector<HTMLInputElement>("#new-notebook-title")!.focus();
      return document.querySelector("[data-testid='pdf-viewer']")!.contains(document.activeElement);
    });
    await expect(inertBackground).toBe(true);
    const nativeKeys = process.env.CORPI_E2E_PROVIDER !== "embedded";
    if (nativeKeys) {
      for (const name of ["Page number", "Next page", "Zoom out", "Zoom in", "Close PDF", "Page number"]) {
        await browser.keys("Tab");
        const focused = await browser.execute(() => ({
          insideViewer: document.querySelector("[data-testid='pdf-viewer']")!.contains(document.activeElement),
          name: document.activeElement?.getAttribute("aria-label") ?? document.activeElement?.textContent?.trim(),
        }));
        await expect(focused).toEqual({ insideViewer: true, name });
      }
    }
    record("modal-keyboard-focus", {
      backgroundFocusBlocked: inertBackground, trustedTabSteps: nativeKeys ? 6 : 0,
      keyboardCoverage: nativeKeys ? "native key input" : "embedded keys are synthetic; modal focus/inert/cancel API checked",
    });
    await expect(first.activeWorkers).toBe(1);
    await expect(matchesPage(first, secondPageColor)).toBe(false); // stale page control
    const blank = await browser.execute(() => {
      const canvas = document.createElement("canvas");
      const pixel = Array.from(canvas.getContext("2d")!.getImageData(0, 0, 1, 1).data);
      return [pixel, pixel, pixel];
    });
    await expect(matchesPage({ ...first, pixels: blank }, firstPageColor)).toBe(false); // allocated canvas control
    record("first-page", first);
    await browser.saveScreenshot(path.join(results, "text-page-one.png"));
    await $("[data-testid='pdf-viewer']").$("button=Next page").click();
    record("second-page", await waitForPage(secondPageColor, 2));
    await browser.saveScreenshot(path.join(results, "text-page-two.png"));
    await $("[data-testid='pdf-viewer']").$("button=Previous page").click();
    await waitForPage(firstPageColor);
    await $("[data-testid='pdf-viewer']").$("button=Zoom in").waitForEnabled();
    await $("[data-testid='pdf-viewer']").$("button=Zoom in").click();
    await expect($("[data-testid='pdf-viewer'] [aria-live='polite']")).toHaveText("110%");
    await waitForPage(firstPageColor);
    await closePdf();
  });

  it("renders a near-25-MB raster PDF under the 20-MP cap and switches saved files", async () => {
    await importPdf(rasterPdf);
    await openPdf(rasterPdf.filename);
    await waitForPage(rasterColor);
    for (let step = 0; step < 17; step += 1) {
      const before = (await viewerSnapshot()).zoom;
      await $("[data-testid='pdf-viewer']").$("button=Zoom in").waitForEnabled();
      await $("[data-testid='pdf-viewer']").$("button=Zoom in").click();
      await browser.waitUntil(async () => (await viewerSnapshot()).zoom > before,
        { timeout: 5000, timeoutMsg: "PDF viewer zoom did not advance" });
    }
    let snapshot: ViewerSnapshot;
    await browser.waitUntil(async () => {
      snapshot = await viewerSnapshot();
      const pixels = snapshot.width * snapshot.height;
      return snapshot.zoom >= 260 && pixels >= 19_000_000 && pixels <= 20_000_000
        && matchesPage(snapshot, rasterColor);
    }, { timeout: 30000, timeoutMsg: "The raster PDF did not finish painting a bounded near-20-MP canvas" });
    record("near-limit-raster", snapshot!);
    await browser.saveScreenshot(path.join(results, "near-limit-raster.png"));
    // Close and switch through the normal modal UI, then reopen the previous saved file.
    await closePdf();
    await openPdf(textPdf.filename);
    const switched = await waitForPage(firstPageColor);
    await expect(switched.activeWorkers).toBe(1);
    record("switched-to-text", switched);
    await closePdf();
    await openPdf(rasterPdf.filename);
    await waitForPage(rasterColor);
    await closePdf();
  });

  it("closes during a real fetch and render, releases work, and reopens first try", async () => {
    await browser.execute(() => {
      const observation = window.__corpiE2E;
      const nativeFetch = window.fetch;
      let release: (() => void) | undefined;
      observation.fetchHeld = false;
      observation.fetchAborted = false;
      window.fetch = async (input, init) => {
        const response = await nativeFetch(input, init);
        if (String(input).endsWith("/pdf")) {
          await new Promise<void>((resolve, reject) => {
            const abort = () => {
              observation.fetchAborted = true;
              reject(new DOMException("Viewer closed", "AbortError"));
            };
            release = () => { init?.signal?.removeEventListener("abort", abort); resolve(); };
            init?.signal?.addEventListener("abort", abort, { once: true });
            observation.fetchHeld = true;
            if (init?.signal?.aborted) abort();
          });
        }
        return response;
      };
      observation.releaseWork = () => { window.fetch = nativeFetch; release?.(); };
    });
    await openPdf(rasterPdf.filename);
    await browser.waitUntil(() => browser.execute(() => window.__corpiE2E.fetchHeld),
      { timeout: 15000, timeoutMsg: "The real PDF response never reached the controlled fetch boundary" });
    await expect($("[data-testid='pdf-viewer'] [role='status']")).toBeDisplayed();
    record("close-during-fetch", await viewerSnapshot());
    await closePdf();
    await expect(await browser.execute(() => window.__corpiE2E.fetchAborted)).toBe(true);
    await browser.execute(() => window.__corpiE2E.releaseWork?.());

    await openPdf(rasterPdf.filename);
    await waitForPage(rasterColor);
    await browser.execute(() => {
      const observation = window.__corpiE2E;
      const nativeFrame = window.requestAnimationFrame;
      const pending: Array<() => void> = [];
      observation.renderHeld = false;
      window.requestAnimationFrame = (callback) => nativeFrame((timestamp) => {
        if (document.querySelector(".pdf-viewer__page")?.getAttribute("aria-busy") === "true") {
          pending.push(() => callback(timestamp));
          observation.renderHeld = true;
        } else callback(timestamp);
      });
      observation.releaseWork = () => {
        window.requestAnimationFrame = nativeFrame;
        for (const callback of pending) callback();
      };
    });
    await $("[data-testid='pdf-viewer']").$("button=Zoom in").click();
    await browser.waitUntil(async () => {
      const held = await browser.execute(() => window.__corpiE2E.renderHeld);
      const snapshot = await viewerSnapshot();
      return held && snapshot.rendering && snapshot.width > 0 && snapshot.activeWorkers === 1;
    }, { timeout: 15000, timeoutMsg: "The real renderer never reached the controlled animation-frame boundary" });
    record("close-during-render", await viewerSnapshot());
    await closePdf();
    await browser.execute(() => window.__corpiE2E.releaseWork?.());
    await openPdf(rasterPdf.filename);
    record("reopened-after-cancel", await waitForPage(rasterColor));
    await closePdf(true);
    await expect(await browser.execute(() => window.__corpiE2E.errors)).toEqual([]);
  });

  it("shows a readable load failure while retaining the saved original", async () => {
    await importPdf(invalidPdf);
    await openPdf(invalidPdf.filename);
    await $("[data-testid='pdf-viewer'] [role='alert']").waitForDisplayed({ timeout: 30000 });
    await expect($("[data-testid='pdf-viewer'] [role='alert']")).toHaveText(/Couldn't open/);
    const layout = await browser.execute(() => {
      const viewer = document.querySelector("[data-testid='pdf-viewer']")!;
      const message = viewer.querySelector<HTMLElement>(".pdf-viewer__message")!;
      const page = viewer.querySelector<HTMLElement>(".pdf-viewer__page")!;
      const rect = message.getBoundingClientRect();
      return {
        pageHidden: page.hidden && page.getClientRects().length === 0,
        wraps: getComputedStyle(message).overflowWrap === "anywhere",
        boundedWidth: rect.width <= viewer.getBoundingClientRect().width,
        noHorizontalOverflow: message.scrollWidth <= message.clientWidth,
      };
    });
    await expect(layout).toEqual({ pageHidden: true, wraps: true, boundedWidth: true, noHorizontalOverflow: true });
    record("fatal-load-layout", layout);
    record("invalid-pdf-error", await viewerSnapshot());
    await browser.saveScreenshot(path.join(results, "invalid-pdf-error.png"));
    await closePdf();
    await expect($(".import-list")).toHaveText(expect.stringContaining(invalidPdf.filename));
    const security = await browser.execute(() => ({
      cspViolations: window.__corpiE2E.cspViolations,
      workerErrors: window.__corpiE2E.errors,
    }));
    await expect(security.cspViolations).toEqual([]);
    await expect(security.workerErrors).toEqual([]);
    record("worker-and-csp", security);
  });
});
