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
    };
  }
}

const results = process.env.CORPI_E2E_RESULTS ?? path.resolve("e2e-results");
const textPdf: PdfFixture = { filename: "native-text.pdf", kind: "text" };
const rasterPdf: PdfFixture = { filename: "native-raster.pdf", kind: "raster" };
const invalidPdf: PdfFixture = { filename: "native-invalid.pdf", kind: "invalid" };
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
}

async function closePdf() {
  await $("[data-testid='pdf-viewer']").$("button=Close PDF").click();
  await browser.waitUntil(async () => {
    const snapshot = await viewerSnapshot();
    return !snapshot.filename && snapshot.activeWorkers === 0 && snapshot.clearedCanvas;
  }, { timeout: 15000, timeoutMsg: "Closing the PDF did not release its worker and canvas" });
  record("closed", await viewerSnapshot());
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

  it("shows a renderer failure while retaining the saved original", async () => {
    await importPdf(invalidPdf);
    await openPdf(invalidPdf.filename);
    await $("[data-testid='pdf-viewer'] [role='alert']").waitForDisplayed({ timeout: 30000 });
    await expect($("[data-testid='pdf-viewer'] [role='alert']")).toHaveText(/Couldn't open/);
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
