import { $, browser, expect } from "@wdio/globals";
import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

interface Sample { kind: string; status: number; ms: number; atMs: number }
interface Measurement {
  filename: string;
  startedWallMs: number | null;
  elapsedMs: number;
  importReadyMs: number | null;
  searchReadyMs: number | null;
  readyMs: number | null;
  cancelledUiMs: number | null;
  completedChunks: number;
  totalChunks: number;
  sourceId: string | null;
  importId: string | null;
  status: string;
  searchStatus: string;
  frames: number;
  maxFrameGapMs: number;
  requests: Sample[];
  requestErrors: number;
  extraction: { outcome: string; pages: Array<{ page_number: number; classification: string; extracted_chars: number }> } | null;
}
declare global {
  interface Window {
    __readiness: { start: () => void; snapshot: () => Measurement; stop: () => Measurement };
    __readinessBytes: string[];
  }
}

const selected = process.env.CORPI_READINESS_CASE;
// Ordinary native smoke tests do not fetch fixtures or run this opt-in measurement.
if (selected) {
  const choice = JSON.parse(selected) as { pdf: string; profile: string; repetitions: number; cancel?: boolean };
  const definitions = JSON.parse(readFileSync(path.resolve("../core/benchmarks/search_readiness.json"), "utf8")) as {
    fixtures: Array<{ id: string; filename: string; sha256: string; bytes: number; pages: number;
      review_exclusions?: Record<string, number[]> }>;
  };
  const fixture = definitions.fixtures.find((item) => item.id === choice.pdf);
  const directory = process.env.CORPI_READINESS_PDFS;
  if (!fixture || !directory || !path.isAbsolute(directory)
    || !["bekko-a8m-quint8-avx2", "minilm-en", "multilingual-e5"].includes(choice.profile)
    || !Number.isInteger(choice.repetitions) || choice.repetitions < 1 || choice.repetitions > 3) {
    throw new Error("Invalid explicit readiness benchmark selection");
  }
  const bytes = readFileSync(path.join(directory, fixture.filename));
  if (bytes.length !== fixture.bytes || createHash("sha256").update(bytes).digest("hex") !== fixture.sha256) {
    throw new Error("Readiness PDF does not match the original pinned fixture");
  }
  const results = process.env.CORPI_E2E_RESULTS!;
  mkdirSync(results, { recursive: true });
  const completed: Array<Record<string, unknown>> = [];
  const save = () => writeFileSync(path.join(results, "readiness.json"), `${JSON.stringify(completed, null, 2)}\n`);

  async function freshNotebook() {
    const title = `Readiness ${choice.profile} ${randomUUID()}`;
    await $("#new-notebook-title").setValue(title);
    await $("#create-notebook").click();
    await browser.waitUntil(() => browser.execute((expected) => {
      const select = document.querySelector<HTMLSelectElement>("#notebook-select");
      return Boolean(select?.value && select.selectedOptions[0]?.textContent === expected
        && document.querySelectorAll(".source-list li,.import-list li").length === 0);
    }, title), { timeout: 15000, timeoutMsg: "Fresh benchmark notebook was not selected" });
    const settings = $("[aria-label='Local search models']");
    await $("#search-model").waitForDisplayed();
    const modelName = await browser.execute((profile) => {
      const option = document.querySelector<HTMLOptionElement>(`#search-model option[value="${profile}"]`);
      if (!option?.textContent) throw new Error("Requested benchmark model is not offered");
      return option.textContent;
    }, choice.profile);
    await browser.waitUntil(() => browser.execute((name) => {
      const rows = document.querySelectorAll("[aria-label='Supported search models'] li");
      return Array.from(rows).some((row) => row.querySelector("strong")?.textContent === name
        && row.querySelector("[role='status']")?.textContent === "installed");
    }, modelName), { timeout: 15000, timeoutMsg: "Requested benchmark model must already be installed; never downloads" });
    if (process.env.CORPI_E2E_PROVIDER === "embedded") {
      await browser.execute((profile) => {
        const select = document.querySelector<HTMLSelectElement>("#search-model")!;
        select.value = profile;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }, choice.profile);
    } else {
      await $("#search-model").selectByAttribute("value", choice.profile);
    }
    await browser.waitUntil(async () => (await $("#search-model").getValue()) === choice.profile
      && (await settings.getAttribute("data-active-profile")) === choice.profile, {
      timeout: 15000, timeoutMsg: "Requested installed model did not become active on empty notebook",
    });
    return String(await $("#notebook-select").getValue());
  }

  async function installMeter(filename: string) {
    await browser.execute((expected) => {
      const data: Measurement = { filename: expected, startedWallMs: null, elapsedMs: 0,
        importReadyMs: null, searchReadyMs: null, readyMs: null, cancelledUiMs: null,
        completedChunks: 0, totalChunks: 0, sourceId: null, importId: null,
        status: "", searchStatus: "", frames: 0, maxFrameGapMs: 0, requests: [], requestErrors: 0, extraction: null };
      let started: number | null = null;
      let stopped = false;
      let lastFrame: number | null = null;
      let frameId = 0;
      const original = window.fetch;
      const update = () => {
        if (started === null || stopped) return;
        data.elapsedMs = performance.now() - started;
        const item = Array.from(document.querySelectorAll(".import-list li"))
          .find((row) => row.querySelector("strong")?.textContent === expected);
        data.status = item?.querySelector("small")?.textContent ?? "";
        data.searchStatus = document.querySelector("[aria-label='Search preparation']")?.textContent ?? "";
        const progress = document.querySelector<HTMLProgressElement>("[aria-label='Text pieces embedded']");
        data.completedChunks = progress?.value ?? 0;
        data.totalChunks = progress?.max ?? 0;
        if (data.cancelledUiMs === null && data.searchStatus.includes("cancelled")
          && data.status.includes("processing failed")) data.cancelledUiMs = data.elapsedMs;
        if (item) data.importId = item.getAttribute("data-testid")?.replace("pdf-import-", "") ?? null;
        if (data.importReadyMs === null && item?.getAttribute("data-processing-state") === "ready"
          && item.getAttribute("data-current") === "true") data.importReadyMs = data.elapsedMs;
        if (data.searchReadyMs === null && item && data.searchStatus.includes("Search data ready")) data.searchReadyMs = data.elapsedMs;
        if (data.readyMs === null && data.importReadyMs !== null && data.searchReadyMs !== null) data.readyMs = data.elapsedMs;
      };
      const observer = new MutationObserver(update);
      observer.observe(document.querySelector("#root")!, { subtree: true, childList: true, characterData: true });
      const frame = (now: number) => {
        if (stopped) return;
        if (started !== null) {
          data.frames += 1;
          if (lastFrame !== null) data.maxFrameGapMs = Math.max(data.maxFrameGapMs, now - lastFrame);
          lastFrame = now;
        }
        frameId = requestAnimationFrame(frame);
      };
      frameId = requestAnimationFrame(frame);
      window.fetch = async (input, init) => {
        const began = performance.now();
        const relativeStart = started;
        try {
          const response = await original.call(window, input, init);
          const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, location.href);
          if (!stopped && relativeStart !== null && url.hostname === "127.0.0.1") {
            const kind = url.pathname.endsWith("/search-progress") ? "search-progress"
              : url.pathname.endsWith("/extraction") ? "extraction"
              : url.pathname.endsWith("/imports") ? "imports" : "other-core";
            if (data.requests.length < 3000) data.requests.push({ kind, status: response.status,
              ms: performance.now() - began, atMs: began - relativeStart });
            // Observe public result identities only. Never inspect or serialize headers/tokens.
            if (url.pathname.endsWith("/content") || kind === "extraction") {
              void response.clone().json().then((payload) => {
                if (stopped) return;
                if (typeof payload.source_id === "string") data.sourceId = payload.source_id;
                if (typeof payload.import_id === "string") data.importId = payload.import_id;
                if (kind === "extraction" && payload.outcome) data.extraction = {
                  outcome: payload.outcome, pages: payload.pages.map((page: {
                    page_number: number; classification: string; extracted_chars: number;
                  }) => ({ page_number: page.page_number, classification: page.classification, extracted_chars: page.extracted_chars })),
                };
              }).catch(() => { data.requestErrors += 1; });
            }
          }
          return response;
        } catch (cause) { data.requestErrors += 1; throw cause; }
      };
      const snapshot = () => { update(); return { ...data, requests: [...data.requests] }; };
      window.__readiness = {
        start: () => { started = performance.now(); data.startedWallMs = Date.now(); }, snapshot,
        stop: () => { update(); stopped = true; observer.disconnect(); cancelAnimationFrame(frameId); window.fetch = original; return snapshot(); },
      };
      window.__readinessBytes = [];
    }, filename);
  }

  async function sendOriginal(filename: string) {
    const injectionStarted = Date.now();
    const encoded = bytes.toString("base64");
    for (let offset = 0; offset < encoded.length; offset += 262144) {
      await browser.execute((piece) => { window.__readinessBytes.push(piece); }, encoded.slice(offset, offset + 262144));
    }
    const sent = await browser.execute((name) => {
      const decoded = atob(window.__readinessBytes.join(""));
      window.__readinessBytes = [];
      const content = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
      const transfer = new DataTransfer();
      transfer.items.add(new File([content], name, { type: "application/pdf" }));
      const input = document.querySelector<HTMLInputElement>("[data-testid='pdf-file-input']")!;
      input.files = transfer.files;
      window.__readiness.start();
      input.dispatchEvent(new Event("change", { bubbles: true }));
      return { bytes: content.byteLength, startedWallMs: window.__readiness.snapshot().startedWallMs! };
    }, filename);
    await expect(sent.bytes).toBe(bytes.length);
    return sent.startedWallMs - injectionStarted;
  }

  describe(`Original PDF readiness: ${choice.pdf}/${choice.profile}`, () => {
    before(async () => {
      await $("#core-status").waitForDisplayed({ timeout: 60000 });
      await expect($("#core-status")).toHaveText(/Core ready/);
      let port = 0;
      await browser.waitUntil(async () => {
        port = await browser.execute(() => {
          const entry = performance.getEntriesByType("resource").find((item) => item.name.startsWith("http://127.0.0.1:"));
          return entry ? Number(new URL(entry.name).port) : 0;
        });
        return port > 0;
      }, { timeout: 15000, timeoutMsg: "No core API resource request completed" });
      if (!Number.isInteger(port) || port <= 0 || port > 65535) throw new Error("No owned core listener was observed");
      const corePid = Number(execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
        `Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction Stop | Select-Object -First 1 -ExpandProperty OwningProcess`],
      { encoding: "utf8", timeout: 20000 }).trim());
      if (!Number.isInteger(corePid) || corePid <= 0) throw new Error("Core listener owner was not identified");
      writeFileSync(path.join(results, "core-owner.json"), JSON.stringify({ pid: corePid, port }));
    });

    for (let repetition = 1; repetition <= choice.repetitions; repetition += 1) it(`imports fresh original PDF, repetition ${repetition}`, async () => {
      const notebookId = await freshNotebook();
      const cancellation = Boolean(choice.cancel && repetition === 1);
      const filename = `${fixture.id}-${choice.profile}-${randomUUID()}.pdf`;
      await installMeter(filename);
      const injectionMs = await sendOriginal(filename);
      let result: Measurement;
      let cancelledAtMs: number | null = null;
      let reviewVisibleMs: number | null = null;
      let confirmedAtMs: number | null = null;
      const confirmReviewedPartial = async (current: Measurement) => {
        if (current.extraction?.outcome !== "partial" || confirmedAtMs !== null) return;
        reviewVisibleMs = current.elapsedMs;
        const exclusions: Record<string, number[]> = {};
        for (const page of current.extraction.pages) if (page.classification !== "extracted") {
          (exclusions[page.classification] ??= []).push(page.page_number);
        }
        await expect(exclusions).toEqual(fixture.review_exclusions);
        await expect(current.extraction.pages.length).toBe(fixture.pages);
        confirmedAtMs = await browser.execute(() => {
          const button = Array.from(document.querySelectorAll<HTMLButtonElement>(".extraction-report button"))
            .find((item) => item.textContent?.trim() === "Confirm");
          if (!button) throw new Error("The reviewed partial extraction has no Confirm control");
          const elapsed = window.__readiness.snapshot().elapsedMs;
          button.click();
          return elapsed;
        });
      };
      try {
        if (cancellation) {
          await browser.waitUntil(async () => {
            const current = await browser.execute(() => window.__readiness.snapshot());
            await confirmReviewedPartial(current);
            return current.searchStatus.includes("Preparing meaning-based search") && current.completedChunks > 0;
          }, {
            timeout: 2600000, timeoutMsg: "Cancellation control never reached actual embedding",
          });
          cancelledAtMs = await browser.execute(() => {
            const button = Array.from(document.querySelectorAll<HTMLButtonElement>(".import-list button"))
              .find((item) => item.textContent?.trim() === "Cancel processing");
            if (!button) throw new Error("Actual import cancellation control is missing");
            const elapsed = window.__readiness.snapshot().elapsedMs;
            button.click();
            return elapsed;
          });
        }
        await browser.waitUntil(async () => {
          result = await browser.execute(() => window.__readiness.snapshot());
          await confirmReviewedPartial(result);
          if (result.sourceId) writeFileSync(path.join(results, "live-progress.json"), JSON.stringify({
            sourceId: result.sourceId, startedWallMs: result.startedWallMs, profile: choice.profile,
          }));
          if (cancellation) return result.searchStatus.includes("cancelled") && result.status.includes("processing failed");
            return result.readyMs !== null || result.status.includes("processing failed")
              || result.searchStatus.includes("cancelled") || result.searchStatus.includes("Search preparation failed");
        }, { timeout: 2600000, interval: 500, timeoutMsg: "Original PDF did not reach a terminal UI state" });
      } finally {
        result = await browser.execute(() => window.__readiness.stop());
        const record = { fixture: fixture.id, sha256: fixture.sha256, pages: fixture.pages,
          profile: choice.profile, notebookId, repetition, cancellation, cancelledAtMs,
          reviewVisibleMs, confirmedAtMs,
          followsCancellation: Boolean(choice.cancel && repetition > 1),
          inputMode: "synthetic File change, original PDF bytes; not trusted physical file selection",
          fixtureInjectionExcludedMs: injectionMs, ...result };
        completed.push(record); save();
        await browser.saveScreenshot(path.join(results, `readiness-${repetition}.png`));
      }
      if (cancellation) {
        await expect(result!.readyMs).toBeNull();
        await expect(result!.searchStatus).toContain("cancelled");
      } else {
        await expect(result!.readyMs).not.toBeNull();
        await expect(result!.extraction?.pages.length).toBe(fixture.pages);
      }
    });
  });
}
