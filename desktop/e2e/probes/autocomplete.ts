// One-off native evidence for PR #19. A screenshot is evidence for review,
// not an assertion that a platform does or does not offer autocomplete.
import { $, browser } from "@wdio/globals";
import { spawnSync } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const title = "Autocomplete CI probe";
const field = "#notebook-search-query";
const phase = process.env.CORPI_AUTOCOMPLETE_PHASE;
const output = process.env.CORPI_AUTOCOMPLETE_RESULTS;
const cases = [
  { label: "browser-default", autocomplete: null },
  { label: "autocomplete-off", autocomplete: "off" },
] as const;
const query = "cobalt fern";

function command(executable: string, args: string[]): string {
  const result = spawnSync(executable, args, { encoding: "utf8", timeout: 10_000 });
  if (result.error || result.status !== 0) {
    throw Error(`${executable}: ${result.error?.message ?? result.stderr?.trim() ?? result.signal ?? result.status}`);
  }
  return result.stdout.trim();
}

function appPid(): number {
  const lines = command("ps", ["-axo", "pid=,comm="]).split("\n");
  const matches = lines.flatMap((line) => {
    const match = line.match(/^\s*(\d+)\s+(.+)$/);
    return match && path.basename(match[2]) === "corpi-desktop" ? [Number(match[1])] : [];
  });
  if (matches.length !== 1) throw Error(`Expected one native app process; found ${matches.length}`);
  return matches[0];
}

async function type(character: string, pid: number): Promise<void> {
  if (process.platform === "darwin") {
    // The embedded driver synthesizes untrusted DOM keyboard events on macOS.
    // Use native keys without granting permissions or changing system settings.
    command("osascript", ["-e", `tell application "System Events"
      if (unix id of (first application process whose frontmost is true)) is not ${pid} then error "Corpi lost focus"
      keystroke "${character}"
    end tell`]);
  } else {
    await browser.keys(character);
  }
}

describe("Notebook autocomplete observation in the native WebView", () => {
  it(`records the ${phase ?? "unconfigured"} process`, async () => {
    if (!output || !path.isAbsolute(output) || !["prime", "observe"].includes(phase ?? "")) {
      throw Error("Missing dedicated probe results path or phase.");
    }
    mkdirSync(output, { recursive: true });
    const log = (data: object) =>
      appendFileSync(path.join(output, "observations.jsonl"), `${JSON.stringify({ phase, ...data })}\n`);
    await browser.waitUntil(async () => {
      for (const handle of await browser.getWindowHandles()) {
        await browser.switchToWindow(handle);
        if (await $("#core-status").isExisting() && (await $("#core-status").getText()).includes("Core ready")) return true;
      }
      return false;
    }, { timeout: 30_000 });
    const pid = appPid();
    log({ pid, platform: process.platform });
    if (phase === "prime") {
      await $("#new-notebook-title").setValue(title);
      await $("#create-notebook").click();
    } else {
      await $("#notebook-select").selectByVisibleText(title);
    }
    await $(field).waitForDisplayed();
    await browser.waitUntil(() => browser.execute((name) => {
      const select = document.querySelector<HTMLSelectElement>("#notebook-select");
      return select?.selectedOptions[0]?.textContent === name;
    }, title), { timeout: 15_000 });
    for (const variant of cases) {
      const input = $(field);
      await input.clearValue();
      await input.click();
      await browser.execute((selector, setting) => {
        const node = document.querySelector<HTMLInputElement>(selector);
        if (!node || document.activeElement !== node) throw Error("Query field is not focused.");
        if (node.value !== "") throw Error("Query field did not clear between observations.");
        if (node.getAttribute("autocomplete") !== "off") throw Error("Expected the candidate's field attribute.");
        if (setting === null) node.removeAttribute("autocomplete");
        const events: boolean[] = [];
        node.addEventListener("input", (event) => events.push(event.isTrusted));
        (node as HTMLInputElement & { probeEvents: boolean[] }).probeEvents = events;
      }, field, variant.autocomplete);
      try {
        for (const [index, character] of [...query].entries()) {
          await type(character, pid);
          const readInput = () => browser.execute((selector) => {
            const node = document.querySelector(selector) as HTMLInputElement & { probeEvents: boolean[] };
            return {
              value: node.value, eventCount: node.probeEvents.length,
              trusted: node.probeEvents.at(-1), autocomplete: node.getAttribute("autocomplete"),
            };
          }, field);
          // AppleScript returns after sending the key, not after WebKit has
          // delivered it. Do not send another key or accept the previous event.
          await browser.waitUntil(async () => (await readInput()).eventCount >= index + 1, {
            timeout: 2_000, interval: 40, timeoutMsg: `No new input event at ${variant.label} character ${index + 1}`,
          });
          const state = await readInput();
          log({ variant: variant.label, index, ...state });
          if (state.eventCount !== index + 1 || !state.trusted ||
              state.value !== query.slice(0, index + 1) || state.autocomplete !== variant.autocomplete) {
            throw Error("Native keystroke or requested autocomplete setting could not be confirmed.");
          }
          if (phase === "observe" && index < query.length - 1) {
            await delay(140);
            const name = path.join(output, `${variant.label}-${String(index + 1).padStart(2, "0")}.png`);
            if (process.platform === "darwin") command("screencapture", ["-x", "-t", "png", name]);
            else command("python3", ["-I", "e2e/probes/x11-screenshot.py", name]);
          }
        }
        await $("button=Search").click();
        await browser.waitUntil(async () => (await $(".notebook-search").getText()).includes("No sources are ready yet"), { timeout: 15_000 });
      } finally {
        await browser.execute((selector) => document.querySelector(selector)?.setAttribute("autocomplete", "off"), field);
      }
    }
  });
});
