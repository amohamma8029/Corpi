import { afterEach, beforeEach, expect, it, vi } from "vitest";

const commands = vi.hoisted(() => ({ execute: vi.fn(), waitUntil: vi.fn(), find: vi.fn() }));
vi.mock("@wdio/globals", () => ({ browser: commands, $: commands.find }));

import { waitForSavedImport } from "../e2e/import-history";

type Read = (filename: string) => boolean;
type Probe = () => Promise<boolean>;

function history(filename: string): HTMLUListElement {
  const list = document.createElement("ul");
  list.className = "import-list";
  list.textContent = filename;
  return list;
}

beforeEach(() => {
  vi.resetAllMocks();
  commands.execute.mockImplementation(async (read: Read, filename: string) => read(filename));
  commands.find.mockImplementation((selector: string) => {
    const element = document.querySelector(selector);
    return { getText: async () => element?.textContent ?? "" };
  });
});
afterEach(() => { document.body.replaceChildren(); });

it("waits through an absent history and the previous source without extending the wait", async () => {
  commands.waitUntil.mockImplementation(async (probe: Probe, options: unknown) => {
    expect(options).toEqual({
      timeout: 15000, timeoutMsg: "native-text.pdf was not listed in saved import history",
    });
    expect(await probe()).toBe(false);
    document.body.replaceChildren(history("native-raster.pdf"));
    expect(await probe()).toBe(false);
    document.body.replaceChildren(history("native-text.pdf"));
    expect(await probe()).toBe(true);
  });
  await waitForSavedImport("native-text.pdf");
});

it("reads the current list when source selection replaces a previously found list", async () => {
  const previous = history("native-raster.pdf");
  document.body.append(previous);
  const remount = () => { document.body.replaceChildren(history("native-text.pdf")); };
  commands.execute.mockImplementation(async (read: Read, filename: string) => {
    remount();
    return read(filename);
  });
  commands.find.mockImplementation((selector: string) => {
    const element = document.querySelector(selector);
    queueMicrotask(remount);
    return { getText: async () => {
      await Promise.resolve();
      // The pinned embedded provider rejects this exact detached-element state.
      if (!element?.isConnected) throw new Error("stale element reference");
      return element.textContent ?? "";
    } };
  });
  commands.waitUntil.mockImplementation(async (probe: Probe) => { expect(await probe()).toBe(true); });
  await waitForSavedImport("native-text.pdf");
  expect(previous.isConnected).toBe(false);
});

it("passes filenames as data rather than building a text selector or script", async () => {
  const filename = "native-'[quoted]\\-\".pdf";
  document.body.append(history(filename));
  commands.waitUntil.mockImplementation(async (probe: Probe) => { expect(await probe()).toBe(true); });
  await waitForSavedImport(filename);
  expect(document.querySelector(".import-list")?.textContent).toBe(filename);
  expect(commands.waitUntil).toHaveBeenCalledOnce();
});
