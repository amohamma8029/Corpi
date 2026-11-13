import { browser } from "@wdio/globals";

export async function waitForSavedImport(filename: string): Promise<void> {
  // A source switch clears and remounts this list. Read it in one page operation,
  // rather than retaining a handle between separate find-element/get-text calls.
  await browser.waitUntil(() => browser.execute((expected) =>
    document.querySelector(".import-list")?.textContent?.includes(expected) ?? false,
  filename), {
    timeout: 15000,
    timeoutMsg: `${filename} was not listed in saved import history`,
  });
}
