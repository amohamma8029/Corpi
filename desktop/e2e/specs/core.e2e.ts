import { $, browser, expect } from "@wdio/globals";

describe("Corpi desktop shell", () => {
  it("launches the shell window", async () => {
    await expect(browser).toHaveTitle(/Corpi/);
  });

  it("reports the local core as ready", async () => {
    await expect($("#core-status")).toHaveText(/Core ready/);
  });

  it("streams demo events through the authenticated core", async () => {
    await $("#run-stream").click();
    await expect($("#stream-events")).toHaveChildren({ gte: 5 });
  });
});
