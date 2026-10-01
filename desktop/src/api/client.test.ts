import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CoreConnection,
  coreBinary,
  coreHealth,
  coreHealthWithRetry,
  coreRequest,
  uploadPdf,
} from "./client";

const CONNECTION: CoreConnection = { base_url: "http://127.0.0.1:9", token: "launch-token" };

afterEach(() => vi.unstubAllGlobals());

describe("core requests", () => {
  it("sends the per-launch bearer token and won't let a caller replace it", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ app: "corpi-core", version: "0.1.0", status: "ok" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const health = await coreRequest<{ status: string }>(CONNECTION, "/health", {
      headers: { Authorization: "Bearer attacker-token" },
    });

    expect(health.status).toBe("ok");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:9/health");
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer launch-token");
  });

  it("preserves sanitized error codes from core responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({ detail: { code: "file_cleanup_pending", message: "Retry after cleanup." } }),
    }));

    await expect(coreHealth(CONNECTION)).rejects.toMatchObject({
      status: 409,
      code: "file_cleanup_pending",
      message: "Retry after cleanup.",
    });
  });
});

describe("coreHealthWithRetry", () => {
  it("retries a transient failure and succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ app: "corpi-core", version: "0.1.0", status: "ok" }),
      });
    vi.stubGlobal("fetch", fetchMock);

    expect((await coreHealthWithRetry(CONNECTION, 3, 0)).status).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("PDF binary operations", () => {
  it("sends bearer authentication on PDF upload and keeps the raw file body", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ state: "completed" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["%PDF-1.4"], "report.pdf", { type: "application/pdf" });

    await uploadPdf(CONNECTION, "acquisition-id", file);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:9/pdf-acquisitions/acquisition-id/content");
    expect(init.method).toBe("PUT");
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer launch-token");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/pdf");
    expect(init.body).toBe(file);
  });

  it("reads exact binary bytes without decoding them as JSON", async () => {
    const bytes = new Uint8Array([0, 37, 80, 68, 70]);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: async () => bytes.buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await coreBinary(CONNECTION, "/sources/source/imports/import/pdf");

    expect(Array.from(result)).toEqual([0, 37, 80, 68, 70]);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer launch-token");
  });
});
