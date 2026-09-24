import { describe, expect, it, vi } from "vitest";

import { CoreConnection, coreHealth, coreHealthWithRetry, demoStream, extractEvents } from "./client";

describe("extractEvents", () => {
  it("returns complete events and keeps the partial tail", () => {
    const { events, buffer } = extractEvents('data: {"step":1}\n\ndata: {"st');
    expect(events).toEqual([{ step: 1 }]);
    expect(buffer).toBe('data: {"st');
  });

  it("returns no events while only a partial block has arrived", () => {
    const { events, buffer } = extractEvents('data: {"step":1}\n');
    expect(events).toEqual([]);
    expect(buffer).toBe('data: {"step":1}\n');
  });

  it("parses several complete blocks from one chunk", () => {
    const { events } = extractEvents(
      'data: {"step":1}\n\ndata: {"step":2}\n\ndata: {"done":true}\n\n',
    );
    expect(events).toEqual([{ step: 1 }, { step: 2 }, { done: true }]);
  });
});

const CONNECTION: CoreConnection = { base_url: "http://127.0.0.1:9", token: "launch-token" };

describe("coreHealth", () => {
  it("sends the bearer token in the Authorization header", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ app: "corpi-core", version: "0.1.0", status: "ok" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const health = await coreHealth(CONNECTION);

    expect(health.status).toBe("ok");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:9/health");
    expect(init.headers).toMatchObject({ Authorization: "Bearer launch-token" });
    vi.unstubAllGlobals();
  });

  it("surfaces non-2xx responses as errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 401 });
    vi.stubGlobal("fetch", fetchMock);
    await expect(coreHealth(CONNECTION)).rejects.toThrow("HTTP 401");
    vi.unstubAllGlobals();
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

    const health = await coreHealthWithRetry(CONNECTION, 3, 0);

    expect(health.status).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    vi.unstubAllGlobals();
  });

  it("surfaces the error after exhausting attempts", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(coreHealthWithRetry(CONNECTION, 3, 0)).rejects.toThrow("Failed to fetch");
    expect(fetchMock).toHaveBeenCalledTimes(3);
    vi.unstubAllGlobals();
  });
});

describe("demoStream", () => {
  it("feeds every event from the authenticated stream to the callback", async () => {
    const sse = 'data: {"step":1,"total":2}\n\ndata: {"step":2,"total":2}\n\ndata: {"done":true}\n\n';
    const reader = {
      read: vi
        .fn()
        .mockResolvedValueOnce({ done: false, value: new TextEncoder().encode(sse) })
        .mockResolvedValueOnce({ done: true, value: undefined }),
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: { getReader: () => reader },
    });
    vi.stubGlobal("fetch", fetchMock);

    const received: unknown[] = [];
    await demoStream(CONNECTION, (event) => received.push(event));

    expect(received).toEqual([{ step: 1, total: 2 }, { step: 2, total: 2 }, { done: true }]);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.headers).toMatchObject({ Authorization: "Bearer launch-token" });
    vi.unstubAllGlobals();
  });
});
