import { invoke } from "@tauri-apps/api/core";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import {
  coreHealthWithRetry,
  createNotebook,
  listNotebooks,
  listNotebookSources,
  storageReport,
} from "./api/client";
import { App } from "./App";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("./api/client", async (loadActual) => {
  const actual = await loadActual<typeof import("./api/client")>();
  return {
    ...actual,
    coreHealthWithRetry: vi.fn(),
    listNotebooks: vi.fn(),
    storageReport: vi.fn(),
    createNotebook: vi.fn(),
    listNotebookSources: vi.fn(),
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("keeps startup responsive while the shell is still connecting", async () => {
  let rejectConnection!: (error: Error) => void;
  vi.mocked(invoke).mockImplementation(
    () => new Promise<never>((_, reject) => {
      rejectConnection = reject;
    }),
  );

  render(<App />);
  expect(screen.getByText("Connecting to core…")).toBeTruthy();

  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  expect(screen.getByText("Connecting to core…")).toBeTruthy();

  await act(async () => {
    rejectConnection(new Error("core persistence startup failed: database remained busy"));
  });
  expect(screen.getByText("Core unavailable")).toBeTruthy();
  expect(screen.getByRole("alert").textContent).toContain("database remained busy");
});

it("creates a notebook through the authenticated core client", async () => {
  const connection = { base_url: "http://127.0.0.1:4567", token: "launch-token" };
  const notebook = {
    id: "0f5098ac-4a1b-47a5-9c20-c2ab1dbfcb3c",
    title: "Field notes",
    created_at: "2026-09-30T00:00:00.000000Z",
    updated_at: "2026-09-30T00:00:00.000000Z",
  };
  vi.mocked(invoke).mockResolvedValue(connection);
  vi.mocked(coreHealthWithRetry).mockResolvedValue({
    app: "corpi-core",
    version: "0.1.0",
    status: "ok",
  });
  vi.mocked(listNotebooks).mockResolvedValue([]);
  vi.mocked(listNotebookSources).mockResolvedValue([]);
  vi.mocked(storageReport).mockResolvedValue({
    original_file_count: 0,
    original_bytes: 0,
    bytes_pending_removal: 0,
    retained_staging_bytes: 0,
    max_upload_bytes: 25_000_000,
    rendered_page_pixel_limit: 20_000_000,
  });
  vi.mocked(createNotebook).mockResolvedValue(notebook);

  render(<App />);
  const title = await screen.findByLabelText("New notebook");
  fireEvent.change(title, { target: { value: "Field notes" } });
  fireEvent.click(screen.getByRole("button", { name: "Create notebook" }));

  await waitFor(() => expect(createNotebook).toHaveBeenCalledWith(connection, "Field notes"));
});
