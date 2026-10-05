import { invoke } from "@tauri-apps/api/core";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import {
  coreHealthWithRetry,
  createNotebook,
  cancelPdfAcquisition,
  createPdfAcquisition,
  decideExtractionReview,
  getExtraction,
  getPdfAcquisition,
  listNotebooks,
  listNotebookSources,
  listPdfAcquisitions,
  listPdfImports,
  reopenExtractionReview,
  retryPdfAcquisition,
  retryPdfProcessing,
  uploadPdf,
  type ExtractionReport,
  type PdfAcquisition,
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
    listPdfAcquisitions: vi.fn(),
    listPdfImports: vi.fn(),
    retryPdfAcquisition: vi.fn(),
    retryPdfProcessing: vi.fn(),
    cancelPdfAcquisition: vi.fn(),
    getPdfAcquisition: vi.fn(),
    createPdfAcquisition: vi.fn(),
    uploadPdf: vi.fn(),
    getExtraction: vi.fn(),
    decideExtractionReview: vi.fn(),
    reopenExtractionReview: vi.fn(),
  };
});

const connection = { base_url: "http://127.0.0.1:4567", token: "launch-token" };
const notebook = {
  id: "0f5098ac-4a1b-47a5-9c20-c2ab1dbfcb3c", title: "Field notes",
  created_at: "2026-09-30T00:00:00.000000Z", updated_at: "2026-09-30T00:00:00.000000Z",
};
const savedAttempt: PdfAcquisition = {
  id: "attempt-1", retry_key: "key-1", notebook_id: notebook.id, intent: "add",
  target_source_id: null, filename: "saved.pdf", state: "waiting_cleanup", generation: 2,
  received_size_bytes: 200, expected_size_bytes: 200, input_sha256: "a".repeat(64),
  planned_managed_file_id: null, outcome_kind: null, source_id: null, import_id: null,
  failure_code: "file_cleanup_pending", updated_at: notebook.updated_at,
  can_retry: true, can_cancel: true, requires_upload: false,
};

const source = {
  id: "source-1", notebook_id: notebook.id, display_name: "Field report",
  created_at: notebook.created_at, updated_at: notebook.updated_at,
  current_import_id: "import-1", pending_import_id: null,
};

const emptyExtraction: ExtractionReport = {
  source_id: source.id,
  extraction_id: null,
  outcome: null,
  failure_code: null,
  failure_detail: null,
  parser_name: null,
  parser_version: null,
  normalization_version: null,
  passage_count: 0,
  pages: [],
  review: null,
};

const partialExtraction: ExtractionReport = {
  source_id: source.id,
  extraction_id: "extraction-1",
  outcome: "partial",
  failure_code: null,
  failure_detail: null,
  parser_name: "pdfminer.six",
  parser_version: "20260107",
  normalization_version: "faithfully-normalized-text-v1",
  passage_count: 1,
  pages: [
    { page_number: 1, classification: "extracted", text_drawn: true, extracted_chars: 120, unmapped_ratio: 0, image_area_ratio: 0 },
    { page_number: 2, classification: "image-only", text_drawn: true, extracted_chars: 1, unmapped_ratio: 0, image_area_ratio: 0.62 },
  ],
  review: { state: "pending", generation: 1, decided_at: null },
};

const sourceB = { ...source, id: "source-2", display_name: "Second source" };

const failedImport = {
  id: "import-1", source_id: source.id, source_version_id: "version-1",
  filename: "failed.pdf", state: "failed", failure_code: null,
  size_bytes: 10, sha256: "a".repeat(64), created_at: notebook.created_at,
  trashed: false, is_current: true, is_pending: false,
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(invoke).mockResolvedValue(connection);
  vi.mocked(coreHealthWithRetry).mockResolvedValue({ app: "corpi-core", version: "0.1.0", status: "ok" });
  vi.mocked(listNotebooks).mockResolvedValue([notebook]);
  vi.mocked(listNotebookSources).mockResolvedValue([]);
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  vi.mocked(listPdfImports).mockResolvedValue([]);
  vi.mocked(getExtraction).mockResolvedValue(emptyExtraction);
  vi.mocked(storageReport).mockResolvedValue({
    original_file_count: 0, original_bytes: 0, bytes_pending_removal: 0, retained_staging_bytes: 200,
    max_upload_bytes: 25_000_000, rendered_page_pixel_limit: 20_000_000,
  });
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

it("restores complete and partial attempts with the correct recovery controls", async () => {
  vi.mocked(listPdfAcquisitions).mockResolvedValue([
    savedAttempt,
    { ...savedAttempt, id: "partial", filename: "partial.pdf", state: "failed", input_sha256: null,
      can_retry: false, requires_upload: true, failure_code: "upload_interrupted" },
  ]);
  render(<App />);
  const complete = await screen.findByRole("region", { name: "PDF import status: saved.pdf" });
  expect(within(complete).getByRole("button", { name: "Retry from saved PDF" })).toBeTruthy();
  const partial = screen.getByRole("region", { name: "PDF import status: partial.pdf" });
  expect(within(partial).getByRole("button", { name: "Select PDF again" })).toBeTruthy();
  expect(within(partial).queryByRole("button", { name: "Retry from saved PDF" })).toBeNull();
  expect(partial.textContent).toContain("no complete file was saved");

  vi.mocked(cancelPdfAcquisition).mockResolvedValue({ ...savedAttempt, state: "cancelled", can_retry: false, can_cancel: false });
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  fireEvent.click(within(complete).getByRole("button", { name: "Cancel import" }));
  await waitFor(() => expect(screen.queryByRole("region", { name: "PDF import status: saved.pdf" })).toBeNull());
  expect(cancelPdfAcquisition).toHaveBeenCalledWith(connection, savedAttempt.id);
});

it("retries a restored attempt and removes its status after a committed result", async () => {
  vi.mocked(listPdfAcquisitions).mockResolvedValue([savedAttempt]);
  vi.mocked(retryPdfAcquisition).mockResolvedValue({ ...savedAttempt, state: "completed", can_retry: false, can_cancel: false });
  render(<App />);
  const retry = await screen.findByRole("button", { name: "Retry from saved PDF" });
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  fireEvent.click(retry);
  await waitFor(() => expect(screen.queryByRole("button", { name: "Retry from saved PDF" })).toBeNull());
  expect(retryPdfAcquisition).toHaveBeenCalledWith(connection, savedAttempt.id);
});

it("ignores a previous notebook's discovery reply after switching notebooks", async () => {
  const other = { ...notebook, id: "other-notebook", title: "Other notebook" };
  vi.mocked(listNotebooks).mockResolvedValue([notebook, other]);
  let resolveFirst!: (items: PdfAcquisition[]) => void;
  vi.mocked(listPdfAcquisitions).mockImplementation((_connection, id) => id === notebook.id
    ? new Promise((resolve) => { resolveFirst = resolve; })
    : Promise.resolve([{ ...savedAttempt, id: "other-attempt", notebook_id: other.id, filename: "other.pdf" }]));
  render(<App />);
  await waitFor(() => expect(listPdfAcquisitions).toHaveBeenCalledWith(connection, notebook.id));
  fireEvent.change(screen.getByLabelText("Open notebook"), { target: { value: other.id } });
  await screen.findByRole("region", { name: "PDF import status: other.pdf" });
  await act(async () => resolveFirst([savedAttempt]));
  expect(screen.queryByRole("region", { name: "PDF import status: saved.pdf" })).toBeNull();
  expect(screen.getByRole("region", { name: "PDF import status: other.pdf" })).toBeTruthy();
});

it("uses fresh persisted state when reopening a notebook after another recovery completed", async () => {
  const other = { ...notebook, id: "other-notebook", title: "Other notebook" };
  vi.mocked(listNotebooks).mockResolvedValue([notebook, other]);
  vi.mocked(listPdfAcquisitions).mockResolvedValue([savedAttempt]);
  vi.mocked(retryPdfAcquisition).mockRejectedValue(new Error("Still waiting for cleanup"));
  vi.mocked(getPdfAcquisition).mockResolvedValue(savedAttempt);
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Retry from saved PDF" }));
  await screen.findByRole("alert");
  await waitFor(() => expect((screen.getByLabelText("Open notebook") as HTMLSelectElement).disabled).toBe(false));
  // Maintenance completed the attempt after the failed retry; discovery is now empty.
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  fireEvent.change(screen.getByLabelText("Open notebook"), { target: { value: other.id } });
  await waitFor(() => expect(listPdfAcquisitions).toHaveBeenCalledWith(connection, other.id));
  fireEvent.change(screen.getByLabelText("Open notebook"), { target: { value: notebook.id } });
  await waitFor(() => expect(listPdfAcquisitions).toHaveBeenCalledTimes(3));
  expect(screen.queryByRole("button", { name: "Retry from saved PDF" })).toBeNull();
});

it("reselects partial input for the existing attempt instead of creating a second action", async () => {
  const partial = { ...savedAttempt, id: "partial", filename: "partial.pdf", state: "failed",
    input_sha256: null, can_retry: false, requires_upload: true, failure_code: "upload_interrupted" };
  vi.mocked(listPdfAcquisitions).mockResolvedValue([partial]);
  vi.mocked(getPdfAcquisition).mockResolvedValue(partial);
  vi.mocked(uploadPdf).mockResolvedValue({ ...partial, state: "completed", requires_upload: false, can_cancel: false });
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Select PDF again" }));
  const file = new File(["%PDF-1.4"], "partial.pdf", { type: "application/pdf" });
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  fireEvent.change(document.querySelector("[data-testid='pdf-file-input']")!, { target: { files: [file] } });
  await waitFor(() => expect(uploadPdf).toHaveBeenCalledWith(connection, partial.id, file, expect.any(AbortSignal)));
  expect(createPdfAcquisition).not.toHaveBeenCalled();
  await waitFor(() => expect(screen.queryByRole("button", { name: "Select PDF again" })).toBeNull());
});

it("shows the page report with Confirm and Decline for a partial extraction", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(getExtraction).mockResolvedValue(partialExtraction);
  render(<App />);
  const report = await screen.findByRole("region", { name: "Text extraction" });
  expect(report.textContent).toContain("Outcome: partial");
  expect(report.textContent).toContain("Page 2 — image-only");
  expect(report.textContent).toContain("62% images");
  expect(within(report).getByRole("button", { name: "Confirm" })).toBeTruthy();
  expect(within(report).getByRole("button", { name: "Decline" })).toBeTruthy();
});

it("confirms the reviewed extraction with its generation and offers Review Again", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(getExtraction).mockResolvedValue(partialExtraction);
  vi.mocked(decideExtractionReview).mockResolvedValue({
    state: "confirmed", generation: 1, decided_at: "2026-10-03T00:00:00.000000Z",
  });
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Confirm" }));
  await waitFor(() =>
    expect(decideExtractionReview).toHaveBeenCalledWith(connection, "extraction-1", {
      decision: "confirmed", generation: 1,
    }),
  );
  expect(await screen.findByText("Review: confirmed")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Review Again" })).toBeTruthy();
});

it("reopens the decision through Review Again", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(getExtraction).mockResolvedValue({
    ...partialExtraction,
    review: { state: "declined", generation: 1, decided_at: "2026-10-03T00:00:00.000000Z" },
  });
  vi.mocked(reopenExtractionReview).mockResolvedValue({ state: "pending", generation: 2, decided_at: null });
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Review Again" }));
  await waitFor(() => expect(reopenExtractionReview).toHaveBeenCalledWith(connection, "extraction-1"));
  expect(await screen.findByRole("button", { name: "Confirm" })).toBeTruthy();
});

it("surfaces extraction failure codes instead of a generic label", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(getExtraction).mockResolvedValue({
    ...emptyExtraction,
    failure_code: "no_usable_text",
    failure_detail: "no_usable_text: the document has no usable text to extract",
  });
  render(<App />);
  expect(await screen.findByText(/no usable text could be read/)).toBeTruthy();
});

it("refetches the extraction report after processing actions", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(getExtraction).mockResolvedValue(emptyExtraction);
  vi.mocked(listPdfImports).mockResolvedValue([failedImport]);
  vi.mocked(retryPdfProcessing).mockResolvedValue({ job_id: "job-1", status: "queued" });
  render(<App />);
  const retry = await screen.findByRole("button", { name: "Retry processing" });
  await waitFor(() => expect(getExtraction).toHaveBeenCalled());
  const before = vi.mocked(getExtraction).mock.calls.length;
  fireEvent.click(retry);
  await waitFor(() => expect(getExtraction).toHaveBeenCalledTimes(before + 1));
  expect(getExtraction).toHaveBeenLastCalledWith(connection, source.id);
});

it("refreshes the report when background processing completes", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    vi.mocked(listNotebookSources).mockResolvedValue([source]);
    vi.mocked(listPdfImports).mockResolvedValue([{ ...failedImport, state: "pending" }]);
    vi.mocked(getExtraction)
      .mockResolvedValueOnce(emptyExtraction)
      .mockResolvedValue(partialExtraction);
    render(<App />);
    await screen.findByRole("button", { name: "Cancel processing" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500);
    });
    expect(await screen.findByRole("region", { name: "Text extraction" })).toBeTruthy();
  } finally {
    vi.useRealTimers();
  }
});

it("refreshes the report after a replacement keeps the same source selected", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source]);
  vi.mocked(listPdfAcquisitions).mockResolvedValue([savedAttempt]);
  vi.mocked(getExtraction)
    .mockResolvedValueOnce(emptyExtraction)
    .mockResolvedValue(partialExtraction);
  render(<App />);
  const retry = await screen.findByRole("button", { name: "Retry from saved PDF" });
  await waitFor(() => expect(getExtraction).toHaveBeenCalled());
  const before = vi.mocked(getExtraction).mock.calls.length;
  vi.mocked(retryPdfAcquisition).mockResolvedValue({
    ...savedAttempt, state: "completed", requires_upload: false, can_retry: false, can_cancel: false,
  });
  vi.mocked(listPdfAcquisitions).mockResolvedValue([]);
  fireEvent.click(retry);
  await waitFor(() => expect(getExtraction).toHaveBeenCalledTimes(before + 1));
  expect(getExtraction).toHaveBeenLastCalledWith(connection, source.id);
  expect(await screen.findByRole("region", { name: "Text extraction" })).toBeTruthy();
});

it("ignores a report that arrives after switching sources", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source, sourceB]);
  let resolveFirst!: (report: ExtractionReport) => void;
  vi.mocked(getExtraction).mockImplementation((_connection, sourceId) =>
    sourceId === source.id
      ? new Promise((resolve) => {
          resolveFirst = resolve;
        })
      : Promise.resolve({
          ...partialExtraction,
          source_id: sourceB.id,
          extraction_id: "extraction-b",
          outcome: "complete",
          review: null,
          pages: [],
        }),
  );
  render(<App />);
  await screen.findByRole("button", { name: "Second source" });
  fireEvent.click(screen.getByRole("button", { name: "Second source" }));
  expect(await screen.findByText("Outcome: complete")).toBeTruthy();
  resolveFirst({ ...partialExtraction, source_id: source.id, extraction_id: "extraction-a" });
  await waitFor(() => expect(screen.queryByText("Outcome: partial")).toBeNull());
  expect(screen.getByText("Outcome: complete")).toBeTruthy();
});

it("ignores a file list that arrives after switching sources", async () => {
  vi.mocked(listNotebookSources).mockResolvedValue([source, sourceB]);
  let resolveStale!: (items: typeof failedImport[]) => void;
  vi.mocked(listPdfImports)
    .mockResolvedValueOnce([failedImport])
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveStale = resolve;
        }),
    )
    .mockResolvedValue([{ ...failedImport, id: "import-2", source_id: sourceB.id, filename: "b.pdf" }]);
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Retry processing" }));
  await waitFor(() => expect(retryPdfProcessing).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Second source" }));
  expect(await screen.findByText(/b\.pdf/)).toBeTruthy();
  resolveStale([{ ...failedImport, filename: "a.pdf" }]);
  await act(async () => {
    await Promise.resolve();
  });
  expect(screen.queryByText(/a\.pdf/)).toBeNull();
  expect(screen.getByText(/b\.pdf/)).toBeTruthy();
});

it("drops polling responses that outlive their effect across a source round-trip", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    vi.mocked(listNotebookSources).mockResolvedValue([source, sourceB]);
    let resolveStaleImports!: (items: typeof failedImport[]) => void;
    let resolveStaleReport!: (report: ExtractionReport) => void;
    const staleImport = { ...failedImport, id: "stale-import", filename: "stale.pdf", state: "pending" };
    const freshImport = { ...failedImport, id: "fresh-import", filename: "a.pdf", state: "pending" };
    let importsCalls = 0;
    vi.mocked(listPdfImports).mockImplementation(() => {
      importsCalls += 1;
      if (importsCalls === 1) return Promise.resolve([staleImport]);
      if (importsCalls === 2)
        return new Promise((resolve) => {
          resolveStaleImports = resolve;
        });
      if (importsCalls === 3)
        return Promise.resolve([{ ...freshImport, id: "import-b", source_id: sourceB.id, filename: "b.pdf" }]);
      return Promise.resolve([freshImport]);
    });
    let reportCalls = 0;
    vi.mocked(getExtraction).mockImplementation((_connection, sourceId) => {
      reportCalls += 1;
      if (reportCalls === 1) return Promise.resolve(emptyExtraction);
      if (reportCalls === 2)
        return new Promise((resolve) => {
          resolveStaleReport = resolve;
        });
      if (sourceId === sourceB.id) return Promise.resolve({ ...emptyExtraction, source_id: sourceB.id });
      return Promise.resolve(partialExtraction);
    });

    render(<App />);
    await screen.findByText(/stale\.pdf/);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500);
    });

    fireEvent.click(screen.getByRole("button", { name: "Second source" }));
    await screen.findByText(/b\.pdf/);
    fireEvent.click(screen.getByRole("button", { name: "Field report" }));
    await screen.findByText(/a\.pdf/);
    expect(await screen.findByText("Outcome: partial")).toBeTruthy();

    resolveStaleImports([{ ...staleImport, filename: "stale.pdf" }]);
    resolveStaleReport({
      ...partialExtraction,
      extraction_id: "stale-report",
      outcome: "complete",
      review: null,
      pages: [],
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/a\.pdf/)).toBeTruthy();
    expect(screen.queryByText(/stale\.pdf/)).toBeNull();
    expect(screen.getByText("Outcome: partial")).toBeTruthy();
    expect(screen.queryByText("Outcome: complete")).toBeNull();
  } finally {
    vi.useRealTimers();
  }
});
