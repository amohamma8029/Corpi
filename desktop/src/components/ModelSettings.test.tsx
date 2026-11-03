import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { ModelSettings } from "./ModelSettings";
import { cancelModelSetup, cancelNotebookModel, getNotebookModel, listModels, setupModel, type NotebookModel, type SupportedModel } from "../api/client";

vi.mock("../api/client", () => ({
  listModels: vi.fn(), getNotebookModel: vi.fn(), setupModel: vi.fn(),
  cancelModelSetup: vi.fn(), chooseNotebookModel: vi.fn(), cancelNotebookModel: vi.fn(),
}));

const connection = { base_url: "http://127.0.0.1:1234", token: "test-only" };
const selection: NotebookModel = { notebook_id: "one", active_profile_id: "minilm-en", pending_profile_id: null, generation: 0, ready_sources: 2, rebuilt_sources: 0, failed_sources: 0, cancelled_sources: 0 };
const model: SupportedModel = { profile_id: "minilm-en", manifest_id: "hash", model: "English", language_scope: "English", is_default: true, download_bytes: 24_000_000, state: "missing", generation: 0, received_bytes: 0, failure_code: null, can_cancel: false, can_retry: true };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(listModels).mockResolvedValue([model]);
  vi.mocked(getNotebookModel).mockResolvedValue(selection);
  vi.mocked(setupModel).mockResolvedValue({ ...model, state: "downloading", can_cancel: true, can_retry: false });
  vi.mocked(cancelModelSetup).mockResolvedValue({ ...model, state: "cancelled", generation: 2 });
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

it("shows missing-model setup but never downloads implicitly", async () => {
  render(<ModelSettings connection={connection} notebookId="one" />);
  expect(await screen.findByRole("button", { name: "Download Fast English search" })).toBeTruthy();
  expect(setupModel).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Download Fast English search" }));
  await waitFor(() => expect(setupModel).toHaveBeenCalledWith(connection, model));
});

it("shows progress and sends cancellation for the exact installation generation", async () => {
  const downloading = { ...model, state: "downloading", generation: 3, received_bytes: 5_000_000, can_retry: false, can_cancel: true };
  vi.mocked(listModels).mockResolvedValue([downloading]);
  render(<ModelSettings connection={connection} notebookId="one" />);
  expect(await screen.findByText("Downloading: 5.0 MB received")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Cancel download" }));
  await waitFor(() => expect(cancelModelSetup).toHaveBeenCalledWith(connection, downloading));
});

it("explains that a notebook rebuild preserves current search", async () => {
  vi.mocked(getNotebookModel).mockResolvedValue({ ...selection, pending_profile_id: "multilingual-e5", generation: 1, rebuilt_sources: 1 });
  vi.mocked(listModels).mockResolvedValue([model, { ...model, profile_id: "multilingual-e5", language_scope: "Multilingual" }]);
  render(<ModelSettings connection={connection} notebookId="one" />);
  expect(await screen.findByText(/Rebuilding 1 of 2 ready sources/)).toBeTruthy();
  expect(screen.getByText(/Current search stays usable/)).toBeTruthy();
  expect(screen.getByText(/Active: MiniLM-L6-v2/)).toBeTruthy();
  expect(screen.getByText(/Requested: Multilingual E5-small/)).toBeTruthy();
  expect(screen.getByRole("region", { name: "Local search models" }).getAttribute("data-active-profile")).toBe("minilm-en");
  expect(screen.getByRole("region", { name: "Local search models" }).getAttribute("data-pending-profile")).toBe("multilingual-e5");
});

it("ignores model polling results belonging to the previous notebook", async () => {
  let resolve!: (value: NotebookModel) => void;
  vi.mocked(getNotebookModel).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  const view = render(<ModelSettings connection={connection} notebookId="one" />);
  await waitFor(() => expect(getNotebookModel).toHaveBeenCalledWith(connection, "one"));
  vi.mocked(getNotebookModel).mockResolvedValue({ ...selection, notebook_id: "two", active_profile_id: "multilingual-e5" });
  view.rerender(<ModelSettings connection={connection} notebookId="two" />);
  expect(await screen.findByText(/Active: Multilingual E5-small/)).toBeTruthy();
  await act(async () => resolve(selection));
  expect(screen.queryByText(/Active: MiniLM-L6-v2/)).toBeNull();
});

it.each([[1, 0], [0, 1], [1, 1]])("reports %i failed and %i cancelled current rebuilds without claiming they are rebuilding", async (failed, cancelled) => {
  const blocked: NotebookModel = { ...selection, pending_profile_id: "multilingual-e5", generation: 4, failed_sources: failed, cancelled_sources: cancelled };
  vi.mocked(getNotebookModel).mockResolvedValue(blocked);
  vi.mocked(cancelNotebookModel).mockResolvedValue({ ...selection, generation: 5 });
  render(<ModelSettings connection={connection} notebookId="one" />);
  const message = await screen.findByRole("alert");
  expect(message.textContent).toContain(`Model change could not finish: ${failed} failed, ${cancelled} cancelled`);
  expect(message.textContent).toContain("Your current model remains active");
  expect(screen.queryByText(/Rebuilding/)).toBeNull();
  vi.mocked(getNotebookModel).mockResolvedValue({ ...selection, generation: 5 });
  fireEvent.click(screen.getByRole("button", { name: "Cancel model change" }));
  await waitFor(() => expect(cancelNotebookModel).toHaveBeenCalledWith(connection, blocked));
  expect(await screen.findByText(/Active: MiniLM-L6-v2/)).toBeTruthy();
});

it("does not let a pre-cancellation poll restore a stale pending failure", async () => {
  vi.useFakeTimers();
  const blocked: NotebookModel = { ...selection, pending_profile_id: "multilingual-e5", generation: 4, failed_sources: 1, cancelled_sources: 0 };
  vi.mocked(getNotebookModel).mockResolvedValue(blocked);
  vi.mocked(cancelNotebookModel).mockResolvedValue({ ...selection, generation: 5 });
  render(<ModelSettings connection={connection} notebookId="one" />);
  await act(async () => {});
  expect(screen.getByRole("alert").textContent).toContain("Model change could not finish");
  let resolve!: (value: NotebookModel) => void;
  vi.mocked(getNotebookModel).mockImplementationOnce(() => new Promise(done => { resolve = done; }));
  await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
  vi.mocked(getNotebookModel).mockResolvedValue({ ...selection, generation: 5 });
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Cancel model change" })); });
  expect(screen.getByText(/Active: MiniLM-L6-v2/)).toBeTruthy();
  await act(async () => resolve(blocked));
  expect(screen.getByText(/Active: MiniLM-L6-v2/)).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
});

it("treats missing terminal counts in an older wire response as zero", async () => {
  const legacy: Partial<NotebookModel> = { ...selection, pending_profile_id: "multilingual-e5" };
  delete legacy.failed_sources;
  delete legacy.cancelled_sources;
  vi.mocked(getNotebookModel).mockResolvedValue(legacy as NotebookModel);
  render(<ModelSettings connection={connection} notebookId="one" />);
  expect(await screen.findByText(/Rebuilding 0 of 2 ready sources/)).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
});

it("retains the failure and allows another explicit cancel attempt after an action error", async () => {
  vi.mocked(getNotebookModel).mockResolvedValue({ ...selection, pending_profile_id: "multilingual-e5", generation: 4, failed_sources: 1 });
  vi.mocked(cancelNotebookModel).mockRejectedValue(new Error("Controlled cancellation failure"));
  render(<ModelSettings connection={connection} notebookId="one" />);
  await screen.findByText(/Model change could not finish/);
  fireEvent.click(screen.getByRole("button", { name: "Cancel model change" }));
  expect(await screen.findByText("Controlled cancellation failure")).toBeTruthy();
  expect(screen.getByText(/Model change could not finish/)).toBeTruthy();
  expect(screen.getByRole("button", { name: "Cancel model change" }).hasAttribute("disabled")).toBe(false);
});
