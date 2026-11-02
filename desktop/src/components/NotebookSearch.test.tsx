import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { searchNotebook, type NotebookSearchResponse, type NotebookSearchResult } from "../api/client";
import { NotebookSearch } from "./NotebookSearch";

vi.mock("../api/client", () => ({ searchNotebook: vi.fn() }));
const connection = { base_url: "http://127.0.0.1:1234", token: "test-only" };
const result: NotebookSearchResult = {
  source_id: "source", source_version_id: "saved-version", index_version_id: "index",
  chunk_id: "chunk", manifest_id: "model", source_name: "Arabic rain study",
  text: "ﻣﻄـﺮ المطر", score: 0.2, import_id: "original-import", filename: "original.pdf",
  citations: [{ passage_id: "passage", ordinal: 0, start_offset: 1, end_offset: 10, page_number: 2 }],
};
const response: NotebookSearchResponse = { notebook_id: "one", mode: "meaning", ready_sources: 3, results: [result] };

beforeEach(() => { vi.resetAllMocks(); vi.mocked(searchNotebook).mockResolvedValue(response); });
afterEach(cleanup);

function submit(query = "rainfall") {
  fireEvent.change(screen.getByRole("searchbox", { name: "Find a passage" }), { target: { value: query } });
  fireEvent.submit(screen.getByRole("button", { name: "Search" }).closest("form")!);
}

it("searches only on submit, preserves canonical RTL text and opens the mapped saved page", async () => {
  const open = vi.fn();
  render(<NotebookSearch connection={connection} notebookId="one" onOpen={open} />);
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "rainfall" } });
  expect(searchNotebook).not.toHaveBeenCalled();
  submit();
  const button = await screen.findByRole("button", { name: "Open original.pdf, page 2" });
  expect(screen.getByText(result.text).getAttribute("dir")).toBe("auto");
  fireEvent.click(button);
  expect(open).toHaveBeenCalledWith(result, 2);
  expect(searchNotebook).toHaveBeenCalledWith(connection, "one", "rainfall", "meaning", expect.any(AbortSignal));
});

it("uses explicit Words mode and distinguishes no ready sources from no matches", async () => {
  vi.mocked(searchNotebook).mockResolvedValueOnce({ ...response, mode: "words", ready_sources: 0, results: [] });
  render(<NotebookSearch connection={connection} notebookId="one" onOpen={() => {}} />);
  fireEvent.change(screen.getByRole("combobox", { name: "Search mode" }), { target: { value: "words" } });
  submit();
  expect(await screen.findByText(/No sources are ready yet/)).toBeTruthy();
  vi.mocked(searchNotebook).mockResolvedValueOnce({ ...response, mode: "words", results: [] });
  submit();
  expect(await screen.findByText(/No matches/)).toBeTruthy();
  expect(searchNotebook).toHaveBeenLastCalledWith(connection, "one", "rainfall", "words", expect.any(AbortSignal));
});

it("offers one PDF button per cited page without dropping the original citation mappings", async () => {
  const mapped = { ...result, citations: [result.citations[0],
    { ...result.citations[0], passage_id: "same-page", ordinal: 1 },
    { ...result.citations[0], passage_id: "next-page", ordinal: 2, page_number: 3 },
  ] };
  vi.mocked(searchNotebook).mockResolvedValue({ ...response, results: [mapped] });
  const open = vi.fn();
  render(<NotebookSearch connection={connection} notebookId="one" onOpen={open} />);
  submit();
  expect(await screen.findAllByRole("button", { name: "Open original.pdf, page 2" })).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Open original.pdf, page 3" }));
  expect(open).toHaveBeenCalledWith(mapped, 3);
  expect(mapped.citations).toHaveLength(3);
});

it("aborts and rejects a late response across notebook changes, including returning to the first notebook", async () => {
  let finish!: (value: NotebookSearchResponse) => void;
  vi.mocked(searchNotebook).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  const view = render(<NotebookSearch connection={connection} notebookId="one" onOpen={() => {}} />);
  submit();
  await waitFor(() => expect(searchNotebook).toHaveBeenCalledTimes(1));
  const signal = vi.mocked(searchNotebook).mock.calls[0][4]!;
  view.rerender(<NotebookSearch connection={connection} notebookId="two" onOpen={() => {}} />);
  view.rerender(<NotebookSearch connection={connection} notebookId="one" onOpen={() => {}} />);
  expect(signal.aborted).toBe(true);
  await act(async () => finish(response));
  expect(screen.queryByText(result.text)).toBeNull();
  expect((screen.getByRole("searchbox") as HTMLInputElement).value).toBe("");
});

it("shows a local worker error and permits another explicit search", async () => {
  vi.mocked(searchNotebook).mockRejectedValueOnce(new Error("The local search worker could not finish."));
  render(<NotebookSearch connection={connection} notebookId="one" onOpen={() => {}} />);
  submit();
  expect((await screen.findByRole("alert")).textContent).toContain("could not finish");
  submit();
  expect(await screen.findByText(result.text)).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
});
