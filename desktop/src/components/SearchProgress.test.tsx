import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { coreRequest } from "../api/client";
import { SearchProgress } from "./SearchProgress";

vi.mock("../api/client", () => ({ coreRequest: vi.fn() }));
const connection = { base_url: "http://127.0.0.1:1234", token: "test-only" };
const running = { source_id: "one", status: "running", step: "embed-search", waiting_reason: null, failure_code: null, completed_chunks: 2, total_chunks: 10, job_id: "original-job", can_cancel: true, can_resume: false };
const cancelled = { ...running, status: "cancelled", can_cancel: false, can_resume: true };
beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

it("cancels the exact current job then explicitly resumes without pretending to reset progress", async () => {
  vi.mocked(coreRequest).mockResolvedValueOnce(running).mockResolvedValueOnce({ job_id: running.job_id }).mockResolvedValueOnce(cancelled)
    .mockResolvedValueOnce({ job_id: "resumed-job" }).mockResolvedValueOnce({ ...running, job_id: "resumed-job" });
  render(<SearchProgress connection={connection} sourceId="one" />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel search preparation" }));
  fireEvent.click(await screen.findByRole("button", { name: "Resume search preparation" }));
  expect(await screen.findByText("2 of 10 text pieces embedded")).toBeTruthy();
  await waitFor(() => expect(coreRequest).toHaveBeenCalledTimes(5));
  const cancel = vi.mocked(coreRequest).mock.calls[1];
  expect(cancel[1]).toBe("/sources/one/search/cancel");
  expect(JSON.parse(cancel[2]!.body as string)).toEqual({ job_id: "original-job" });
  const resume = vi.mocked(coreRequest).mock.calls[3];
  expect(resume[1]).toBe("/sources/one/search/resume");
  expect(JSON.parse(resume[2]!.body as string)).toEqual({ job_id: "original-job", request_key: expect.any(String) });
});

it("ignores an action reply after the source changes and comes back", async () => {
  let finish!: (value: unknown) => void;
  vi.mocked(coreRequest).mockResolvedValueOnce(running).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
    .mockResolvedValueOnce({ ...running, source_id: "two" }).mockResolvedValueOnce(running).mockResolvedValueOnce(cancelled);
  const view = render(<SearchProgress connection={connection} sourceId="one" />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel search preparation" }));
  view.rerender(<SearchProgress connection={connection} sourceId="two" />);
  view.rerender(<SearchProgress connection={connection} sourceId="one" />);
  await act(async () => finish({ job_id: "original-job" }));
  expect(screen.queryByText("Search preparation cancelled")).toBeNull();
});

it("shows an action conflict instead of silently claiming cancellation", async () => {
  vi.mocked(coreRequest).mockResolvedValueOnce(running).mockRejectedValueOnce(new Error("Preparation changed; refresh."));
  render(<SearchProgress connection={connection} sourceId="one" />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel search preparation" }));
  expect((await screen.findByRole("alert")).textContent).toContain("Preparation changed");
  expect(screen.queryByText("Search preparation cancelled")).toBeNull();
});

it.each([
  ["review reopened", "Review Again reopened the text review"],
  ["model selection changed", "The notebook model changed"],
  ["user cancelled PDF processing", "PDF processing was cancelled"],
])("explains why preparation stopped: %s", async (reason, explanation) => {
  vi.mocked(coreRequest).mockResolvedValue({ ...cancelled, waiting_reason: reason, can_resume: false });
  render(<SearchProgress connection={connection} sourceId="one" />);
  expect(await screen.findByText(new RegExp(explanation))).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Resume search preparation" })).toBeNull();
});

it.each(["embed-search", "build-search", "publish-search"])("does not call fully embedded pieces ready during %s", async step => {
  vi.mocked(coreRequest).mockResolvedValue({ ...running, step, completed_chunks: 10 });
  render(<SearchProgress connection={connection} sourceId="one" />);
  expect(await screen.findByText(/10 of 10 text pieces embedded/)).toBeTruthy();
  expect(screen.getByText(/Search is not ready yet/)).toBeTruthy();
  expect(screen.queryByText("Search data ready")).toBeNull();
});

it("shows a terminal failure detail when there is no waiting reason", async () => {
  vi.mocked(coreRequest).mockResolvedValue({ ...running, status: "failed", failure_code: "search_time_limit", failure_reason: "search preparation allowance exhausted", can_cancel: false });
  render(<SearchProgress connection={connection} sourceId="one" />);
  expect(await screen.findByText(/search preparation allowance exhausted/)).toBeTruthy();
});

it("names the exact pending PDF and model being prepared", async () => {
  vi.mocked(coreRequest).mockResolvedValue({ ...running, filename: "replacement-fr.pdf", pdf_role: "pending", profile_id: "multilingual-e5" });
  render(<SearchProgress connection={connection} sourceId="one" />);
  expect(await screen.findByText(/Saved PDF: replacement-fr.pdf · Pending replacement/)).toBeTruthy();
  expect(screen.getByText("Search model: Multilingual E5-small")).toBeTruthy();
});

it.each(["completed", "failed", "cancelled"])("does not let an old waiting reason override %s status", async status => {
  vi.mocked(coreRequest).mockResolvedValue({ ...running, status, waiting_reason: "model_not_installed", completed_chunks: 10, can_cancel: false });
  render(<SearchProgress connection={connection} sourceId="one" />);
  expect(await screen.findByText(status === "completed" ? "Search data ready" : `Search preparation ${status}`)).toBeTruthy();
  expect(screen.queryByText(/Waiting for model setup/)).toBeNull();
  expect(screen.queryByText(/Search is not ready yet/)).toBeNull();
});
