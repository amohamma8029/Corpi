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
  expect(await screen.findByText("2 of 10 search pieces prepared")).toBeTruthy();
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
