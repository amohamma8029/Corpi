import { invoke } from "@tauri-apps/api/core";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { App } from "./App";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("keeps the startup screen responsive while the core is pending, then shows busy", async () => {
  let rejectConnection!: (error: Error) => void;
  vi.mocked(invoke).mockImplementation(
    () => new Promise<never>((_, reject) => {
      rejectConnection = reject;
    }),
  );

  render(<App />);
  expect(screen.getByText("Connecting to core…")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Run demo stream" }).hasAttribute("disabled")).toBe(true);

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
