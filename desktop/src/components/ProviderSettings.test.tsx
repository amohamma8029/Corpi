import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import {
  clearProviderCredential,
  getProviderSettings,
  listProviderModels,
  saveProviderSettings,
  testProvider,
  type ProviderSettings as ProviderSettingsPayload,
} from "../api/client";
import { ProviderSettings } from "./ProviderSettings";

vi.mock("../api/client", () => ({
  getProviderSettings: vi.fn(),
  listProviderModels: vi.fn(),
  saveProviderSettings: vi.fn(),
  testProvider: vi.fn(),
  clearProviderCredential: vi.fn(),
}));

const connection = { base_url: "http://127.0.0.1:1234", token: "test-only" };
const settings: ProviderSettingsPayload = {
  provider_id: "google-gemini",
  provider_label: "Google Gemini",
  generation_model: "gemini-3.8-flash",
  checker_model: "gemini-3.5-flash-lite",
  free_tier_acknowledged: false,
  output_token_allowance: 1024,
  safety_margin_tokens: 512,
  call_policies: {},
  credential_configured: false,
  generation_models: ["gemini-3.8-flash", "gemini-4-hydrogen"],
  checker_models: ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getProviderSettings).mockResolvedValue(settings);
  vi.mocked(listProviderModels).mockResolvedValue({
    generation_models: settings.generation_models,
    checker_models: settings.checker_models,
    discovered: true,
    warning: null,
  });
  vi.mocked(saveProviderSettings).mockResolvedValue(settings);
});

afterEach(() => {
  cleanup();
});

it("shows free-tier consent and never pre-fills a stored key", async () => {
  vi.mocked(getProviderSettings).mockResolvedValue({
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: true,
  });

  render(<ProviderSettings connection={connection} />);

  expect(await screen.findByLabelText("Generation model")).toBeTruthy();
  expect(screen.queryByLabelText("API key")).toBeNull();
  expect(screen.getByText(/is stored in the operating-system keychain/)).toBeTruthy();
  expect(screen.getByLabelText(/free tier may use prompts/)).toHaveProperty("checked", true);

  fireEvent.click(screen.getByRole("button", { name: "Replace key" }));
  expect(screen.getByLabelText("API key")).toHaveProperty("value", "");
});

it("shows recommended request limits and resets edited values", async () => {
  vi.mocked(getProviderSettings).mockResolvedValue({
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: true,
  });

  render(<ProviderSettings connection={connection} />);
  const output = await screen.findByLabelText("Output limit (tokens)");
  const margin = screen.getByLabelText("Safety margin (tokens)");
  expect((output as HTMLInputElement).value).toBe("1024");
  expect((margin as HTMLInputElement).value).toBe("512");
  expect(screen.getByText(/Recommended: 1024 tokens/)).toBeTruthy();
  expect(screen.getByText(/Recommended: 512 tokens/)).toBeTruthy();
  const reset = screen.getByRole("button", { name: "Reset to recommended limits" });
  expect(reset.hasAttribute("disabled")).toBe(true);

  fireEvent.change(output, { target: { value: "4096" } });
  fireEvent.change(margin, { target: { value: "128" } });
  expect(reset.hasAttribute("disabled")).toBe(false);
  fireEvent.click(reset);

  expect((output as HTMLInputElement).value).toBe("1024");
  expect((margin as HTMLInputElement).value).toBe("512");
  expect(reset.hasAttribute("disabled")).toBe(true);
});

it("offers a discovered model and saves the selected model", async () => {
  const discovered = ["gemini-3.8-flash", "gemini-9-experimental"];
  const configured = {
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: true,
    generation_models: discovered,
    checker_models: settings.checker_models,
  };
  vi.mocked(getProviderSettings).mockResolvedValue(configured);
  vi.mocked(listProviderModels).mockResolvedValue({
    generation_models: discovered,
    checker_models: settings.checker_models,
    discovered: true,
    warning: null,
  });
  vi.mocked(saveProviderSettings).mockResolvedValue({
    ...configured,
    generation_model: "gemini-9-experimental",
  });

  render(<ProviderSettings connection={connection} />);
  const model = await screen.findByLabelText("Generation model");
  fireEvent.change(model, { target: { value: "gemini-9-experimental" } });
  fireEvent.click(screen.getByRole("button", { name: "Save provider settings" }));

  await waitFor(() => expect(saveProviderSettings).toHaveBeenCalledWith(
    connection,
    expect.objectContaining({ generation_model: "gemini-9-experimental" }),
  ));
});

it("saves the key and consent without rendering the key afterward", async () => {
  const saved = {
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: true,
  };
  vi.mocked(saveProviderSettings).mockResolvedValue(saved);

  render(<ProviderSettings connection={connection} />);
  const key = await screen.findByLabelText("API key");
  expect(screen.queryByLabelText("Generation model")).toBeNull();
  fireEvent.change(key, { target: { value: "secret-free-tier-key" } });
  expect(screen.getByLabelText("Generation model")).toBeTruthy();
  fireEvent.click(screen.getByLabelText(/free tier may use prompts/));
  fireEvent.click(screen.getByRole("button", { name: "Save provider settings" }));

  await waitFor(() => expect(saveProviderSettings).toHaveBeenCalledWith(connection, {
    generation_model: "gemini-3.8-flash",
    checker_model: "gemini-3.5-flash-lite",
    free_tier_acknowledged: true,
    output_token_allowance: 1024,
    safety_margin_tokens: 512,
    api_key: "secret-free-tier-key",
  }));
  expect(await screen.findByText("Provider settings and key saved.")).toBeTruthy();
  expect(screen.queryByLabelText("API key")).toBeNull();
  expect(screen.getByText(/is stored in the operating-system keychain/)).toBeTruthy();
  expect(document.body.textContent).not.toContain("secret-free-tier-key");
});

it("saves the current model before testing, then removes a configured key", async () => {
  const hydrogen = {
    ...settings,
    generation_model: "gemini-4-hydrogen",
    free_tier_acknowledged: true,
    credential_configured: true,
  };
  vi.mocked(getProviderSettings).mockResolvedValue({
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: true,
  });
  vi.mocked(saveProviderSettings).mockResolvedValue(hydrogen);
  vi.mocked(testProvider).mockResolvedValue({
    ok: true,
    provider_id: "google-gemini",
    model: "gemini-4-hydrogen",
    request_id: "request-1",
    provider_request_id: "provider-request-1",
    input_tokens: 4,
    output_tokens: 1,
    total_tokens: 5,
  });
  vi.mocked(clearProviderCredential).mockResolvedValue({
    ...settings,
    free_tier_acknowledged: true,
    credential_configured: false,
  });

  render(<ProviderSettings connection={connection} />);
  fireEvent.change(await screen.findByLabelText("Generation model"), {
    target: { value: "gemini-4-hydrogen" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
  await waitFor(() => expect(saveProviderSettings).toHaveBeenCalledWith(
    connection,
    expect.objectContaining({ generation_model: "gemini-4-hydrogen" }),
  ));
  await waitFor(() => expect(testProvider).toHaveBeenCalledWith(connection));
  expect(await screen.findByText(/Provider test passed: gemini-4-hydrogen/)).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Remove key" }));
  await waitFor(() => expect(clearProviderCredential).toHaveBeenCalledWith(connection));
  expect(await screen.findByText("Provider key removed.")).toBeTruthy();
  expect(await screen.findByLabelText("API key")).toBeTruthy();
});
