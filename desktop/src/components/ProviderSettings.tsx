import { useEffect, useRef, useState } from "react";

import {
  clearProviderCredential,
  getProviderSettings,
  listProviderModels,
  saveProviderSettings,
  testProvider,
  type CoreConnection,
  type ProviderSettings as ProviderSettingsPayload,
} from "../api/client";

const RECOMMENDED_OUTPUT_TOKENS = 1024;
const RECOMMENDED_SAFETY_MARGIN = 512;

function messageFor(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

export function ProviderSettings({ connection }: { connection: CoreConnection }) {
  const [settings, setSettings] = useState<ProviderSettingsPayload | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [editingKey, setEditingKey] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [modelWarning, setModelWarning] = useState<string | null>(null);
  const epoch = useRef(0);

  async function loadCatalog(base: ProviderSettingsPayload): Promise<ProviderSettingsPayload> {
    if (!base.credential_configured) {
      setModelWarning(null);
      return base;
    }
    try {
      const catalog = await listProviderModels(connection);
      setModelWarning(catalog.warning);
      return {
        ...base,
        generation_models: catalog.generation_models,
        checker_models: catalog.checker_models,
      };
    } catch (cause) {
      setModelWarning(messageFor(cause));
      return base;
    }
  }

  useEffect(() => {
    const current = ++epoch.current;
    let cancelled = false;
    setSettings(null);
    setError(null);
    setStatus(null);
    setModelWarning(null);
    setApiKey("");
    setEditingKey(false);
    void getProviderSettings(connection)
      .then(async (next) => {
        const withModels = await loadCatalog(next);
        if (!cancelled && epoch.current === current) {
          setSettings(withModels);
          setAcknowledged(withModels.free_tier_acknowledged);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled && epoch.current === current) setError(messageFor(cause));
      });
    return () => {
      cancelled = true;
      ++epoch.current;
    };
  }, [connection]);

  async function action(operation: () => Promise<ProviderSettingsPayload>, success: string) {
    if (busy) return;
    const current = epoch.current;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const next = await operation();
      if (epoch.current === current) {
        setSettings(next);
        setAcknowledged(next.free_tier_acknowledged);
      }
      if (epoch.current === current) setStatus(success);
    } catch (cause) {
      if (epoch.current === current) setError(messageFor(cause));
    } finally {
      if (epoch.current === current) setBusy(false);
    }
  }

  function settingsInput(current: ProviderSettingsPayload) {
    return {
      generation_model: current.generation_model,
      checker_model: current.checker_model,
      free_tier_acknowledged: acknowledged,
      output_token_allowance: current.output_token_allowance,
      safety_margin_tokens: current.safety_margin_tokens,
      ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
    };
  }

  function save() {
    if (!settings) return;
    void action(
      () => saveProviderSettings(connection, settingsInput(settings)).then(async (next) => {
        setApiKey("");
        setEditingKey(false);
        return await loadCatalog(next);
      }),
      apiKey.trim() ? "Provider settings and key saved." : "Provider settings saved.",
    );
  }

  async function testConnection() {
    if (!settings || busy) return;
    const current = epoch.current;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      // The test uses persisted settings, so save the current form first. This
      // makes a model change or replacement key take effect in one click.
      const saved = await saveProviderSettings(connection, settingsInput(settings));
      if (epoch.current === current) {
        setSettings(saved);
        setAcknowledged(saved.free_tier_acknowledged);
        setApiKey("");
        setEditingKey(false);
      }
      const result = await testProvider(connection);
      const next = await getProviderSettings(connection);
      const withModels = await loadCatalog(next);
      if (epoch.current === current) {
        setSettings(withModels);
        const tokens = result.total_tokens === null ? "" : ` (${result.total_tokens} tokens)`;
        setStatus(`Provider test passed: ${result.model}${tokens}.`);
      }
    } catch (cause) {
      if (epoch.current === current) setError(messageFor(cause));
    } finally {
      if (epoch.current === current) setBusy(false);
    }
  }

  if (!settings) {
    return (
      <section
        aria-label="Generation provider settings"
        className="model-settings provider-settings"
      >
        <h3>Generation provider</h3>
        <p role={error ? "alert" : "status"}>{error ?? "Loading provider settings…"}</p>
      </section>
    );
  }

  const keyReady = settings.credential_configured || apiKey.trim().length > 0;
  const limitsAreRecommended =
    settings.output_token_allowance === RECOMMENDED_OUTPUT_TOKENS &&
    settings.safety_margin_tokens === RECOMMENDED_SAFETY_MARGIN;

  function resetLimits() {
    setSettings((current) =>
      current
        ? {
            ...current,
            output_token_allowance: RECOMMENDED_OUTPUT_TOKENS,
            safety_margin_tokens: RECOMMENDED_SAFETY_MARGIN,
          }
        : current,
    );
  }

  return (
    <section
      aria-label="Generation provider settings"
      className="model-settings provider-settings"
    >
      <h3>Generation provider</h3>
      {error ? <p role="alert">{error}</p> : null}
      {status ? <p role="status">{status}</p> : null}

      {settings.credential_configured && !editingKey ? (
        <div>
          <p>
            <strong>API key</strong> is stored in the operating-system keychain.
            It is never shown again.
          </p>
          <button type="button" disabled={busy} onClick={() => setEditingKey(true)}>
            Replace key
          </button>
        </div>
      ) : (
        <>
          <label htmlFor="provider-api-key">API key</label>
          <input
            id="provider-api-key"
            type="password"
            autoComplete="off"
            value={apiKey}
            disabled={busy}
            placeholder={settings.credential_configured
              ? "Paste a replacement key"
              : "Paste a provider API key"}
            onChange={(event) => setApiKey(event.currentTarget.value)}
          />
          <small>
            The key is sent once to the local core and stored in your operating-system keychain.
          </small>
          {editingKey ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                setEditingKey(false);
                setApiKey("");
              }}
            >
              Cancel replacement
            </button>
          ) : null}
        </>
      )}

      <label htmlFor="provider-free-tier-consent" className="provider-consent">
        <input
          id="provider-free-tier-consent"
          type="checkbox"
          checked={acknowledged}
          disabled={busy}
          onChange={(event) => setAcknowledged(event.currentTarget.checked)}
        />{" "}
        I understand that Google&apos;s Gemini free tier may use prompts and responses to improve its products.
      </label>

      <div>
        <button type="button" disabled={busy || !acknowledged} onClick={save}>
          {busy ? "Saving…" : "Save provider settings"}
        </button>
        <button
          type="button"
          disabled={busy || !keyReady}
          onClick={() => void testConnection()}
        >
          Test connection
        </button>
        {settings.credential_configured ? (
          <button
            type="button"
            disabled={busy}
            onClick={() => void action(
              () => clearProviderCredential(connection).then((next) => {
                setEditingKey(false);
                return next;
              }),
              "Provider key removed.",
            )}
          >
            Remove key
          </button>
        ) : null}
      </div>

      {keyReady ? (
        <fieldset>
          <legend>Models and request limits</legend>
          <label htmlFor="provider-generation-model">Generation model</label>
          <select
            id="provider-generation-model"
            value={settings.generation_model}
            disabled={busy}
            onChange={(event) => {
              const value = event.currentTarget.value;
              setSettings((current) => current ? { ...current, generation_model: value } : current);
            }}
          >
            {settings.generation_models.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>

          <label htmlFor="provider-checker-model">Claim-checker model</label>
          <select
            id="provider-checker-model"
            value={settings.checker_model}
            disabled={busy}
            onChange={(event) => {
              const value = event.currentTarget.value;
              setSettings((current) => current ? { ...current, checker_model: value } : current);
            }}
          >
            {settings.checker_models.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>

          <label htmlFor="provider-output-tokens">Output limit (tokens)</label>
          <input
            id="provider-output-tokens"
            type="number"
            min={1}
            max={1_000_000}
            value={settings.output_token_allowance}
            disabled={busy}
            onChange={(event) => {
              const value = Number(event.currentTarget.value);
              setSettings((current) =>
                current ? { ...current, output_token_allowance: value } : current);
            }}
          />
          <small>
            Recommended: {RECOMMENDED_OUTPUT_TOKENS} tokens. Lower values can shorten
            or truncate answers.
          </small>

          <label htmlFor="provider-safety-margin">Safety margin (tokens)</label>
          <input
            id="provider-safety-margin"
            type="number"
            min={0}
            max={1_000_000}
            value={settings.safety_margin_tokens}
            disabled={busy}
            onChange={(event) => {
              const value = Number(event.currentTarget.value);
              setSettings((current) =>
                current ? { ...current, safety_margin_tokens: value } : current);
            }}
          />
          <small>
            Recommended: {RECOMMENDED_SAFETY_MARGIN} tokens. Lower values leave less
            room for context-limit protection.
          </small>

          <button
            type="button"
            disabled={busy || limitsAreRecommended}
            onClick={resetLimits}
          >
            Reset to recommended limits
          </button>
        </fieldset>
      ) : (
        <p>Enter a provider key to load its available models.</p>
      )}

      {modelWarning ? <small role="status">{modelWarning}</small> : null}

      <small>
        Free-tier rate limits apply to supported models. If the provider reports high demand,
        retry later or choose another supported model.
      </small>
    </section>
  );
}
