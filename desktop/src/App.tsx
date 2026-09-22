import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { CoreConnection, CoreHealth, coreHealth, demoStream, StreamEvent } from "./api/client";

type CoreStatus = "connecting" | "ok" | "error";

const STATUS_TEXT: Record<CoreStatus, string> = {
  connecting: "Connecting to core…",
  ok: "Core ready",
  error: "Core unavailable",
};

export function App() {
  const [connection, setConnection] = useState<CoreConnection | null>(null);
  const [status, setStatus] = useState<CoreStatus>("connecting");
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const conn = await invoke<CoreConnection>("core_connection");
        if (cancelled) return;
        const health: CoreHealth = await coreHealth(conn);
        if (cancelled) return;
        setConnection(conn);
        setVersion(health.version);
        setStatus("ok");
      } catch (cause) {
        if (!cancelled) {
          setStatus("error");
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const runStream = async () => {
    if (!connection || streaming) return;
    setStreaming(true);
    setEvents([]);
    try {
      await demoStream(connection, (event: StreamEvent) =>
        setEvents((previous) => [...previous, event]),
      );
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setStreaming(false);
    }
  };

  return (
    <main>
      <h1>Corpi</h1>
      <p id="core-status">{STATUS_TEXT[status]}{version ? ` (v${version})` : ""}</p>
      {error ? <p id="core-error" role="alert">{error}</p> : null}
      <button id="run-stream" type="button" onClick={runStream} disabled={!connection || streaming}>
        {streaming ? "Streaming…" : "Run demo stream"}
      </button>
      <ol id="stream-events">
        {events.map((event, index) => (
          <li key={index}>{event.step != null ? `Step ${String(event.step)}/${String(event.total)}` : "Done"}</li>
        ))}
      </ol>
    </main>
  );
}
