export interface CoreConnection {
  base_url: string;
  token: string;
}

export interface CoreHealth {
  app: string;
  version: string;
  status: string;
}

export type StreamEvent = Record<string, unknown>;

export function extractEvents(buffer: string): { events: StreamEvent[]; buffer: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const events: StreamEvent[] = [];
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (line.startsWith("data: ")) {
        events.push(JSON.parse(line.slice("data: ".length)) as StreamEvent);
      }
    }
  }
  return { events, buffer: rest };
}

export async function coreRequest<T>(
  connection: CoreConnection,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${connection.base_url}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      Authorization: `Bearer ${connection.token}`,
    },
  });
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function coreHealth(connection: CoreConnection): Promise<CoreHealth> {
  return coreRequest<CoreHealth>(connection, "/health");
}

export async function demoStream(
  connection: CoreConnection,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${connection.base_url}/demo/stream`, {
    headers: { Authorization: `Bearer ${connection.token}` },
  });
  if (!response.ok || !response.body) {
    throw new Error(`/demo/stream: HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const { events, buffer: rest } = extractEvents(buffer);
    buffer = rest;
    for (const event of events) {
      onEvent(event);
    }
  }
}
