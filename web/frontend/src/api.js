import { createSSEParser } from "./sse.js";

async function jsonOrThrow(res) {
  if (res.ok) return res.json();
  let msg = `HTTP ${res.status}`;
  try {
    const j = await res.json();
    msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch { /* keep default message */ }
  throw new Error(msg);
}

export const getStatus = () => fetch("/api/status").then(jsonOrThrow);
export const getCheckpoints = () => fetch("/api/checkpoints").then(jsonOrThrow);
export const loadCheckpoint = (name) =>
  fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  }).then(jsonOrThrow);

/** POST /api/generate and call back for every SSE event. Abort with the AbortSignal to stop. */
export async function streamGenerate(body, { signal, onStart, onToken, onDone, onError }) {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    await jsonOrThrow(res); // throws with the server's message
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const feed = createSSEParser((event, data) => {
    if (event === "start") onStart?.(data);
    else if (event === "token") onToken?.(data.text);
    else if (event === "done") onDone?.(data);
    else if (event === "error") onError?.(data.message);
  });
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    feed(decoder.decode(value, { stream: true }));
  }
}
