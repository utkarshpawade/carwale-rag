import type { ChatMessage, Conversation } from "./types";

export const API = ((import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000").replace(/\/$/, "");

export function getUserId(): string {
  let id = localStorage.getItem("cw_user_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("cw_user_id", id);
  }
  return id;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export const health = (signal?: AbortSignal) =>
  json<{ status: string; models: number; variants: number; chunks: number; data_snapshot: string }>("/api/health", { signal });

export const getMeta = () => json<{ starters: string[] }>("/api/meta");

export const listConversations = (userId: string) =>
  json<{ conversations: Conversation[] }>(`/api/conversations?user_id=${encodeURIComponent(userId)}`).then((r) => r.conversations);

export async function getConversation(id: string, userId: string): Promise<ChatMessage[]> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r = await json<{ messages: any[] }>(`/api/conversations/${id}?user_id=${encodeURIComponent(userId)}`);
  return r.messages.map((m) => ({
    key: `m${m.id}`, id: m.id, role: m.role, content: m.content,
    cars: m.cars || undefined, sources: m.sources || undefined, followups: m.followups || undefined, trace: m.trace || undefined,
  }));
}

export const deleteConversation = (id: string, userId: string) =>
  fetch(`${API}/api/conversations/${id}?user_id=${encodeURIComponent(userId)}`, { method: "DELETE" });

export const sendFeedback = (messageId: number, rating: number) =>
  fetch(`${API}/api/feedback`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, rating }),
  });

/** POST + Server-Sent Events (EventSource can't POST, so parse the stream manually). */
export async function streamChat(
  body: { user_id: string; conversation_id?: string | null; message: string },
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onEvent: (event: string, data: any) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let msg = text;
    try { msg = JSON.parse(text).error || JSON.parse(text).detail || text; } catch { /* plain text */ }
    throw new Error(typeof msg === "string" && msg ? msg : `Request failed (HTTP ${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}
