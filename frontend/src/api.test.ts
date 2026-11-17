import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";

function streamResponse(chunks: string[], init: ResponseInit = {}) {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" }, ...init });
}

function mockFetch(impl: (...args: Parameters<typeof fetch>) => Promise<Response>) {
  const fn = vi.fn(impl);
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("getUserId", () => {
  it("creates a UUID once, stores it and reuses it", () => {
    const id = api.getUserId();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    expect(id).toMatch(/^[A-Za-z0-9_-]{6,64}$/); // satisfies the backend's user_id rule
    expect(api.getUserId()).toBe(id);
    expect(localStorage.getItem("cw_user_id")).toBe(id);
  });

  it("keeps an id that already exists", () => {
    localStorage.setItem("cw_user_id", "existing-user-1");
    expect(api.getUserId()).toBe("existing-user-1");
  });
});

describe("streamChat (SSE over POST)", () => {
  const run = async (chunks: string[]) => {
    mockFetch(async () => streamResponse(chunks));
    const events: [string, unknown][] = [];
    await api.streamChat({ user_id: "u-123456", message: "hi" }, (ev, data) => events.push([ev, data]), new AbortController().signal);
    return events;
  };

  it("posts JSON and parses several events in one chunk", async () => {
    const fetchFn = mockFetch(async () =>
      streamResponse(['event: conversation\ndata: {"conversation_id":"c1"}\n\nevent: token\ndata: {"text":"Hel"}\n\n']));
    const events: [string, unknown][] = [];
    await api.streamChat({ user_id: "u-123456", conversation_id: null, message: "hi" }, (e, d) => events.push([e, d]),
      new AbortController().signal);
    expect(events).toEqual([["conversation", { conversation_id: "c1" }], ["token", { text: "Hel" }]]);
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe(`${api.API}/api/chat`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ user_id: "u-123456", conversation_id: null, message: "hi" });
  });

  it("reassembles events split across network chunks, including mid-JSON and mid-unicode", async () => {
    const full = 'event: token\ndata: {"text":"₹5 lakh 🚗"}\n\nevent: done\ndata: {"message_id":7}\n\n';
    const bytes = new TextEncoder().encode(full);
    const enc = (b: Uint8Array) => b;
    const pieces = [bytes.slice(0, 7), bytes.slice(7, 27), bytes.slice(27, 30), bytes.slice(30)].map(enc);
    mockFetch(async () => new Response(new ReadableStream({
      start(c) { for (const p of pieces) c.enqueue(p); c.close(); },
    }), { status: 200 }));
    const events: [string, unknown][] = [];
    await api.streamChat({ user_id: "u-123456", message: "x" }, (e, d) => events.push([e, d]), new AbortController().signal);
    expect(events).toEqual([["token", { text: "₹5 lakh 🚗" }], ["done", { message_id: 7 }]]);
  });

  it("accepts CRLF line endings and ignores comment/keep-alive frames", async () => {
    const events = await run([": keep-alive\r\n\r\n", 'event: status\r\ndata: {"text":"ok"}\r\n\r\n']);
    expect(events).toEqual([["status", { text: "ok" }]]);
  });

  it("defaults the event name to 'message'", async () => {
    expect(await run(['data: {"a":1}\n\n'])).toEqual([["message", { a: 1 }]]);
  });

  it.each([
    [429, '{"error":"Too many requests. Please retry in 30 seconds."}', "Too many requests. Please retry in 30 seconds."],
    [403, '{"error":"conversation belongs to another user"}', "conversation belongs to another user"],
    [422, '{"detail":[{"loc":["body","message"],"msg":"too long"}]}', "Request failed (HTTP 422)"],
    [400, '{"detail":"invalid conversation id"}', "invalid conversation id"],
    [500, "Internal Server Error", "Internal Server Error"],
    [502, "", "Request failed (HTTP 502)"],
  ])("turns HTTP %s into a readable error", async (status, body, message) => {
    mockFetch(async () => new Response(body, { status }));
    await expect(api.streamChat({ user_id: "u-123456", message: "x" }, () => undefined, new AbortController().signal))
      .rejects.toThrow(message);
  });

  it("rejects when the request is aborted", async () => {
    mockFetch(async (_u, init) => {
      if (init?.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      return streamResponse([]);
    });
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(api.streamChat({ user_id: "u-123456", message: "x" }, () => undefined, ctrl.signal)).rejects.toThrow("Aborted");
  });

  it("surfaces a malformed data frame as an error instead of hanging", async () => {
    mockFetch(async () => streamResponse(["event: token\ndata: {not json}\n\n"]));
    await expect(api.streamChat({ user_id: "u-123456", message: "x" }, () => undefined, new AbortController().signal))
      .rejects.toThrow();
  });
});

describe("JSON endpoints", () => {
  it("health / listConversations / getConversation", async () => {
    const fetchFn = mockFetch(async (url) => {
      const u = String(url);
      if (u.endsWith("/api/health")) return Response.json({ status: "ok", models: 288 });
      if (u.includes("/api/conversations?")) return Response.json({ conversations: [{ id: "c1", title: "t" }] });
      return Response.json({ messages: [{ id: 5, role: "assistant", content: "hi", cars: null, sources: [], followups: null, trace: null }] });
    });
    expect((await api.health()).models).toBe(288);
    expect(await api.listConversations("a b&c")).toEqual([{ id: "c1", title: "t" }]);
    expect(String(fetchFn.mock.calls[1][0])).toContain("user_id=a%20b%26c"); // user id is URL-encoded
    const msgs = await api.getConversation("c1", "u-123456");
    expect(msgs).toEqual([{ key: "m5", id: 5, role: "assistant", content: "hi", cars: undefined, sources: [], followups: undefined, trace: undefined }]);
  });

  it("throws on non-OK responses", async () => {
    mockFetch(async () => new Response("nope", { status: 503 }));
    await expect(api.health()).rejects.toThrow("HTTP 503");
    await expect(api.getConversation("c1", "u")).rejects.toThrow("HTTP 503");
  });

  it("delete and feedback use the right method and body", async () => {
    const fetchFn = mockFetch(async () => Response.json({ ok: true }));
    await api.deleteConversation("c1", "u-123456");
    await api.sendFeedback(9, -1);
    expect(fetchFn.mock.calls[0][1]?.method).toBe("DELETE");
    expect(String(fetchFn.mock.calls[0][0])).toContain("/api/conversations/c1?user_id=u-123456");
    expect(JSON.parse(String(fetchFn.mock.calls[1][1]?.body))).toEqual({ message_id: 9, rating: -1 });
  });
});
