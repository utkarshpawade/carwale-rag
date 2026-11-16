import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import Composer from "./components/Composer";
import { FlowIcon, MenuIcon, PlusIcon, SparkIcon } from "./components/Icons";
import MessageView from "./components/MessageView";
import Sidebar from "./components/Sidebar";
import TracePanel from "./components/TracePanel";
import type { ChatMessage, Conversation, Trace } from "./types";

const STARTERS = [
  "Smart hybrid cars under 30 lakhs",
  "Creta vs Seltos — which is better?",
  "Mileage of Maruti Swift",
  "Safest SUVs under 15 lakh",
  "Does the Tata Nexon have a sunroof?",
  "Warranty and service of Hyundai Venue",
];
let keySeq = 0;
const newKey = () => `k${Date.now()}_${keySeq++}`;

export default function App() {
  const userId = useRef(api.getUserId()).current;
  const [server, setServer] = useState<"waking" | "ready" | "down">("waking");
  const [stats, setStats] = useState<{ models?: number; variants?: number; data_snapshot?: string }>({});
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState<string | null>(() => localStorage.getItem("cw_conv_id"));
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [trace, setTrace] = useState<{ key: string; trace: Trace } | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // wake the (free-tier, sleeping) backend as soon as the page loads
  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    const ping = async () => {
      try {
        const h = await api.health(AbortSignal.timeout(15000));
        if (cancelled) return;
        setStats(h);
        setServer("ready");
      } catch {
        if (cancelled) return;
        if (Date.now() - started > 150000) setServer("down");
        else setTimeout(ping, 3000);
      }
    };
    ping();
    return () => { cancelled = true; };
  }, []);

  const refreshConversations = useCallback(() => {
    api.listConversations(userId).then(setConversations).catch(() => undefined);
  }, [userId]);

  useEffect(() => {
    if (server !== "ready") return;
    refreshConversations();
    if (convId) {
      api.getConversation(convId, userId).then(setMessages).catch(() => { localStorage.removeItem("cw_conv_id"); setConvId(null); });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [server]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages]);

  const patch = (key: string, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((ms) => ms.map((m) => (m.key === key ? fn(m) : m)));

  const send = async (text: string) => {
    if (streaming || server !== "ready") return;
    const aKey = newKey();
    setMessages((ms) => [...ms, { key: newKey(), role: "user", content: text },
      { key: aKey, role: "assistant", content: "", streaming: true, status: "Understanding your question…" }]);
    setStreaming(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await api.streamChat({ user_id: userId, conversation_id: convId, message: text }, (ev, data) => {
        switch (ev) {
          case "conversation":
            if (data.conversation_id !== convId) { setConvId(data.conversation_id); localStorage.setItem("cw_conv_id", data.conversation_id); }
            break;
          case "status": patch(aKey, (m) => ({ ...m, status: data.text })); break;
          case "trace": patch(aKey, (m) => ({ ...m, trace: data, cached: !!data.cache })); break;
          case "cars": patch(aKey, (m) => ({ ...m, cars: data.cars })); break;
          case "sources": patch(aKey, (m) => ({ ...m, sources: data.sources })); break;
          case "token": patch(aKey, (m) => ({ ...m, content: m.content + data.text })); break;
          case "followups": patch(aKey, (m) => ({ ...m, followups: data.followups })); break;
          case "error": patch(aKey, (m) => ({ ...m, error: data.message })); break;
          case "done": patch(aKey, (m) => ({ ...m, id: data.message_id, streaming: false })); break;
        }
      }, ctrl.signal);
    } catch (e) {
      const aborted = (e as Error).name === "AbortError";
      patch(aKey, (m) => ({ ...m, error: aborted ? undefined : (e as Error).message || "Network error", content: m.content + (aborted ? "\n\n_(stopped)_" : "") }));
    } finally {
      patch(aKey, (m) => ({ ...m, streaming: false }));
      setStreaming(false);
      abortRef.current = null;
      refreshConversations();
    }
  };

  const newChat = () => {
    abortRef.current?.abort();
    setMessages([]); setConvId(null); setTrace(null); setSidebar(false);
    localStorage.removeItem("cw_conv_id");
  };
  const openConversation = async (id: string) => {
    abortRef.current?.abort();
    setSidebar(false); setTrace(null);
    setConvId(id); localStorage.setItem("cw_conv_id", id);
    setMessages(await api.getConversation(id, userId).catch(() => []));
  };
  const removeConversation = async (id: string) => {
    await api.deleteConversation(id, userId);
    if (id === convId) newChat();
    refreshConversations();
  };
  const feedback = (m: ChatMessage, rating: number) => {
    if (!m.id) return;
    api.sendFeedback(m.id, rating);
    patch(m.key, (x) => ({ ...x, feedback: rating }));
  };
  const lastTrace = [...messages].reverse().find((m) => m.trace);

  return (
    <div className="h-full flex overflow-hidden">
      {/* sidebar */}
      <div className={`fixed inset-y-0 left-0 z-30 transform transition md:static md:translate-x-0 ${sidebar ? "translate-x-0" : "-translate-x-full"}`}>
        <Sidebar conversations={conversations} activeId={convId} onNew={newChat} onOpen={openConversation} onDelete={removeConversation}
                 onClose={() => setSidebar(false)} snapshot={stats.data_snapshot} stats={stats} />
      </div>
      {sidebar && <div className="fixed inset-0 z-20 bg-black/40 md:hidden" onClick={() => setSidebar(false)} />}

      {/* main */}
      <main className="flex-1 min-w-0 flex flex-col">
        <header className="flex items-center gap-2 px-3 sm:px-5 py-2.5 border-b border-slate-200 bg-white">
          <button onClick={() => setSidebar(true)} className="md:hidden p-2 rounded-lg hover:bg-slate-100" aria-label="Open menu"><MenuIcon /></button>
          <div className="min-w-0">
            <h1 className="font-semibold text-slate-900 truncate">Car advisor</h1>
            <p className="text-[11px] text-slate-500 truncate">Prices · mileage · safety · features · comparisons — grounded in CarWale data</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className={`hidden sm:inline-flex items-center gap-1.5 text-[11px] rounded-full px-2 py-0.5 ${server === "ready" ? "bg-emerald-50 text-emerald-700" : server === "down" ? "bg-rose-50 text-rose-700" : "bg-amber-50 text-amber-800"}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${server === "ready" ? "bg-emerald-500" : server === "down" ? "bg-rose-500" : "bg-amber-500 animate-pulse"}`} />
              {server === "ready" ? "Online" : server === "down" ? "Server unavailable" : "Waking server…"}
            </span>
            {lastTrace && (
              <button onClick={() => setTrace(trace ? null : { key: lastTrace.key, trace: lastTrace.trace! })}
                      className="flex items-center gap-1 text-xs rounded-lg border border-slate-200 px-2.5 py-1.5 hover:bg-slate-50">
                <FlowIcon className="w-3.5 h-3.5" /> <span className="hidden sm:inline">RAG trace</span>
              </button>
            )}
            <button onClick={newChat} className="md:hidden p-2 rounded-lg hover:bg-slate-100" aria-label="New chat"><PlusIcon /></button>
          </div>
        </header>

        {server === "waking" && (
          <div className="bg-amber-50 border-b border-amber-100 text-amber-900 text-xs px-4 py-2 text-center">
            Waking up the server (free hosting sleeps when idle) — this can take up to a minute…
          </div>
        )}

        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <div className="max-w-3xl mx-auto px-3 sm:px-6 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="pt-8 sm:pt-16 text-center">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-gradient-to-br from-brand-500 to-accent-500 text-white flex items-center justify-center shadow-lg">
                  <SparkIcon className="w-7 h-7" />
                </div>
                <h2 className="mt-4 text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">Ask anything about new cars in India</h2>
                <p className="mt-2 text-slate-500 text-sm max-w-lg mx-auto">
                  I search {stats.models ? `${stats.models} models and ${stats.variants?.toLocaleString()} variants` : "CarWale's catalog"} — prices, specs,
                  mileage, safety, features, owner reviews — and remember our conversation for follow-ups.
                </p>
                <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto">
                  {STARTERS.map((q) => (
                    <button key={q} onClick={() => send(q)} disabled={server !== "ready"}
                            className="text-left text-sm rounded-xl border border-slate-200 bg-white hover:border-brand-500 hover:shadow-sm px-3.5 py-2.5 disabled:opacity-50 transition">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) => (
              <MessageView key={m.key} m={m} onFollowup={send} onFeedback={feedback}
                           traceOpen={trace?.key === m.key}
                           onTrace={(x) => setTrace(trace?.key === x.key ? null : { key: x.key, trace: x.trace! })} />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <Composer onSend={send} onStop={() => abortRef.current?.abort()} streaming={streaming} disabled={server !== "ready"} />
      </main>

      {/* trace drawer */}
      {trace && (
        <div className="fixed inset-y-0 right-0 z-40 w-full sm:w-[420px] lg:static lg:w-[400px] shadow-xl lg:shadow-none">
          <TracePanel trace={messages.find((m) => m.key === trace.key)?.trace || trace.trace} onClose={() => setTrace(null)} />
        </div>
      )}
    </div>
  );
}
