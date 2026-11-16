import { useState } from "react";
import type { ChatMessage } from "../types";
import CarCards from "./CarCards";
import Markdown from "./Markdown";
import { CopyIcon, DownIcon, FlowIcon, SparkIcon, UpIcon } from "./Icons";

type Props = {
  m: ChatMessage;
  onFollowup: (q: string) => void;
  onTrace: (m: ChatMessage) => void;
  onFeedback: (m: ChatMessage, rating: number) => void;
  traceOpen: boolean;
};

export default function MessageView({ m, onFollowup, onTrace, onFeedback, traceOpen }: Props) {
  const [copied, setCopied] = useState(false);
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 text-white px-4 py-2.5 text-[15px] whitespace-pre-wrap shadow-sm">{m.content}</div>
      </div>
    );
  }
  const showStatus = m.streaming && !m.content;
  return (
    <div className="flex gap-3">
      <div className="shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-accent-500 text-white flex items-center justify-center shadow-sm">
        <SparkIcon className="w-4 h-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md bg-white border border-slate-200 px-4 py-3 shadow-sm">
          {showStatus && (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <span className="flex gap-0.5"><span className="dot">●</span><span className="dot">●</span><span className="dot">●</span></span>
              {m.status || "Thinking…"}
            </div>
          )}
          {m.content && <Markdown text={m.content} sources={m.sources} />}
          {m.streaming && m.content && <span className="inline-block w-2 h-4 bg-brand-500 animate-pulse align-middle ml-0.5" />}
          {m.error && <div className="mt-2 text-sm text-rose-700 bg-rose-50 border border-rose-100 rounded-lg px-3 py-2">{m.error}</div>}
          {m.cars && m.cars.length > 0 && <CarCards cars={m.cars} />}
          {!m.streaming && m.sources && m.sources.length > 0 && (
            <div className="mt-3 pt-2 border-t border-slate-100">
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1.5">Sources (CarWale)</div>
              <div className="flex flex-wrap gap-1.5">
                {m.sources.map((s) => (
                  <a key={s.n} href={s.url || "#"} target="_blank" rel="noreferrer"
                     className="text-[11px] rounded-md border border-slate-200 bg-slate-50 hover:bg-brand-50 hover:border-brand-100 px-1.5 py-0.5 text-slate-600">
                    <b className="text-brand-700">{s.n}</b> {s.car} · {s.section}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
        {!m.streaming && (
          <div className="flex flex-wrap items-center gap-1 mt-1.5 text-slate-400">
            {m.trace && (
              <button onClick={() => onTrace(m)}
                      className={`flex items-center gap-1 text-xs rounded-md px-2 py-1 hover:bg-slate-100 hover:text-brand-700 ${traceOpen ? "text-brand-700 bg-brand-50" : ""}`}>
                <FlowIcon className="w-3.5 h-3.5" /> How I answered this
              </button>
            )}
            {m.content && (
              <button onClick={() => { navigator.clipboard.writeText(m.content); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
                      className="flex items-center gap-1 text-xs rounded-md px-2 py-1 hover:bg-slate-100 hover:text-slate-600">
                <CopyIcon className="w-3.5 h-3.5" /> {copied ? "Copied" : "Copy"}
              </button>
            )}
            {m.id && (
              <>
                <button aria-label="Helpful" onClick={() => onFeedback(m, 1)} className={`p-1 rounded-md hover:bg-slate-100 ${m.feedback === 1 ? "text-emerald-600" : ""}`}><UpIcon className="w-3.5 h-3.5" /></button>
                <button aria-label="Not helpful" onClick={() => onFeedback(m, -1)} className={`p-1 rounded-md hover:bg-slate-100 ${m.feedback === -1 ? "text-rose-600" : ""}`}><DownIcon className="w-3.5 h-3.5" /></button>
              </>
            )}
            {m.cached && <span className="text-[11px] text-emerald-600 ml-1">⚡ cached</span>}
          </div>
        )}
        {!m.streaming && m.followups && m.followups.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {m.followups.map((q) => (
              <button key={q} onClick={() => onFollowup(q)}
                      className="text-[13px] rounded-full border border-brand-100 bg-brand-50 text-brand-700 hover:bg-brand-100 px-3 py-1 transition">
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
