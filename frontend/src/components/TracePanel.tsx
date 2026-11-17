import { useState } from "react";
import type { Trace } from "../types";
import { XIcon } from "./Icons";

const Badge = ({ children, tone = "slate" }: { children: React.ReactNode; tone?: string }) => {
  const tones: Record<string, string> = {
    slate: "bg-slate-100 text-slate-700", blue: "bg-brand-50 text-brand-700 border border-brand-100",
    green: "bg-emerald-50 text-emerald-700 border border-emerald-100", amber: "bg-amber-50 text-amber-800 border border-amber-100",
  };
  return <span className={`inline-block rounded-md px-1.5 py-0.5 text-[11px] font-medium mr-1 mb-1 ${tones[tone]}`}>{children}</span>;
};

function Step({ n, title, meta, children }: { n: number; title: string; meta?: string; children: React.ReactNode }) {
  return (
    <div className="relative pl-8 pb-5">
      <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-brand-600 text-white text-xs font-bold flex items-center justify-center">{n}</div>
      <div className="absolute left-[11px] top-6 bottom-0 w-px bg-slate-200" />
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        {meta && <span className="text-[11px] text-slate-400 whitespace-nowrap">{meta}</span>}
      </div>
      <div className="mt-1.5 text-xs text-slate-600 space-y-1">{children}</div>
    </div>
  );
}

const clean = (o: Record<string, unknown> | undefined) =>
  Object.entries(o || {}).filter(([, v]) => v !== null && v !== "launched" && !(Array.isArray(v) && v.length === 0));

export default function TracePanel({ trace, onClose }: { trace: Trace; onClose: () => void }) {
  const [raw, setRaw] = useState(false);
  const plan = trace.plan || {};
  const llm: Trace[] = trace.llm || [];
  const planner = llm.find((x) => x.role === "planner");
  const answer = llm.find((x) => x.role === "answer");
  const t = trace.timings_ms || {};
  const mem = trace.memory || {};
  const fmtLakh = (k: string, v: unknown) => (typeof v === "number" && k.startsWith("price") ? `₹${(v / 1e5).toFixed(2)} L` : Array.isArray(v) ? v.join(", ") : String(v));
  return (
    <aside className="h-full flex flex-col bg-white border-l border-slate-200 w-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div>
          <h3 className="font-semibold text-slate-900 text-sm">How this answer was built</h3>
          <p className="text-[11px] text-slate-500">RAG pipeline trace for this message</p>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-100" aria-label="Close trace"><XIcon /></button>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        {trace.cache && <div className="mb-4 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-800 text-xs p-2">⚡ {trace.cache}</div>}

        <Step n={1} title="Load memory" meta={`${mem.history_messages ?? 0} past messages`}>
          {mem.focus_before && <div>Focus car: <Badge>{mem.focus_before}</Badge></div>}
          {mem.active_cars_before?.length > 0 && (
            <div>Active cars: {mem.active_cars_before.map((c: string, i: number) => <Badge key={c}>{i + 1}. {c}</Badge>)}</div>
          )}
          {clean(mem.filters_before).length > 0 && (
            <div>Previous filters: {clean(mem.filters_before).map(([k, v]) => <Badge key={k}>{k}: {fmtLakh(k, v)}</Badge>)}</div>
          )}
          {mem.summary_used && <div><Badge tone="amber">rolling summary used</Badge></div>}
          {!mem.history_messages && <div className="text-slate-400">New conversation — no history yet.</div>}
        </Step>

        <Step n={2} title="Plan the query (LLM #1)" meta={planner ? `${planner.model} · ${planner.latency_ms} ms · ${planner.prompt_tokens + planner.completion_tokens} tok` : "rule-based"}>
          <div>Intent: <Badge tone="blue">{trace.intent}</Badge> {plan.refers_to && plan.refers_to !== "none" && <Badge tone="amber">refers to: {plan.refers_to} {plan.active_indices?.join(",")}</Badge>}</div>
          <div>Rewritten query: <span className="italic text-slate-800">“{trace.rewritten_query}”</span></div>
          {clean(plan.filters).length > 0 && (
            <div>Filters: {clean(plan.filters).map(([k, v]) => <Badge key={k} tone="green">{k}: {fmtLakh(k, v)}</Badge>)} {plan.filter_mode !== "none" && <Badge>{plan.filter_mode}</Badge>}</div>
          )}
          {plan.prefer_tags?.length > 0 && <div>Soft preferences: {plan.prefer_tags.map((x: string) => <Badge key={x}>{x}</Badge>)}</div>}
          {plan.sort_by && plan.sort_by !== "relevance" && <div>Sort: <Badge>{plan.sort_by}</Badge></div>}
          {plan.aspects?.length > 0 && <div>Aspects: {plan.aspects.map((x: string) => <Badge key={x}>{x}</Badge>)}</div>}
          {(planner?.fallbacks?.length ?? 0) > 1 && <div className="text-amber-700">Fallback chain used: {planner!.fallbacks.join(" → ")}</div>}
        </Step>

        <Step n={3} title="Resolve car names">
          {trace.resolved_cars?.length ? trace.resolved_cars.map((r: Trace) => (
            <div key={r.mention}>“{r.mention}” → <Badge tone="blue">{r.name}</Badge><span className="text-slate-400">score {r.score}</span>
              {r.ambiguous?.length > 0 && <span className="text-amber-700"> (also: {r.ambiguous.join(", ")})</span>}</div>
          )) : <div className="text-slate-400">No car names in this message{plan.refers_to !== "none" ? " — resolved from memory" : ""}.</div>}
          {trace.unresolved?.length > 0 && <div className="text-rose-600">Not in catalog: {trace.unresolved.join(", ")}</div>}
          {trace.notes?.map((n: string) => <div key={n} className="text-amber-700">{n}</div>)}
        </Step>

        <Step n={4} title="Retrieve" meta={t.retrieve ? `${t.retrieve - (t.plan || 0)} ms` : undefined}>
          <div>Route: <Badge tone="blue">{trace.route}</Badge></div>
          {trace.sql_filters?.length > 0 && <div>SQL filters: {trace.sql_filters.map((x: string) => <Badge key={x} tone="green">{x}</Badge>)}</div>}
          {trace.relaxed?.length > 0 && <div className="text-amber-700">Relaxed: {trace.relaxed.join("; ")}</div>}
          {trace.retrieved?.length > 0 && (
            <table className="w-full mt-1 text-[11px] border border-slate-200">
              <thead><tr className="bg-slate-50 text-slate-500"><th className="p-1 text-left">#</th><th className="p-1 text-left">Chunk</th><th className="p-1 text-right">RRF</th><th className="p-1 text-left">Matched by</th></tr></thead>
              <tbody>
                {trace.retrieved.map((s: Trace) => (
                  <tr key={s.n} className="border-t border-slate-100">
                    <td className="p-1 font-semibold">{s.n}</td>
                    <td className="p-1"><a href={s.url} target="_blank" rel="noreferrer" className="hover:text-brand-600">{s.car}</a><div className="text-slate-400">{s.section}</div></td>
                    <td className="p-1 text-right tabular-nums">{s.score?.toFixed(4)}</td>
                    <td className="p-1 text-slate-500">{(s.via || []).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {trace.context_tokens_est > 0 && <div className="text-slate-400">Context ≈ {trace.context_tokens_est} tokens</div>}
        </Step>

        <Step n={5} title="Generate answer (LLM #2, streamed)" meta={answer ? `${answer.model} · ${answer.prompt_tokens + answer.completion_tokens} tok` : undefined}>
          {t.first_token > 0 && <div>Time to first token: <Badge>{t.first_token} ms</Badge> Total: <Badge>{t.total} ms</Badge></div>}
          {(answer?.fallbacks?.length ?? 0) > 1 && <div className="text-amber-700">Fallback chain used: {answer!.fallbacks.join(" → ")}</div>}
          <div className="text-slate-400">Grounded on CAR DATA + numbered sources; memory updated afterwards.</div>
        </Step>

        <button onClick={() => setRaw(!raw)} className="text-xs text-brand-600 hover:underline">{raw ? "Hide" : "Show"} raw trace JSON</button>
        {raw && <pre className="mt-2 text-[10px] bg-slate-900 text-slate-100 rounded-lg p-3 overflow-x-auto scrollbar-thin">{JSON.stringify(trace, null, 2)}</pre>}
      </div>
    </aside>
  );
}
