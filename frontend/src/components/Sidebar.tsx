import type { Conversation } from "../types";
import { PlusIcon, TrashIcon, XIcon } from "./Icons";

type Props = {
  conversations: Conversation[];
  activeId: string | null;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
  snapshot?: string;
  stats?: { models?: number; variants?: number };
};

function ago(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function Sidebar({ conversations, activeId, onNew, onOpen, onDelete, onClose, snapshot, stats }: Props) {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-slate-200 w-72">
      <div className="flex items-center justify-between px-4 pt-4 pb-3">
        <div>
          <div className="text-white font-bold tracking-tight">CarWale <span className="text-accent-500">AI</span></div>
          <div className="text-[11px] text-slate-400">RAG assistant for new cars</div>
        </div>
        <button onClick={onClose} className="md:hidden p-1.5 rounded-lg hover:bg-slate-800" aria-label="Close menu"><XIcon /></button>
      </div>
      <div className="px-3">
        <button onClick={onNew} className="w-full flex items-center gap-2 rounded-lg border border-slate-700 hover:bg-slate-800 px-3 py-2 text-sm">
          <PlusIcon /> New chat
        </button>
      </div>
      <div className="px-4 pt-4 pb-1 text-[11px] uppercase tracking-wider text-slate-500">Recent chats</div>
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-2">
        {conversations.length === 0 && <div className="px-2 py-3 text-xs text-slate-500">Your conversations appear here and are saved across visits.</div>}
        {conversations.map((c) => (
          <div key={c.id} className={`group flex items-center rounded-lg pr-2 text-sm ${c.id === activeId ? "bg-slate-800 text-white" : "hover:bg-slate-800/60"}`}>
            {/* a real button: rows are reachable with Tab and open with Enter / Space */}
            <button type="button" onClick={() => onOpen(c.id)} aria-current={c.id === activeId ? "true" : undefined}
                    className="flex-1 min-w-0 text-left px-2 py-2 cursor-pointer rounded-lg">
              <div className="truncate">{c.title || "New chat"}</div>
              <div className="text-[11px] text-slate-500">{ago(c.updated_at)} · {c.n_messages} msgs</div>
            </button>
            <button type="button" onClick={() => onDelete(c.id)} aria-label="Delete chat"
                    className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-rose-300">
              <TrashIcon className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
      <div className="px-4 py-3 border-t border-slate-800 text-[11px] text-slate-500 leading-relaxed">
        {stats?.models ? <div>{stats.models} models · {stats.variants?.toLocaleString()} variants indexed</div> : null}
        <div>Data: CarWale.com{snapshot ? ` · snapshot ${snapshot}` : ""}</div>
        <div>LLM: Groq · Retrieval: pgvector + full-text (RRF)</div>
      </div>
    </div>
  );
}
