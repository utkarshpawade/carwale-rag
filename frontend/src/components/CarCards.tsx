import type { Car } from "../types";
import { ExternalIcon } from "./Icons";

export default function CarCards({ cars }: { cars: Car[] }) {
  if (!cars?.length) return null;
  return (
    <div className="flex gap-3 overflow-x-auto scrollbar-thin pb-2 mt-3 -mx-1 px-1">
      {cars.map((c) => (
        <a key={c.id} href={c.url} target="_blank" rel="noreferrer"
           className="group shrink-0 w-48 rounded-xl border border-slate-200 bg-white hover:border-brand-500 hover:shadow-md transition overflow-hidden">
          <div className="h-24 bg-gradient-to-br from-slate-100 to-slate-50 flex items-center justify-center overflow-hidden">
            {c.image ? (
              <img src={c.image} alt={c.name} loading="lazy" referrerPolicy="no-referrer" className="h-full w-full object-contain group-hover:scale-105 transition" />
            ) : (
              <span className="text-3xl">🚗</span>
            )}
          </div>
          <div className="p-2.5">
            <div className="text-[13px] font-semibold leading-tight text-slate-900 line-clamp-2">{c.name}</div>
            <div className="text-xs text-brand-700 font-medium mt-1">{c.price}</div>
            <div className="flex items-center justify-between mt-1.5 text-[11px] text-slate-500">
              <span>{c.body || ""}</span>
              <span className="flex items-center gap-1">
                {c.ncap ? <span title="NCAP rating">{c.ncap}★</span> : null}
                {c.rating ? <span title="Owner rating">· {c.rating.toFixed(1)}/5</span> : null}
              </span>
            </div>
            {c.status && c.status !== "launched" && (
              <span className="inline-block mt-1 text-[10px] uppercase tracking-wide bg-amber-100 text-amber-800 rounded px-1.5 py-0.5">{c.status}</span>
            )}
            <div className="mt-1.5 text-[11px] text-slate-400 group-hover:text-brand-600 flex items-center gap-1">
              View on CarWale <ExternalIcon className="w-3 h-3" />
            </div>
          </div>
        </a>
      ))}
    </div>
  );
}
