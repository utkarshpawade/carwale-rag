import { useEffect, useRef, useState } from "react";
import { SendIcon, StopIcon } from "./Icons";

type Props = { onSend: (text: string) => void; onStop: () => void; streaming: boolean; disabled: boolean };

export default function Composer({ onSend, onStop, streaming, disabled }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [text]);
  const submit = () => {
    const t = text.trim();
    if (!t || streaming || disabled) return;
    onSend(t);
    setText("");
  };
  return (
    <div className="border-t border-slate-200 bg-white/90 backdrop-blur px-3 sm:px-6 py-3">
      <div className="max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border border-slate-300 bg-white px-3 py-2 shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100">
        <textarea
          ref={ref} rows={1} value={text} maxLength={1000}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          placeholder={disabled ? "Connecting to the server…" : "Ask about prices, mileage, safety, features, comparisons…"}
          className="flex-1 resize-none outline-none text-[15px] py-1.5 max-h-40 bg-transparent placeholder:text-slate-400"
        />
        {streaming ? (
          <button onClick={onStop} className="shrink-0 rounded-xl bg-slate-800 text-white p-2.5 hover:bg-slate-700" aria-label="Stop"><StopIcon /></button>
        ) : (
          <button onClick={submit} disabled={!text.trim() || disabled}
                  className="shrink-0 rounded-xl bg-brand-600 text-white p-2.5 hover:bg-brand-700 disabled:opacity-40" aria-label="Send"><SendIcon /></button>
        )}
      </div>
      <p className="max-w-3xl mx-auto text-center text-[11px] text-slate-400 mt-1.5">
        Answers are generated from CarWale's catalog and may contain mistakes. Prices are ex-showroom unless stated.
      </p>
    </div>
  );
}
