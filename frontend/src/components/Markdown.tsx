import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../types";

/** Renders the answer; turns [n] citations into small links to the CarWale source page. */
export default function Markdown({ text, sources }: { text: string; sources?: Source[] }) {
  const linked = text
    .replace(/【(\d{1,2})(?:†[^】]*)?】/g, "[$1]") // gpt-oss sometimes emits its native 【4†L1-L4】 citation style
    .replace(/\[(\d{1,2})\](?!\()/g, (_m, n) => `[${n}](#cite-${n})`);
  return (
    <div className="answer prose prose-slate prose-sm max-w-none prose-p:my-2 prose-li:my-0.5 prose-ul:my-2 prose-table:my-3 prose-headings:mt-3 prose-headings:mb-1.5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            if (href?.startsWith("#cite-")) {
              const n = Number(href.slice(6));
              const s = sources?.find((x) => x.n === n);
              return (
                <a className="cite" href={s?.url || "#"} target="_blank" rel="noreferrer" title={s ? `${s.car} — ${s.section}` : `Source ${n}`}>
                  {n}
                </a>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
          table({ children }) {
            return <div className="overflow-x-auto scrollbar-thin -mx-1 px-1"><table>{children}</table></div>;
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}
