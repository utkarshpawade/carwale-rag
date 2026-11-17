import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import TracePanel from "./TracePanel";

const fullTrace = {
  intent: "recommend", rewritten_query: "smart hybrid cars under 30 lakh", route: "structured_search",
  plan: { refers_to: "none", filters: { price_max: 3000000, fuel_types: ["Hybrid"], status: "launched", body_types: [] },
          filter_mode: "new", prefer_tags: ["adas"], sort_by: "mileage", aspects: ["mileage"] },
  memory: { history_messages: 2, focus_before: "Hyundai Creta", active_cars_before: ["Hyundai Creta", "Kia Seltos"],
            filters_before: { seats_min: 7 }, summary_used: true },
  resolved_cars: [{ mention: "creta", name: "Hyundai Creta", score: 100, ambiguous: ["hyundai/creta-n-line"] }],
  unresolved: ["Batmobile"], notes: ["Backstop: fuel filter taken from the message text."],
  sql_filters: ["price <= 30.00 L"], relaxed: ["budget stretched"],
  retrieved: [{ n: 1, car: "Hyundai Creta", section: "Overview", url: "https://u/1", score: 0.0333, via: ["vector#1", "keyword#2"] }],
  context_tokens_est: 1800, timings_ms: { plan: 300, retrieve: 900, first_token: 1200, total: 2500 },
  llm: [{ role: "planner", model: "qwen/qwen3.8-27b", latency_ms: 300, prompt_tokens: 1700, completion_tokens: 200,
          fallbacks: ["a:rate_limited", "b:timeout"] },
        { role: "answer", model: "openai/gpt-oss-120b", prompt_tokens: 2800, completion_tokens: 700, fallbacks: [] }],
  cache: "hit (answer served from cache, 0 LLM tokens)",
};

describe("TracePanel", () => {
  it("renders an empty trace without crashing", () => {
    render(<TracePanel trace={{}} onClose={vi.fn()} />);
    expect(screen.getByText("New conversation — no history yet.")).toBeInTheDocument();
    expect(screen.getByText("rule-based")).toBeInTheDocument();
  });

  it("renders every pipeline step of a full trace", () => {
    render(<TracePanel trace={fullTrace} onClose={vi.fn()} />);
    expect(screen.getByText(/hit \(answer served from cache/)).toBeInTheDocument();
    expect(screen.getByText("2 past messages")).toBeInTheDocument();
    expect(screen.getByText("2. Kia Seltos")).toBeInTheDocument();
    expect(screen.getByText("price_max: ₹30.00 L")).toBeInTheDocument();
    expect(screen.getByText("fuel_types: Hybrid")).toBeInTheDocument();
    expect(screen.queryByText(/status:/)).toBeNull(); // the 'launched' default is hidden
    expect(screen.getByText("qwen/qwen3.8-27b · 300 ms · 1900 tok")).toBeInTheDocument();
    expect(screen.getByText(/Fallback chain used: a:rate_limited → b:timeout/)).toBeInTheDocument();
    expect(screen.getByText("Not in catalog: Batmobile")).toBeInTheDocument();
    expect(screen.getByText("600 ms")).toBeInTheDocument(); // retrieve - plan
    expect(screen.getByText("0.0333")).toBeInTheDocument();
    expect(screen.getByText("Context ≈ 1800 tokens")).toBeInTheDocument();
    expect(screen.getByText("1200 ms")).toBeInTheDocument();
  });

  it("toggles the raw JSON and closes", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<TracePanel trace={fullTrace} onClose={onClose} />);
    await user.click(screen.getByText("Show raw trace JSON"));
    expect(screen.getByText(/"route": "structured_search"/)).toBeInTheDocument();
    await user.click(screen.getByText("Hide raw trace JSON"));
    expect(screen.queryByText(/"route": "structured_search"/)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Close trace" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("does not print a stray 0 when the context is empty", () => {
    // B-18: `{trace.context_tokens_est && ...}` renders the number 0 as text
    const { container } = render(<TracePanel trace={{ route: "semantic_search", context_tokens_est: 0 }} onClose={vi.fn()} />);
    const step4 = Array.from(container.querySelectorAll("h4")).find((h) => h.textContent === "Retrieve")!.closest("div.relative")!;
    expect(step4.textContent).not.toMatch(/semantic_search0|^0$|\b0\b(?! ms)/);
  });
});
