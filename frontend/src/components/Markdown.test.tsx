import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Source } from "../types";
import Markdown from "./Markdown";

const sources: Source[] = [
  { n: 1, car: "Hyundai Creta", section: "Mileage", url: "https://www.carwale.com/hyundai-cars/creta/" },
  { n: 4, car: "Kia Seltos", section: "Safety", url: "https://www.carwale.com/kia-cars/seltos/" },
];

describe("Markdown answer renderer", () => {
  it("turns [n] into a citation link to that source", () => {
    render(<Markdown text="Claimed 17.4 kmpl [1]." sources={sources} />);
    const cite = screen.getByRole("link", { name: "1" });
    expect(cite).toHaveClass("cite");
    expect(cite).toHaveAttribute("href", sources[0].url);
    expect(cite).toHaveAttribute("title", "Hyundai Creta — Mileage");
    expect(cite).toHaveAttribute("target", "_blank");
    expect(cite).toHaveAttribute("rel", "noreferrer");
  });

  it("links an unknown citation number to '#'", () => {
    render(<Markdown text="Fact [3]" sources={sources} />);
    expect(screen.getByRole("link", { name: "3" })).toHaveAttribute("href", "#");
    expect(screen.getByRole("link", { name: "3" })).toHaveAttribute("title", "Source 3");
  });

  it("normalises gpt-oss native citations and drops pseudo-citations", () => {
    const { container } = render(<Markdown text="Safe 【4†L1-L4】 car 【CAR DATA】 and [CAR DATA] or [data]." sources={sources} />);
    expect(screen.getByRole("link", { name: "4" })).toHaveAttribute("href", sources[1].url);
    expect(container.textContent).not.toMatch(/CAR DATA|【|\[data\]/);
  });

  it("keeps real markdown links and opens them in a new tab", () => {
    render(<Markdown text="See [12](https://example.com/x) and [CarWale](https://www.carwale.com)" />);
    expect(screen.getByRole("link", { name: "12" })).toHaveAttribute("href", "https://example.com/x");
    expect(screen.getByRole("link", { name: "CarWale" })).toHaveAttribute("target", "_blank");
  });

  it("never renders a javascript: URL", () => {
    const { container } = render(<Markdown text="[click me](javascript:alert(1)) [x](JAVASCRIPT:alert(2))" />);
    const anchors = container.querySelectorAll("a");
    expect(anchors).toHaveLength(2);
    for (const a of anchors) {
      expect((a.getAttribute("href") || "").toLowerCase()).not.toContain("javascript:");
    }
  });

  it("does not render raw HTML from the model output", () => {
    const { container } = render(<Markdown text={'Hello <img src=x onerror="window.__xss=1"> <script>window.__xss=2</script> <b>bold</b>'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect((window as unknown as { __xss?: number }).__xss).toBeUndefined();
  });

  it("renders GFM tables inside a horizontal scroller", () => {
    const { container } = render(<Markdown text={"| Car | Price |\n|---|---|\n| Creta | ₹11 L |"} />);
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    expect(table?.parentElement).toHaveClass("overflow-x-auto");
    expect(screen.getByRole("cell", { name: "₹11 L" })).toBeInTheDocument();
  });

  it("handles empty and very long text", () => {
    const { container } = render(<Markdown text="" />);
    expect(container.textContent).toBe("");
    render(<Markdown text={"word ".repeat(20000)} />);
    expect(screen.getByText(/word word/)).toBeInTheDocument();
  });
});
