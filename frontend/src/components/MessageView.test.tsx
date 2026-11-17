import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { captureUnhandledRejections, settle } from "../test/unhandled";
import type { ChatMessage } from "../types";
import MessageView from "./MessageView";

const assistant: ChatMessage = {
  key: "a1", id: 42, role: "assistant", content: "**Creta** is a good pick [1].",
  sources: [{ n: 1, car: "Hyundai Creta", section: "Overview", url: "https://www.carwale.com/hyundai-cars/creta/" },
            { n: 2, car: "Glossary", section: "ADAS", url: null }],
  followups: ["Is the Creta safe?", "Mileage of the Creta"],
  cars: [{ id: "hyundai/creta", name: "Hyundai Creta", url: "https://www.carwale.com/hyundai-cars/creta/", price: "₹11.00 L" }],
  trace: { intent: "review" },
};

const setup = (m: ChatMessage, traceOpen = false) => {
  const props = { onFollowup: vi.fn(), onTrace: vi.fn(), onFeedback: vi.fn() };
  render(<MessageView m={m} traceOpen={traceOpen} {...props} />);
  return { ...props, user: userEvent.setup() };
};

afterEach(() => vi.useRealTimers());

describe("MessageView", () => {
  it("renders user text literally (no markdown, no HTML)", () => {
    const { container } = render(<MessageView m={{ key: "u", role: "user", content: '**bold** <img src=x onerror="alert(1)">' }}
                                              onFollowup={vi.fn()} onTrace={vi.fn()} onFeedback={vi.fn()} traceOpen={false} />);
    expect(screen.getByText('**bold** <img src=x onerror="alert(1)">')).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("shows the live status before the first token, and hides actions while streaming", () => {
    setup({ key: "s", role: "assistant", content: "", streaming: true, status: "Searching 2,202 variants…" });
    expect(screen.getByText("Searching 2,202 variants…")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Helpful" })).toBeNull();
    expect(screen.queryByText("How I answered this")).toBeNull();
  });

  it("hides sources and follow-ups until the answer is complete", () => {
    setup({ ...assistant, streaming: true });
    expect(screen.queryByText("Sources (CarWale)")).toBeNull();
    expect(screen.queryByText("Is the Creta safe?")).toBeNull();
  });

  it("shows sources, cards, follow-ups, trace and feedback when done", async () => {
    const { onFollowup, onTrace, onFeedback, user } = setup(assistant);
    expect(screen.getByText("Sources (CarWale)")).toBeInTheDocument();
    const srcLinks = screen.getAllByRole("link").filter((a) => a.textContent?.includes("Overview"));
    expect(srcLinks[0]).toHaveAttribute("href", "https://www.carwale.com/hyundai-cars/creta/");
    expect(screen.getByText(/Glossary · ADAS/).closest("a")).toHaveAttribute("href", "#");
    expect(screen.getByText("Hyundai Creta", { selector: "div" })).toBeInTheDocument(); // car card
    await user.click(screen.getByText("Is the Creta safe?"));
    expect(onFollowup).toHaveBeenCalledWith("Is the Creta safe?");
    await user.click(screen.getByText("How I answered this"));
    expect(onTrace).toHaveBeenCalledWith(assistant);
    await user.click(screen.getByRole("button", { name: "Helpful" }));
    await user.click(screen.getByRole("button", { name: "Not helpful" }));
    expect(onFeedback.mock.calls).toEqual([[assistant, 1], [assistant, -1]]);
  });

  it("only offers feedback once the server has given the message an id", () => {
    setup({ ...assistant, id: undefined });
    expect(screen.queryByRole("button", { name: "Helpful" })).toBeNull();
  });

  it("shows errors and the cached badge", () => {
    setup({ ...assistant, error: "The AI service is busy", cached: true });
    expect(screen.getByText("The AI service is busy")).toBeInTheDocument();
    expect(screen.getByText("⚡ cached")).toBeInTheDocument();
  });

  it("copies the answer and confirms", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { user } = setup(assistant);
    const writeText = vi.fn().mockResolvedValue(undefined);  // after userEvent.setup(), which installs its own stub
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    await user.click(screen.getByRole("button", { name: /Copy/ }));
    expect(writeText).toHaveBeenCalledWith(assistant.content);
    expect(await screen.findByText("Copied")).toBeInTheDocument();
    await act(async () => { vi.advanceTimersByTime(1300); });
    expect(screen.queryByText("Copied")).toBeNull();
  });

  it("does not claim 'Copied' when the clipboard refuses", async () => {
    // B-17: the button said "Copied" even when writeText() was rejected, and the rejection went unhandled
    const { user } = setup(assistant);
    const writeText = vi.fn().mockRejectedValue(new DOMException("denied", "NotAllowedError"));
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    const unhandled = captureUnhandledRejections();
    await user.click(screen.getByRole("button", { name: /Copy/ }));
    expect(writeText).toHaveBeenCalled();
    await settle();
    unhandled.stop();
    expect(unhandled.reasons).toEqual([]);
    expect(screen.queryByText("Copied")).toBeNull();
  });
});
