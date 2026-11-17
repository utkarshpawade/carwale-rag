import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => ({
  API: "http://api.test",
  getUserId: vi.fn(() => "test-user-123"),
  health: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
  sendFeedback: vi.fn(),
  streamChat: vi.fn(),
}));

import App from "./App";
import * as api from "./api";
import { captureUnhandledRejections, settle } from "./test/unhandled";

const m = vi.mocked(api);
type OnEvent = (event: string, data: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any

const HEALTH = { status: "ok", models: 288, variants: 2202, chunks: 30602, data_snapshot: "2026-11-17" };

/** A streamChat that replays a normal answer. */
const happyStream = (convId = "conv-1") => async (_b: unknown, on: OnEvent) => {
  on("conversation", { conversation_id: convId });
  on("status", { text: "Searching 2,202 variants…" });
  on("trace", { intent: "recommend", route: "structured_search" });
  on("cars", { cars: [{ id: "hyundai/creta", name: "Hyundai Creta", url: "https://u/creta", price: "₹11.00 L" }] });
  on("sources", { sources: [{ n: 1, car: "Hyundai Creta", section: "Overview", url: "https://u/creta" }] });
  on("token", { text: "The **Creta** " });
  on("token", { text: "fits [1]." });
  on("followups", { followups: ["Which one has the best mileage?"] });
  on("done", { message_id: 77, conversation_id: convId });
};

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  m.health.mockResolvedValue(HEALTH);
  m.listConversations.mockResolvedValue([]);
  m.getConversation.mockResolvedValue([]);
  m.deleteConversation.mockResolvedValue(new Response(null, { status: 200 }));
  m.sendFeedback.mockResolvedValue(new Response(null, { status: 200 }));
  m.streamChat.mockImplementation(happyStream());
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("App", () => {
  it("goes online after the health check and enables the starters", async () => {
    render(<App />);
    expect(await screen.findByText("Online")).toBeInTheDocument();
    expect(screen.getByText(/288 models and 2,202 variants/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" })).toBeEnabled();
    expect(m.listConversations).toHaveBeenCalledWith("test-user-123");
  });

  it("shows the wake-up banner and keeps retrying while the server sleeps", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    m.health.mockRejectedValueOnce(new Error("down")).mockRejectedValueOnce(new Error("down")).mockResolvedValue(HEALTH);
    render(<App />);
    expect(await screen.findByText(/Waking up the server/)).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toHaveAttribute("placeholder", "Connecting to the server…");
    expect(screen.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" })).toBeDisabled();
    await act(async () => { await vi.advanceTimersByTimeAsync(6500); });
    expect(await screen.findByText("Online")).toBeInTheDocument();
    expect(m.health).toHaveBeenCalledTimes(3);
    expect(screen.queryByText(/Waking up the server/)).toBeNull();
  });

  it("gives up after 150 s and says the server is unavailable", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    m.health.mockRejectedValue(new Error("down"));
    render(<App />);
    await act(async () => { await vi.advanceTimersByTimeAsync(155_000); });
    expect(screen.getByText("Server unavailable")).toBeInTheDocument();
    const calls = m.health.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(m.health.mock.calls.length).toBe(calls); // stopped retrying
  });

  it("streams an answer end to end and remembers the conversation", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("Online");
    await user.click(screen.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" }));
    expect(m.streamChat).toHaveBeenCalledWith({ user_id: "test-user-123", conversation_id: null, message: "Smart hybrid cars under 30 lakhs" },
      expect.any(Function), expect.any(AbortSignal));
    expect(await screen.findByText("Creta")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "1" })).toHaveAttribute("href", "https://u/creta");
    expect(screen.getByText("Which one has the best mileage?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Helpful" })).toBeInTheDocument();
    expect(localStorage.getItem("cw_conv_id")).toBe("conv-1");
    await waitFor(() => expect(m.listConversations).toHaveBeenCalledTimes(2)); // sidebar refreshed after the turn

    // follow-up chip continues the same conversation
    await user.click(screen.getByText("Which one has the best mileage?"));
    expect(m.streamChat.mock.calls[1][0]).toEqual({ user_id: "test-user-123", conversation_id: "conv-1", message: "Which one has the best mileage?" });
  });

  it("sends feedback for the finished answer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "creta vs seltos{Enter}");
    await user.click(await screen.findByRole("button", { name: "Helpful" }));
    expect(m.sendFeedback).toHaveBeenCalledWith(77, 1);
  });

  it("restores the saved conversation after a reload, and forgets it if it is gone", async () => {
    localStorage.setItem("cw_conv_id", "conv-9");
    m.getConversation.mockResolvedValueOnce([{ key: "m1", id: 1, role: "user", content: "Mileage of Swift" },
                                             { key: "m2", id: 2, role: "assistant", content: "About 25 kmpl." }]);
    const { unmount } = render(<App />);
    expect(await screen.findByText("About 25 kmpl.")).toBeInTheDocument();
    expect(m.getConversation).toHaveBeenCalledWith("conv-9", "test-user-123");
    unmount();

    m.getConversation.mockRejectedValueOnce(new Error("HTTP 404"));
    render(<App />);
    await screen.findByText("Online");
    await waitFor(() => expect(localStorage.getItem("cw_conv_id")).toBeNull());
  });

  it("shows server errors and network errors on the assistant bubble", async () => {
    const user = userEvent.setup();
    m.streamChat.mockImplementationOnce(async (_b, on: OnEvent) => {
      on("conversation", { conversation_id: "c2" });
      on("error", { message: "The AI service is at its free-tier limit right now." });
    });
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "first{Enter}");
    expect(await screen.findByText("The AI service is at its free-tier limit right now.")).toBeInTheDocument();

    m.streamChat.mockRejectedValueOnce(new Error("Too many requests. Please retry in 30 seconds."));
    await user.type(screen.getByRole("textbox"), "second{Enter}");
    expect(await screen.findByText("Too many requests. Please retry in 30 seconds.")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toBeEnabled();
  });

  it("stop keeps the partial answer and marks it stopped", async () => {
    const user = userEvent.setup();
    m.streamChat.mockImplementationOnce((_b, on: OnEvent, signal: AbortSignal) => new Promise((_res, rej) => {
      on("token", { text: "Partial answer" });
      signal.addEventListener("abort", () => rej(new DOMException("Aborted", "AbortError")));
    }));
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "long question{Enter}");
    await screen.findByText("Partial answer");
    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(await screen.findByText("(stopped)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
  });

  it("new chat clears the thread and the saved conversation", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "creta{Enter}");
    await screen.findByText("Creta");
    await user.click(screen.getAllByRole("button", { name: /New chat/ })[0]);
    expect(screen.queryByText("Creta")).toBeNull();
    expect(localStorage.getItem("cw_conv_id")).toBeNull();
    expect(screen.getByText("Ask anything about new cars in India")).toBeInTheDocument();
  });

  it("deleting the open chat resets the view", async () => {
    const user = userEvent.setup();
    m.listConversations.mockResolvedValue([{ id: "conv-1", title: "creta", updated_at: new Date().toISOString(), n_messages: 2 }]);
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "creta{Enter}");
    await screen.findByText("Creta");
    await user.click(screen.getByRole("button", { name: "Delete chat" }));
    expect(m.deleteConversation).toHaveBeenCalledWith("conv-1", "test-user-123");
    await waitFor(() => expect(screen.queryByText("Creta")).toBeNull());
  });

  it("flags an answer whose stream ended without a 'done' event", async () => {
    // B-23: a connection closed mid-answer (proxy timeout, deploy) left a truncated answer that looked complete
    const user = userEvent.setup();
    m.streamChat.mockImplementationOnce(async (_b, on: OnEvent) => {
      on("conversation", { conversation_id: "c3" });
      on("token", { text: "The Creta is" });
    });
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "creta{Enter}");
    expect(await screen.findByText("The Creta is")).toBeInTheDocument();
    expect(await screen.findByText(/interrupted/i)).toBeInTheDocument();
  });

  it("opening two past chats quickly shows the one clicked last", async () => {
    // B-24: responses arriving out of order replaced the newer chat's messages with the older one's
    const user = userEvent.setup();
    const now = new Date().toISOString();
    m.listConversations.mockResolvedValue([{ id: "slow", title: "Slow chat", updated_at: now, n_messages: 2 },
                                           { id: "fast", title: "Fast chat", updated_at: now, n_messages: 2 }]);
    let releaseSlow: (v: unknown) => void = () => undefined;
    m.getConversation.mockImplementation((id: string) => id === "slow"
      ? new Promise((res) => { releaseSlow = res; }).then(() => [{ key: "s1", id: 1, role: "assistant" as const, content: "SLOW ANSWER" }])
      : Promise.resolve([{ key: "f1", id: 2, role: "assistant" as const, content: "FAST ANSWER" }]));
    render(<App />);
    await screen.findByText("Online");
    await user.click(screen.getByText("Slow chat"));
    await user.click(screen.getByText("Fast chat"));
    expect(await screen.findByText("FAST ANSWER")).toBeInTheDocument();
    await act(async () => { releaseSlow(undefined); await settle(); });
    expect(screen.queryByText("SLOW ANSWER")).toBeNull();
    expect(screen.getByText("FAST ANSWER")).toBeInTheDocument();
  });

  it("does not leave unhandled promise rejections when delete or feedback fail", async () => {
    // B-19: deleteConversation / sendFeedback were fire-and-forget; a network error became an unhandled rejection
    const unhandled = captureUnhandledRejections();
    const user = userEvent.setup();
    m.listConversations.mockResolvedValue([{ id: "conv-1", title: "creta", updated_at: new Date().toISOString(), n_messages: 2 }]);
    m.sendFeedback.mockRejectedValue(new TypeError("Failed to fetch"));
    m.deleteConversation.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    await screen.findByText("Online");
    await user.type(screen.getByRole("textbox"), "creta{Enter}");
    await user.click(await screen.findByRole("button", { name: "Helpful" }));
    await user.click(screen.getByRole("button", { name: "Delete chat" }));
    await settle();
    unhandled.stop();
    expect(unhandled.reasons).toEqual([]);
    expect(screen.getByRole("button", { name: "Delete chat" })).toBeInTheDocument(); // the chat is still listed
  });
});
