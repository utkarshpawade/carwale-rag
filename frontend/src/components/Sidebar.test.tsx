import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Conversation } from "../types";
import Sidebar from "./Sidebar";

const iso = (secondsAgo: number) => new Date(Date.now() - secondsAgo * 1000).toISOString();
const convs: Conversation[] = [
  { id: "c1", title: "Creta vs Seltos", updated_at: iso(30), n_messages: 2 },
  { id: "c2", title: "", updated_at: iso(5 * 60), n_messages: 4 },
  { id: "c3", title: "Safest SUVs", updated_at: iso(3 * 3600), n_messages: 6 },
  { id: "c4", title: "Old chat", updated_at: iso(2 * 86400), n_messages: 8 },
];

const setup = (props: Partial<Parameters<typeof Sidebar>[0]> = {}) => {
  const handlers = { onNew: vi.fn(), onOpen: vi.fn(), onDelete: vi.fn(), onClose: vi.fn() };
  render(<Sidebar conversations={convs} activeId="c1" {...handlers} snapshot="2026-11-17"
                  stats={{ models: 288, variants: 2202 }} {...props} />);
  return { ...handlers, user: userEvent.setup() };
};

describe("Sidebar", () => {
  it("shows an empty-state hint", () => {
    setup({ conversations: [] });
    expect(screen.getByText(/Your conversations appear here/)).toBeInTheDocument();
  });

  it("lists chats with relative times, message counts and a fallback title", () => {
    setup();
    expect(screen.getByText("just now · 2 msgs")).toBeInTheDocument();
    expect(screen.getByText("5m ago · 4 msgs")).toBeInTheDocument();
    expect(screen.getByText("3h ago · 6 msgs")).toBeInTheDocument();
    expect(screen.getByText("2d ago · 8 msgs")).toBeInTheDocument();
    expect(screen.getByText("New chat", { selector: "div" })).toBeInTheDocument();
    expect(screen.getByText("288 models · 2,202 variants indexed")).toBeInTheDocument();
    expect(screen.getByText(/snapshot 2026-11-17/)).toBeInTheDocument();
  });

  it("opens a chat on click and deletes without opening it", async () => {
    const { onOpen, onDelete, user } = setup();
    await user.click(screen.getByText("Safest SUVs"));
    expect(onOpen).toHaveBeenCalledWith("c3");
    await user.click(screen.getAllByRole("button", { name: "Delete chat" })[0]);
    expect(onDelete).toHaveBeenCalledWith("c1");
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it("new chat and close buttons call back", async () => {
    const { onNew, onClose, user } = setup();
    await user.click(screen.getByRole("button", { name: "New chat" })); // exact: untitled chat rows also read "New chat …"
    await user.click(screen.getByRole("button", { name: "Close menu" }));
    expect(onNew).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("chat rows can be opened with the keyboard", async () => {
    // B-16: rows were click-only <div>s, so keyboard and screen-reader users could not reach past chats at all
    const { onOpen, user } = setup();
    const row = screen.getByRole("button", { name: /Safest SUVs/ });
    row.focus();
    await user.keyboard("{Enter}");
    expect(onOpen).toHaveBeenCalledWith("c3");
    await user.keyboard(" ");
    expect(onOpen).toHaveBeenCalledTimes(2);
  });

  it("marks the active chat", () => {
    setup();
    expect(screen.getByText("Creta vs Seltos").closest("[class*='bg-slate-800']")).not.toBeNull();
  });
});
