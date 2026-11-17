import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import Composer from "./Composer";

const setup = (props: Partial<Parameters<typeof Composer>[0]> = {}) => {
  const onSend = vi.fn();
  const onStop = vi.fn();
  render(<Composer onSend={onSend} onStop={onStop} streaming={false} disabled={false} {...props} />);
  return { onSend, onStop, box: screen.getByRole("textbox"), user: userEvent.setup() };
};

describe("Composer", () => {
  it("sends trimmed text on Enter and clears the box", async () => {
    const { onSend, box, user } = setup();
    await user.type(box, "  creta mileage  {Enter}");
    expect(onSend).toHaveBeenCalledWith("creta mileage");
    expect(box).toHaveValue("");
  });

  it("Shift+Enter adds a new line instead of sending", async () => {
    const { onSend, box, user } = setup();
    await user.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(onSend).not.toHaveBeenCalled();
    expect(box).toHaveValue("line one\nline two");
  });

  it("does not send empty or whitespace-only text", async () => {
    const { onSend, box, user } = setup();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await user.type(box, "   {Enter}");
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("sends with the button too", async () => {
    const { onSend, box, user } = setup();
    await user.type(box, "Creta vs Seltos");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("Creta vs Seltos");
  });

  it("is blocked while the server is not ready", async () => {
    const { onSend, box, user } = setup({ disabled: true });
    expect(box).toHaveAttribute("placeholder", "Connecting to the server…");
    await user.type(box, "hello{Enter}");
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("shows Stop while streaming and never sends then", async () => {
    const { onSend, onStop, box, user } = setup({ streaming: true });
    await user.type(box, "next question{Enter}");
    expect(onSend).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
  });

  it("caps input at 1,000 characters (the server's limit)", async () => {
    const { box, user } = setup();
    expect(box).toHaveAttribute("maxLength", "1000");
    await user.click(box);
    await user.paste("x".repeat(1500));
    expect((box as HTMLTextAreaElement).value.length).toBe(1000);
  });

  it("keeps unicode and emoji intact", async () => {
    const { onSend, box, user } = setup();
    await user.type(box, "क्रेटा का माइलेज 🚗{Enter}");
    expect(onSend).toHaveBeenCalledWith("क्रेटा का माइलेज 🚗");
  });
});
