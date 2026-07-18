import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnimatedSessionTitle } from "./AnimatedSessionTitle";

afterEach(() => {
  vi.useRealTimers();
});

describe("AnimatedSessionTitle", () => {
  it("renders New Chat while the session is untitled", () => {
    render(<AnimatedSessionTitle title={null} fallback="New Chat" />);
    expect(screen.getByText("New Chat")).toBeInTheDocument();
  });

  it("reveals the first generated title character by character", () => {
    vi.useFakeTimers();
    const { rerender } = render(<AnimatedSessionTitle title={null} fallback="New session" />);
    expect(screen.getByText("New session")).toBeInTheDocument();

    rerender(<AnimatedSessionTitle title="OAuth Fix" fallback="New session" />);
    act(() => vi.advanceTimersByTime(48));
    expect(screen.getByText("O")).toBeInTheDocument();

    act(() => vi.runAllTimers());
    expect(screen.getByText("OAuth Fix")).toBeInTheDocument();
  });

  it("treats an absent streamed title as untitled", () => {
    vi.useFakeTimers();
    const { rerender } = render(<AnimatedSessionTitle title={undefined} fallback="New session" />);
    rerender(<AnimatedSessionTitle title="Generated" fallback="New session" />);

    act(() => vi.advanceTimersByTime(48));
    expect(screen.getByText("G")).toBeInTheDocument();
  });

  it("shows an existing title immediately on initial render", () => {
    render(<AnimatedSessionTitle title="Existing title" fallback="New session" />);
    expect(screen.getByText("Existing title")).toBeInTheDocument();
  });

  it("applies later manual renames immediately", () => {
    const { rerender } = render(
      <AnimatedSessionTitle title="Generated title" fallback="New session" />,
    );
    rerender(<AnimatedSessionTitle title="Manual title" fallback="New session" />);
    expect(screen.getByText("Manual title")).toBeInTheDocument();
  });
});
