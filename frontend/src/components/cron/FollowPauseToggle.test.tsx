import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { FollowPauseToggle } from "./FollowPauseToggle";

afterEach(() => cleanup());

describe("FollowPauseToggle (phase-44.7)", () => {
  it("renders 'Following' label when following=true", () => {
    const { container } = render(
      <FollowPauseToggle following={true} onToggle={() => {}} />,
    );
    expect(container.textContent).toContain("Following");
  });

  it("renders 'Paused' label when following=false", () => {
    const { container } = render(
      <FollowPauseToggle following={false} onToggle={() => {}} />,
    );
    expect(container.textContent).toContain("Paused");
  });

  it("calls onToggle when clicked", () => {
    const onToggle = vi.fn();
    const { container } = render(
      <FollowPauseToggle following={true} onToggle={onToggle} />,
    );
    fireEvent.click(container.querySelector("button")!);
    expect(onToggle).toHaveBeenCalled();
  });

  it("aria-pressed reflects following state", () => {
    const { container, rerender } = render(
      <FollowPauseToggle following={true} onToggle={() => {}} />,
    );
    expect(container.querySelector("button")?.getAttribute("aria-pressed")).toBe("true");
    rerender(<FollowPauseToggle following={false} onToggle={() => {}} />);
    expect(container.querySelector("button")?.getAttribute("aria-pressed")).toBe("false");
  });

  it("aria-label reflects action (the OPPOSITE of current state)", () => {
    const { container, rerender } = render(
      <FollowPauseToggle following={true} onToggle={() => {}} />,
    );
    expect(container.querySelector("button")?.getAttribute("aria-label")).toBe("Pause auto-scroll");
    rerender(<FollowPauseToggle following={false} onToggle={() => {}} />);
    expect(container.querySelector("button")?.getAttribute("aria-label")).toBe("Resume auto-scroll");
  });

  it("meets WCAG 2.2 24px target-size", () => {
    const { container } = render(
      <FollowPauseToggle following={true} onToggle={() => {}} />,
    );
    expect(container.querySelector("button")?.className).toContain("min-h-[24px]");
  });
});
