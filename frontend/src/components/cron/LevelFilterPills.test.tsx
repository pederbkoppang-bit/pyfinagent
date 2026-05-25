import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { LevelFilterPills } from "./LevelFilterPills";
import type { LogLevel } from "./density-helpers";

afterEach(() => cleanup());

describe("LevelFilterPills (phase-44.7)", () => {
  it("renders 3 pills in a role=group", () => {
    const { container } = render(
      <LevelFilterPills active={new Set()} onToggle={() => {}} />,
    );
    const group = container.querySelector('[role="group"]');
    expect(group).not.toBeNull();
    const buttons = container.querySelectorAll("button");
    expect(buttons.length).toBe(3);
    expect(Array.from(buttons).map((b) => b.textContent)).toEqual(["ERROR", "WARN", "INFO"]);
  });

  it("each pill has aria-pressed reflecting active state", () => {
    const active = new Set<LogLevel>(["WARN"]);
    const { container } = render(
      <LevelFilterPills active={active} onToggle={() => {}} />,
    );
    const buttons = Array.from(container.querySelectorAll("button"));
    expect(buttons[0].getAttribute("aria-pressed")).toBe("false");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");
    expect(buttons[2].getAttribute("aria-pressed")).toBe("false");
  });

  it("calls onToggle with the correct level when clicked", () => {
    const onToggle = vi.fn();
    const { container } = render(
      <LevelFilterPills active={new Set()} onToggle={onToggle} />,
    );
    const buttons = container.querySelectorAll("button");
    fireEvent.click(buttons[0]);
    expect(onToggle).toHaveBeenCalledWith("ERROR");
    fireEvent.click(buttons[1]);
    expect(onToggle).toHaveBeenCalledWith("WARN");
    fireEvent.click(buttons[2]);
    expect(onToggle).toHaveBeenCalledWith("INFO");
  });

  it("group has aria-label", () => {
    const { container } = render(
      <LevelFilterPills active={new Set()} onToggle={() => {}} />,
    );
    const group = container.querySelector('[role="group"]');
    expect(group?.getAttribute("aria-label")).toBe("Log level filter");
  });

  it("each pill meets WCAG 2.2 24px target-size", () => {
    const { container } = render(
      <LevelFilterPills active={new Set()} onToggle={() => {}} />,
    );
    const buttons = Array.from(container.querySelectorAll("button"));
    for (const b of buttons) expect(b.className).toContain("min-h-[24px]");
  });

  it("each pill has aria-label for screen readers", () => {
    const { container } = render(
      <LevelFilterPills active={new Set()} onToggle={() => {}} />,
    );
    const buttons = Array.from(container.querySelectorAll("button"));
    expect(buttons[0].getAttribute("aria-label")).toBe("Toggle ERROR filter");
    expect(buttons[1].getAttribute("aria-label")).toBe("Toggle WARN filter");
    expect(buttons[2].getAttribute("aria-label")).toBe("Toggle INFO filter");
  });
});
