import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import {
  TimeRangeSelector,
  filterByTimeRange,
  type TimeRange,
} from "./TimeRangeSelector";

afterEach(() => cleanup());

describe("TimeRangeSelector (phase-44.4)", () => {
  it("renders 4 options as radio buttons", () => {
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={() => {}} />,
    );
    const radios = container.querySelectorAll('[role="radio"]');
    expect(radios.length).toBe(4);
  });

  it("marks the active option with aria-checked=true", () => {
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={() => {}} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    const checked = radios.filter((r) => r.getAttribute("aria-checked") === "true");
    expect(checked.length).toBe(1);
    expect(checked[0].getAttribute("aria-label")).toBe("30 days");
  });

  it("emits onChange when a non-active radio is clicked", () => {
    const onChange = vi.fn();
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={onChange} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    fireEvent.click(radios[0]);
    expect(onChange).toHaveBeenCalledWith("7d");
  });

  it("ArrowRight cycles to next option", () => {
    const onChange = vi.fn();
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={onChange} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    fireEvent.keyDown(radios[1], { key: "ArrowRight" });
    expect(onChange).toHaveBeenCalledWith("90d");
  });

  it("ArrowLeft cycles to previous option (wraps)", () => {
    const onChange = vi.fn();
    const { container } = render(
      <TimeRangeSelector value="7d" onChange={onChange} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    fireEvent.keyDown(radios[0], { key: "ArrowLeft" });
    expect(onChange).toHaveBeenCalledWith("all");
  });

  it("Home/End jump to first/last", () => {
    const onChange = vi.fn();
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={onChange} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    fireEvent.keyDown(radios[1], { key: "Home" });
    expect(onChange).toHaveBeenCalledWith("7d");
    fireEvent.keyDown(radios[1], { key: "End" });
    expect(onChange).toHaveBeenCalledWith("all");
  });

  it("has role=radiogroup + aria-label", () => {
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={() => {}} />,
    );
    const group = container.querySelector('[role="radiogroup"]');
    expect(group).not.toBeNull();
    expect(group?.getAttribute("aria-label")).toBe("Time range");
  });

  it("each radio meets WCAG 2.2 24px target-size", () => {
    const { container } = render(
      <TimeRangeSelector value="30d" onChange={() => {}} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    for (const r of radios) {
      expect(r.className).toContain("min-h-[32px]");
    }
  });

  it("roving tabindex: only the active radio has tabIndex=0", () => {
    const { container } = render(
      <TimeRangeSelector value="90d" onChange={() => {}} />,
    );
    const radios = Array.from(container.querySelectorAll('[role="radio"]'));
    const tabIndices = radios.map((r) => r.getAttribute("tabindex"));
    expect(tabIndices.filter((t) => t === "0").length).toBe(1);
    expect(tabIndices.filter((t) => t === "-1").length).toBe(3);
  });
});

describe("filterByTimeRange", () => {
  type Row = { date: string; v: number };
  const NOW = Date.now();
  const day = (d: number) => new Date(NOW - d * 86_400_000).toISOString();
  const items: Row[] = [
    { date: day(1), v: 1 },
    { date: day(10), v: 2 },
    { date: day(40), v: 3 },
    { date: day(80), v: 4 },
    { date: day(120), v: 5 },
  ];

  it("returns all items when range=all", () => {
    expect(filterByTimeRange(items, "all" as TimeRange, "date").length).toBe(5);
  });

  it("filters to last 7 days", () => {
    expect(filterByTimeRange(items, "7d" as TimeRange, "date").length).toBe(1);
  });

  it("filters to last 30 days", () => {
    expect(filterByTimeRange(items, "30d" as TimeRange, "date").length).toBe(2);
  });

  it("filters to last 90 days", () => {
    expect(filterByTimeRange(items, "90d" as TimeRange, "date").length).toBe(4);
  });

  it("returns empty when no items match", () => {
    expect(filterByTimeRange([], "7d" as TimeRange, "date")).toEqual([]);
  });

  it("skips items with non-string dates", () => {
    const bad = [{ date: 12345 as unknown as string, v: 1 }];
    expect(filterByTimeRange(bad as { date: string; v: number }[], "7d" as TimeRange, "date").length).toBe(0);
  });

  it("skips items with un-parseable dates", () => {
    const bad = [{ date: "not a date", v: 1 }];
    expect(filterByTimeRange(bad, "7d" as TimeRange, "date").length).toBe(0);
  });
});
