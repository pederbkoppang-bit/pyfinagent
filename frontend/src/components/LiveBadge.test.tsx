import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { LiveBadge } from "./LiveBadge";

afterEach(() => cleanup());

describe("LiveBadge (phase-44.0 foundation)", () => {
  it("renders green band with 'live' label by default", () => {
    const { container } = render(<LiveBadge band="green" />);
    const status = container.querySelector('[role="status"]');
    expect(status).not.toBeNull();
    expect(status?.getAttribute("aria-label")).toBe("live");
    expect(status?.textContent).toContain("live");
  });

  it("renders amber band with 'stale' label", () => {
    const { container } = render(<LiveBadge band="amber" />);
    const status = container.querySelector('[role="status"]');
    expect(status?.getAttribute("aria-label")).toContain("amber");
    expect(status?.textContent).toContain("stale");
  });

  it("renders red band as stale", () => {
    const { container } = render(<LiveBadge band="red" />);
    const status = container.querySelector('[role="status"]');
    expect(status?.getAttribute("aria-label")).toContain("red");
    expect(status?.textContent).toContain("stale");
  });

  it("renders unknown band with 'unknown' label", () => {
    const { container } = render(<LiveBadge band="unknown" />);
    const status = container.querySelector('[role="status"]');
    expect(status?.textContent).toContain("unknown");
  });

  it("compact mode renders dot only, no label", () => {
    const { container } = render(<LiveBadge band="green" compact />);
    const status = container.querySelector('[role="status"]');
    expect(status?.textContent).toBe("");
    expect(status?.getAttribute("aria-label")).toBe("live");
  });

  it("formats age in seconds for small values", () => {
    const { container } = render(<LiveBadge band="green" ageSec={45} />);
    expect(container.textContent).toContain("45s");
  });

  it("formats age in minutes for sub-hour values", () => {
    const { container } = render(<LiveBadge band="green" ageSec={300} />);
    expect(container.textContent).toContain("5m");
  });

  it("formats age in hours for multi-hour values", () => {
    const { container } = render(<LiveBadge band="amber" ageSec={7200} />);
    expect(container.textContent).toContain("2.0h");
  });

  it("formats age in days for very stale values", () => {
    const { container } = render(<LiveBadge band="red" ageSec={172800} />);
    expect(container.textContent).toContain("2.0d");
  });

  it("accepts custom label override", () => {
    const { container } = render(<LiveBadge band="green" label="custom-label" />);
    expect(container.textContent).toContain("custom-label");
  });
});
