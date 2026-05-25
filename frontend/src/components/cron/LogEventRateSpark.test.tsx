import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { LogEventRateSpark } from "./LogEventRateSpark";

afterEach(() => cleanup());

describe("LogEventRateSpark (phase-44.7)", () => {
  it("renders nothing when lines empty", () => {
    const { container } = render(<LogEventRateSpark lines={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when no timestamps in the recent window", () => {
    const { container } = render(
      <LogEventRateSpark lines={["plain text", "no timestamp here"]} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the spark when recent timestamped lines exist", () => {
    const now = new Date();
    const fmt = (offsetSec: number) => {
      const d = new Date(now.getTime() - offsetSec * 1000);
      return d.toISOString();
    };
    const lines = [
      `${fmt(10)} INFO event 1`,
      `${fmt(60)} INFO event 2`,
      `${fmt(120)} INFO event 3`,
    ];
    const { container } = render(<LogEventRateSpark lines={lines} />);
    expect(container.firstChild).not.toBeNull();
    expect(container.textContent).toContain("Event rate");
    expect(container.textContent).toContain("peak");
  });

  it("has region role + aria-label", () => {
    const fmt = (offsetSec: number) => {
      const d = new Date(Date.now() - offsetSec * 1000);
      return d.toISOString();
    };
    const lines = [`${fmt(10)} INFO`, `${fmt(60)} INFO`];
    const { container } = render(<LogEventRateSpark lines={lines} />);
    const region = container.querySelector('[role="region"]');
    expect(region).not.toBeNull();
    expect(region?.getAttribute("aria-label")).toBe(
      "Log event rate over last 60 minutes",
    );
  });

  it("renders an SVG with polyline + polygon", () => {
    const fmt = (offsetSec: number) => {
      const d = new Date(Date.now() - offsetSec * 1000);
      return d.toISOString();
    };
    const lines = Array.from({ length: 5 }, (_, i) => `${fmt(i * 60)} INFO`);
    const { container } = render(<LogEventRateSpark lines={lines} />);
    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelector("polyline")).not.toBeNull();
    expect(container.querySelector("polygon")).not.toBeNull();
  });
});
