import { describe, it, expect, afterEach, beforeAll } from "vitest";
import { render, cleanup } from "@testing-library/react";
import { StrategyDetail } from "./StrategyDetail";
import type {
  StrategyEquityPoint,
  StrategyKillSwitchEvent,
  StrategyOverride,
} from "@/lib/api";

beforeAll(() => {
  // jsdom polyfill -- Recharts ResponsiveContainer requires ResizeObserver.
  globalThis.ResizeObserver =
    globalThis.ResizeObserver ||
    (class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver);
});

const EQUITY: StrategyEquityPoint[] = [
  { date: "2026-04-15", nav: 9500 },
  { date: "2026-04-16", nav: 9550 },
  { date: "2026-04-17", nav: 9530 },
  { date: "2026-04-18", nav: 9580 },
  { date: "2026-04-19", nav: 9600 },
];

const OVERRIDES: StrategyOverride[] = [
  { date: "2026-04-12", param: "learning_rate", from_value: "0.01", to_value: "0.02" },
  { date: "2026-04-15", param: "rolling_window", from_value: "252", to_value: "504" },
];

const EVENTS: StrategyKillSwitchEvent[] = [
  { date: "2026-04-14", label: "auto_demoted", detail: "dd=-0.20 threshold=0.15" },
  { date: "2026-04-16", label: "first_week_armed", detail: null },
  { date: "2026-04-18", label: "kill_switch_check", detail: "dd=-0.05 threshold=0.15" },
];

describe("StrategyDetail", () => {
  afterEach(cleanup);

  it("renders_without_crash", () => {
    const { container } = render(
      <StrategyDetail
        strategyId="alpha_test"
        equity={[]}
        overrides={[]}
        events={[]}
      />,
    );
    const root = container.querySelector('[data-testid="strategy-detail"]');
    expect(root).not.toBeNull();
    // All three section anchors are present even with empty data.
    expect(container.querySelector('[data-testid="equity-curve"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="override-timeline"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="kill-switch-events"]')).not.toBeNull();
  });

  it("equity_curve_scoped_by_strategy", () => {
    const { container } = render(
      <StrategyDetail
        strategyId="alpha_a"
        equity={EQUITY}
        overrides={[]}
        events={[]}
      />,
    );
    const eq = container.querySelector('[data-testid="equity-curve"]');
    expect(eq).not.toBeNull();
    // The header reflects the per-strategy point count from props.
    const text = container.textContent || "";
    expect(text).toContain(`Equity curve (${EQUITY.length})`);
    // Strategy ID is rendered as the page title (proves scoping).
    expect(text).toContain("alpha_a");
  });

  it("param_override_timeline_rendered", () => {
    const { container } = render(
      <StrategyDetail
        strategyId="alpha_b"
        equity={[]}
        overrides={OVERRIDES}
        events={[]}
      />,
    );
    const tl = container.querySelector('[data-testid="override-timeline"]');
    expect(tl).not.toBeNull();
    const items = tl!.querySelectorAll("[data-override]");
    expect(items.length).toBe(OVERRIDES.length);
    const text = tl!.textContent || "";
    expect(text).toContain("learning_rate");
    expect(text).toContain("rolling_window");
  });

  it("kill_switch_events_scoped", () => {
    const { container } = render(
      <StrategyDetail
        strategyId="alpha_c"
        equity={[]}
        overrides={[]}
        events={EVENTS}
      />,
    );
    const ks = container.querySelector('[data-testid="kill-switch-events"]');
    expect(ks).not.toBeNull();
    const items = ks!.querySelectorAll("[data-event]");
    expect(items.length).toBe(EVENTS.length);
    const text = ks!.textContent || "";
    expect(text).toContain("auto_demoted");
    expect(text).toContain("first_week_armed");
    expect(text).toContain("kill_switch_check");
  });
});
