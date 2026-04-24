import { describe, it, expect, afterEach, beforeAll } from "vitest";
import { render, cleanup } from "@testing-library/react";
import {
  ComputeCostBreakdown,
  CostTooltip,
  PROVIDERS,
  PROVIDER_COLORS,
} from "./ComputeCostBreakdown";
import type { ProviderCostPoint } from "./ComputeCostBreakdown";

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

const COST_DATA: ProviderCostPoint[] = [
  { date: "2026-04-19", anthropic: 0.12, vertex: 0.08, openai: 0.05, bigquery: 0.02, altdata: 0.01 },
  { date: "2026-04-20", anthropic: 0.15, vertex: 0.09, openai: 0.06, bigquery: 0.03, altdata: 0.02 },
  { date: "2026-04-21", anthropic: 0.10, vertex: 0.07, openai: 0.04, bigquery: 0.01, altdata: 0.01 },
];

describe("ComputeCostBreakdown", () => {
  afterEach(cleanup);

  it("provider_colors_exported", () => {
    // criterion 1: deterministic_color_map_present
    expect(typeof PROVIDER_COLORS).toBe("object");
    const keys = Object.keys(PROVIDER_COLORS).sort();
    expect(keys).toEqual(["altdata", "anthropic", "bigquery", "openai", "vertex"]);
    // Each value is a non-empty hex string.
    for (const k of keys) {
      const v = PROVIDER_COLORS[k as keyof typeof PROVIDER_COLORS];
      expect(typeof v).toBe("string");
      expect(v.length).toBeGreaterThan(0);
      expect(v.startsWith("#")).toBe(true);
    }
  });

  it("providers_cover_anthropic_vertex_openai_bq_altdata", () => {
    // criterion 2: PROVIDERS contains exactly these 5 keys.
    expect([...PROVIDERS].sort()).toEqual([
      "altdata",
      "anthropic",
      "bigquery",
      "openai",
      "vertex",
    ]);
  });

  it("chart_renders_when_data_present", () => {
    const { container } = render(
      <ComputeCostBreakdown data={COST_DATA} grandTotal={0.86} window="30d" />,
    );
    const chart = container.querySelector('[data-testid="compute-cost-chart"]');
    expect(chart).not.toBeNull();
  });

  it("empty_state_renders_when_data_empty", () => {
    const { container } = render(
      <ComputeCostBreakdown data={[]} grandTotal={0} window="7d" />,
    );
    const empty = container.querySelector('[data-testid="compute-cost-empty"]');
    expect(empty).not.toBeNull();
  });

  it("tooltip_shows_usd_and_percent", () => {
    // criterion 3: custom Tooltip output contains $ + %.
    const row = COST_DATA[0]; // total = 0.28
    const { container } = render(
      <CostTooltip
        active={true}
        label={row.date}
        payload={[
          { name: "anthropic", dataKey: "anthropic", value: row.anthropic, payload: row },
        ]}
      />,
    );
    const tip = container.querySelector('[data-testid="cost-tooltip"]');
    expect(tip).not.toBeNull();
    const text = tip!.textContent || "";
    // Anthropic = 0.12 / 0.28 ~= 42.9%.
    expect(text).toContain("$0.12");
    expect(text).toContain("%");
    expect(text).toContain("anthropic");
    expect(text).toContain("day total");
  });
});
