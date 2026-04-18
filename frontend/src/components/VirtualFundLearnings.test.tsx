import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import {
  VirtualFundLearnings,
  type VirtualFundLearningsData,
} from "./VirtualFundLearnings";

// 15 divergences -- component should render the TOP 10 by abs drift.
const MANY_DIVERGENCES = Array.from({ length: 15 }, (_, i) => ({
  symbol: `TKR${i.toString().padStart(2, "0")}`,
  side: (i % 2 === 0 ? "buy" : "sell") as "buy" | "sell",
  paper_fill: 100 + i,
  sim_fill: 100 + i + (i * 0.3 + 0.1) * (i % 2 === 0 ? 1 : -1),
  drift_pct: (i * 0.3 + 0.1) * (i % 2 === 0 ? 1 : -1),
  ts: `2026-04-${(i % 28) + 1}T12:00:00Z`,
}));

const SAMPLE: VirtualFundLearningsData = {
  reconciliation_divergences: MANY_DIVERGENCES,
  kill_switch_triggers: [
    { reason: "daily_loss_pct", count: 5 },
    { reason: "trailing_dd_pct", count: 2 },
    { reason: "position_cap", count: 7 },
    { reason: "manual", count: 1 },
  ],
  regime_buckets: [
    { regime: "bull", n_trades: 42, return_pct: 3.21, sharpe: 1.52 },
    { regime: "sideways", n_trades: 18, return_pct: -0.87, sharpe: -0.12 },
    { regime: "bear", n_trades: 11, return_pct: -4.4, sharpe: -0.98 },
  ],
  window_days: 30,
};

describe("VirtualFundLearnings", () => {
  afterEach(() => cleanup());

  it("learnings_page_landed: page header h2 identifies the dashboard", () => {
    render(<VirtualFundLearnings data={SAMPLE} />);
    const header = document.querySelector('[data-section="page-header"] h2');
    expect(header).not.toBeNull();
    expect(header?.textContent).toMatch(/Virtual-Fund Learnings/);
  });

  it("reconciliation_divergences_top10_rendered: exactly 10 rows, sorted by abs drift desc", () => {
    render(<VirtualFundLearnings data={SAMPLE} />);
    const section = document.querySelector('[data-section="reconciliation-divergences"]');
    expect(section).not.toBeNull();
    const rows = section!.querySelectorAll("tbody tr[data-row-index]");
    expect(rows.length).toBe(10);
    // Top row must be the largest abs drift in the input.
    const largestAbs = Math.max(
      ...SAMPLE.reconciliation_divergences.map((d) => Math.abs(d.drift_pct)),
    );
    const topDriftCell = rows[0].querySelector('td[data-cell="drift_pct"]');
    const topNum = parseFloat(
      (topDriftCell?.textContent ?? "").replace(/[+%]/g, ""),
    );
    expect(Math.abs(topNum)).toBeCloseTo(largestAbs, 2);
    // And the 10th row's abs drift must be >= any row not included (rank 11+).
    const includedAbs = Array.from(rows).map((r) => {
      const t = r.querySelector('td[data-cell="drift_pct"]')?.textContent ?? "";
      return Math.abs(parseFloat(t.replace(/[+%]/g, "")));
    });
    const tenthAbs = includedAbs[9];
    const excludedAbs = SAMPLE.reconciliation_divergences
      .map((d) => Math.abs(d.drift_pct))
      .sort((a, b) => b - a)
      .slice(10);
    for (const e of excludedAbs) {
      expect(tenthAbs).toBeGreaterThanOrEqual(e - 1e-9);
    }
  });

  it("kill_switch_trigger_distribution_rendered: all buckets render, counts sum to total", () => {
    render(<VirtualFundLearnings data={SAMPLE} />);
    const section = document.querySelector('[data-section="kill-switch-distribution"]');
    expect(section).not.toBeNull();
    const list = section!.querySelector('[data-testid="killswitch-list"]');
    expect(list).not.toBeNull();
    const items = list!.querySelectorAll("[data-killswitch-reason]");
    expect(items.length).toBe(SAMPLE.kill_switch_triggers.length);
    const totalEl = section!.querySelector('[data-testid="total-triggers"]');
    const total = parseInt(totalEl?.textContent ?? "0", 10);
    const expected = SAMPLE.kill_switch_triggers.reduce(
      (s, t) => s + t.count,
      0,
    );
    expect(total).toBe(expected);
    // At least one reason must be "daily_loss_pct" (input guaranteed).
    const reasons = Array.from(items).map((i) =>
      i.getAttribute("data-killswitch-reason"),
    );
    expect(reasons).toContain("daily_loss_pct");
  });

  it("regime_underperformance_buckets_rendered: bear/sideways flagged with negative-return styling", () => {
    render(<VirtualFundLearnings data={SAMPLE} />);
    const section = document.querySelector('[data-section="regime-underperformance"]');
    expect(section).not.toBeNull();
    const rows = section!.querySelectorAll("tr[data-regime]");
    expect(rows.length).toBe(SAMPLE.regime_buckets.length);
    const bearRow = section!.querySelector('tr[data-regime="bear"]');
    expect(bearRow).not.toBeNull();
    const bearReturn = bearRow!.querySelector('td[data-cell="return_pct"]');
    expect(bearReturn?.textContent).toMatch(/-4\.40%/);
    expect(bearReturn?.className).toMatch(/text-rose-400/);
  });

  it("renders empty states when data is absent", () => {
    render(<VirtualFundLearnings />);
    expect(document.querySelector('[data-testid="reconciliation-empty"]')).not.toBeNull();
    expect(document.querySelector('[data-testid="killswitch-empty"]')).not.toBeNull();
    expect(document.querySelector('[data-testid="regime-empty"]')).not.toBeNull();
  });
});
