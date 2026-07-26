/**
 * phase-80.36 -- UNKNOWN is not NOMINAL.
 *
 * MEASURED 2026-07-26 with the backend killed: the Risk Monitor rendered
 * "Kill switch (-15%) SAFE" in emerald, "Position size OK", "Sector
 * concentration OK" and "Drawdown 0% / -15%" -- all from ZERO data --
 * because `perf?.max_drawdown_pct ?? 0` turns absence into the most
 * reassuring possible observation and `0 > -10` is true.
 *
 * Capture: handoff/current/captures_done_definition/backend_dead_positions.png
 *
 * There was no prior test asserting SAFE/OK/WARNING/DANGER anywhere in the
 * repo, so the healthy path had no regression net at all. These tests are
 * that net as much as they are the fix's guard.
 */
import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { RiskMonitorCard } from "./cockpit-helpers";
import type { PaperPerformance, PaperPortfolio, PaperPosition } from "@/lib/types";

afterEach(() => cleanup());

const portfolio = { total_nav: 23838.16 } as unknown as PaperPortfolio;
const pos = (ticker: string, mv: number): PaperPosition =>
  ({ ticker, quantity: 1, current_price: mv, cost_basis: mv, market: "US" }) as unknown as PaperPosition;
const perfWith = (dd: number) => ({ max_drawdown_pct: dd }) as unknown as PaperPerformance;

describe("RiskMonitorCard -- unknown vs nominal (phase-80.36)", () => {
  it("does NOT claim SAFE when there is no performance data", () => {
    // THE DEFECT. This is the most trust-bearing pixel on the cockpit, and it
    // compounds with step 36.7 where the kill switch cannot currently fire.
    render(<RiskMonitorCard perf={null} positions={[]} portfolio={null} tickerMeta={{}} />);
    expect(screen.queryByText("SAFE"), "no data must never render SAFE").toBeNull();
    expect(screen.getAllByText("NO DATA").length).toBeGreaterThan(0);
  });

  it("STILL renders SAFE for a genuine zero drawdown -- presence, not value", () => {
    // HARD STOP guarded: max_drawdown_pct === 0 is a legitimate healthy reading
    // (a fund at its high-water mark). Discriminating on VALUE rather than
    // PRESENCE would flip this real SAFE to unknown and change the healthy path.
    render(
      <RiskMonitorCard perf={perfWith(0)} positions={[]} portfolio={portfolio} tickerMeta={{}} />,
    );
    expect(screen.getByText("SAFE")).toBeTruthy();
    expect(screen.queryByText("NO DATA")).toBeNull();
  });

  it("still escalates a real breach", () => {
    render(
      <RiskMonitorCard perf={perfWith(-14)} positions={[]} portfolio={portfolio} tickerMeta={{}} />,
    );
    expect(screen.getByText("DANGER")).toBeTruthy();
  });

  it("keeps a LIVE position-size breach visible when only perf failed", () => {
    // HARD STOP guarded: a card-level `if (!perf) return <Unknown/>` would hide
    // this. Position size reads positions+portfolio, never perf, and
    // getPaperPerformance() has its own .catch(() => null) at layout.tsx:193 --
    // so "perf null, portfolio healthy" is a REAL state, not a hypothetical.
    render(
      <RiskMonitorCard
        perf={null}
        positions={[pos("AAA", 9000)]}
        portfolio={portfolio}
        tickerMeta={{}}
      />,
    );
    expect(
      screen.getByText("HIGH (>20%)"),
      "a genuine concentration breach was suppressed because perf was unavailable",
    ).toBeTruthy();
  });

  it("a genuinely FLAT book still reads OK -- empty is not unknown", () => {
    // The other direction: portfolio present, no positions held. That is a real
    // flat book and must NOT be reported as unknown, or the fix cries wolf.
    render(
      <RiskMonitorCard perf={perfWith(-1)} positions={[]} portfolio={portfolio} tickerMeta={{}} />,
    );
    expect(screen.getAllByText("OK").length).toBeGreaterThan(0);
  });

  it("does not render a measured-looking 0% drawdown with no data", () => {
    const { container } = render(
      <RiskMonitorCard perf={null} positions={[]} portfolio={null} tickerMeta={{}} />,
    );
    expect(container.textContent).not.toContain("0.0% / -15%");
    expect(container.textContent).toContain("— / -15%");
  });
});
