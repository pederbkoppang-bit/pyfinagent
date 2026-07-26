/**
 * phase-80.40 -- the drawdown row, driven by the value the backend now actually
 * returns.
 *
 * Companion to cockpit-helpers.risk.test.tsx (phase-80.36), which is NOT edited
 * by this step: those 6 tests are criterion 5's evidence that the presence-vs-
 * value contract is unchanged, and they pass untouched. This file adds what only
 * 80.40 can assert -- the row fed the REAL measured payload.
 *
 * -5.31 is not an invented fixture. It is the verbatim value returned by
 * GET /api/paper-trading/performance on 2026-07-26 against the live 64-row
 * paper_portfolio_snapshots series, cross-checked out-of-band by a hand-rolled
 * peak-to-trough over the same rows (-5.3112). The same series walked backwards
 * yields -61.51 -- the exact phantom that paged a P1 in phase-66.2 -- which is
 * why the backend helper sorts chronologically before differencing.
 */
import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { RiskMonitorCard } from "./cockpit-helpers";
import type { PaperPerformance, PaperPortfolio } from "@/lib/types";

afterEach(() => cleanup());

const portfolio = { total_nav: 23838.16 } as unknown as PaperPortfolio;
// The live payload shape, verbatim from the :8001 rig curl (corrected citation --
// an earlier revision said :8009, a stale port number from a different rig).
const LIVE_PERF = {
  sharpe_ratio: 3.44,
  max_drawdown_pct: -5.31,
} as unknown as PaperPerformance;

describe("RiskMonitorCard drawdown row (phase-80.40)", () => {
  it("renders a REAL verdict from the live measured drawdown", () => {
    const { container } = render(
      <RiskMonitorCard perf={LIVE_PERF} positions={[]} portfolio={portfolio} tickerMeta={{}} />,
    );
    expect(screen.getByText("SAFE")).toBeTruthy();
    expect(screen.queryByText("NO DATA")).toBeNull();
    // The readout is measured, not an em-dash placeholder.
    expect(container.textContent).toContain("-5.3% / -15%");
  });

  it("still renders NO DATA when the field is absent -- 80.36 unchanged", () => {
    // Criterion 5's other half, asserted here so this step owns both directions
    // without editing the 80.36 file.
    render(<RiskMonitorCard perf={null} positions={[]} portfolio={portfolio} tickerMeta={{}} />);
    expect(screen.queryByText("SAFE")).toBeNull();
    expect(screen.getAllByText("NO DATA").length).toBeGreaterThan(0);
  });

  it("treats a perf object WITHOUT the field as unknown, not as zero", () => {
    // The pre-80.40 reality: the object arrived, that one key never did.
    const noField = { sharpe_ratio: 3.44 } as unknown as PaperPerformance;
    render(<RiskMonitorCard perf={noField} positions={[]} portfolio={portfolio} tickerMeta={{}} />);
    expect(screen.queryByText("SAFE")).toBeNull();
    expect(screen.getAllByText("NO DATA").length).toBeGreaterThan(0);
  });

  it("escalates on the live-magnitude phantom the backend sort prevents", () => {
    // -61.51 is what the SAME live series produces when read DESC (phase-66.2).
    // If that ever reaches the UI the row must scream, not reassure -- this is
    // the frontend half of the order-trap guard.
    const phantom = { max_drawdown_pct: -61.51 } as unknown as PaperPerformance;
    render(<RiskMonitorCard perf={phantom} positions={[]} portfolio={portfolio} tickerMeta={{}} />);
    expect(screen.getByText("DANGER")).toBeTruthy();
  });

  it("would render a POSITIVE-magnitude drawdown as SAFE -- why the sign is pinned in pytest", () => {
    // NOT an endorsement: this documents the inversion hazard. paper_go_live_gate
    // ._snapshot_max_dd_pct returns +5.31 for this same book, and +14 / +99 all
    // satisfy `maxDd > -10`. The frontend cannot defend itself here, so the sign
    // is pinned backend-side by
    // backend/tests/test_phase_80_40_perf_metrics_drawdown.py::
    // test_sign_is_the_exact_negation_of_the_positive_magnitude_sibling.
    const inverted = { max_drawdown_pct: 14.0 } as unknown as PaperPerformance;
    render(<RiskMonitorCard perf={inverted} positions={[]} portfolio={portfolio} tickerMeta={{}} />);
    expect(screen.getByText("SAFE")).toBeTruthy();
  });

  it("no longer labels the row as the kill switch", () => {
    // The row renders the MCP buy-block ladder (-15 / -10, signals_server.py
    // :1370-1379), not the kill switch's 10% trailing-DD halt limit
    // (settings.py:537). It is also ALL-TIME MAX drawdown, which the kill switch
    // never trips on. See the ladder-reconciliation comment in cockpit-helpers.tsx.
    render(
      <RiskMonitorCard perf={LIVE_PERF} positions={[]} portfolio={portfolio} tickerMeta={{}} />,
    );
    expect(screen.getByText("Max drawdown (-15%)")).toBeTruthy();
    expect(screen.queryByText("Kill switch (-15%)")).toBeNull();
  });
});
