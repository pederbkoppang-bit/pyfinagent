import { describe, it, expect, vi, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import {
  AutoresearchLeaderboard,
  AUTORESEARCH_REFRESH_MS,
  type LeaderboardCandidate,
} from "./AutoresearchLeaderboard";
import { mapExperimentsToCandidates } from "./AutoresearchLeaderboardMap";

const SAMPLE: LeaderboardCandidate[] = [
  {
    index: 0,
    run_id: "run-a",
    param_changed: "rsi_weight",
    dsr: 0.97,
    pbo: 0.12,
    realized_pnl_if_promoted: 12_500,
    starting_capital: 100_000,
    status: "kept",
  },
  {
    index: 1,
    run_id: "run-a",
    param_changed: "holding_days",
    dsr: 0.82,
    pbo: 0.63,
    realized_pnl_if_promoted: -8_000,
    starting_capital: 100_000,
    status: "discarded",
  },
];

describe("AutoresearchLeaderboard", () => {
  afterEach(() => cleanup());

  it("renders the DSR column header (dsr_column_present)", () => {
    render(<AutoresearchLeaderboard candidates={SAMPLE} />);
    const dsrHeader = document.querySelector('th[data-col="dsr"]');
    expect(dsrHeader).not.toBeNull();
    expect(dsrHeader?.textContent).toContain("DSR");
  });

  it("renders the PBO column header (pbo_column_present)", () => {
    render(<AutoresearchLeaderboard candidates={SAMPLE} />);
    const pboHeader = document.querySelector('th[data-col="pbo"]');
    expect(pboHeader).not.toBeNull();
    expect(pboHeader?.textContent).toContain("PBO");
  });

  it("renders the Realized P&L if-promoted column (realized_pnl_if_promoted)", () => {
    render(<AutoresearchLeaderboard candidates={SAMPLE} />);
    const pnlHeader = document.querySelector('th[data-col="realized_pnl"]');
    expect(pnlHeader).not.toBeNull();
    expect(pnlHeader?.textContent).toMatch(/Realized\s+P&amp;L|Realized\s+P&L/);
    // Cells must render the real numeric values per row, not a blanket
    // "$" placeholder. A regression that collapsed the mapping to $0
    // everywhere (qa-evaluator Cycle 73 CONDITIONAL) would fail here.
    const cells = document.querySelectorAll('td[data-cell="realized_pnl"]');
    expect(cells.length).toBe(SAMPLE.length);
    // First row: the ranked top candidate (pbo<0.5) = SAMPLE[0] = +12,500
    expect(cells[0].textContent).toMatch(/\$12,500/);
    // Last row: the vetoed candidate (pbo>0.5) = SAMPLE[1] = -8,000
    expect(cells[cells.length - 1].textContent).toMatch(/-\$8,000/);
    // Defensive: none of the cells should be exactly "$0"
    for (const c of cells) {
      expect(c.textContent?.trim()).not.toBe("$0");
    }
  });

  it("pins PBO-vetoed candidates (>0.5) to the bottom", () => {
    render(<AutoresearchLeaderboard candidates={SAMPLE} />);
    const rankedIndices = Array.from(
      document.querySelectorAll("tbody tr"),
    ).map((row) => row.getAttribute("data-candidate-index"));
    // The candidate with pbo=0.63 (index 1) must be last; the pbo=0.12 one first.
    expect(rankedIndices[0]).toBe("0");
    expect(rankedIndices[rankedIndices.length - 1]).toBe("1");
  });

  it("refresh interval <= 10s (leaderboard_refresh_le_10s)", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(async () => SAMPLE);
    render(<AutoresearchLeaderboard fetcher={fetcher} />);
    // first call fires from the refresh() effect
    await vi.advanceTimersByTimeAsync(0);
    expect(fetcher).toHaveBeenCalledTimes(1);
    // must refresh again within 10s
    await vi.advanceTimersByTimeAsync(AUTORESEARCH_REFRESH_MS);
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(AUTORESEARCH_REFRESH_MS).toBeLessThanOrEqual(10_000);
    // DOM carries the attribute for independent verification
    const el = document.querySelector('[data-testid="autoresearch-leaderboard"]');
    expect(Number(el?.getAttribute("data-refresh-ms"))).toBeLessThanOrEqual(10_000);
    vi.useRealTimers();
  });

  it("renders empty state when no candidates are provided", () => {
    render(<AutoresearchLeaderboard candidates={[]} />);
    const empty = document.querySelector('[data-testid="leaderboard-empty"]');
    expect(empty).not.toBeNull();
    expect(empty?.textContent).toMatch(/No optimizer candidates/);
  });

  // Guards the page.tsx wiring: mapExperimentsToCandidates must pass
  // the backend's `pbo` field through rather than hardcoding null.
  // This is the regression qa-evaluator flagged in Cycle 73: production
  // rows always showed "--" because the mapping ignored exp.pbo.
  it("passes through backend PBO field (no hardcoded null)", () => {
    const experiments = [
      {
        run_id: "r1",
        param_changed: "x",
        dsr: "0.91",
        pbo: 0.42,
        metric_after: "1.5",
        status: "KEPT",
      },
      {
        run_id: "r1",
        param_changed: "y",
        dsr: "0.77",
        pbo: null,
        metric_after: "0.8",
        status: "DISCARDED",
      },
    ];
    const candidates = mapExperimentsToCandidates(experiments);
    expect(candidates[0].pbo).toBe(0.42);
    expect(candidates[1].pbo).toBeNull();
    expect(candidates[0].dsr).toBe(0.91);
    expect(candidates[0].realized_pnl_if_promoted).toBe(1.5 * 100_000);
    expect(candidates[1].realized_pnl_if_promoted).toBe(0.8 * 100_000);
  });
});
