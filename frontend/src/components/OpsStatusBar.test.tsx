import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, act } from "@testing-library/react";
import { OpsStatusBar } from "./OpsStatusBar";

const {
  getPaperGateMock,
  getPaperKillSwitchStateMock,
  getPaperFreshnessMock,
  getPaperCyclesHistoryMock,
} = vi.hoisted(() => ({
  getPaperGateMock: vi.fn(),
  getPaperKillSwitchStateMock: vi.fn(),
  getPaperFreshnessMock: vi.fn(),
  getPaperCyclesHistoryMock: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getPaperGate: getPaperGateMock,
  getPaperKillSwitchState: getPaperKillSwitchStateMock,
  getPaperFreshness: getPaperFreshnessMock,
  getPaperCyclesHistory: getPaperCyclesHistoryMock,
  postPaperKillSwitchAction: vi.fn(),
}));

function rejectAllFour() {
  getPaperGateMock.mockRejectedValue(new Error("down"));
  getPaperKillSwitchStateMock.mockRejectedValue(new Error("down"));
  getPaperFreshnessMock.mockRejectedValue(new Error("down"));
  getPaperCyclesHistoryMock.mockRejectedValue(new Error("down"));
}

beforeEach(() => {
  // vi.restoreAllMocks() below only affects vi.spyOn() mocks -- these
  // hoisted vi.fn()s keep call-count history across tests otherwise.
  vi.resetAllMocks();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("OpsStatusBar failRef circuit breaker (phase-75.12 frontend-05)", () => {
  it("renders the stale segment after 5 all-null rounds and stops the interval poll", async () => {
    vi.useFakeTimers();
    rejectAllFour();

    const { container } = render(<OpsStatusBar />);

    // Mount fire = round #1.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(container.querySelector('[data-testid="ops-stale-segment"]')).toBeNull();

    // 3 more interval rounds (#2, #3, #4) -- still not stale.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000 * 3);
    });
    expect(container.querySelector('[data-testid="ops-stale-segment"]')).toBeNull();
    expect(getPaperGateMock).toHaveBeenCalledTimes(4);

    // 5th round trips the circuit.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getPaperGateMock).toHaveBeenCalledTimes(5);
    expect(container.querySelector('[data-testid="ops-stale-segment"]')).not.toBeNull();
    expect(container.textContent?.toLowerCase()).toContain("stale");

    // A 6th interval tick must NOT poll -- the interval-driven path stops.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getPaperGateMock).toHaveBeenCalledTimes(5);
  });

  it("recovers (clears the stale segment) on a visibility-regain refresh that succeeds", async () => {
    vi.useFakeTimers();
    rejectAllFour();

    const { container } = render(<OpsStatusBar />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000 * 4);
    });
    expect(container.querySelector('[data-testid="ops-stale-segment"]')).not.toBeNull();

    // Now the backend recovers; a visibility-regain event (not the dead
    // interval) is the recovery trigger.
    getPaperGateMock.mockResolvedValue({
      booleans: {},
      details: { n_round_trips: 0, n_obs: 0, latest_reconciliation_divergence_pct: 0, realized_max_dd_pct: 0 },
      thresholds: { trades: 100, psr: 0.95, psr_sustained_days: 30, dsr: 0.95, sr_gap: 0.3, max_dd_pct: 20 },
      promote_eligible: false,
    });
    getPaperKillSwitchStateMock.mockResolvedValue({
      paused: false,
      pause_reason: null,
      current_nav: 100_000,
      breach: { any_breached: false, daily_loss_pct: 0, daily_loss_limit_pct: 4, trailing_dd_pct: 0, trailing_dd_limit_pct: 10 },
    });
    getPaperFreshnessMock.mockResolvedValue({
      sources: {},
      heartbeat: { band: "green" },
      thresholds: { warn_ratio: 1.5, critical_ratio: 3 },
    });
    getPaperCyclesHistoryMock.mockResolvedValue({ cycles: [] });

    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(container.querySelector('[data-testid="ops-stale-segment"]')).toBeNull();
  });
});
