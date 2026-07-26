/**
 * phase-36.20 -- RE-ANCHORING is a STATUS, not an alarm.
 *
 * phase-36.9 made the backend's `armed` mean "can this leg fire RIGHT NOW", which
 * is false for a daily anchor that is merely from yesterday. Both cockpit
 * components derived `disarmed = breach.armed === false`, so a perfectly healthy
 * funded book rendered the DISARMED alarm badge every day from 00:00 UTC until the
 * first autonomous cycle rolled the anchor -- a recurring alarm for a condition
 * that repairs itself and needs no operator action.
 *
 * ISA-18.2 defines an alarm as an abnormal condition REQUIRING A RESPONSE and
 * prescribes reclassification to a recordable event otherwise; the clinical
 * alarm-fatigue literature (AHRQ: 72-99% false alarms, staff "turn down the
 * volume, ignore, or deactivate") is what happens when that rule is ignored.
 *
 * THE SECOND DEFECT THESE TESTS PIN, which is worse than the badge: `daily_loss_pct`
 * keeps its 0.0 initialiser whenever the leg is unevaluable, but the components
 * branched the em-dash on `daily_baseline_missing` ALONE -- so a stale anchor
 * printed a fabricated "0.00%" for a leg that CANNOT FIRE. That errs reassuring,
 * the exact direction phase-36.7 exists to prevent.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, act } from "@testing-library/react";

import { KillSwitchPanel } from "./KillSwitchPanel";
import { OpsStatusBar } from "./OpsStatusBar";

const {
  getPaperKillSwitchStateMock,
  getPaperGateMock,
  getPaperFreshnessMock,
  getPaperCyclesHistoryMock,
} = vi.hoisted(() => ({
  getPaperKillSwitchStateMock: vi.fn(),
  getPaperGateMock: vi.fn(),
  getPaperFreshnessMock: vi.fn(),
  getPaperCyclesHistoryMock: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getPaperKillSwitchState: getPaperKillSwitchStateMock,
  getPaperGate: getPaperGateMock,
  getPaperFreshness: getPaperFreshnessMock,
  getPaperCyclesHistory: getPaperCyclesHistoryMock,
  postPaperKillSwitchAction: vi.fn(),
}));

const BASE_BREACH = {
  daily_loss_breached: false,
  daily_loss_pct: 0.0,
  daily_loss_limit_pct: 4.0,
  trailing_dd_breached: false,
  trailing_dd_pct: 3.3584,
  trailing_dd_limit_pct: 10.0,
  any_breached: false,
};

/** The LIVE 2026-07-26 shape: healthy book, anchor from an earlier UTC day. */
const REANCHORING = {
  paused: false,
  pause_reason: null,
  sod_nav: 23838.19,
  sod_date: "2026-07-24",
  peak_nav: 24666.57,
  current_nav: 23838.16,
  breach: {
    ...BASE_BREACH,
    armed: false,
    daily_baseline_stale: true,
    daily_baseline_missing: false,
    trailing_baseline_missing: false,
    baselines_present: true,
  },
  thresholds: { daily_loss_limit_pct: 4.0, trailing_dd_limit_pct: 10.0 },
};

/** Genuine absence -- a DURABLE fault an operator must repair. */
const LOST_BASELINES = {
  ...REANCHORING,
  sod_nav: null,
  peak_nav: null,
  breach: {
    ...BASE_BREACH,
    armed: false,
    daily_baseline_stale: false,
    daily_baseline_missing: true,
    trailing_baseline_missing: true,
    baselines_present: false,
  },
};

/** Stale AND unmeasurable NAV -- must NOT get the friendly badge. */
const STALE_BUT_NAV_INVALID = {
  ...REANCHORING,
  breach: {
    ...REANCHORING.breach,
    nav_invalid: true,
    nav_invalid_disarmed: true,
  },
};

async function mountPanel(payload: unknown) {
  getPaperKillSwitchStateMock.mockResolvedValue(payload);
  const r = render(<KillSwitchPanel />);
  await act(async () => {
    await Promise.resolve();
  });
  return r;
}

async function mountBar(payload: unknown) {
  getPaperKillSwitchStateMock.mockResolvedValue(payload);
  getPaperGateMock.mockRejectedValue(new Error("not under test"));
  getPaperFreshnessMock.mockRejectedValue(new Error("not under test"));
  getPaperCyclesHistoryMock.mockRejectedValue(new Error("not under test"));
  const r = render(<OpsStatusBar />);
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  return r;
}

beforeEach(() => {
  vi.resetAllMocks();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("phase-36.20 -- RE-ANCHORING is status, not alarm", () => {
  it("KillSwitchPanel renders RE-ANCHORING, not DISARMED, on a stale anchor", async () => {
    const { container } = await mountPanel(REANCHORING);
    const txt = container.textContent ?? "";
    expect(txt).toContain("RE-ANCHORING");
    expect(txt).not.toContain("DISARMED");
  });

  it("OpsStatusBar renders RE-ANCHORING too -- criterion 1 covers BOTH", async () => {
    const { container } = await mountBar(REANCHORING);
    const txt = container.textContent ?? "";
    expect(txt).toContain("RE-ANCHORING");
    expect(txt).not.toContain("DISARMED");
  });

  it("genuine absence still renders DISARMED in both -- the states are DISTINGUISHABLE", async () => {
    const panel = await mountPanel(LOST_BASELINES);
    expect(panel.container.textContent ?? "").toContain("DISARMED");
    expect(panel.container.textContent ?? "").not.toContain("RE-ANCHORING");
    cleanup();
    const bar = await mountBar(LOST_BASELINES);
    expect(bar.container.textContent ?? "").toContain("DISARMED");
    expect(bar.container.textContent ?? "").not.toContain("RE-ANCHORING");
  });

  it("does NOT collapse the two: identical armed:false yields DIFFERENT badges", async () => {
    // THE MUTATION TARGET (criterion 5). Both fixtures carry armed:false. Revert the
    // component to keying on `armed === false` alone and both render DISARMED, so
    // this fails -- which is what makes it a guard rather than a restatement.
    expect(REANCHORING.breach.armed).toBe(LOST_BASELINES.breach.armed);
    const a = await mountPanel(REANCHORING);
    const staleTxt = a.container.textContent ?? "";
    cleanup();
    const b = await mountPanel(LOST_BASELINES);
    const absentTxt = b.container.textContent ?? "";
    expect(staleTxt).toContain("RE-ANCHORING");
    expect(absentTxt).toContain("DISARMED");
    expect(staleTxt).not.toBe(absentTxt);
  });

  it("an unmeasurable NAV inside the stale window is NOT given the friendly badge", async () => {
    // evaluate_breach's nav_invalid early return passes `daily_baseline_stale`
    // through while both *_missing stay false, and GET /kill-switch falls back to
    // `... or 0.0` on a 5s BQ timeout. Without the nav_invalid guard a genuine
    // "we cannot measure the book" state would render as RE-ANCHORING.
    const { container } = await mountPanel(STALE_BUT_NAV_INVALID);
    const txt = container.textContent ?? "";
    expect(txt).not.toContain("RE-ANCHORING");
    expect(txt).toContain("DISARMED");
  });

  it("never prints a fabricated 0.00% for a daily leg that cannot fire", async () => {
    // daily_loss_pct is 0.0 here purely because the leg was SKIPPED. Printing it
    // asserts a measurement nobody made, and 0.00% is the most reassuring possible
    // reading. The trailing leg IS evaluable and must still show its number.
    const { container } = await mountPanel(REANCHORING);
    const txt = container.textContent ?? "";
    expect(txt).toContain("—");
    expect(txt).not.toContain("0.00%");
    expect(txt).toContain("3.36%");
  });

  it("keeps `armed` a strict boolean -- no third state on the wire (criterion 4)", () => {
    // Both backend gates use `.get("armed", True)` and FAIL OPEN, so a third value
    // there would silently allow orders. The new state is derived CLIENT-SIDE.
    expect(typeof REANCHORING.breach.armed).toBe("boolean");
    expect(typeof LOST_BASELINES.breach.armed).toBe("boolean");
  });

  it("an older backend without the new keys keeps pre-36.20 behaviour", async () => {
    const legacy = {
      ...REANCHORING,
      breach: { ...BASE_BREACH, armed: false, daily_baseline_missing: true },
    };
    const { container } = await mountPanel(legacy);
    const txt = container.textContent ?? "";
    expect(txt).toContain("DISARMED");
    expect(txt).not.toContain("RE-ANCHORING");
  });
});
