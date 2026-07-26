/**
 * phase-36.7: DISARMED must never render as ACTIVE.
 *
 * Between the 2026-07-26 audit-log rotation and the phase-36.7 fix, the backend
 * returned sod_nav:null / peak_nav:null with a breach dict carrying no marker at
 * all, so both of these components computed `alarm = paused || any_breached` =
 * false and painted an emerald "ACTIVE" badge next to "daily 0.00% / 4%   trail
 * 0.00% / 10%" -- while no drawdown of any size could pause trading. Absence
 * rendered as the most reassuring possible reading.
 *
 * Same rule as phase-80.36 on the cockpit: DISCRIMINATE ON PRESENCE, NEVER ON
 * VALUE. Both directions are asserted here:
 *   - armed:false            -> DISARMED (amber), Resume disabled
 *   - armed:true, pct 0.00   -> ACTIVE (a book at its high-water mark is healthy)
 *   - armed absent (old API) -> ACTIVE, i.e. today's behaviour, unchanged
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, act, screen } from "@testing-library/react";

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

// The measured 2026-07-26 payload from the running backend, plus the phase-36.7
// keys. `current_nav` 23838.16 is the real live NAV.
const DISARMED = {
  paused: false,
  pause_reason: null,
  sod_nav: null,
  sod_date: null,
  peak_nav: null,
  current_nav: 23838.16,
  breach: {
    daily_loss_breached: false,
    daily_loss_pct: 0.0,
    daily_loss_limit_pct: 4.0,
    trailing_dd_breached: false,
    trailing_dd_pct: 0.0,
    trailing_dd_limit_pct: 10.0,
    any_breached: false,
    armed: false,
    daily_baseline_missing: true,
    trailing_baseline_missing: true,
  },
  thresholds: { daily_loss_limit_pct: 4.0, trailing_dd_limit_pct: 10.0 },
};

// ARMED and sitting exactly at the high-water mark: 0.00% on both legs is a
// legitimate HEALTHY reading, not absence.
const ARMED_AT_HIGH_WATER = {
  ...DISARMED,
  sod_nav: 23838.19,
  sod_date: "2026-07-26",
  peak_nav: 23838.16,
  breach: {
    ...DISARMED.breach,
    armed: true,
    daily_baseline_missing: false,
    trailing_baseline_missing: false,
  },
};

// An older backend that predates the phase-36.7 keys entirely.
const LEGACY_NO_MARKER = {
  ...DISARMED,
  sod_nav: 23838.19,
  sod_date: "2026-07-26",
  peak_nav: 23838.16,
  breach: {
    daily_loss_breached: false,
    daily_loss_pct: 0.0,
    daily_loss_limit_pct: 4.0,
    trailing_dd_breached: false,
    trailing_dd_pct: 0.0,
    trailing_dd_limit_pct: 10.0,
    any_breached: false,
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
  vi.useRealTimers();
});

describe("KillSwitchPanel -- phase-36.7 disarmed state", () => {
  it("renders DISARMED, not ACTIVE, when armed:false", async () => {
    const { container } = await mountPanel(DISARMED);
    const txt = container.textContent ?? "";
    expect(txt).toContain("DISARMED");
    expect(txt).not.toContain("ACTIVE");
  });

  it("shows an em-dash, never 0.00%, for a leg with no baseline", async () => {
    const { container } = await mountPanel(DISARMED);
    const txt = container.textContent ?? "";
    // The pre-fix readout literally rendered "daily 0.00% / 4%".
    expect(txt).not.toContain("0.00%");
    expect(txt).toContain("—");
    // Thresholds still shown, and still the authoritative 4 / 10 pair.
    expect(txt).toContain("4%");
    expect(txt).toContain("10%");
  });

  it("paints amber (not emerald, not rose) while disarmed but not paused", async () => {
    const { container } = await mountPanel(DISARMED);
    const html = container.innerHTML;
    expect(html).toContain("amber");
    expect(html).not.toContain("bg-emerald-500/20");
  });

  it("STILL renders ACTIVE for a genuine 0.00% drawdown -- presence, not value", async () => {
    const { container } = await mountPanel(ARMED_AT_HIGH_WATER);
    const txt = container.textContent ?? "";
    expect(txt).toContain("ACTIVE");
    expect(txt).not.toContain("DISARMED");
    expect(txt).toContain("0.00%");
    expect(container.innerHTML).toContain("bg-emerald-500/20");
  });

  it("keeps today's behaviour when the backend omits the marker entirely", async () => {
    const { container } = await mountPanel(LEGACY_NO_MARKER);
    const txt = container.textContent ?? "";
    expect(txt).toContain("ACTIVE");
    expect(txt).not.toContain("DISARMED");
    expect(txt).toContain("0.00%");
  });

  it("disables Resume while disarmed and paused (mirrors the server 409)", async () => {
    await mountPanel({ ...DISARMED, paused: true, pause_reason: "manual" });
    const btn = screen.getByRole("button", { name: /resume/i }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.title).toMatch(/DISARMED/);
  });

  // phase-36.12: the two operator tooltips used to promise "The next cycle
  // re-anchors them" -- i.e. they recommended the very defect 36.12 fixes. The
  // /DISARMED/ regex above is satisfied by BOTH the old and the new wording, so it
  // could not have caught a revert. These assert the rendered DOM text itself.
  it("the resume-button tooltip describes the BLOCK, not an automatic re-anchor", async () => {
    await mountPanel({ ...DISARMED, paused: true, pause_reason: "manual" });
    const btn = screen.getByRole("button", { name: /resume/i }) as HTMLButtonElement;
    expect(btn.title).toContain("blocks new orders");
    expect(btn.title).not.toContain("re-anchors them");
    expect(btn.title).not.toContain("next cycle re-anchor");
  });

  it("the DISARMED badge tooltip describes the BLOCK, not an automatic re-anchor", async () => {
    const { container } = await mountPanel(DISARMED);
    const badge = Array.from(container.querySelectorAll("[title]")).find((el) =>
      (el.getAttribute("title") ?? "").startsWith("DISARMED:"),
    );
    expect(badge, "the DISARMED badge tooltip is not in the DOM").toBeTruthy();
    const title = badge!.getAttribute("title") ?? "";
    expect(title).toContain("blocks new orders");
    expect(title).toContain("baseline_anchor_on_lost_history");
    expect(title).not.toContain("re-anchors them");
    expect(title).not.toContain("Resume is blocked until the next cycle");
  });

  it("leaves Resume enabled when armed, paused and healthy", async () => {
    await mountPanel({
      ...ARMED_AT_HIGH_WATER,
      paused: true,
      pause_reason: "manual",
    });
    const btn = screen.getByRole("button", { name: /resume/i }) as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });
});

describe("OpsStatusBar -- phase-36.7 disarmed state", () => {
  it("renders DISARMED in the Kill segment when armed:false", async () => {
    const { container } = await mountBar(DISARMED);
    const txt = container.textContent ?? "";
    expect(txt).toContain("DISARMED");
    expect(txt).not.toContain("ACTIVE");
    expect(container.innerHTML).not.toContain("bg-emerald-500/15");

    // SCOPED, not container-wide. `mountBar` deliberately rejects three
    // sibling data sources (Gate/Freshness/CyclesHistory) so THEY each render
    // their own em-dash -- a container-wide `toContain("—")` assertion is
    // vacuous, since it passes even if the Kill segment's readout never uses
    // one at all. The Kill segment's daily/trailing readout is the only
    // element carrying a "Daily:" title; scope to it specifically.
    const killReadout = container.querySelector('[title^="Daily:"]');
    expect(killReadout, "Kill segment daily/trailing readout not found").not.toBeNull();
    expect(killReadout!.textContent).toContain("—");
    expect(killReadout!.textContent).not.toMatch(/0\.0%/);
  });

  it("STILL renders ACTIVE for a genuine 0.0% drawdown", async () => {
    const { container } = await mountBar(ARMED_AT_HIGH_WATER);
    const txt = container.textContent ?? "";
    expect(txt).toContain("ACTIVE");
    expect(txt).not.toContain("DISARMED");
    expect(txt).toContain("0.0%");
  });

  it("keeps today's behaviour when the marker is absent", async () => {
    const { container } = await mountBar(LEGACY_NO_MARKER);
    const txt = container.textContent ?? "";
    expect(txt).toContain("ACTIVE");
    expect(txt).not.toContain("DISARMED");
  });

  it("disables the Resume affordance while disarmed and paused", async () => {
    await mountBar({ ...DISARMED, paused: true, pause_reason: "manual" });
    const btn = screen.getByRole("button", {
      name: /resume paper trading/i,
    }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});
