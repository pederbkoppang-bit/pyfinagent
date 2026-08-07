import { describe, it, expect } from "vitest";
import { ageSecFromIso, bandFromAgeSec, bandFromMarkAgeSec } from "./paper-trading-utils";

/**
 * phase-61.3 — the mark-freshness band.
 *
 * These thresholds ARE the honesty signal: they decide whether a stale P&L reads as
 * normal, notable, or broken. Asserting only the chip's text (as the component spec
 * does) leaves them free to be anything, so they are pinned here directly — including
 * the property that makes the function worth having, namely that it is NOT
 * bandFromAgeSec.
 */

const H = 3600;

describe("bandFromMarkAgeSec (criterion 4)", () => {
  it("is green within one scheduled cycle", () => {
    expect(bandFromMarkAgeSec(0)).toBe("green");
    expect(bandFromMarkAgeSec(6 * H)).toBe("green");
    expect(bandFromMarkAgeSec(25 * H)).toBe("green");
  });

  it("turns amber past a day and stays amber across a weekend", () => {
    expect(bandFromMarkAgeSec(26 * H)).toBe("amber");
    expect(bandFromMarkAgeSec(72 * H)).toBe("amber");
  });

  it("turns red once no plausible schedule explains the gap", () => {
    expect(bandFromMarkAgeSec(74 * H)).toBe("red");
    expect(bandFromMarkAgeSec(30 * 24 * H)).toBe("red");
  });

  it("reports unknown for a missing age rather than guessing", () => {
    expect(bandFromMarkAgeSec(null)).toBe("unknown");
    expect(bandFromMarkAgeSec(undefined)).toBe("unknown");
  });

  it("is NOT the live-price band — that is the whole reason it exists", () => {
    // A 6-hour-old mark is healthy; a 6-hour-old live PRICE is not. If these two
    // ever agree at cycle scale, every healthy mark renders red and the signal dies.
    const sixHours = 6 * H;
    expect(bandFromMarkAgeSec(sixHours)).toBe("green");
    expect(bandFromAgeSec(sixHours)).toBe("red");
    expect(bandFromMarkAgeSec(sixHours)).not.toBe(bandFromAgeSec(sixHours));
  });
});

describe("ageSecFromIso", () => {
  it("computes age in seconds against an injected now", () => {
    const now = Date.parse("2026-08-07T12:00:00Z");
    expect(ageSecFromIso("2026-08-07T09:00:00Z", now)).toBe(3 * H);
  });

  it("returns null for absent or unparseable input", () => {
    expect(ageSecFromIso(null)).toBeNull();
    expect(ageSecFromIso(undefined)).toBeNull();
    expect(ageSecFromIso("not a timestamp")).toBeNull();
  });

  it("clamps a future timestamp to zero instead of a negative age", () => {
    const now = Date.parse("2026-08-07T12:00:00Z");
    expect(ageSecFromIso("2026-08-07T13:00:00Z", now)).toBe(0);
  });
});
