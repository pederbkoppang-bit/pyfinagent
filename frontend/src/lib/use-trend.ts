// phase-76 (2026-05-26) -- trend tracker for NumberFlow color tint.
//
// NumberFlow's custom element does NOT ship `::part(up)` /
// `::part(down)` selectors (verified by researcher
// ae08ef2407507449a against lite.ts source). To color the changing
// digits emerald (up-tick) or rose (down-tick), we set a custom
// `data-pyfa-trend` host attribute on each NumberFlow consumer and
// target it via `number-flow-react[data-pyfa-trend="up"]::part(digit)`
// in globals.css. (Cycle 77 bugfix: the React wrapper renders
// <number-flow-react>, NOT <number-flow> -- cycle 76 had the wrong
// element name and the tint silently never matched.)
//
// The hook tracks the previous value via useRef, derives "up" /
// "down" / "flat" on each tick, and auto-resets to "flat" after
// 900ms (matches the CSS keyframe + NumberFlow transformTiming
// duration; cycle 77 bump from 700ms per researcher
// a750bbbd767273170) so consecutive identical ticks do not re-flash.
// setTimeout is cleared on subsequent change AND on unmount to
// prevent leaks (same cleanup discipline as the removed cycle-74
// useFlashOnChange hook).

import { useEffect, useRef, useState } from "react";

export type Trend = "up" | "down" | "flat";

export function useTrend(
  value: number | null | undefined,
  durationMs: number = 900,
): Trend {
  const prev = useRef<number | null | undefined>(value);
  const [trend, setTrend] = useState<Trend>("flat");
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (value == null || prev.current == null) {
      prev.current = value;
      return;
    }
    if (value === prev.current) return;
    const next: Trend = value > prev.current ? "up" : "down";
    prev.current = value;
    setTrend(next);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setTrend("flat");
      timeoutRef.current = null;
    }, durationMs);
  }, [value, durationMs]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return trend;
}
