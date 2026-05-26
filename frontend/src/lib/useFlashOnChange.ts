// phase-74 (2026-05-26) -- Google-Finance-style flash-on-change hook.
//
// Wraps numeric values that update with live stock prices so the cell
// briefly tints green (up-tick) or rose (down-tick) for ~500ms then
// fades back. Matches the lab49/react-value-flash pattern (200ms hold
// + 100ms fade ~= 500ms total; researcher a3f10c3c35c087f50 Section 1).
//
// JIT-safety (cycle-68 lesson): callers map the returned direction to
// a static literal Tailwind class. Do NOT compose classes via template
// strings.
//
// A11y: SC 2.3.3 does NOT apply (passive price ticks are not
// user-initiated interaction; researcher Section 3). SC 2.2.2 governs
// and a 500ms flash is well under the 5s ceiling. We honor
// `prefers-reduced-motion: reduce` defensively in the hook itself
// (returns null so callers skip the animation class entirely) AND in
// globals.css (sets `animation: none !important` on the .animate-flash-*
// classes -- defense in depth per researcher Section 3).
//
// ARIA: callers should set `aria-live="off"` on the flashing region
// (MDN stock-ticker default; do NOT announce every tick or screen
// readers flood).

import { useEffect, useRef, useState } from "react";

export type FlashDirection = "up" | "down" | null;

export interface UseFlashOnChangeOptions {
  // Decimal precision for the comparison. 0.001 -> 0.002 rounding noise
  // is filtered out when decimals=2 (researcher Section 1: lab49 ships
  // a similar decimals knob to prevent strobe on sub-cent jitter).
  decimals?: number;
  // Animation length. Matches the keyframe duration in tailwind.config.js.
  durationMs?: number;
}

export function useFlashOnChange(
  value: number | null | undefined,
  opts: UseFlashOnChangeOptions = {},
): FlashDirection {
  const { decimals = 2, durationMs = 500 } = opts;
  const previousRef = useRef<string | null>(null);
  const [direction, setDirection] = useState<FlashDirection>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (value == null) {
      previousRef.current = null;
      return;
    }
    const current = value.toFixed(decimals);
    const previous = previousRef.current;

    // First render: populate ref, no flash. Prevents an initial flash
    // when the SSR-empty value populates from null -> $124.50.
    if (previous === null) {
      previousRef.current = current;
      return;
    }
    if (current === previous) return;

    // Update prev BEFORE the early-return so subsequent ticks compare
    // against the latest, not the never-replaced first value.
    previousRef.current = current;

    // Respect prefers-reduced-motion (defense in depth -- globals.css
    // also overrides). Returns null so no animate-flash class lands.
    if (
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }

    const next: FlashDirection =
      Number(current) > Number(previous) ? "up" : "down";

    // Clear any prior in-flight flash so we don't strobe on rapid ticks
    // arriving inside the duration window.
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    // Set direction=null first, then promote to "up"/"down" in the next
    // frame. The intermediate null state removes any prior animate-flash
    // className so the CSS animation restarts (browsers do not re-run a
    // CSS animation if the class is identical across renders).
    setDirection(null);
    rafRef.current = requestAnimationFrame(() => {
      setDirection(next);
      rafRef.current = null;
    });

    timeoutRef.current = setTimeout(() => {
      setDirection(null);
      timeoutRef.current = null;
    }, durationMs);
  }, [value, decimals, durationMs]);

  // Cleanup on unmount: both timer + raf to prevent leaks across route
  // changes (researcher Section 2 hook-cleanup requirement).
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return direction;
}

// JIT-safe static literal map. Callers consume this via direct property
// lookup -- NEVER via template-string concatenation like
// `animate-flash-${dir}` (Tailwind v3 JIT does not scan template
// strings; cycle-68 lesson). Both class literals appear verbatim so JIT
// compiles them into the bundle.
export const FLASH_CLASS: Record<"up" | "down", string> = {
  up: "animate-flash-up",
  down: "animate-flash-down",
};

// Helper for consumers: convert direction to className (empty string
// when direction is null). Encapsulates the FLASH_CLASS lookup so
// consumer call sites stay one line.
export function flashClassName(direction: FlashDirection): string {
  if (direction === null) return "";
  return FLASH_CLASS[direction];
}
