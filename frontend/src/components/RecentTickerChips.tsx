"use client";

// phase-44.6 -- recent-tickers chip row.
//
// Persists last N tickers (default 5) to localStorage, deduped + LRU.
// Each chip is clickable -> calls `onSelect(ticker)`. Used on the
// /signals page (and reusable on future ticker-input surfaces).
//
// Per research_brief topic 4: role="group" (not toolbar -- click-to-fill
// doesn't need arrow-key nav). WCAG 2.2 24px target-size via py-1.5
// + leading numeric class.

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "pyfinagent.signals.recentTickers";
const MAX_CHIPS = 5;

function readStorage(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((x): x is string => typeof x === "string" && x.length > 0)
      .slice(0, MAX_CHIPS);
  } catch {
    return [];
  }
}

function writeStorage(tickers: string[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(tickers));
  } catch {
    /* localStorage quota / disabled -- ignore */
  }
}

export interface RecentTickerChipsProps {
  // Called when a chip is clicked. The consumer is responsible for
  // updating its own input state + triggering the fetch.
  onSelect: (ticker: string) => void;
  // External notification that a ticker was just submitted. The component
  // updates its own list + persists. Pass null/undefined to not record.
  recentlySubmitted?: string | null;
  // Override storage key for tests / other ticker surfaces.
  storageKey?: string;
}

export function RecentTickerChips({
  onSelect,
  recentlySubmitted,
  storageKey = STORAGE_KEY,
}: RecentTickerChipsProps) {
  const [chips, setChips] = useState<string[]>([]);

  // Hydrate from localStorage on mount (SSR-safe via the useEffect guard).
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setChips(
            parsed
              .filter((x): x is string => typeof x === "string" && x.length > 0)
              .slice(0, MAX_CHIPS),
          );
        }
      }
    } catch {
      /* ignore parse errors */
    }
  }, [storageKey]);

  // When a parent reports a submission, prepend the ticker.
  useEffect(() => {
    if (!recentlySubmitted) return;
    const normalized = recentlySubmitted.trim().toUpperCase();
    if (!normalized) return;
    setChips((prev) => {
      const next = [normalized, ...prev.filter((t) => t !== normalized)].slice(0, MAX_CHIPS);
      if (typeof window !== "undefined") {
        try {
          window.localStorage.setItem(storageKey, JSON.stringify(next));
        } catch {
          /* ignore quota errors */
        }
      }
      return next;
    });
  }, [recentlySubmitted, storageKey]);

  const handleClick = useCallback(
    (t: string) => {
      onSelect(t);
    },
    [onSelect],
  );

  if (chips.length === 0) return null;

  return (
    <div
      role="group"
      aria-label="Recent tickers"
      className="mb-4 flex flex-wrap gap-2"
    >
      <span className="text-xs uppercase tracking-wider text-slate-500 self-center">Recent</span>
      {chips.map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => handleClick(t)}
          className="rounded-full border border-navy-700 bg-navy-800/70 px-3 py-1.5 font-mono text-xs text-slate-200 hover:bg-navy-700/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px] min-w-[24px]"
          aria-label={`Analyze ${t}`}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

// Exported for tests + external consumers who want to read the persisted list.
export const _internals = { readStorage, writeStorage, STORAGE_KEY, MAX_CHIPS };
