"use client";

// goal-multimarket-ux #1: global market filter. WAI-ARIA APG radiogroup (single-
// select): container role="radiogroup", each option role="radio" + aria-checked, a
// roving tabindex, and Arrow/Home/End keyboard nav with selection-follows-focus.
// (Not a tablist -- a tablist is for switching content panels; this filters the
// current view. Per researcher brief + W3C APG.) Styled like the tab-pill pattern
// (frontend-layout.md §5). NO flag emoji -- a colored dot + the market code; the
// exchange name is exposed via the title tooltip.
//
// Options are data-driven: the layout passes the markets present in holdings (plus
// the core configured set), so the control shows All + US/EU/KR today and grows to
// include the Nordics (NO/SE/DK/FI/IS) once those positions exist.

import { useMemo, useRef, type KeyboardEvent } from "react";
import { clsx } from "clsx";
import { MARKET_DOT_CLASS, MARKET_EXCHANGE } from "@/lib/format";

export function MarketFilter({
  value,
  onChange,
  markets,
  sessionOpen,
  className,
}: {
  value: string;
  onChange: (market: string) => void;
  // Market codes present (canonical order), excluding "ALL".
  markets: string[];
  // goal-market-filter-in-gate-bar: optional per-market open/closed map. When
  // present, each non-"All" pill's dot reflects session state (emerald=open,
  // slate=closed) instead of the per-market identity colour, folding the
  // retired MarketSessionStrip signal into the pills. Pass `undefined` (the
  // pre-mount state) to keep the neutral per-market dot and avoid a hydration
  // mismatch. When absent entirely (e.g. a future caller), the per-market
  // MARKET_DOT_CLASS dot is used as before.
  sessionOpen?: Record<string, boolean>;
  className?: string;
}) {
  const options = useMemo(
    () => ["ALL", ...markets.filter((m) => m && m !== "ALL")],
    [markets],
  );
  const btnRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const activeIdx = Math.max(0, options.indexOf(value));

  const focusAndSelect = (idx: number) => {
    const next = (idx + options.length) % options.length;
    btnRefs.current[next]?.focus();
    onChange(options[next]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      focusAndSelect(idx + 1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      focusAndSelect(idx - 1);
    } else if (e.key === "Home") {
      e.preventDefault();
      focusAndSelect(0);
    } else if (e.key === "End") {
      e.preventDefault();
      focusAndSelect(options.length - 1);
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label="Filter by market"
      className={clsx("inline-flex flex-wrap gap-1 rounded-lg bg-navy-800/50 p-1", className)}
    >
      {options.map((opt, i) => {
        const checked = opt === value;
        const isAll = opt === "ALL";
        // Dot colour: when a session map is supplied, emerald=open / slate=closed
        // (the folded MarketSessionStrip signal); otherwise the per-market
        // identity colour. `sessionOpen === undefined` (pre-mount) falls through
        // to the per-market colour so the first client render matches the server.
        const open = sessionOpen ? sessionOpen[opt] : undefined;
        const dot = isAll
          ? null
          : open === undefined
            ? (MARKET_DOT_CLASS[opt] ?? "bg-slate-400")
            : open
              ? "bg-emerald-400"
              : "bg-slate-600";
        const exchange = isAll ? "All markets" : (MARKET_EXCHANGE[opt] ?? opt);
        const title =
          isAll || open === undefined
            ? exchange
            : `${exchange} — ${open ? "OPEN" : "CLOSED"}`;
        return (
          <button
            key={opt}
            ref={(el) => {
              btnRefs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            tabIndex={i === activeIdx ? 0 : -1}
            title={title}
            onClick={() => onChange(opt)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={clsx(
              "flex items-center gap-1.5 min-h-[24px] rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40",
              checked
                ? "bg-sky-500/10 text-sky-400"
                : "text-slate-400 hover:text-slate-200",
            )}
          >
            {dot && (
              <span className={clsx("h-1.5 w-1.5 rounded-full", dot)} aria-hidden="true" />
            )}
            {isAll ? "All" : opt}
          </button>
        );
      })}
    </div>
  );
}
