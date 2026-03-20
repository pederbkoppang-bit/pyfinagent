"use client";

import { useState } from "react";
import type { Citation } from "@/lib/types";

// ── Source tag color mapping ─────────────────────────────────────

const SOURCE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  YFIN: { bg: "bg-sky-500/10 border-sky-500/20",  text: "text-sky-400",    label: "Yahoo Finance"     },
  SEC:  { bg: "bg-amber-500/10 border-amber-500/20", text: "text-amber-400",  label: "SEC EDGAR"        },
  FRED: { bg: "bg-violet-500/10 border-violet-500/20", text: "text-violet-400", label: "Federal Reserve"  },
  AV:   { bg: "bg-emerald-500/10 border-emerald-500/20", text: "text-emerald-400", label: "Alpha Vantage" },
};

function getSourceStyle(source: string) {
  const upper = source.toUpperCase();
  if (upper in SOURCE_STYLES) return SOURCE_STYLES[upper];
  // URL from web grounding
  if (source.startsWith("http")) return { bg: "bg-slate-500/10 border-slate-500/20", text: "text-slate-400", label: "Web" };
  return { bg: "bg-slate-500/10 border-slate-500/20", text: "text-slate-400", label: source };
}

// ── CitationBadge ────────────────────────────────────────────────

export function CitationBadge({ source }: { source: string }) {
  const style = getSourceStyle(source);
  const label = source.startsWith("http") ? "WEB" : source.toUpperCase();
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${style.bg} ${style.text}`}
      title={getSourceStyle(source).label}
    >
      {label}
    </span>
  );
}

// ── CitationList ─────────────────────────────────────────────────

export function CitationList({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState(false);

  if (!citations || citations.length === 0) return null;

  const visible = expanded ? citations : citations.slice(0, 4);

  return (
    <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Data Citations ({citations.length})
        </span>
        {citations.length > 4 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[11px] text-sky-400 hover:text-sky-300 transition-colors"
          >
            {expanded ? "Show less" : `+${citations.length - 4} more`}
          </button>
        )}
      </div>
      <div className="space-y-1.5">
        {visible.map((c, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            <CitationBadge source={c.source} />
            <span className="flex-1 text-slate-400 leading-relaxed">
              {c.claim}
              {c.value && (
                <span className="ml-1.5 font-mono text-slate-300">{c.value}</span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
