"use client";

// phase-44.7 -- event-rate mini sparkline above the cron log container.
//
// Per researcher topic 2: Tremor SparkAreaChart binned by minute over
// the last 60 buckets. Inline Tailwind-SVG fallback (cycle-64 MiniSpark
// pattern) keeps the bundle small + avoids the Tremor BarList
// per-item-color limitation that bit us in cycle 63. Renders nothing
// when the log has <2 events.

import { useMemo } from "react";

export interface LogEventRateSparkProps {
  // Log lines used for the rate calculation. Each line is expected to
  // start with an ISO-ish timestamp; if absent, the rate is undefined
  // for that line (skipped). We tolerate noisy lines.
  lines: string[];
  className?: string;
}

// Look for an ISO 8601 timestamp anywhere near the start of the line.
// Capture group covers the date+time plus optional ms + Z so timezone
// information is preserved when present (otherwise Date.parse treats
// the input as LOCAL time which is wrong for server-side logs).
const TS_PATTERN = /(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?Z?)/;

function extractTimestamp(line: string): number | null {
  const m = line.match(TS_PATTERN);
  if (!m) return null;
  const t = Date.parse(m[1].replace(" ", "T"));
  return Number.isNaN(t) ? null : t;
}

export function LogEventRateSpark({ lines, className }: LogEventRateSparkProps) {
  const series = useMemo(() => {
    if (!lines || lines.length < 2) return [];
    // Bin events into 1-minute buckets relative to NOW (last 60 minutes).
    const now = Date.now();
    const buckets = new Array(60).fill(0);
    for (const line of lines) {
      const t = extractTimestamp(line);
      if (t === null) continue;
      const ageMin = Math.floor((now - t) / 60_000);
      if (ageMin < 0 || ageMin >= 60) continue;
      buckets[59 - ageMin] += 1;
    }
    return buckets;
  }, [lines]);

  if (series.length === 0 || series.every((v) => v === 0)) return null;

  const max = Math.max(...series);
  const W = 600;
  const H = 36;
  const stepX = W / (series.length - 1);
  const points = series
    .map((v, i) => {
      const x = i * stepX;
      const y = H - (max > 0 ? (v / max) * H : 0);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const areaPoints = `0,${H} ${points} ${W},${H}`;

  return (
    <div
      className={`rounded-md border border-navy-700 bg-navy-800/40 p-2 ${className ?? ""}`}
      role="region"
      aria-label="Log event rate over last 60 minutes"
    >
      <div className="mb-1 flex items-center justify-between text-[10px] text-slate-500">
        <span>Event rate (last 60 min)</span>
        <span className="font-mono">peak {max}/min</span>
      </div>
      <svg
        aria-hidden="true"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="h-9 w-full"
      >
        <polygon points={areaPoints} fill="rgba(56, 189, 248, 0.18)" />
        <polyline
          fill="none"
          stroke="#38bdf8"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
    </div>
  );
}
