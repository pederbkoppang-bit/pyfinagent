/**
 * phase-10.5.3 RedLineMonitor.
 *
 * Props-driven Recharts chart over the phase-10.5.0 `/api/sovereign/red-line`
 * series + events shape. The sovereign page owns the fetch and passes
 * `series`, `events`, `window`, and `onWindowChange` callbacks.
 *
 * Window-selector: 3-button group for 7d / 30d / 90d.
 * ReferenceLine: horizontal y=0 baseline marking the kill-switch
 *   threshold (delta-mode interpretation).
 * ReferenceDot: one per event (kill_switch / parameter flip annotations).
 */
"use client";

import { BentoCard } from "@/components/BentoCard";
import { TrendDown } from "@phosphor-icons/react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type RedLineWindow = "7d" | "30d" | "90d";

export interface RedLinePoint {
  date: string;
  nav: number;
  source: string;
}

export interface RedLineEvent {
  date: string;
  label: string;
  detail?: string | null;
}

export interface RedLineMonitorProps {
  series: RedLinePoint[];
  events: RedLineEvent[];
  window: RedLineWindow;
  onWindowChange: (w: RedLineWindow) => void;
  /** phase-10.5.7 compact variant: hides the window selector and lets
   * the chart container fill its parent (`h-full` instead of `h-64`).
   * Used by the homepage hero embed. */
  compact?: boolean;
}

const WINDOW_OPTIONS: RedLineWindow[] = ["7d", "30d", "90d"];

export function RedLineMonitor({
  series,
  events,
  window,
  onWindowChange,
  compact = false,
}: RedLineMonitorProps) {
  // Compute a numeric y for each event so the dot lands on the line.
  // Match the event date to the closest series point; fall back to
  // first NAV if unmatched (defensive).
  const navByDate = new Map<string, number>();
  for (const p of series) navByDate.set(p.date, p.nav);
  const fallbackNav = series.length > 0 ? series[0].nav : 0;

  return (
    <BentoCard>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <TrendDown size={18} className="text-rose-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">Red Line Monitor</h3>
        {!compact && (
          <div
            data-testid="window-selector"
            className="ml-auto inline-flex items-center gap-1 rounded-md bg-navy-900/60 p-1"
          >
            {WINDOW_OPTIONS.map((w) => (
              <button
                key={w}
                type="button"
                data-window={w}
                aria-pressed={w === window}
                onClick={() => onWindowChange(w)}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  w === window
                    ? "bg-sky-500/15 text-sky-400"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {w}
              </button>
            ))}
          </div>
        )}
        {compact && (
          <span className="ml-auto font-mono text-xs text-slate-500">{window}</span>
        )}
      </div>

      <div
        data-testid="red-line-chart"
        role="img"
        aria-label={`Red-line NAV chart, ${window} window, ${series.length} points, ${events.length} events`}
        className={compact ? "h-full min-h-[16rem]" : "h-64"}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={series}
            margin={{ top: 8, right: 16, bottom: 16, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} />
            <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 8,
                color: "#e2e8f0",
              }}
            />
            {/* phase-10.5.3 reference line at the kill-switch baseline (delta mode). */}
            <ReferenceLine
              y={0}
              stroke="#f43f5e"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: "kill-switch", position: "right", fill: "#f43f5e", fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="nav"
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            {events.map((ev, i) => (
              <ReferenceDot
                key={`${ev.date}-${i}`}
                x={ev.date}
                y={navByDate.get(ev.date) ?? fallbackNav}
                r={5}
                fill="#fbbf24"
                stroke="#0f172a"
                strokeWidth={1}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-[11px] text-slate-500">
        Window <span className="font-mono text-slate-400">{window}</span> ·
        {" "}{series.length} points · {events.length} events
      </p>
    </BentoCard>
  );
}
