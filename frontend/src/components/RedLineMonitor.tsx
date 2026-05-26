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
import { TrendDown } from "@/lib/icons";
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
  /** phase-73 (2026-05-26): chart-side SSOT overlay. When supplied AND
   * today's date is later than the last actual snapshot in `series`,
   * append a synthetic "live_now" point and render a pulsating marker
   * + dashed connector. Lets the chart honor the live cockpit NAV the
   * cycle-72 LivePortfolioProvider exposes, instead of silently
   * forward-filling the stale persisted snapshot. */
  liveNav?: number | null;
  /** Freshness band drives the live-marker color (green/amber/rose). */
  liveBand?: "green" | "amber" | "red" | "unknown";
}

const WINDOW_OPTIONS: RedLineWindow[] = ["7d", "30d", "90d"];

// phase-73: per-band color tokens for the live-now marker. Static literals
// so Tailwind JIT and Recharts both pick them up (cycle-68 JIT-safe lesson).
const LIVE_MARKER_COLOR: Record<
  NonNullable<RedLineMonitorProps["liveBand"]>,
  string
> = {
  green: "#34d399", // emerald-400
  amber: "#fbbf24", // amber-400
  red: "#fb7185", // rose-400
  unknown: "#94a3b8", // slate-400
};

export function RedLineMonitor({
  series,
  events,
  window,
  onWindowChange,
  compact = false,
  liveNav,
  liveBand,
}: RedLineMonitorProps) {
  // phase-73: chart-side SSOT overlay. When liveNav is supplied AND today's
  // ISO date is strictly later than the last actual snapshot date, append a
  // synthetic { source: "live_now" } point so the chart ends at the live
  // value instead of forward-filling the stale snapshot. The rendering
  // distinguishes this point with a pulsating ReferenceDot + dashed
  // connector so the operator sees that the last segment is "now", not
  // "today's close".
  const todayIso = new Date().toISOString().slice(0, 10);
  const lastActual = series.length > 0 ? series[series.length - 1] : null;
  const shouldOverlay =
    liveNav != null &&
    liveNav > 0 &&
    lastActual != null &&
    lastActual.date < todayIso;
  const overlaySeries: RedLinePoint[] = shouldOverlay
    ? [...series, { date: todayIso, nav: liveNav as number, source: "live_now" }]
    : series;
  const liveColor = liveBand ? LIVE_MARKER_COLOR[liveBand] : LIVE_MARKER_COLOR.amber;

  // Compute a numeric y for each event so the dot lands on the line.
  // Match the event date to the closest series point; fall back to
  // first NAV if unmatched (defensive).
  const navByDate = new Map<string, number>();
  for (const p of overlaySeries) navByDate.set(p.date, p.nav);
  const fallbackNav = overlaySeries.length > 0 ? overlaySeries[0].nav : 0;

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
        className={compact ? "h-72" : "h-64"}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={overlaySeries}
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
              formatter={(value: number, name: string, item: { payload?: RedLinePoint }) => {
                const isLive = item?.payload?.source === "live_now";
                return [`${value.toFixed(2)}${isLive ? " (live now)" : ""}`, name];
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
              // phase-73: dashed segment between the last actual snapshot
              // and the synthetic "live_now" point. `segment` prop is a
              // Recharts pattern -- here we use strokeDasharray on a
              // per-segment basis by overlaying a 2nd Line for the live
              // segment (cleaner than a custom shape).
            />
            {shouldOverlay && lastActual && (
              <Line
                type="monotone"
                dataKey="nav"
                data={[
                  { date: lastActual.date, nav: lastActual.nav, source: lastActual.source },
                  { date: todayIso, nav: liveNav as number, source: "live_now" },
                ]}
                stroke={liveColor}
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={false}
                isAnimationActive={false}
                legendType="none"
              />
            )}
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
            {/* phase-73: pulsating "live now" marker. Recharts ReferenceDot with
                an inline SVG animate driving radius oscillation (Gaurav Gupta
                Medium pattern; researcher Section 7). */}
            {shouldOverlay && (
              <ReferenceDot
                x={todayIso}
                y={liveNav as number}
                r={6}
                fill={liveColor}
                stroke="#0f172a"
                strokeWidth={1.5}
                ifOverflow="visible"
                shape={({ cx, cy }: { cx?: number; cy?: number }) => (
                  <g>
                    {/* Outer halo */}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={6}
                      fill={liveColor}
                      opacity={0.35}
                    >
                      <animate
                        attributeName="r"
                        values="6;12;6"
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                      <animate
                        attributeName="opacity"
                        values="0.35;0.05;0.35"
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                    </circle>
                    {/* Core dot */}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill={liveColor}
                      stroke="#0f172a"
                      strokeWidth={1.5}
                    />
                  </g>
                )}
              />
            )}
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
