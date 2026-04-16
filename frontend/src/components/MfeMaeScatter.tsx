"use client";

import { useEffect, useState } from "react";
import { getPaperMfeMaeScatter } from "@/lib/api";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

interface ScatterPoint {
  ticker: string;
  entry_date: string | null;
  exit_date: string | null;
  mfe_pct: number;
  mae_pct: number;
  mae_abs_pct: number;
  capture_ratio: number;
  realized_pnl_pct: number;
  holding_days: number;
  leakage_flag: boolean;
}

interface Response {
  points: ScatterPoint[];
  summary: {
    edge_ratio: number;
    avg_capture_ratio: number;
    mfe_p75: number | null;
    leakage_threshold_capture: number;
    n_points: number;
    n_leakers: number;
  };
  computed_at: string;
}

/**
 * MFE x MAE scatter per closed round-trip (4.5.9).
 * Axes: X = |MAE|, Y = MFE (AFML Ch.13 convention).
 * Color: green = winner (realized_pnl>0), red = loser.
 * Stroke highlight: amber stroke when leakage_flag (high MFE + low capture).
 * Reference line: 45-degree MFE = |MAE|.
 */
export function MfeMaeScatter() {
  const [data, setData] = useState<Response | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getPaperMfeMaeScatter()
      .then((j) => {
        if (!cancelled) setData(j as Response);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "load failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <p className="text-sm text-slate-400">Loading exit-quality scatter...</p>;
  }
  if (error) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
        <p className="text-sm text-rose-300">{error}</p>
      </div>
    );
  }
  if (!data || data.summary.n_points === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="mt-4 text-lg text-slate-400">No closed round-trips yet</p>
        <p className="mt-1 text-sm text-slate-600">
          Exit-quality analysis needs at least one BUY-SELL pair.
        </p>
      </div>
    );
  }

  const winners = data.points
    .filter((p) => p.realized_pnl_pct > 0)
    .map((p) => ({ ...p, x: p.mae_abs_pct, y: p.mfe_pct, z: Math.max(4, Math.min(16, p.holding_days || 4)) }));
  const losers = data.points
    .filter((p) => p.realized_pnl_pct <= 0)
    .map((p) => ({ ...p, x: p.mae_abs_pct, y: p.mfe_pct, z: Math.max(4, Math.min(16, p.holding_days || 4)) }));
  const leakers = data.points
    .filter((p) => p.leakage_flag)
    .map((p) => ({ ...p, x: p.mae_abs_pct, y: p.mfe_pct, z: Math.max(6, Math.min(20, p.holding_days || 6)) }));

  const maxX = Math.max(1, ...data.points.map((p) => p.mae_abs_pct));
  const maxY = Math.max(1, ...data.points.map((p) => p.mfe_pct));
  const lim = Math.max(maxX, maxY);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Edge ratio" value={data.summary.edge_ratio.toFixed(2)} hint="mean(MFE / |MAE|)" />
        <StatCard
          label="Avg capture"
          value={`${(data.summary.avg_capture_ratio * 100).toFixed(0)}%`}
          hint="realized_pnl / MFE"
        />
        <StatCard label="Round-trips" value={String(data.summary.n_points)} hint="closed only" />
        <StatCard
          label="Leakers"
          value={String(data.summary.n_leakers)}
          hint={`capture < ${(data.summary.leakage_threshold_capture * 100).toFixed(0)}% & MFE > P75`}
          emphasize={data.summary.n_leakers > 0}
        />
      </div>

      <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-300">
          MFE vs |MAE| per round-trip
        </h3>
        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number"
              dataKey="x"
              name="MAE"
              domain={[0, lim]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "|MAE| (%)", fill: "#64748b", fontSize: 11, position: "insideBottom", offset: -6 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="MFE"
              domain={[0, lim]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "MFE (%)", fill: "#64748b", fontSize: 11, angle: -90, position: "insideLeft" }}
            />
            <ZAxis type="number" dataKey="z" range={[20, 240]} />
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: lim, y: lim }]}
              stroke="#475569"
              strokeDasharray="4 4"
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const p = payload[0]?.payload as ScatterPoint | undefined;
                if (!p) return null;
                return (
                  <div className="rounded-lg border border-slate-700 bg-navy-900/95 p-2 text-xs">
                    <p className="font-mono font-semibold text-slate-100">{p.ticker}</p>
                    <p className="text-slate-400">
                      MFE {p.mfe_pct.toFixed(2)}% &nbsp; |MAE| {p.mae_abs_pct.toFixed(2)}%
                    </p>
                    <p className="text-slate-400">
                      Capture {(p.capture_ratio * 100).toFixed(0)}% &nbsp; PnL {p.realized_pnl_pct.toFixed(2)}%
                    </p>
                    <p className="text-slate-500">holding {p.holding_days}d</p>
                    {p.leakage_flag && (
                      <p className="mt-1 text-amber-300">exit leakage</p>
                    )}
                  </div>
                );
              }}
            />
            <Scatter name="Winners" data={winners} fill="#10b981" fillOpacity={0.7} />
            <Scatter name="Losers" data={losers} fill="#f43f5e" fillOpacity={0.7} />
            <Scatter
              name="Leakage"
              data={leakers}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={2}
              shape="circle"
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="mt-2 text-[11px] text-slate-500">
          Green = winners, red = losers, amber ring = exit leakage (high MFE + low capture). Dashed
          line: MFE = |MAE|. Point size = holding days.
        </p>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  hint,
  emphasize = false,
}: {
  label: string;
  value: string;
  hint?: string;
  emphasize?: boolean;
}) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</p>
      <p
        className={
          "mt-1 text-2xl font-bold " + (emphasize ? "text-amber-400" : "text-slate-100")
        }
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-[10px] text-slate-500">{hint}</p>}
    </div>
  );
}
