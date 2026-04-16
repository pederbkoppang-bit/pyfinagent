"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { clsx } from "clsx";
import { SkeletonCard } from "@/components/Skeleton";

interface SourceHealth {
  last_tick_age_sec: number | null;
  ratio: number | null;
  band: "green" | "amber" | "red" | "unknown";
}

interface Freshness {
  sources: Record<string, SourceHealth>;
  heartbeat: {
    updated_at: string | null;
    event: string | null;
    cycle_id: string | null;
    age_sec: number | null;
    ratio: number | null;
    band: "green" | "amber" | "red" | "unknown";
  };
  bq_ingest_lag_sec: number | null;
  thresholds: { warn_ratio: number; critical_ratio: number; cycle_interval_sec: number };
  computed_at: string;
}

interface CycleRow {
  cycle_id: string;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  status: string;
  n_trades: number;
  error_count: number;
  bq_ingest_lag_sec: number | null;
}

const BAND_CLS: Record<string, string> = {
  green: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  amber: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  red: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  unknown: "bg-slate-700/50 text-slate-400 border-slate-600/40",
};

function humanAge(sec: number | null): string {
  if (sec == null) return "—";
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h`;
  return `${Math.round(sec / 86400)}d`;
}

function bandFromRatio(
  ageSec: number | null | undefined,
  t: Freshness["thresholds"],
): "green" | "amber" | "red" | "unknown" {
  if (ageSec == null || t.cycle_interval_sec <= 0) return "unknown";
  const ratio = ageSec / t.cycle_interval_sec;
  if (ratio >= t.critical_ratio) return "red";
  if (ratio >= t.warn_ratio) return "amber";
  return "green";
}

export function CycleHealthStrip() {
  const [fresh, setFresh] = useState<Freshness | null>(null);
  const [cycles, setCycles] = useState<CycleRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const failRef = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/paper-trading/freshness", { credentials: "include" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setFresh((prev) =>
        prev && prev.computed_at === j.computed_at ? prev : j,
      );
      setError(null);
      failRef.current = 0;
    } catch (e) {
      failRef.current += 1;
      if (failRef.current >= 5) setError(e instanceof Error ? e.message : "freshness failed");
    }
    try {
      const r = await fetch("/api/paper-trading/cycles/history?limit=10", {
        credentials: "include",
      });
      if (r.ok) {
        const j = await r.json();
        setCycles((prev) => {
          const next: CycleRow[] = j.cycles || [];
          if (prev && prev.length === next.length && prev[0]?.cycle_id === next[0]?.cycle_id) {
            return prev;
          }
          return next;
        });
      }
    } catch {
      // freshness is the primary signal; cycle-history is informational
    }
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const tick = () => {
      if (document.hidden) return;
      void refresh();
    };
    void refresh();
    const id = window.setInterval(tick, 30_000);
    const onVis = () => {
      if (!document.hidden) tick();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [refresh]);

  if (!fresh) {
    if (error) {
      return (
        <div className="rounded-xl border border-rose-500/30 bg-rose-950/30 p-3">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
            Cycle health
          </p>
          <p className="mt-1 text-xs text-rose-300">Failed to load: {error}</p>
          <button
            type="button"
            onClick={() => {
              failRef.current = 0;
              setError(null);
              void refresh();
            }}
            className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60"
          >
            Retry
          </button>
        </div>
      );
    }
    return (
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Cycle health
        </p>
        <SkeletonCard h="h-14" className="mt-2" />
      </div>
    );
  }

  const hb = fresh.heartbeat;
  const pills: Array<{ label: string; age: number | null; band: string }> = [
    { label: "heartbeat", age: hb.age_sec, band: hb.band },
    ...Object.entries(fresh.sources).map(([name, s]) => ({
      label: name,
      age: s.last_tick_age_sec,
      band: s.band,
    })),
    {
      label: "bq ingest",
      age: fresh.bq_ingest_lag_sec ?? null,
      band: bandFromRatio(fresh.bq_ingest_lag_sec, fresh.thresholds),
    },
  ];

  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
          Cycle health
        </p>
        <span className="ml-auto font-mono text-[10px] text-slate-500">
          warn {fresh.thresholds.warn_ratio}x &nbsp;/&nbsp; crit {fresh.thresholds.critical_ratio}x
        </span>
      </div>

      <div className="mt-2 flex flex-wrap gap-2">
        {pills.map((p) => (
          <span
            key={p.label}
            className={clsx(
              "rounded-full border px-2 py-0.5 text-[11px]",
              BAND_CLS[p.band] || BAND_CLS.unknown,
            )}
          >
            {p.label} <span className="ml-1 font-mono">{humanAge(p.age)}</span>
          </span>
        ))}
      </div>

      {cycles && cycles.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-[11px] text-slate-500">
            last {cycles.length} cycles
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[10px] uppercase text-slate-500">
                <tr>
                  <th className="py-1 pr-3">id</th>
                  <th className="py-1 pr-3">start</th>
                  <th className="py-1 pr-3">dur</th>
                  <th className="py-1 pr-3">status</th>
                  <th className="py-1 pr-3">trades</th>
                  <th className="py-1 pr-3">err</th>
                </tr>
              </thead>
              <tbody>
                {cycles.map((c) => (
                  <tr key={c.cycle_id} className="text-slate-300">
                    <td className="py-1 pr-3 font-mono text-slate-500">{c.cycle_id}</td>
                    <td className="py-1 pr-3 text-slate-400">
                      {c.started_at ? new Date(c.started_at).toLocaleTimeString() : "—"}
                    </td>
                    <td className="py-1 pr-3 font-mono">
                      {c.duration_ms != null ? `${(c.duration_ms / 1000).toFixed(1)}s` : "—"}
                    </td>
                    <td
                      className={clsx(
                        "py-1 pr-3",
                        c.status === "success" || c.status === "ok"
                          ? "text-emerald-300"
                          : c.status === "failed"
                            ? "text-rose-300"
                            : "text-amber-300",
                      )}
                    >
                      {c.status}
                    </td>
                    <td className="py-1 pr-3 font-mono">{c.n_trades}</td>
                    <td className="py-1 pr-3 font-mono">{c.error_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
