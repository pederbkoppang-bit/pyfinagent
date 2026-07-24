"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Database } from "@/lib/icons";
import { getObservabilityDataFreshness } from "@/lib/api";
import type { FreshnessBand, FreshnessResponse, FreshnessSource } from "@/lib/types";

// phase-49.3: this client page fetches live data at request time and must NOT
// be statically prerendered at build (SSG prerender hit a webpack-runtime
// "a[d] is not a function" -- a pre-existing production-build break, latent
// because the app runs on `next dev`). Render dynamically instead.
export const dynamic = "force-dynamic";

const BAND_CLASS: Record<FreshnessBand, string> = {
  green: "bg-emerald-500/15 text-emerald-300 border-emerald-700/40",
  amber: "bg-amber-500/15 text-amber-300 border-amber-700/40",
  red: "bg-rose-500/15 text-rose-300 border-rose-700/40",
  unknown: "bg-slate-700/30 text-slate-400 border-slate-700/40",
};

const BAND_LABEL: Record<FreshnessBand, string> = {
  green: "Fresh",
  amber: "Lagging",
  red: "Stale",
  unknown: "Unknown",
};

function fmtAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

function fmtInterval(seconds: number | null | undefined): string {
  return fmtAge(seconds);
}

function fmtRatio(r: number | null | undefined): string {
  if (r === null || r === undefined || Number.isNaN(r)) return "--";
  return `${r.toFixed(2)}x`;
}

function BandPill({ band }: { band: FreshnessBand }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${BAND_CLASS[band]}`}
    >
      {BAND_LABEL[band]}
    </span>
  );
}

const MAX_CONSECUTIVE_FAILURES = 5;

export default function ObservabilityPage() {
  const [data, setData] = useState<FreshnessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // phase-75.12 (frontend-06): cron-page failuresRef/stoppedRef template --
  // stop polling a dead backend after 5 consecutive failures. The manual
  // "Refresh" button still calls load() directly (bypassing stoppedRef),
  // so it doubles as the recovery action.
  const failuresRef = useRef(0);
  const stoppedRef = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await getObservabilityDataFreshness();
      setData(resp);
      setError(null);
      failuresRef.current = 0;
      stoppedRef.current = false;
    } catch (err) {
      failuresRef.current += 1;
      const msg = err instanceof Error ? err.message : "Failed to load freshness";
      if (failuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        stoppedRef.current = true;
        setError(
          `Freshness polling stopped after ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Last error: ${msg}`,
        );
      } else {
        setError(msg);
      }
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const id = setInterval(() => {
      if (!stoppedRef.current) void load();
    }, 30_000);
    return () => clearInterval(id);
  }, [load]);

  const sources: [string, FreshnessSource][] = data
    ? Object.entries(data.sources).sort((a, b) => a[0].localeCompare(b[0]))
    : [];

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Data Freshness</h2>
              <p className="text-sm text-slate-500">
                Per-table age + SLA bands across the warehouse (phase-25.C7)
              </p>
            </div>
            <div className="flex items-center gap-3">
              {data && (
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span>Overall</span>
                  <BandPill band={data.overall_band ?? "unknown"} />
                </div>
              )}
              <button
                onClick={() => void load()}
                disabled={loading}
                className="rounded-lg border border-navy-700 bg-navy-800/60 px-3 py-1.5 text-xs text-slate-300 hover:bg-navy-700 disabled:opacity-50"
              >
                {loading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {error && (
            <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
              <p className="text-sm text-rose-300">{error}</p>
            </div>
          )}

          {loading && !data && (
            <div className="flex items-center gap-3 py-12 text-slate-400">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
              Loading freshness...
            </div>
          )}

          {!loading && !error && sources.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <Database size={48} weight="duotone" className="text-slate-600" />
              <p className="mt-4 text-lg text-slate-400">No freshness data available</p>
              <p className="mt-1 text-sm text-slate-600">Backend has not reported any tables.</p>
            </div>
          )}

          {sources.length > 0 && (
            <div className="overflow-hidden rounded-xl border border-navy-700">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-navy-700 bg-navy-800/80">
                  <tr>
                    <th className="px-4 py-3 font-medium text-slate-400">Source</th>
                    <th className="px-4 py-3 font-medium text-slate-400">Age</th>
                    <th className="px-4 py-3 font-medium text-slate-400">SLA Interval</th>
                    <th className="px-4 py-3 font-medium text-slate-400">Ratio</th>
                    <th className="px-4 py-3 font-medium text-slate-400">Band</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-navy-700/50">
                  {sources.map(([name, src]) => (
                    <tr key={name} className="transition-colors hover:bg-navy-700/40">
                      <td className="px-4 py-3 font-mono text-slate-200">{name}</td>
                      <td className="px-4 py-3 text-slate-300">{fmtAge(src.last_tick_age_sec)}</td>
                      <td className="px-4 py-3 text-slate-300">{fmtInterval(src.interval_sec)}</td>
                      <td className="px-4 py-3 text-slate-300">{fmtRatio(src.ratio)}</td>
                      <td className="px-4 py-3">
                        <BandPill band={src.band ?? "unknown"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data?.computed_at && (
            <p className="mt-4 text-xs text-slate-600">
              Computed at {data.computed_at}
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
