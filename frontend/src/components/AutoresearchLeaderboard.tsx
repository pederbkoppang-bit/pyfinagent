"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export const AUTORESEARCH_REFRESH_MS = 5_000; // well under the 10s contract ceiling

export interface LeaderboardCandidate {
  index: number;
  run_id: string;
  param_changed: string | null;
  dsr: number | null;
  pbo: number | null;
  realized_pnl_if_promoted: number | null;
  starting_capital: number | null;
  status: "kept" | "discarded" | string;
}

interface Props {
  candidates?: LeaderboardCandidate[];
  fetcher?: () => Promise<LeaderboardCandidate[]>;
  refreshMs?: number;
}

type SortKey = "rank" | "dsr" | "pbo" | "realized";

function fmt(v: number | null, digits = 2, suffix = "") {
  if (v == null || Number.isNaN(v)) return "\u2014";
  return `${v.toFixed(digits)}${suffix}`;
}

function fmtUsd(v: number | null) {
  if (v == null || Number.isNaN(v)) return "\u2014";
  const sign = v >= 0 ? "" : "-";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`;
}

function rankCandidates(cs: LeaderboardCandidate[]): LeaderboardCandidate[] {
  // Composite rank: lower PBO is better, higher DSR is better, higher P&L is
  // better. We use DSR as primary (it already accounts for trials) and break
  // ties on realized P&L. Candidates with PBO > 0.5 are pinned to the bottom
  // regardless of DSR -- that's the structural overfitting veto.
  return [...cs].sort((a, b) => {
    const aVeto = (a.pbo ?? 0) > 0.5 ? 1 : 0;
    const bVeto = (b.pbo ?? 0) > 0.5 ? 1 : 0;
    if (aVeto !== bVeto) return aVeto - bVeto;
    const dsrA = a.dsr ?? 0;
    const dsrB = b.dsr ?? 0;
    if (dsrB !== dsrA) return dsrB - dsrA;
    return (b.realized_pnl_if_promoted ?? 0) - (a.realized_pnl_if_promoted ?? 0);
  });
}

export function AutoresearchLeaderboard({
  candidates: controlledCandidates,
  fetcher,
  refreshMs = AUTORESEARCH_REFRESH_MS,
}: Props) {
  const [candidates, setCandidates] = useState<LeaderboardCandidate[]>(
    controlledCandidates ?? [],
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>("rank");
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    if (!fetcher) return;
    setLoading(true);
    setError(null);
    try {
      const fresh = await fetcher();
      if (mountedRef.current) setCandidates(fresh);
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (controlledCandidates) {
      setCandidates(controlledCandidates);
    }
  }, [controlledCandidates]);

  useEffect(() => {
    if (!fetcher) return;
    refresh();
    const id = setInterval(refresh, refreshMs);
    return () => clearInterval(id);
  }, [fetcher, refresh, refreshMs]);

  const ranked = useMemo(() => rankCandidates(candidates), [candidates]);
  const display = useMemo(() => {
    if (sort === "dsr") return [...ranked].sort((a, b) => (b.dsr ?? 0) - (a.dsr ?? 0));
    if (sort === "pbo") return [...ranked].sort((a, b) => (a.pbo ?? 1) - (b.pbo ?? 1));
    if (sort === "realized")
      return [...ranked].sort(
        (a, b) => (b.realized_pnl_if_promoted ?? 0) - (a.realized_pnl_if_promoted ?? 0),
      );
    return ranked;
  }, [ranked, sort]);

  return (
    <div
      className="overflow-hidden rounded-xl border border-navy-700 bg-navy-800/30"
      data-testid="autoresearch-leaderboard"
      data-refresh-ms={refreshMs}
    >
      <div className="flex items-center justify-between border-b border-navy-700 bg-navy-800/80 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Autoresearch Leaderboard
          </h3>
          <p className="text-xs text-slate-500">
            Ranked by DSR; PBO &gt; 0.5 candidates pinned to bottom (overfitting veto).
            Refresh every {(refreshMs / 1000).toFixed(0)}s.
          </p>
        </div>
        {loading && (
          <span className="text-xs text-sky-400" data-testid="leaderboard-loading">
            refreshing...
          </span>
        )}
      </div>
      {error && (
        <div className="border-b border-rose-500/30 bg-rose-950/30 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
      <table className="w-full text-left text-sm">
        <thead className="border-b border-navy-700 bg-navy-800/60">
          <tr>
            <th className="px-3 py-2 font-medium text-slate-400" data-col="rank">
              <button onClick={() => setSort("rank")} className="hover:text-sky-400">
                Rank
              </button>
            </th>
            <th className="px-3 py-2 font-medium text-slate-400">Param Change</th>
            <th className="px-3 py-2 font-medium text-slate-400" data-col="dsr">
              <button onClick={() => setSort("dsr")} className="hover:text-sky-400">
                DSR
              </button>
            </th>
            <th className="px-3 py-2 font-medium text-slate-400" data-col="pbo">
              <button onClick={() => setSort("pbo")} className="hover:text-sky-400">
                PBO
              </button>
            </th>
            <th className="px-3 py-2 font-medium text-slate-400" data-col="realized_pnl">
              <button onClick={() => setSort("realized")} className="hover:text-sky-400">
                Realized P&amp;L (if promoted)
              </button>
            </th>
            <th className="px-3 py-2 font-medium text-slate-400" data-col="status">
              Status
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-navy-700/50">
          {display.length === 0 && (
            <tr>
              <td
                colSpan={6}
                className="px-3 py-8 text-center text-slate-600"
                data-testid="leaderboard-empty"
              >
                No optimizer candidates yet. Start an autoresearch run.
              </td>
            </tr>
          )}
          {display.map((c, i) => {
            const vetoed = (c.pbo ?? 0) > 0.5;
            return (
              <tr
                key={`${c.run_id}-${c.index}`}
                className={`transition-colors hover:bg-navy-700/40 ${
                  vetoed ? "opacity-60" : ""
                }`}
                data-candidate-index={c.index}
              >
                <td className="px-3 py-2 font-mono text-slate-300" data-cell="rank">
                  {i + 1}
                </td>
                <td className="px-3 py-2 text-slate-200">{c.param_changed ?? "\u2014"}</td>
                <td className="px-3 py-2 font-mono text-slate-100" data-cell="dsr">
                  {fmt(c.dsr, 3)}
                </td>
                <td
                  className={`px-3 py-2 font-mono ${vetoed ? "text-rose-400" : "text-slate-100"}`}
                  data-cell="pbo"
                >
                  {fmt(c.pbo, 3)}
                </td>
                <td
                  className={`px-3 py-2 font-mono ${
                    (c.realized_pnl_if_promoted ?? 0) >= 0
                      ? "text-emerald-400"
                      : "text-rose-400"
                  }`}
                  data-cell="realized_pnl"
                >
                  {fmtUsd(c.realized_pnl_if_promoted)}
                </td>
                <td className="px-3 py-2 text-xs uppercase" data-cell="status">
                  <span
                    className={
                      c.status === "kept"
                        ? "text-emerald-400"
                        : c.status === "discarded"
                          ? "text-slate-500"
                          : "text-amber-400"
                    }
                  >
                    {c.status}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
