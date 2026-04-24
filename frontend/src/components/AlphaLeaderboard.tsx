/**
 * phase-10.5.5 AlphaLeaderboard.
 *
 * Sortable + filterable strategy-deployment table for the Sovereign
 * page. 7 columns matching the backend response shape:
 *   strategy_id | sharpe | dsr | pbo | max_dd | status | allocation_pct
 *
 * Status cell renders as a Phosphor pill (CheckCircle / Warning /
 * XCircle); clicking any pill filters the table to that status.
 * Sort state is local component state (`{key, dir}`); clicking a
 * sortable column header toggles asc/desc.
 */
"use client";

import { useMemo, useState } from "react";
import { BentoCard } from "@/components/BentoCard";
import {
  Trophy,
  CaretUp,
  CaretDown,
  CheckCircle,
  XCircle,
  Warning,
  X,
} from "@phosphor-icons/react";
import type { SovereignLeaderboardEntry } from "@/lib/api";

export type SortDir = "asc" | "desc";
export type SortKey = "sharpe" | "dsr" | "pbo" | "max_dd" | "allocation_pct" | "strategy_id";

export interface AlphaLeaderboardProps {
  entries: SovereignLeaderboardEntry[];
  loading?: boolean;
  error?: string | null;
}

interface ColumnDef {
  key: keyof SovereignLeaderboardEntry;
  label: string;
  numeric: boolean;
  sortable: boolean;
  // For pbo, lower is better; null sorts to worst (Infinity).
  // For everything else, higher is better; null sorts to worst (-Infinity).
  lowerIsBetter?: boolean;
}

const COLUMNS: ColumnDef[] = [
  { key: "strategy_id",   label: "Strategy",  numeric: false, sortable: true },
  { key: "sharpe",        label: "Sharpe",    numeric: true,  sortable: true },
  { key: "dsr",           label: "DSR",       numeric: true,  sortable: true },
  { key: "pbo",           label: "PBO",       numeric: true,  sortable: true, lowerIsBetter: true },
  { key: "max_dd",        label: "Max DD",    numeric: true,  sortable: true, lowerIsBetter: true },
  { key: "status",        label: "Status",    numeric: false, sortable: false },
  { key: "allocation_pct", label: "Alloc %",  numeric: true,  sortable: true },
];

function getNumeric(v: number | null | undefined, lowerIsBetter: boolean): number {
  if (v == null) return lowerIsBetter ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  return v;
}

function statusVisuals(status: string | null | undefined): {
  Icon: typeof CheckCircle;
  cls: string;
  iconCls: string;
} {
  const s = (status || "").toLowerCase();
  if (s === "champion" || s === "active" || s === "deployed") {
    return {
      Icon: CheckCircle,
      cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
      iconCls: "text-emerald-400",
    };
  }
  if (s === "challenger" || s === "deploying" || s === "pending") {
    return {
      Icon: Warning,
      cls: "border-amber-500/40 bg-amber-500/10 text-amber-300",
      iconCls: "text-amber-400",
    };
  }
  if (s === "retired" || s === "stopped" || s === "demoted" || s === "rejected") {
    return {
      Icon: XCircle,
      cls: "border-rose-500/40 bg-rose-500/10 text-rose-300",
      iconCls: "text-rose-400",
    };
  }
  return {
    Icon: Warning,
    cls: "border-slate-600/50 bg-slate-700/40 text-slate-300",
    iconCls: "text-slate-400",
  };
}

function fmtCell(v: unknown, numeric: boolean): string {
  if (v == null || v === "") return "--";
  if (numeric && typeof v === "number") return v.toFixed(4);
  return String(v);
}

export function AlphaLeaderboard({
  entries,
  loading,
  error,
}: AlphaLeaderboardProps) {
  const [sortState, setSortState] = useState<{ key: SortKey; dir: SortDir }>({
    key: "sharpe",
    dir: "desc",
  });
  const [statusFilter, setStatusFilter] = useState<string | null>(null);

  const filteredSorted = useMemo(() => {
    const filtered = statusFilter
      ? entries.filter((e) => (e.status || "").toLowerCase() === statusFilter.toLowerCase())
      : entries;
    const col = COLUMNS.find((c) => c.key === sortState.key);
    const lowerIsBetter = !!col?.lowerIsBetter;
    const numeric = !!col?.numeric;
    const out = [...filtered];
    out.sort((a, b) => {
      const ak = a[sortState.key as keyof SovereignLeaderboardEntry];
      const bk = b[sortState.key as keyof SovereignLeaderboardEntry];
      let cmp: number;
      if (numeric) {
        cmp =
          getNumeric(ak as number | null, lowerIsBetter) -
          getNumeric(bk as number | null, lowerIsBetter);
      } else {
        cmp = String(ak ?? "").localeCompare(String(bk ?? ""));
      }
      return sortState.dir === "asc" ? cmp : -cmp;
    });
    return out;
  }, [entries, sortState, statusFilter]);

  function requestSort(key: SortKey) {
    setSortState((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" },
    );
  }

  return (
    <BentoCard>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Trophy size={18} className="text-amber-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">Alpha Leaderboard</h3>
        <span className="ml-auto font-mono text-xs text-slate-500">
          {entries.length} strategies
        </span>
      </div>

      {/* Active-filter chip */}
      {statusFilter && (
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs text-slate-500">Filter:</span>
          <button
            type="button"
            data-testid="status-filter-chip"
            aria-label={`Clear status filter: ${statusFilter}`}
            onClick={() => setStatusFilter(null)}
            className="inline-flex items-center gap-1.5 rounded-full border border-sky-500/40 bg-sky-500/10 px-2.5 py-0.5 text-xs font-medium text-sky-300 hover:bg-sky-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <span>{statusFilter}</span>
            <X size={12} weight="bold" aria-hidden="true" />
          </button>
          <span className="text-xs text-slate-500">
            ({filteredSorted.length} of {entries.length})
          </span>
        </div>
      )}

      {loading ? (
        <div className="py-8 text-center text-sm text-slate-500">Loading...</div>
      ) : error ? (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-rose-400">
          {error}
        </div>
      ) : entries.length === 0 ? (
        <div
          data-testid="alpha-leaderboard-empty"
          className="flex flex-col items-center justify-center py-8 text-center"
        >
          <Trophy size={32} weight="duotone" className="text-slate-600" aria-hidden="true" />
          <p className="mt-3 text-sm text-slate-400">No strategies recorded yet</p>
        </div>
      ) : (
        <div data-testid="alpha-leaderboard" className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-navy-700 bg-navy-800/80">
              <tr>
                {COLUMNS.map((col) => {
                  const isActive = sortState.key === col.key;
                  return (
                    <th
                      key={String(col.key)}
                      className={`px-3 py-2.5 font-medium text-slate-400 ${
                        col.numeric ? "text-right" : ""
                      }`}
                      aria-sort={
                        isActive
                          ? sortState.dir === "asc"
                            ? "ascending"
                            : "descending"
                          : "none"
                      }
                    >
                      {col.sortable ? (
                        <button
                          type="button"
                          data-col={String(col.key)}
                          aria-label={`Sort by ${col.label}`}
                          onClick={() => requestSort(col.key as SortKey)}
                          className={`inline-flex items-center gap-1 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${
                            col.numeric ? "ml-auto" : ""
                          } ${isActive ? "text-sky-400" : "hover:text-slate-200"}`}
                        >
                          {col.label}
                          {isActive && (sortState.dir === "asc" ? (
                            <CaretUp size={11} weight="bold" />
                          ) : (
                            <CaretDown size={11} weight="bold" />
                          ))}
                        </button>
                      ) : (
                        <span>{col.label}</span>
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/50">
              {filteredSorted.map((row) => {
                const v = statusVisuals(row.status);
                const StatusIcon = v.Icon;
                const allocStr =
                  row.allocation_pct == null
                    ? "--"
                    : `${(row.allocation_pct * 100).toFixed(1)}%`;
                return (
                  <tr
                    key={row.strategy_id}
                    data-row={row.strategy_id}
                    className="transition-colors hover:bg-navy-700/40"
                  >
                    <td data-cell="strategy_id" className="px-3 py-2.5 font-mono text-xs text-slate-200">
                      {row.strategy_id}
                    </td>
                    <td data-cell="sharpe" className="px-3 py-2.5 text-right font-mono text-xs text-slate-300">
                      {fmtCell(row.sharpe, true)}
                    </td>
                    <td data-cell="dsr" className="px-3 py-2.5 text-right font-mono text-xs text-slate-300">
                      {fmtCell(row.dsr, true)}
                    </td>
                    <td data-cell="pbo" className="px-3 py-2.5 text-right font-mono text-xs text-slate-300">
                      {fmtCell(row.pbo, true)}
                    </td>
                    <td data-cell="max_dd" className="px-3 py-2.5 text-right font-mono text-xs text-slate-300">
                      {fmtCell(row.max_dd, true)}
                    </td>
                    <td data-cell="status" className="px-3 py-2.5">
                      {row.status ? (
                        <button
                          type="button"
                          data-testid="status-pill"
                          data-status={row.status}
                          aria-label={`Filter leaderboard by status: ${row.status}`}
                          onClick={() => setStatusFilter(row.status ?? null)}
                          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${v.cls}`}
                        >
                          <StatusIcon size={12} weight="fill" className={v.iconCls} aria-hidden="true" />
                          {row.status}
                        </button>
                      ) : (
                        <span className="text-xs text-slate-500">--</span>
                      )}
                    </td>
                    <td data-cell="allocation_pct" className="px-3 py-2.5 text-right font-mono text-xs text-slate-300">
                      {allocStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </BentoCard>
  );
}
