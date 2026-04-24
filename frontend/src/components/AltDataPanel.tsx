/**
 * phase-15.7 Alt-data panel.
 *
 * Renders the response from `GET /api/signals/{ticker}/alt-data`:
 *   - Congress trades (Senate only; House deferred)
 *   - 13F top holdings (latest period across all filers -- the backing
 *     table has no ticker column, so this surfaces aggregate
 *     institutional signal rather than ticker-specific holdings)
 *   - Information-coefficient evaluation stub
 *
 * Empty state when both arrays are empty. Navy palette; Phosphor icons.
 */
"use client";

import { BentoCard } from "@/components/BentoCard";
import type { AltDataResponse } from "@/lib/types";
import { Bank, Buildings, ChartBar } from "@phosphor-icons/react";

export interface AltDataPanelProps {
  data: AltDataResponse | null;
}

function fmtUsd(amount: number): string {
  if (!Number.isFinite(amount)) return "--";
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(2)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

export function AltDataPanel({ data }: AltDataPanelProps) {
  if (!data) return null;
  const congress = data.congress ?? [];
  const f13 = data.f13 ?? [];
  const ic = data.ic_eval;
  const empty = congress.length === 0 && f13.length === 0;

  return (
    <BentoCard>
      <div className="mb-4 flex items-center gap-2">
        <Bank size={18} className="text-sky-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">
          Alt-data ({data.ticker})
        </h3>
      </div>

      {empty ? (
        <div
          data-altdata-state="empty"
          className="flex flex-col items-center justify-center py-8 text-center"
        >
          <Bank
            size={32}
            weight="duotone"
            className="text-slate-600"
            aria-hidden="true"
          />
          <p className="mt-3 text-sm text-slate-400">
            No Congress trades or 13F holdings found for {data.ticker}
          </p>
          <p className="mt-1 text-xs text-slate-600">
            Congress data covers Senate only. 13F surfaces the latest
            period&apos;s top institutional holdings across all filers.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Congress */}
          <section
            data-altdata-section="congress"
            className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4"
          >
            <div className="mb-3 flex items-center gap-2">
              <Buildings size={14} className="text-sky-400" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Congress trades ({congress.length})
              </h4>
            </div>
            {congress.length === 0 ? (
              <p className="py-4 text-center text-xs text-slate-500">
                No Senate trades recorded
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-navy-700/50">
                    <tr>
                      <th className="py-2 font-medium text-slate-500">
                        Senator
                      </th>
                      <th className="py-2 font-medium text-slate-500">Type</th>
                      <th className="py-2 text-right font-medium text-slate-500">
                        Amount
                      </th>
                      <th className="py-2 text-right font-medium text-slate-500">
                        Date
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-navy-700/30">
                    {congress.slice(0, 10).map((t, i) => (
                      <tr key={`${t.senator}-${t.transaction_date}-${i}`}>
                        <td className="py-1.5 text-xs text-slate-300">
                          {t.senator}
                        </td>
                        <td className="py-1.5 text-xs text-slate-400">
                          {t.type}
                        </td>
                        <td className="py-1.5 text-right font-mono text-xs text-slate-300">
                          {fmtUsd(t.amount_mid)}
                        </td>
                        <td className="py-1.5 text-right font-mono text-xs text-slate-500">
                          {t.transaction_date?.slice(0, 10) || "--"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {congress.length > 10 && (
                  <p className="mt-2 text-xs text-slate-500">
                    Showing newest 10 of {congress.length}
                  </p>
                )}
              </div>
            )}
          </section>

          {/* 13F */}
          <section
            data-altdata-section="f13"
            className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4"
          >
            <div className="mb-3 flex items-center gap-2">
              <Buildings size={14} className="text-emerald-400" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                13F top holdings (latest period)
              </h4>
            </div>
            {f13.length === 0 ? (
              <p className="py-4 text-center text-xs text-slate-500">
                No 13F holdings loaded
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-navy-700/50">
                    <tr>
                      <th className="py-2 font-medium text-slate-500">
                        Filer
                      </th>
                      <th className="py-2 text-right font-medium text-slate-500">
                        Value
                      </th>
                      <th className="py-2 text-right font-medium text-slate-500">
                        Period
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-navy-700/30">
                    {f13.map((h, i) => (
                      <tr key={`${h.filer_name}-${h.period}-${i}`}>
                        <td className="py-1.5 text-xs text-slate-300">
                          {h.filer_name}
                        </td>
                        <td className="py-1.5 text-right font-mono text-xs text-emerald-400">
                          ${(h.value_usd_thousands / 1000).toFixed(1)}M
                        </td>
                        <td className="py-1.5 text-right font-mono text-xs text-slate-500">
                          {h.period?.slice(0, 10) || "--"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      )}

      {/* IC eval footer */}
      {ic && (
        <div
          data-altdata-section="ic"
          className="mt-4 flex flex-wrap items-center gap-4 border-t border-navy-700/50 pt-3 text-xs text-slate-500"
        >
          <ChartBar size={14} className="text-slate-500" />
          <span>
            IC mean:{" "}
            <span className="font-mono text-slate-300">
              {ic.ic_mean.toFixed(4)}
            </span>
          </span>
          <span>
            IC_IR:{" "}
            <span className="font-mono text-slate-300">
              {ic.ic_ir.toFixed(2)}
            </span>
          </span>
          <span>
            window: <span className="font-mono">{ic.window_days}d</span>
          </span>
          <span className="ml-auto italic text-slate-600">{ic.note}</span>
        </div>
      )}
    </BentoCard>
  );
}
