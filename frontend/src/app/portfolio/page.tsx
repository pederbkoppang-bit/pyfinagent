"use client";

import { useCallback, useEffect, useState } from "react";
import { clsx } from "clsx";
import { Sidebar } from "@/components/Sidebar";
import {
  listPortfolioPositions,
  addPortfolioPosition,
  deletePortfolioPosition,
  getPortfolioPerformance,
} from "@/lib/api";
import type { PortfolioPosition, PortfolioPerformance } from "@/lib/types";

function PnlValue({ value, pct }: { value?: number; pct?: number }) {
  if (value == null) return <span className="text-slate-500">—</span>;
  const isPositive = value >= 0;
  return (
    <span className={isPositive ? "text-emerald-400" : "text-rose-400"}>
      {isPositive ? "+" : ""}${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
      {pct != null && (
        <span className="ml-1 text-xs opacity-70">
          ({isPositive ? "+" : ""}{pct.toFixed(2)}%)
        </span>
      )}
    </span>
  );
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [perf, setPerf] = useState<PortfolioPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add position form
  const [ticker, setTicker] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");
  const [adding, setAdding] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [pos, p] = await Promise.all([
        listPortfolioPositions(),
        getPortfolioPerformance(),
      ]);
      setPositions(pos);
      setPerf(p);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load portfolio");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async () => {
    if (!ticker.trim() || !qty || !price) return;
    setAdding(true);
    try {
      await addPortfolioPosition({
        ticker: ticker.toUpperCase(),
        quantity: parseFloat(qty),
        avg_entry_price: parseFloat(price),
      });
      setTicker("");
      setQty("");
      setPrice("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add position");
    } finally {
      setAdding(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deletePortfolioPosition(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to remove position");
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-100">Portfolio Tracker</h2>
          <p className="text-sm text-slate-500">
            Track positions, P&amp;L, and recommendation accuracy
          </p>
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-rose-900 bg-rose-950/50 p-4">
            <p className="text-sm font-medium text-rose-200">{error}</p>
            {error.includes("Cannot reach backend") && (
              <p className="mt-2 text-xs text-rose-300/60">
                Run: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
              </p>
            )}
          </div>
        )}

        {/* Summary Cards */}
        {perf && (
          <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
              <p className="text-xs text-slate-500">Total Cost Basis</p>
              <p className="mt-1 text-lg font-semibold text-slate-200">
                ${perf.total_cost_basis.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
              <p className="text-xs text-slate-500">Market Value</p>
              <p className="mt-1 text-lg font-semibold text-slate-200">
                ${perf.total_market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
              <p className="text-xs text-slate-500">Total P&amp;L</p>
              <p className="mt-1 text-lg font-semibold">
                <PnlValue value={perf.total_pnl} pct={perf.total_pnl_pct} />
              </p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
              <p className="text-xs text-slate-500">Rec. Accuracy</p>
              <p className="mt-1 text-lg font-semibold text-slate-200">
                {perf.recommendation_accuracy != null
                  ? `${perf.recommendation_accuracy}%`
                  : "—"}
              </p>
            </div>
          </div>
        )}

        {/* Add Position Form */}
        <div className="mb-6 rounded-xl border border-navy-700 bg-navy-800/70 p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">Add Position</h3>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label className="mb-1 block text-xs text-slate-500">Ticker</label>
              <input
                type="text"
                placeholder="AAPL"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="w-28 rounded-lg border border-navy-700 bg-navy-900 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-500">Quantity</label>
              <input
                type="number"
                placeholder="100"
                value={qty}
                onChange={(e) => setQty(e.target.value)}
                className="w-28 rounded-lg border border-navy-700 bg-navy-900 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-500">Avg Entry Price</label>
              <input
                type="number"
                step="0.01"
                placeholder="150.00"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                className="w-36 rounded-lg border border-navy-700 bg-navy-900 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handleAdd}
              disabled={adding || !ticker.trim() || !qty || !price}
              className="rounded-lg bg-sky-600 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {adding ? "Adding..." : "Add"}
            </button>
          </div>
        </div>

        {/* Positions Table */}
        <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-navy-700 text-xs uppercase text-slate-500">
                  <th className="px-4 py-3">Ticker</th>
                  <th className="px-4 py-3">Qty</th>
                  <th className="px-4 py-3">Entry</th>
                  <th className="px-4 py-3">Current</th>
                  <th className="px-4 py-3">Cost Basis</th>
                  <th className="px-4 py-3">Market Value</th>
                  <th className="px-4 py-3">P&amp;L</th>
                  <th className="px-4 py-3"></th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                      Loading...
                    </td>
                  </tr>
                ) : positions.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                      No positions yet. Add one above.
                    </td>
                  </tr>
                ) : (
                  positions.map((pos) => (
                    <tr
                      key={pos.id}
                      className="border-b border-navy-700/50 text-slate-300 hover:bg-navy-700/30"
                    >
                      <td className="px-4 py-3 font-mono font-semibold text-slate-100">
                        {pos.ticker}
                      </td>
                      <td className="px-4 py-3">{pos.quantity}</td>
                      <td className="px-4 py-3">${pos.avg_entry_price.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {pos.current_price != null ? `$${pos.current_price.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-4 py-3">${pos.cost_basis.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {pos.market_value != null ? `$${pos.market_value.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-4 py-3">
                        <PnlValue value={pos.unrealized_pnl} pct={pos.unrealized_pnl_pct} />
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => handleDelete(pos.id)}
                          className="rounded px-2 py-1 text-xs text-rose-400 transition-colors hover:bg-rose-500/10"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Allocation breakdown (simple bar chart) */}
        {perf && perf.allocation.length > 0 && (
          <div className="mt-6 rounded-xl border border-navy-700 bg-navy-800/70 p-6">
            <h3 className="mb-4 text-sm font-semibold text-slate-300">Allocation</h3>
            <div className="space-y-2">
              {perf.allocation.map((a) => {
                const pct =
                  perf.total_market_value > 0
                    ? (a.market_value / perf.total_market_value) * 100
                    : 0;
                return (
                  <div key={a.ticker} className="flex items-center gap-3">
                    <span className="w-16 font-mono text-xs text-slate-300">{a.ticker}</span>
                    <div className="h-3 flex-1 rounded-full bg-slate-700">
                      <div
                        className={clsx(
                          "h-3 rounded-full transition-all",
                          a.pnl >= 0 ? "bg-sky-500" : "bg-rose-500"
                        )}
                        style={{ width: `${Math.max(pct, 1)}%` }}
                      />
                    </div>
                    <span className="w-12 text-right font-mono text-xs text-slate-400">
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
