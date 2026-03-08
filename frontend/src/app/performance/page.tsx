"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { evaluateOutcomes, getPerformanceStats } from "@/lib/api";
import type { PerformanceStats } from "@/lib/types";

export default function PerformancePage() {
  const [stats, setStats] = useState<PerformanceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPerformanceStats()
      .then(setStats)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleEvaluate = async () => {
    setEvaluating(true);
    try {
      const result = await evaluateOutcomes();
      // Refresh stats after evaluation
      const updated = await getPerformanceStats();
      setStats(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">
              Recommendation Performance
            </h2>
            <p className="text-sm text-slate-500">
              Track whether past recommendations were correct (learning loop)
            </p>
          </div>
          <button
            onClick={handleEvaluate}
            disabled={evaluating}
            className="rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
          >
            {evaluating ? "Evaluating..." : "Evaluate Outcomes"}
          </button>
        </div>

        {loading && <p className="text-slate-400">Loading performance data...</p>}
        {error && <p className="text-rose-400">{error}</p>}

        {stats && (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 md:col-span-4">
              <BentoCard glow>
                <p className="text-sm text-slate-400">Win Rate</p>
                <p className="mt-2 font-mono text-5xl font-bold text-emerald-400">
                  {(stats.win_rate * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {stats.wins}W / {stats.losses}L of{" "}
                  {stats.total_recommendations} total
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12 md:col-span-4">
              <BentoCard>
                <p className="text-sm text-slate-400">Avg Return</p>
                <p
                  className={`mt-2 font-mono text-5xl font-bold ${
                    stats.avg_return >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {stats.avg_return >= 0 ? "+" : ""}
                  {stats.avg_return.toFixed(2)}%
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12 md:col-span-4">
              <BentoCard>
                <p className="text-sm text-slate-400">Beat Benchmark (SPY)</p>
                <p className="mt-2 font-mono text-5xl font-bold text-sky-400">
                  {(stats.benchmark_beat_rate * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  of recommendations outperformed S&P 500
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12">
              <BentoCard>
                <h3 className="mb-2 text-lg font-semibold text-slate-300">
                  🧠 How the Learning Loop Works
                </h3>
                <p className="text-sm leading-relaxed text-slate-400">
                  Click <strong>Evaluate Outcomes</strong> to compare each past
                  recommendation against actual price performance. The system
                  checks: did the stock go up after a Buy? Did it go down after a
                  Sell? Did it beat the S&P 500? Over time, this data reveals
                  which agents and pillars are most predictive — allowing pillar
                  weights to be calibrated empirically.
                </p>
              </BentoCard>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
