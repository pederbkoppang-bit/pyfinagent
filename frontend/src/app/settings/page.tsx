"use client";

import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";

export default function SettingsPage() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <h2 className="mb-2 text-2xl font-bold text-slate-100">Settings</h2>
        <p className="mb-8 text-sm text-slate-500">
          Configure scoring weights and preferences
        </p>

        <div className="grid max-w-2xl grid-cols-1 gap-6">
          <BentoCard>
            <h3 className="mb-4 text-lg font-semibold text-slate-300">
              Pillar Weights
            </h3>
            <p className="mb-4 text-sm text-slate-400">
              Adjust how each scoring pillar contributes to the final score.
              These weights are applied server-side and can be calibrated based
              on outcome performance data.
            </p>
            <div className="space-y-4">
              {[
                { label: "Corporate", key: "weight_corporate", default: 35 },
                { label: "Industry", key: "weight_industry", default: 20 },
                { label: "Valuation", key: "weight_valuation", default: 20 },
                { label: "Sentiment", key: "weight_sentiment", default: 15 },
                { label: "Governance", key: "weight_governance", default: 10 },
              ].map((w) => (
                <div key={w.key} className="flex items-center gap-4">
                  <label className="w-28 text-sm text-slate-300">
                    {w.label}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={50}
                    defaultValue={w.default}
                    className="flex-1 accent-sky-500"
                  />
                  <span className="w-12 text-right font-mono text-sm text-sky-300">
                    {w.default}%
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-slate-600">
              Note: Weights must sum to 100%. Save functionality will be added
              in the next iteration.
            </p>
          </BentoCard>

          <BentoCard>
            <h3 className="mb-4 text-lg font-semibold text-slate-300">
              API Configuration
            </h3>
            <p className="text-sm text-slate-400">
              Backend API keys and service URLs are managed via environment
              variables. See{" "}
              <code className="rounded bg-slate-800 px-1 py-0.5 text-xs text-sky-300">
                backend/.env
              </code>{" "}
              for configuration.
            </p>
          </BentoCard>
        </div>
      </main>
    </div>
  );
}
