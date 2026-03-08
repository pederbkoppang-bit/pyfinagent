"use client";

import { clsx } from "clsx";

interface Anomaly {
  metric: string;
  value: number;
  z_score: number;
  mean?: number;
  std?: number;
  note?: string;
}

interface HorizonData {
  var_95: number;
  var_99: number;
  expected_shortfall_95: number;
  prob_gain_20_pct: number;
  prob_loss_20_pct: number;
  prob_positive: number;
  median_return: number;
  mean_return: number;
  std_return: number;
}

interface MonteCarloData {
  ticker: string;
  signal: string;
  summary: string;
  current_price?: number;
  annualized_volatility?: number;
  horizons?: Record<string, HorizonData>;
}

interface AnomalyData {
  ticker: string;
  signal: string;
  summary: string;
  anomaly_count?: number;
  anomalies?: Anomaly[];
}

export interface RiskDataPayload {
  monte_carlo: MonteCarloData;
  anomalies: AnomalyData;
}

function VarGauge({ label, value }: { label: string; value: number }) {
  // value is a negative % (loss). Map to gauge: 0 → -50%
  const severity = Math.min(100, Math.abs(value) * 2);
  const color =
    severity < 30
      ? "bg-emerald-500"
      : severity < 60
      ? "bg-amber-500"
      : "bg-rose-500";
  const textColor =
    severity < 30
      ? "text-emerald-400"
      : severity < 60
      ? "text-amber-400"
      : "text-rose-400";

  return (
    <div className="rounded-lg border border-navy-700 bg-navy-900/50 p-3">
      <div className="mb-1 text-xs text-slate-500">{label}</div>
      <div className={clsx("text-lg font-bold", textColor)}>
        {value > 0 ? "+" : ""}
        {value}%
      </div>
      <div className="mt-1 h-1.5 rounded-full bg-slate-700">
        <div
          className={clsx("h-1.5 rounded-full transition-all", color)}
          style={{ width: `${severity}%` }}
        />
      </div>
    </div>
  );
}

function ProbabilityCard({
  label,
  value,
  icon,
  positive,
}: {
  label: string;
  value: number;
  icon: string;
  positive?: boolean;
}) {
  return (
    <div className="rounded-lg border border-navy-700 bg-navy-900/50 p-3 text-center">
      <div className="text-lg">{icon}</div>
      <div
        className={clsx(
          "text-xl font-bold",
          positive ? "text-emerald-400" : "text-rose-400"
        )}
      >
        {value}%
      </div>
      <div className="text-[10px] text-slate-500">{label}</div>
    </div>
  );
}

function AnomalyCard({ anomaly }: { anomaly: Anomaly }) {
  const isRisk = anomaly.z_score < -2 || ["debt_to_equity", "short_ratio", "growth_margin_divergence"].includes(anomaly.metric);
  const borderColor = isRisk
    ? "border-rose-500/30"
    : "border-emerald-500/30";
  const badgeColor = isRisk
    ? "bg-rose-500/20 text-rose-400"
    : "bg-emerald-500/20 text-emerald-400";

  return (
    <div className={clsx("rounded-lg border bg-navy-900/50 p-3", borderColor)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-300">
          {anomaly.metric.replace(/_/g, " ")}
        </span>
        <span className={clsx("rounded-full px-2 py-0.5 text-[10px] font-bold", badgeColor)}>
          Z={anomaly.z_score.toFixed(1)}
        </span>
      </div>
      <div className="mt-1 text-sm font-bold text-slate-200">{anomaly.value}</div>
      {anomaly.note && (
        <p className="mt-1 text-[10px] text-slate-500">{anomaly.note}</p>
      )}
    </div>
  );
}

export function RiskDashboard({ data }: { data: RiskDataPayload }) {
  if (!data) return null;

  const mc = data.monte_carlo;
  const anomalyData = data.anomalies;
  const hasHorizons = mc?.horizons && Object.keys(mc.horizons).length > 0;
  const hasAnomalies = anomalyData?.anomalies && anomalyData.anomalies.length > 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
          🎯 Risk Dashboard
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Monte Carlo VaR simulation ({mc?.signal === "ERROR" ? "unavailable" : "1,000 GBM paths"})
          + Statistical anomaly detection
        </p>
        {mc?.annualized_volatility && (
          <div className="mt-2 flex items-center gap-4 text-xs text-slate-400">
            <span>
              Current Price:{" "}
              <span className="font-medium text-slate-200">
                ${mc.current_price?.toFixed(2)}
              </span>
            </span>
            <span>
              Annualized Vol:{" "}
              <span className="font-medium text-slate-200">
                {mc.annualized_volatility}%
              </span>
            </span>
          </div>
        )}
      </div>

      {/* VaR Gauges + Probabilities by Horizon */}
      {hasHorizons && (
        <div className="space-y-4">
          {(["3M", "6M", "1Y"] as const).map((horizon) => {
            const h = mc.horizons?.[horizon];
            if (!h) return null;
            return (
              <div
                key={horizon}
                className="rounded-xl border border-navy-700 bg-navy-800/60 p-4"
              >
                <h4 className="mb-3 text-sm font-semibold text-slate-300">
                  {horizon} Horizon
                </h4>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <VarGauge label="VaR (95%)" value={h.var_95} />
                  <VarGauge label="VaR (99%)" value={h.var_99} />
                  <VarGauge label="Expected Shortfall" value={h.expected_shortfall_95} />
                  <div className="rounded-lg border border-navy-700 bg-navy-900/50 p-3">
                    <div className="mb-1 text-xs text-slate-500">Median Return</div>
                    <div
                      className={clsx(
                        "text-lg font-bold",
                        h.median_return >= 0 ? "text-emerald-400" : "text-rose-400"
                      )}
                    >
                      {h.median_return > 0 ? "+" : ""}
                      {h.median_return}%
                    </div>
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3">
                  <ProbabilityCard
                    label="P(Positive)"
                    value={h.prob_positive}
                    icon="📈"
                    positive
                  />
                  <ProbabilityCard
                    label="P(≥+20%)"
                    value={h.prob_gain_20_pct}
                    icon="🚀"
                    positive
                  />
                  <ProbabilityCard
                    label="P(≤−20%)"
                    value={h.prob_loss_20_pct}
                    icon="⚠️"
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Anomaly Alerts */}
      {hasAnomalies && (
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-300">
              <span>🔍</span> Statistical Anomalies
            </h4>
            <span
              className={clsx(
                "rounded-full border px-2.5 py-0.5 text-xs font-semibold",
                anomalyData.signal === "ANOMALY_RISK"
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
                  : anomalyData.signal === "ANOMALY_OPPORTUNITY"
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  : "border-slate-500/30 bg-slate-500/10 text-slate-400"
              )}
            >
              {anomalyData.signal}
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {anomalyData.anomalies!.map((a, i) => (
              <AnomalyCard key={i} anomaly={a} />
            ))}
          </div>
        </div>
      )}

      {/* No Data Fallback */}
      {!hasHorizons && !hasAnomalies && (
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
          Risk data unavailable. Monte Carlo and anomaly detection require sufficient price history.
        </div>
      )}
    </div>
  );
}
