"use client";

import { useEffect, useState, useMemo } from "react";
import {
  ComposedChart,
  Area,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { BentoCard } from "./BentoCard";

interface OHLCVRow {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

interface StockChartProps {
  ticker: string;
  currentPrice?: number;
  analysisDate?: string;
}

type Period = "1mo" | "3mo" | "6mo" | "1y" | "2y";

function computeSMA(data: OHLCVRow[], window: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < window - 1) return null;
    const slice = data.slice(i - window + 1, i + 1);
    return slice.reduce((s, d) => s + d.Close, 0) / window;
  });
}

function computeRSI(data: OHLCVRow[], period: number): (number | null)[] {
  const rsi: (number | null)[] = new Array(data.length).fill(null);
  if (data.length < period + 1) return rsi;

  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 1; i <= period; i++) {
    const diff = data[i].Close - data[i - 1].Close;
    if (diff > 0) avgGain += diff;
    else avgLoss += Math.abs(diff);
  }
  avgGain /= period;
  avgLoss /= period;

  rsi[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

  for (let i = period + 1; i < data.length; i++) {
    const diff = data[i].Close - data[i - 1].Close;
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? Math.abs(diff) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return rsi;
}

function formatVol(v: number): string {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
  return String(v);
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function StockChart({ ticker, currentPrice, analysisDate }: StockChartProps) {
  const [raw, setRaw] = useState<OHLCVRow[]>([]);
  const [period, setPeriod] = useState<Period>("1y");
  const [showSMA50, setShowSMA50] = useState(true);
  const [showSMA200, setShowSMA200] = useState(false);
  const [showRSI, setShowRSI] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/api/charts/${encodeURIComponent(ticker)}?period=${period}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.text();
          throw new Error(`${res.status}: ${body}`);
        }
        return res.json();
      })
      .then((rows: OHLCVRow[]) => setRaw(rows))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [ticker, period]);

  const chartData = useMemo(() => {
    if (!raw.length) return [];
    const sma50 = computeSMA(raw, 50);
    const sma200 = computeSMA(raw, 200);
    const rsi = computeRSI(raw, 14);

    return raw.map((row, i) => ({
      date: new Date(row.Date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      fullDate: row.Date,
      close: row.Close,
      high: row.High,
      low: row.Low,
      volume: row.Volume,
      sma50: sma50[i],
      sma200: sma200[i],
      rsi: rsi[i],
      // For candlestick-like area: use high/low range
      range: [row.Low, row.High],
    }));
  }, [raw]);

  const periods: { value: Period; label: string }[] = [
    { value: "1mo", label: "1M" },
    { value: "3mo", label: "3M" },
    { value: "6mo", label: "6M" },
    { value: "1y", label: "1Y" },
    { value: "2y", label: "2Y" },
  ];

  return (
    <BentoCard>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-400">
          <span>📈</span> {ticker} Price Chart
        </h3>

        <div className="flex items-center gap-4">
          {/* Period picker */}
          <div className="flex gap-1 rounded-lg border border-slate-700 p-0.5">
            {periods.map((p) => (
              <button
                key={p.value}
                onClick={() => setPeriod(p.value)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  period === p.value
                    ? "bg-sky-500/20 text-sky-400"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Indicator toggles */}
          <label className="flex items-center gap-1.5 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={showSMA50}
              onChange={(e) => setShowSMA50(e.target.checked)}
              className="accent-sky-500"
            />
            SMA 50
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={showSMA200}
              onChange={(e) => setShowSMA200(e.target.checked)}
              className="accent-sky-500"
            />
            SMA 200
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={showRSI}
              onChange={(e) => setShowRSI(e.target.checked)}
              className="accent-sky-500"
            />
            RSI
          </label>
        </div>
      </div>

      {loading && <p className="py-12 text-center text-slate-400">Loading chart data...</p>}
      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
          <pre className="whitespace-pre-wrap text-xs text-rose-300">{error}</pre>
        </div>
      )}

      {!loading && !error && chartData.length > 0 && (
        <div>
          {/* Price + Volume chart */}
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "#64748b" }}
                interval="preserveStartEnd"
                tickCount={8}
              />
              <YAxis
                yAxisId="price"
                orientation="right"
                domain={["auto", "auto"]}
                tick={{ fontSize: 10, fill: "#64748b" }}
                tickFormatter={(v: number) => `$${v.toFixed(0)}`}
              />
              <YAxis
                yAxisId="vol"
                orientation="left"
                domain={[0, (max: number) => max * 4]}
                tick={{ fontSize: 10, fill: "#64748b" }}
                tickFormatter={formatVol}
                hide
              />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#94a3b8" }}
                formatter={(val: number, name: string) => {
                  if (name === "volume") return [formatVol(val), "Volume"];
                  if (name === "close") return [`$${val.toFixed(2)}`, "Close"];
                  if (name === "sma50") return [`$${val.toFixed(2)}`, "SMA 50"];
                  if (name === "sma200") return [`$${val.toFixed(2)}`, "SMA 200"];
                  return [val, name];
                }}
              />

              {/* Volume bars */}
              <Bar yAxisId="vol" dataKey="volume" fill="#334155" opacity={0.4} />

              {/* Price area */}
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="close"
                stroke="#38bdf8"
                strokeWidth={2}
                fill="url(#priceGrad)"
              />

              {/* SMA lines */}
              {showSMA50 && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="sma50"
                  stroke="#f97316"
                  strokeWidth={1}
                  dot={false}
                  connectNulls={false}
                />
              )}
              {showSMA200 && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="sma200"
                  stroke="#a855f7"
                  strokeWidth={1}
                  dot={false}
                  connectNulls={false}
                />
              )}

              {/* Current price reference */}
              {currentPrice && (
                <ReferenceLine
                  yAxisId="price"
                  y={currentPrice}
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  label={{
                    value: `$${currentPrice.toFixed(2)}`,
                    position: "right",
                    fill: "#ef4444",
                    fontSize: 11,
                  }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>

          {/* RSI chart (optional) */}
          {showRSI && (
            <div className="mt-2">
              <ResponsiveContainer width="100%" height={100}>
                <ComposedChart data={chartData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" hide />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: "#64748b" }}
                    ticks={[30, 50, 70]}
                  />
                  <Line type="monotone" dataKey="rsi" stroke="#60a5fa" strokeWidth={1.5} dot={false} connectNulls={false} />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                  <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #334155",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(val: number) => [`${val.toFixed(1)}`, "RSI"]}
                  />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex justify-between px-2 text-[10px] text-slate-600">
                <span>Oversold &lt; 30</span>
                <span>RSI (14)</span>
                <span>Overbought &gt; 70</span>
              </div>
            </div>
          )}
        </div>
      )}

      {!loading && !error && chartData.length === 0 && (
        <p className="py-12 text-center text-sm text-slate-500">
          No price data available for {ticker}
        </p>
      )}
    </BentoCard>
  );
}
