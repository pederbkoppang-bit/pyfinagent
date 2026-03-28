"use client";

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { listReports, getReport } from "@/lib/api";
import type { ReportSummary, SynthesisReport } from "@/lib/types";
import {
  IconChart, IconStar, IconCheck, IconScoringMatrix, IconSearch,
} from "@/lib/icons";
import { Trophy, ChartPolar, NotePencil } from "@phosphor-icons/react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* Ticker-indexed colour palette */
const PALETTE = [
  { line: "#38bdf8", fill: "#38bdf820", label: "text-sky-400", bg: "bg-sky-500/15 border-sky-500/40" },
  { line: "#a78bfa", fill: "#a78bfa20", label: "text-violet-400", bg: "bg-violet-500/15 border-violet-500/40" },
  { line: "#34d399", fill: "#34d39920", label: "text-emerald-400", bg: "bg-emerald-500/15 border-emerald-500/40" },
  { line: "#fb923c", fill: "#fb923c20", label: "text-orange-400", bg: "bg-orange-500/15 border-orange-500/40" },
  { line: "#f472b6", fill: "#f472b620", label: "text-pink-400", bg: "bg-pink-500/15 border-pink-500/40" },
];

function scoreColor(action: string): string {
  const lower = action.toLowerCase();
  if (lower.includes("strong buy")) return "text-emerald-400";
  if (lower.includes("buy")) return "text-emerald-300";
  if (lower.includes("sell")) return "text-rose-400";
  return "text-slate-300";
}

function scoreBg(score: number): string {
  if (score >= 7.5) return "bg-emerald-500";
  if (score >= 5) return "bg-sky-500";
  if (score >= 3) return "bg-amber-500";
  return "bg-rose-500";
}

interface FullReport {
  ticker: string;
  company_name: string;
  analysis_date: string;
  synthesis: SynthesisReport;
}

interface PriceRow { Date: string; Close: number }
interface NormalizedRow { date: string; [ticker: string]: string | number }

const pillars = [
  { key: "pillar_1_corporate", label: "Corporate" },
  { key: "pillar_2_industry", label: "Industry" },
  { key: "pillar_3_valuation", label: "Valuation" },
  { key: "pillar_4_sentiment", label: "Sentiment" },
  { key: "pillar_5_governance", label: "Governance" },
] as const;

export default function ComparePage() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState<FullReport[]>([]);
  const [priceData, setPriceData] = useState<Record<string, PriceRow[]>>({});
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listReports(50)
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const tickers = useMemo(
    () => [...new Set([...selected].map((k) => k.split("|")[0]))],
    [selected],
  );

  const startCompare = async () => {
    setComparing(true);
    setError(null);
    const results: FullReport[] = [];
    const prices: Record<string, PriceRow[]> = {};

    for (const key of selected) {
      const [ticker, analysisDate] = key.split("|");
      try {
        const full = (await getReport(ticker, analysisDate)) as Record<string, any>;
        const synth =
          full.full_report_json?.final_synthesis ??
          full.full_report_json ??
          {};
        results.push({
          ticker: full.ticker ?? ticker,
          company_name: full.company_name ?? ticker,
          analysis_date: full.analysis_date ?? key.split("|")[1] ?? "",
          synthesis: synth as SynthesisReport,
        });
      } catch (e) {
        setError(
          `Failed to load ${ticker}: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }

    // Fetch price histories in parallel for all unique tickers
    const uniqueTickers = [...new Set(results.map((r) => r.ticker))];
    await Promise.all(
      uniqueTickers.map(async (t) => {
        try {
          const res = await fetch(
            `${API_BASE}/api/charts/${encodeURIComponent(t)}?period=1y`,
          );
          if (res.ok) prices[t] = await res.json();
        } catch {
          /* ignore chart failures */
        }
      }),
    );

    setLoaded(results);
    setPriceData(prices);
    setComparing(false);
  };

  /* Normalize prices to % change for overlay comparison */
  const normalizedChart = useMemo<NormalizedRow[]>(() => {
    const tickerList = Object.keys(priceData);
    if (tickerList.length === 0) return [];

    // Build a date-indexed map { "2025-03-08": { AAPL: %, GOOGL: % } }
    const dateMap = new Map<string, Record<string, number>>();
    for (const t of tickerList) {
      const rows = priceData[t];
      if (!rows?.length) continue;
      const base = rows[0].Close;
      for (const r of rows) {
        const d = new Date(r.Date).toISOString().slice(0, 10);
        const entry = dateMap.get(d) ?? {};
        entry[t] = ((r.Close - base) / base) * 100;
        dateMap.set(d, entry);
      }
    }

    return [...dateMap.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, vals]) => ({
        date: new Date(date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        ...vals,
      }));
  }, [priceData]);

  /* Radar chart data */
  const radarData = useMemo(() => {
    return pillars.map((p) => {
      const entry: Record<string, string | number> = { pillar: p.label };
      loaded.forEach((r, i) => {
        const sm = r.synthesis.scoring_matrix as unknown as Record<string, number>;
        entry[`${r.ticker}-${i}`] = sm?.[p.key] ?? 0;
      });
      return entry;
    });
  }, [loaded]);

  /* Bar chart data for overall scores */
  const scoreBarData = useMemo(() => {
    return loaded.map((r) => ({
      ticker: r.ticker,
      score: r.synthesis.final_weighted_score ?? 0,
    }));
  }, [loaded]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
        <h2 className="mb-2 text-2xl font-bold text-slate-100">
          Compare Companies
        </h2>
        <p className="mb-6 text-sm text-slate-500">
          Select companies to compare their analysis, scoring, and price
          performance side-by-side
        </p>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
            <pre className="whitespace-pre-wrap text-xs text-rose-300">
              {error}
            </pre>
          </div>
        )}

        {/* ── Selection ────────────────────────────── */}
        {loading && <p className="text-slate-400">Loading reports...</p>}
        {!loading && reports.length === 0 && (
          <p className="text-slate-500">No reports found yet.</p>
        )}

        {!loading && reports.length > 0 && loaded.length === 0 && (
          <>
            <div className="mb-4 space-y-2">
              {reports.map((r) => {
                const key = `${r.ticker}|${r.analysis_date}`;
                const isSelected = selected.has(key);
                return (
                  <button
                    key={key}
                    onClick={() => toggle(key)}
                    className={`flex w-full items-center justify-between rounded-lg border p-3 text-left transition-colors ${
                      isSelected
                        ? "border-sky-500/50 bg-sky-500/10"
                        : "border-slate-800 bg-slate-900/50 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={`flex h-4 w-4 items-center justify-center rounded border text-[10px] ${
                          isSelected
                            ? "border-sky-400 bg-sky-400 text-white"
                            : "border-slate-600"
                        }`}
                      >
                        {isSelected && <IconCheck size={12} weight="bold" />}
                      </span>
                      <span className="font-mono font-bold text-slate-200">
                        {r.ticker}
                      </span>
                      <span className="text-sm text-slate-400">
                        {r.company_name}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date(r.analysis_date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-sm text-sky-300">
                        {r.final_score.toFixed(2)}
                      </span>
                      <span
                        className={`text-xs font-medium ${scoreColor(r.recommendation)}`}
                      >
                        {r.recommendation}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            <button
              onClick={startCompare}
              disabled={selected.size < 2 || comparing}
              className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {comparing
                ? "Loading..."
                : `Compare ${selected.size} Companies`}
            </button>
          </>
        )}

        {/* ── Comparison View ─────────────────────── */}
        {loaded.length > 0 && (
          <div className="mt-6 space-y-6">
            <button
              onClick={() => {
                setLoaded([]);
                setPriceData({});
              }}
              className="text-sm text-sky-400 hover:underline"
            >
              ← Back to selection
            </button>

            {/* Company header badges */}
            <div className="flex flex-wrap items-center gap-3">
              {loaded.map((r, i) => {
                const c = PALETTE[i % PALETTE.length];
                return (
                  <div
                    key={`${r.ticker}-${i}`}
                    className={`flex items-center gap-2 rounded-full border px-4 py-1.5 ${c.bg}`}
                  >
                    <span
                      className="inline-block h-3 w-3 rounded-full"
                      style={{ backgroundColor: c.line }}
                    />
                    <span className={`font-mono font-bold ${c.label}`}>
                      {r.ticker}
                    </span>
                    <span className="text-xs text-slate-400">
                      {r.company_name}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* ── 1. Normalized Price Comparison ─── */}
            {normalizedChart.length > 0 && (
              <BentoCard>
                <h3 className="mb-1 text-lg font-semibold text-slate-300">
                  <IconChart size={20} weight="duotone" className="inline text-slate-400" /> Price Performance (% Change, 1Y)
                </h3>
                <p className="mb-4 text-xs text-slate-500">
                  Normalized to starting date — compare how each stock moved
                  relative to itself
                </p>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart
                    data={normalizedChart}
                    margin={{ top: 5, right: 20, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10, fill: "#64748b" }}
                      interval="preserveStartEnd"
                      tickCount={10}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#64748b" }}
                      tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(val: number, name: string) => [
                        `${val > 0 ? "+" : ""}${val.toFixed(2)}%`,
                        name,
                      ]}
                    />
                    {Object.keys(priceData).map((t, i) => (
                      <Line
                        key={t}
                        type="monotone"
                        dataKey={t}
                        stroke={PALETTE[i % PALETTE.length].line}
                        strokeWidth={2}
                        dot={false}
                        name={t}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </BentoCard>
            )}

            {/* ── 2. Overall Score Bar + Radar ───── */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Overall score bars */}
              <BentoCard>
                <h3 className="mb-4 text-lg font-semibold text-slate-300">
                  <Trophy size={20} weight="duotone" className="inline text-slate-400" /> Overall Score
                </h3>
                <ResponsiveContainer width="100%" height={loaded.length * 60 + 40}>
                  <BarChart
                    data={scoreBarData}
                    layout="vertical"
                    margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 10]}
                      tick={{ fontSize: 11, fill: "#64748b" }}
                    />
                    <YAxis
                      type="category"
                      dataKey="ticker"
                      tick={{ fontSize: 13, fill: "#e2e8f0", fontFamily: "monospace", fontWeight: 700 }}
                      width={70}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(val: number) => [val.toFixed(2), "Score"]}
                    />
                    <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={28}>
                      {scoreBarData.map((d, i) => (
                        <Cell
                          key={d.ticker}
                          fill={PALETTE[i % PALETTE.length].line}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </BentoCard>

              {/* Radar overlay */}
              <BentoCard>
                <h3 className="mb-4 text-lg font-semibold text-slate-300">
                  <ChartPolar size={20} weight="duotone" className="inline text-slate-400" /> Pillar Radar
                </h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={radarData} outerRadius="75%">
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis
                      dataKey="pillar"
                      tick={{ fontSize: 11, fill: "#94a3b8" }}
                    />
                    <PolarRadiusAxis
                      domain={[0, 10]}
                      tick={{ fontSize: 9, fill: "#475569" }}
                      axisLine={false}
                    />
                    {loaded.map((r, i) => (
                      <Radar
                        key={`${r.ticker}-${i}`}
                        name={`${r.ticker} (#${i + 1})`}
                        dataKey={`${r.ticker}-${i}`}
                        stroke={PALETTE[i % PALETTE.length].line}
                        fill={PALETTE[i % PALETTE.length].line}
                        fillOpacity={0.15}
                        strokeWidth={2}
                      />
                    ))}
                    <Legend
                      wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </BentoCard>
            </div>

            {/* ── 3. Score Comparison Table ──────── */}
            <BentoCard>
              <h3 className="mb-4 text-lg font-semibold text-slate-300">
                <IconScoringMatrix size={20} weight="duotone" className="inline text-slate-400" /> Detailed Score Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-left">
                      <th className="px-3 py-2 text-slate-400">Company</th>
                      <th className="px-3 py-2 text-slate-400">Score</th>
                      <th className="px-3 py-2 text-slate-400">Verdict</th>
                      {pillars.map((p) => (
                        <th key={p.key} className="px-3 py-2 text-slate-400">
                          {p.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {loaded.map((r, i) => {
                      const color = PALETTE[i % PALETTE.length];
                      return (
                        <tr key={`${r.ticker}-${i}`} className="border-b border-slate-800">
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <span
                                className="inline-block h-2.5 w-2.5 rounded-full"
                                style={{ backgroundColor: color.line }}
                              />
                              <span className={`font-mono font-bold ${color.label}`}>
                                {r.ticker}
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-2 font-mono font-bold text-sky-300">
                            {r.synthesis.final_weighted_score?.toFixed(2) ?? "N/A"}
                          </td>
                          <td
                            className={`px-3 py-2 font-medium ${scoreColor(
                              r.synthesis.recommendation?.action ?? "",
                            )}`}
                          >
                            {r.synthesis.recommendation?.action ?? "N/A"}
                          </td>
                          {pillars.map((p) => {
                            const sm = r.synthesis.scoring_matrix as unknown as Record<string, number>;
                            const val = sm?.[p.key] as number | undefined;
                            // Find the max for this pillar across all loaded
                            const allVals = loaded.map(
                              (lr) => ((lr.synthesis.scoring_matrix as unknown as Record<string, number>)?.[p.key] ?? 0) as number,
                            );
                            const isMax = val != null && val === Math.max(...allVals) && allVals.filter((v) => v === val).length === 1;
                            return (
                              <td key={p.key} className="px-3 py-2">
                                <div className="flex items-center gap-2">
                                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800">
                                    <div
                                      className={`h-full rounded-full ${scoreBg(val ?? 0)}`}
                                      style={{ width: `${((val ?? 0) / 10) * 100}%` }}
                                    />
                                  </div>
                                  <span className={`font-mono text-xs ${isMax ? "font-bold text-emerald-400" : "text-slate-300"}`}>
                                    {val?.toFixed(1) ?? "—"}
                                    {isMax && <IconStar size={12} weight="fill" className="ml-1 inline text-emerald-400" />}
                                  </span>
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </BentoCard>

            {/* ── 4. Side-by-side Qualitative ───── */}
            <div>
              <h3 className="mb-4 text-lg font-semibold text-slate-300">
                <NotePencil size={20} weight="duotone" className="inline text-slate-400" /> Side-by-Side Analysis
              </h3>
              <div className={`grid gap-6 ${loaded.length === 2 ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
                {loaded.map((r, i) => {
                  const color = PALETTE[i % PALETTE.length];
                  return (
                    <BentoCard key={`${r.ticker}-${i}`}>
                      <div className="mb-4 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className="inline-block h-3 w-3 rounded-full"
                            style={{ backgroundColor: color.line }}
                          />
                          <span className={`font-mono text-lg font-bold ${color.label}`}>
                            {r.ticker}
                          </span>
                          <span className="text-sm text-slate-400">
                            {r.company_name}
                          </span>
                        </div>
                        <span className="font-mono text-lg font-bold text-sky-400">
                          {r.synthesis.final_weighted_score?.toFixed(2) ?? "—"}
                        </span>
                      </div>

                      <div className="mb-4">
                        <span
                          className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${scoreColor(
                            r.synthesis.recommendation?.action ?? "",
                          )} bg-slate-800`}
                        >
                          {r.synthesis.recommendation?.action ?? "N/A"}
                        </span>
                        <span className="ml-2 text-xs text-slate-500">
                          {new Date(r.analysis_date).toLocaleDateString()}
                        </span>
                      </div>

                      <div className="mb-4">
                        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                          Justification
                        </p>
                        <p className="text-sm leading-relaxed text-slate-300">
                          {r.synthesis.recommendation?.justification ?? "N/A"}
                        </p>
                      </div>

                      <div className="mb-4">
                        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                          Summary
                        </p>
                        <p className="text-sm leading-relaxed text-slate-400">
                          {r.synthesis.final_summary ?? "N/A"}
                        </p>
                      </div>

                      {r.synthesis.key_risks?.length > 0 && (
                        <div>
                          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                            Key Risks
                          </p>
                          <ul className="space-y-1">
                            {r.synthesis.key_risks.map((risk, ri) => (
                              <li
                                key={ri}
                                className="flex items-start gap-2 text-sm text-slate-400"
                              >
                                <span className="mt-1 text-rose-400">•</span>
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </BentoCard>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
