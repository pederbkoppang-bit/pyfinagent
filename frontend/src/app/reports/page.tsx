"use client";

import { useEffect, useMemo, useState, Suspense } from "react";
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
import { PageSkeleton } from "@/components/Skeleton";
import { DataTable } from "@/components/DataTable";
import { EmptyState } from "@/components/states/EmptyState";
import { ReportCompareDrawer } from "@/components/ReportCompareDrawer";
import { reportsColumns, buildTickerHistory } from "@/components/reports-columns";
import { useURLState } from "@/lib/hooks";
import { listReports, getReport } from "@/lib/api";
import type { ReportSummary, SynthesisReport } from "@/lib/types";
import {
  IconChart, IconStar, IconScoringMatrix,
} from "@/lib/icons";
import { Trophy, ChartPolar, NotePencil, Files, ArrowsLeftRight } from "@/lib/icons";
import type { Icon } from "@/lib/icons";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Shared utilities ── */
function scoreColor(recommendation: string): string {
  const lower = recommendation?.toLowerCase() ?? "";
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

const PALETTE = [
  { line: "#38bdf8", fill: "#38bdf820", label: "text-sky-400", bg: "bg-sky-500/15 border-sky-500/40" },
  { line: "#a78bfa", fill: "#a78bfa20", label: "text-violet-400", bg: "bg-violet-500/15 border-violet-500/40" },
  { line: "#34d399", fill: "#34d39920", label: "text-emerald-400", bg: "bg-emerald-500/15 border-emerald-500/40" },
  { line: "#fb923c", fill: "#fb923c20", label: "text-orange-400", bg: "bg-orange-500/15 border-orange-500/40" },
  { line: "#f472b6", fill: "#f472b620", label: "text-pink-400", bg: "bg-pink-500/15 border-pink-500/40" },
];

const pillars = [
  { key: "pillar_1_corporate", label: "Corporate" },
  { key: "pillar_2_industry", label: "Industry" },
  { key: "pillar_3_valuation", label: "Valuation" },
  { key: "pillar_4_sentiment", label: "Sentiment" },
  { key: "pillar_5_governance", label: "Governance" },
] as const;

interface FullReport {
  ticker: string;
  company_name: string;
  analysis_date: string;
  synthesis: SynthesisReport;
}

interface PriceRow { Date: string; Close: number }
interface NormalizedRow { date: string; [ticker: string]: string | number }

type Tab = "history" | "compare";

const TABS: { id: Tab; label: string; icon: Icon }[] = [
  { id: "history", label: "History", icon: Files },
  { id: "compare", label: "Compare", icon: ArrowsLeftRight },
];

export default function ReportsPage() {
  return (
    <Suspense>
      <ReportsContent />
    </Suspense>
  );
}

function ReportsContent() {
  // phase-44.4: URL-state migration -- replaces manual `searchParams.get` boilerplate.
  // useURLState (cycle 44.1 foundation) keeps URL <-> state bidirectionally synced
  // so links like /reports?tab=compare&ticker=AAPL round-trip correctly.
  const [activeTab, setActiveTab] = useURLState<Tab>("tab", "history", {
    parser: (raw) => (raw === "compare" || raw === "history" ? raw : "history"),
    serializer: (v) => (v === "history" ? null : v),
  });
  // phase-44.2 cycle-69: drop the `.toUpperCase()` URL parser -- filter is
  // now case-insensitive AND matches company names (mixed case in source),
  // so forcing uppercase strips information.
  const [filter, setFilter] = useURLState<string>("ticker", "", {
    parser: (raw) => (raw ?? "").trim(),
    serializer: (v) => (v === "" ? null : v),
  });

  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // History state
  const [expanded, setExpanded] = useState<string | null>(null);

  // Compare state
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState<FullReport[]>([]);
  const [priceData, setPriceData] = useState<Record<string, PriceRow[]>>({});
  const [comparing, setComparing] = useState(false);
  // phase-44.4: drawer-open state for the compare wizard. Selection step
  // lives in the drawer (per master_design Section 3.8); results below.
  const [compareDrawerOpen, setCompareDrawerOpen] = useState(false);

  useEffect(() => {
    listReports(50)
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // phase-44.2 cycle-69 UX-audit: filter matches ticker OR company_name
  // OR recommendation (case-insensitive substring). Operator-flagged
  // 2026-05-26 that filtering by ticker alone was too restrictive --
  // company name is the natural search term.
  const filtered = (() => {
    if (!filter) return reports;
    const q = filter.toLowerCase();
    return reports.filter((r) => {
      const ticker = (r.ticker ?? "").toLowerCase();
      const company = (r.company_name ?? "").toLowerCase();
      const recommendation = (r.recommendation ?? "").toLowerCase();
      return (
        ticker.includes(q) || company.includes(q) || recommendation.includes(q)
      );
    });
  })();

  const tickers = [...new Set(reports.map((r) => r.ticker))];

  /* ── Compare logic ── */
  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const startCompare = async () => {
    setComparing(true);
    setError(null);
    const prices: Record<string, PriceRow[]> = {};

    // Parallel fetch all selected reports
    const keys = [...selected];
    const fetched = await Promise.all(
      keys.map(async (key) => {
        const [ticker, analysisDate] = key.split("|");
        try {
          const full = (await getReport(ticker, analysisDate)) as Record<string, unknown>;
          const synth =
            (full.full_report_json as Record<string, unknown>)?.final_synthesis ??
            full.full_report_json ??
            {};
          return {
            ticker: (full.ticker as string) ?? ticker,
            company_name: (full.company_name as string) ?? ticker,
            analysis_date: (full.analysis_date as string) ?? key.split("|")[1] ?? "",
            synthesis: synth as SynthesisReport,
          } as FullReport;
        } catch (e) {
          setError(`Failed to load ${ticker}: ${e instanceof Error ? e.message : String(e)}`);
          return null;
        }
      }),
    );
    const results = fetched.filter((r): r is FullReport => r !== null);

    const uniqueTickers = [...new Set(results.map((r) => r.ticker))];
    await Promise.all(
      uniqueTickers.map(async (t) => {
        try {
          const res = await fetch(`${API_BASE}/api/charts/${encodeURIComponent(t)}?period=1y`);
          if (res.ok) prices[t] = await res.json();
        } catch { /* ignore chart failures */ }
      }),
    );

    setLoaded(results);
    setPriceData(prices);
    setComparing(false);
  };

  const normalizedChart = useMemo<NormalizedRow[]>(() => {
    const tickerList = Object.keys(priceData);
    if (tickerList.length === 0) return [];
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

  // phase-44.4: ticker -> score-history map for the DataTable's sparkline
  // column. Derived from the same reports array so no extra fetch.
  const tickerHistory = useMemo(() => buildTickerHistory(reports), [reports]);
  const historyColumns = useMemo(
    () => reportsColumns(tickerHistory),
    [tickerHistory],
  );

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

  const scoreBarData = useMemo(() => {
    return loaded.map((r) => ({
      ticker: r.ticker,
      score: r.synthesis.final_weighted_score ?? 0,
    }));
  }, [loaded]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* phase-16.49: canonical two-zone shell -- header + tab bar pinned, content scrolls */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Reports</h2>
            <p className="text-sm text-slate-500">Browse past analyses or compare companies side-by-side</p>
          </div>

          {/* phase-44.4: ARIA tablist (role=tablist + role=tab + aria-selected
              + aria-controls per W3C WAI-ARIA APG). Roving tabindex: only the
              active tab is in focus order. */}
          <div
            role="tablist"
            aria-label="Reports view"
            className="mb-6 flex gap-1 rounded-lg bg-navy-800/60 p-1"
          >
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  id={`tab-${tab.id}`}
                  aria-selected={isActive}
                  aria-controls={`panel-${tab.id}`}
                  tabIndex={isActive ? 0 : -1}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px] ${
                    isActive ? "bg-sky-500/10 text-sky-400" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <tab.icon size={16} weight={isActive ? "fill" : "regular"} />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
        {error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
            <pre className="whitespace-pre-wrap text-xs text-rose-300">{error}</pre>
          </div>
        )}

        {loading && <PageSkeleton />}

        {/* phase-44.4: HISTORY TAB -- DataTable foundation + EmptyState */}
        {activeTab === "history" && !loading && (
          <div
            role="tabpanel"
            id="panel-history"
            aria-labelledby="tab-history"
            tabIndex={0}
          >
            {/* Ticker filter */}
            <div className="mb-6 flex flex-wrap items-center gap-3">
              <input
                type="text"
                placeholder="Filter ticker, company, or recommendation..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                aria-label="Filter reports by ticker, company, or recommendation"
                className="w-40 rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
              />
              {tickers.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setFilter(filter === t ? "" : t)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    filter === t
                      ? "bg-sky-500/20 text-sky-300"
                      : "bg-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {t}
                </button>
              ))}
              {filter && (
                <button
                  type="button"
                  onClick={() => setFilter("")}
                  className="text-xs text-slate-500 hover:text-slate-300"
                >
                  Clear
                </button>
              )}
            </div>

            {filtered.length === 0 ? (
              <EmptyState
                icon={Files}
                title={filter ? `No reports for ${filter}` : "No reports yet"}
                description={
                  filter
                    ? "Try clearing the ticker filter or running a new analysis."
                    : "Reports will appear here after the first analysis completes."
                }
              />
            ) : (
              <div className="rounded-xl border border-navy-700 bg-navy-800/40 p-4">
                <DataTable
                  data={filtered}
                  columns={historyColumns}
                  ariaLabel="Reports history"
                  onRowClick={(r) => {
                    const key = `${r.ticker}|${r.analysis_date}`;
                    setExpanded((cur) => (cur === key ? null : key));
                  }}
                  emptyState={null}
                />
                {expanded && (() => {
                  const [exTicker, exDate] = expanded.split("|");
                  const exReport = filtered.find(
                    (r) => r.ticker === exTicker && r.analysis_date === exDate,
                  );
                  if (!exReport) return null;
                  return (
                    <div className="mt-4 rounded-lg border border-sky-500/20 bg-sky-950/20 p-4">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="font-mono font-semibold text-slate-200">
                          {exReport.ticker} -- {new Date(exReport.analysis_date).toLocaleDateString()}
                        </span>
                        <button
                          type="button"
                          onClick={() => setExpanded(null)}
                          className="text-xs text-slate-400 hover:text-slate-200"
                        >
                          Close
                        </button>
                      </div>
                      <p className="text-sm leading-relaxed text-slate-400">
                        {exReport.summary}
                      </p>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* phase-44.4: COMPARE TAB -- wizard moved into ReportCompareDrawer overlay.
            Selection step is in the drawer; results render below when comparison
            data is loaded. */}
        {activeTab === "compare" && !loading && (
          <div
            role="tabpanel"
            id="panel-compare"
            aria-labelledby="tab-compare"
            tabIndex={0}
          >
            {reports.length === 0 && (
              <EmptyState
                icon={Files}
                title="No reports yet"
                description="Reports will appear here after the first analysis completes; then return to compare them side-by-side."
              />
            )}

            {reports.length > 0 && loaded.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12">
                <p className="mb-4 text-sm text-slate-400">
                  {selected.size} of {reports.length} reports selected.
                </p>
                <button
                  type="button"
                  onClick={() => setCompareDrawerOpen(true)}
                  className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 min-h-[24px]"
                >
                  {selected.size >= 2 ? "Re-open selection" : "Select reports to compare"}
                </button>
                {comparing && (
                  <p className="mt-3 text-xs text-slate-500">Loading comparison data...</p>
                )}
              </div>
            )}

            {/* Comparison results */}
            {loaded.length > 0 && (
              <div className="space-y-6">
                <button
                  onClick={() => { setLoaded([]); setPriceData({}); }}
                  className="text-sm text-sky-400 hover:underline"
                >
                  ← Back to selection
                </button>

                {/* Company header badges */}
                <div className="flex flex-wrap items-center gap-3">
                  {loaded.map((r, i) => {
                    const c = PALETTE[i % PALETTE.length];
                    return (
                      <div key={`${r.ticker}-${i}`} className={`flex items-center gap-2 rounded-full border px-4 py-1.5 ${c.bg}`}>
                        <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: c.line }} />
                        <span className={`font-mono font-bold ${c.label}`}>{r.ticker}</span>
                        <span className="text-xs text-slate-400">{r.company_name}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Price comparison chart */}
                {normalizedChart.length > 0 && (
                  <BentoCard>
                    <h3 className="mb-1 text-lg font-semibold text-slate-300">
                      <IconChart size={20} weight="duotone" className="inline text-slate-400" /> Price Performance (% Change, 1Y)
                    </h3>
                    <p className="mb-4 text-xs text-slate-500">Normalized to starting date</p>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={normalizedChart} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} interval="preserveStartEnd" tickCount={10} />
                        <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`} />
                        <Tooltip
                          contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                          formatter={(val: number, name: string) => [`${val > 0 ? "+" : ""}${val.toFixed(2)}%`, name]}
                        />
                        {Object.keys(priceData).map((t, i) => (
                          <Line key={t} type="monotone" dataKey={t} stroke={PALETTE[i % PALETTE.length].line} strokeWidth={2} dot={false} name={t} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </BentoCard>
                )}

                {/* Score bar + Radar */}
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">
                      <Trophy size={20} weight="duotone" className="inline text-slate-400" /> Overall Score
                    </h3>
                    <ResponsiveContainer width="100%" height={loaded.length * 60 + 40}>
                      <BarChart data={scoreBarData} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                        <XAxis type="number" domain={[0, 10]} tick={{ fontSize: 11, fill: "#64748b" }} />
                        <YAxis type="category" dataKey="ticker" tick={{ fontSize: 13, fill: "#e2e8f0", fontFamily: "monospace", fontWeight: 700 }} width={70} />
                        <Tooltip
                          contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                          formatter={(val: number) => [val.toFixed(2), "Score"]}
                        />
                        <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={28}>
                          {scoreBarData.map((d, i) => (
                            <Cell key={d.ticker} fill={PALETTE[i % PALETTE.length].line} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </BentoCard>

                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">
                      <ChartPolar size={20} weight="duotone" className="inline text-slate-400" /> Pillar Radar
                    </h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <RadarChart data={radarData} outerRadius="75%">
                        <PolarGrid stroke="#334155" />
                        <PolarAngleAxis dataKey="pillar" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                        <PolarRadiusAxis domain={[0, 10]} tick={{ fontSize: 9, fill: "#475569" }} axisLine={false} />
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
                        <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </BentoCard>
                </div>

                {/* Score Comparison Table */}
                <BentoCard>
                  <h3 className="mb-4 text-lg font-semibold text-slate-300">
                    <IconScoringMatrix size={20} weight="duotone" className="inline text-slate-400" /> Detailed Score Comparison
                  </h3>
                  <div className="overflow-x-auto scrollbar-thin">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700 text-left">
                          <th className="px-3 py-2 text-slate-400">Company</th>
                          <th className="px-3 py-2 text-slate-400">Score</th>
                          <th className="px-3 py-2 text-slate-400">Verdict</th>
                          {pillars.map((p) => (
                            <th key={p.key} className="px-3 py-2 text-slate-400">{p.label}</th>
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
                                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color.line }} />
                                  <span className={`font-mono font-bold ${color.label}`}>{r.ticker}</span>
                                </div>
                              </td>
                              <td className="px-3 py-2 font-mono font-bold text-sky-300">
                                {r.synthesis.final_weighted_score?.toFixed(2) ?? "N/A"}
                              </td>
                              <td className={`px-3 py-2 font-medium ${scoreColor(r.synthesis.recommendation?.action ?? "")}`}>
                                {r.synthesis.recommendation?.action ?? "N/A"}
                              </td>
                              {pillars.map((p) => {
                                const sm = r.synthesis.scoring_matrix as unknown as Record<string, number>;
                                const val = sm?.[p.key] as number | undefined;
                                const allVals = loaded.map(
                                  (lr) => ((lr.synthesis.scoring_matrix as unknown as Record<string, number>)?.[p.key] ?? 0) as number,
                                );
                                const isMax = val != null && val === Math.max(...allVals) && allVals.filter((v) => v === val).length === 1;
                                return (
                                  <td key={p.key} className="px-3 py-2">
                                    <div className="flex items-center gap-2">
                                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800">
                                        <div className={`h-full rounded-full ${scoreBg(val ?? 0)}`} style={{ width: `${((val ?? 0) / 10) * 100}%` }} />
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

                {/* Side-by-side qualitative */}
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
                              <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: color.line }} />
                              <span className={`font-mono text-lg font-bold ${color.label}`}>{r.ticker}</span>
                              <span className="text-sm text-slate-400">{r.company_name}</span>
                            </div>
                            <span className="font-mono text-lg font-bold text-sky-400">
                              {r.synthesis.final_weighted_score?.toFixed(2) ?? "—"}
                            </span>
                          </div>
                          <div className="mb-4">
                            <span className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${scoreColor(r.synthesis.recommendation?.action ?? "")} bg-slate-800`}>
                              {r.synthesis.recommendation?.action ?? "N/A"}
                            </span>
                            <span className="ml-2 text-xs text-slate-500">{new Date(r.analysis_date).toLocaleDateString()}</span>
                          </div>
                          <div className="mb-4">
                            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Justification</p>
                            <p className="text-sm leading-relaxed text-slate-300">{r.synthesis.recommendation?.justification ?? "N/A"}</p>
                          </div>
                          <div className="mb-4">
                            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Summary</p>
                            <p className="text-sm leading-relaxed text-slate-400">{r.synthesis.final_summary ?? "N/A"}</p>
                          </div>
                          {r.synthesis.key_risks?.length > 0 && (
                            <div>
                              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Key Risks</p>
                              <ul className="space-y-1">
                                {r.synthesis.key_risks.map((risk, ri) => (
                                  <li key={ri} className="flex items-start gap-2 text-sm text-slate-400">
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
          </div>
        )}
        </div>
      </main>
      {/* phase-44.4: ReportCompareDrawer overlay -- selection wizard. */}
      <ReportCompareDrawer
        open={compareDrawerOpen}
        onClose={() => setCompareDrawerOpen(false)}
        reports={reports}
        selected={selected}
        onToggle={toggle}
        onStartCompare={startCompare}
        comparing={comparing}
      />
    </div>
  );
}
