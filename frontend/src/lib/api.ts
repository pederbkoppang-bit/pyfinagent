/**
 * API client for the PyFinAgent FastAPI backend.
 */

import type {
  AllSignals,
  AnalysisResponse,
  AnalysisStatusResponse,
  BacktestResults,
  BacktestRunSummary,
  BacktestStatus,
  BacktestWindowResult,
  CacheStats,
  CostHistoryEntry,
  FullSettings,
  IngestionStatus,
  LatestCostSummary,
  ModelConfig,
  ModelPricing,
  OptimizerBest,
  OptimizerExperiment,
  OptimizerInsights,
  OptimizerRunSummary,
  OptimizerStatus,
  PaperPerformance,
  PaperPortfolio,
  PaperPosition,
  PaperSnapshot,
  PaperTrade,
  PaperTradingStatus,
  PerfExperiment,
  PerfOptimizerStatus,
  PerfSummary,
  PerformanceStats,
  PortfolioPerformance,
  PortfolioPosition,
  ReportSummary,
  SharpeHistoryResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getAuthToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  try {
    const res = await fetch("/api/auth/session");
    if (!res.ok) return null;
    const session = await res.json();
    // NextAuth JWT — the session cookie itself is the token
    // For backend auth, we pass the raw session token cookie
    const cookies = document.cookie.split(";").map((c) => c.trim());
    const tokenCookie = cookies.find(
      (c) => c.startsWith("__Secure-authjs.session-token=") || c.startsWith("authjs.session-token=")
    );
    if (tokenCookie) return tokenCookie.split("=").slice(1).join("=");
    return session?.user ? "session-active" : null;
  } catch {
    return null;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    ...Object.fromEntries(
      Object.entries(init?.headers || {}).filter(([, v]) => v != null) as [string, string][]
    ),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Timeout: abort fetch after 30 seconds to prevent infinite hangs
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      signal: init?.signal ?? controller.signal,
    });
  } catch (err) {
    // Abort / timeout
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        `Request to ${path} timed out after 30 seconds. ` +
        "The backend may be overloaded or unresponsive."
      );
    }
    // Network-level failure (CORS, DNS, refused, etc.)
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
      throw new Error(
        `Cannot reach backend at ${API_BASE}. ` +
        "Make sure the FastAPI server is running (uvicorn backend.main:app --port 8000)."
      );
    }
    throw new Error(`Network error calling ${path}: ${msg}`);
  } finally {
    clearTimeout(timeoutId);
  }
  if (!res.ok) {
    // 401 → redirect to login
    if (res.status === 401) {
      if (typeof window !== "undefined") {
        window.location.href = "/login";
      }
      throw new Error("Session expired. Redirecting to login.");
    }
    let detail: string;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    if (res.status === 422) {
      throw new Error(`Invalid request to ${path}: ${detail}`);
    }
    if (res.status === 500) {
      throw new Error(`Server error on ${path}. Check the backend logs for details.`);
    }
    if (res.status === 404) {
      throw new Error(`Endpoint not found: ${path}. The backend may be running an older version.`);
    }
    throw new Error(`API ${res.status} on ${path}: ${detail}`);
  }
  return res.json();
}

// ── Analysis ─────────────────────────────────────────────────────

export function startAnalysis(ticker: string): Promise<AnalysisResponse> {
  return apiFetch("/api/analysis/", {
    method: "POST",
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });
}

export function getAnalysisStatus(
  analysisId: string
): Promise<AnalysisStatusResponse> {
  return apiFetch(`/api/analysis/${analysisId}`);
}

// ── Reports ──────────────────────────────────────────────────────

export function listReports(limit = 20): Promise<ReportSummary[]> {
  return apiFetch(`/api/reports/?limit=${limit}`);
}

export function getReport(ticker: string, analysisDate?: string): Promise<Record<string, unknown>> {
  const params = analysisDate ? `?analysis_date=${encodeURIComponent(analysisDate)}` : "";
  return apiFetch(`/api/reports/${ticker}${params}`);
}

// ── Performance ──────────────────────────────────────────────────

export function getPerformanceStats(): Promise<PerformanceStats> {
  return apiFetch("/api/reports/performance");
}

export function evaluateOutcomes(): Promise<{
  evaluated: number;
  outcomes: unknown[];
}> {
  return apiFetch("/api/reports/evaluate", { method: "POST" });
}

export function getCostHistory(limit = 50): Promise<CostHistoryEntry[]> {
  return apiFetch(`/api/reports/cost-history?limit=${limit}`);
}

// ── Health ───────────────────────────────────────────────────────

export function healthCheck(): Promise<{ status: string; service: string; version?: string }> {
  return apiFetch("/api/health");
}

// ── Signals ─────────────────────────────────────────────────────

export function getAllSignals(ticker: string): Promise<AllSignals> {
  return apiFetch(`/api/signals/${encodeURIComponent(ticker.toUpperCase())}`);
}

export function getSignal(ticker: string, signal: string): Promise<Record<string, unknown>> {
  return apiFetch(`/api/signals/${encodeURIComponent(ticker.toUpperCase())}/${signal}`);
}

export function getMacroIndicators(): Promise<Record<string, unknown>> {
  return apiFetch("/api/signals/macro/indicators");
}

// ── Portfolio ───────────────────────────────────────────────────

export function listPortfolioPositions(): Promise<PortfolioPosition[]> {
  return apiFetch("/api/portfolio/");
}

export function addPortfolioPosition(body: {
  ticker: string;
  quantity: number;
  avg_entry_price: number;
  recommendation?: string;
  recommendation_score?: number;
}): Promise<PortfolioPosition> {
  return apiFetch("/api/portfolio/", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function deletePortfolioPosition(id: string): Promise<{ message: string }> {
  return apiFetch(`/api/portfolio/${id}`, { method: "DELETE" });
}

export function getPortfolioPerformance(): Promise<PortfolioPerformance> {
  return apiFetch("/api/portfolio/performance");
}

// ── Settings ────────────────────────────────────────────────────

export function getModelConfig(): Promise<ModelConfig> {
  return apiFetch("/api/settings/models");
}

export function getAvailableModels(): Promise<ModelPricing[]> {
  return apiFetch("/api/settings/models/available");
}

export function updateModelConfig(body: { gemini_model?: string; deep_think_model?: string }): Promise<ModelConfig> {
  return apiFetch("/api/settings/models", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getFullSettings(): Promise<FullSettings> {
  return apiFetch("/api/settings/");
}

export function updateSettings(body: Partial<FullSettings>): Promise<FullSettings> {
  return apiFetch("/api/settings/", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function getLatestCostSummary(): Promise<LatestCostSummary> {
  return apiFetch("/api/reports/latest-cost-summary");
}

// ── Paper Trading ───────────────────────────────────────────────

export function startPaperTrading(): Promise<{ status: string; portfolio_id: string; starting_capital: number; scheduler_active: boolean }> {
  return apiFetch("/api/paper-trading/start", { method: "POST" });
}

export function stopPaperTrading(): Promise<{ status: string; message?: string }> {
  return apiFetch("/api/paper-trading/stop", { method: "POST" });
}

export function getPaperTradingStatus(): Promise<PaperTradingStatus> {
  return apiFetch("/api/paper-trading/status");
}

export function getPaperPortfolio(): Promise<{ portfolio: PaperPortfolio; positions: PaperPosition[] }> {
  return apiFetch("/api/paper-trading/portfolio");
}

export function getPaperTrades(limit = 100): Promise<{ trades: PaperTrade[]; count: number }> {
  return apiFetch(`/api/paper-trading/trades?limit=${limit}`);
}

export function getPaperSnapshots(limit = 365): Promise<{ snapshots: PaperSnapshot[]; count: number }> {
  return apiFetch(`/api/paper-trading/snapshots?limit=${limit}`);
}

export function getPaperPerformance(): Promise<PaperPerformance> {
  return apiFetch("/api/paper-trading/performance");
}

export function triggerPaperTradingCycle(): Promise<{ status: string; message: string }> {
  return apiFetch("/api/paper-trading/run-now", { method: "POST" });
}

// ── Backtest ────────────────────────────────────────────────────

export function runBacktest(params?: Record<string, unknown>): Promise<{ status: string; run_id: string }> {
  return apiFetch("/api/backtest/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
}

export function getBacktestStatus(): Promise<BacktestStatus> {
  return apiFetch("/api/backtest/status");
}

export function getBacktestResults(): Promise<BacktestResults> {
  return apiFetch("/api/backtest/results");
}

export function getBacktestWindowDetail(windowId: number): Promise<BacktestWindowResult> {
  return apiFetch(`/api/backtest/results/${windowId}`);
}

export function runDataIngestion(params?: { start_date?: string; end_date?: string }): Promise<{ status: string; result: Record<string, unknown> }> {
  return apiFetch("/api/backtest/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
}

export function getIngestionStatus(): Promise<IngestionStatus> {
  return apiFetch("/api/backtest/ingest/status");
}

export function startOptimizer(params?: { max_iterations?: number; use_llm?: boolean }): Promise<{ status: string; max_iterations: number }> {
  return apiFetch("/api/backtest/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
}

export function stopOptimizer(): Promise<{ status: string }> {
  return apiFetch("/api/backtest/optimize/stop", { method: "POST" });
}

export function getOptimizerStatus(): Promise<OptimizerStatus> {
  return apiFetch("/api/backtest/optimize/status");
}

export function getOptimizerExperiments(runId?: string, runIndex?: number): Promise<{ experiments: OptimizerExperiment[] }> {
  const qs = new URLSearchParams();
  if (runId) qs.set("run_id", runId);
  if (runIndex !== undefined) qs.set("run_index", String(runIndex));
  const params = qs.toString() ? `?${qs}` : "";
  return apiFetch(`/api/backtest/optimize/experiments${params}`);
}

export function getOptimizerBest(): Promise<OptimizerBest> {
  return apiFetch("/api/backtest/optimize/best");
}

export function getOptimizerRuns(): Promise<{ runs: OptimizerRunSummary[] }> {
  return apiFetch("/api/backtest/optimize/runs");
}

export function deleteOptimizerHistory(): Promise<{ status: string; files: string[] }> {
  return apiFetch("/api/backtest/optimize/history", { method: "DELETE" });
}

export function getOptimizerInsights(): Promise<OptimizerInsights> {
  return apiFetch("/api/backtest/optimize/insights");
}

export function getBacktestRuns(): Promise<{ runs: BacktestRunSummary[] }> {
  return apiFetch("/api/backtest/runs");
}

export function loadBacktestRun(runId: string): Promise<BacktestResults> {
  return apiFetch(`/api/backtest/runs/${encodeURIComponent(runId)}`);
}

export function deleteBacktestRun(runId: string): Promise<{ status: string; run_id: string }> {
  return apiFetch(`/api/backtest/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export function getSharpeHistory(): Promise<SharpeHistoryResponse> {
  return apiFetch("/api/backtest/sharpe-history");
}

// ── API Cache & Latency ─────────────────────────────────────────

export function getPerfSummary(windowSeconds = 300): Promise<PerfSummary> {
  return apiFetch(`/api/perf/summary?window_seconds=${windowSeconds}`);
}

export function getCacheStats(): Promise<CacheStats> {
  return apiFetch("/api/perf/cache");
}

export function clearCache(): Promise<{ cleared: number }> {
  return apiFetch("/api/perf/cache/clear", { method: "POST" });
}

export function startPerfOptimizer(): Promise<{ status: string }> {
  return apiFetch("/api/perf/optimize", { method: "POST" });
}

export function stopPerfOptimizer(): Promise<{ status: string }> {
  return apiFetch("/api/perf/optimize/stop", { method: "POST" });
}

export function getPerfOptimizerStatus(): Promise<PerfOptimizerStatus> {
  return apiFetch("/api/perf/optimize/status");
}

export function getPerfOptimizerExperiments(): Promise<{ experiments: PerfExperiment[] }> {
  return apiFetch("/api/perf/optimize/experiments");
}
