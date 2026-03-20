/**
 * API client for the PyFinAgent FastAPI backend.
 */

import type {
  AllSignals,
  AnalysisResponse,
  AnalysisStatusResponse,
  CostHistoryEntry,
  FullSettings,
  LatestCostSummary,
  ModelConfig,
  ModelPricing,
  PaperPerformance,
  PaperPortfolio,
  PaperPosition,
  PaperSnapshot,
  PaperTrade,
  PaperTradingStatus,
  PerformanceStats,
  PortfolioPerformance,
  PortfolioPosition,
  ReportSummary,
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

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
    });
  } catch (err) {
    // Network-level failure (CORS, DNS, refused, etc.)
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
      throw new Error(
        `Cannot reach backend at ${API_BASE}. ` +
        "Make sure the FastAPI server is running (uvicorn backend.main:app --port 8000)."
      );
    }
    throw new Error(`Network error calling ${path}: ${msg}`);
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

export function healthCheck(): Promise<{ status: string; service: string }> {
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
