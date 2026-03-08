/**
 * API client for the PyFinAgent FastAPI backend.
 */

import type {
  AllSignals,
  AnalysisResponse,
  AnalysisStatusResponse,
  PerformanceStats,
  PortfolioPerformance,
  PortfolioPosition,
  ReportSummary,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
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

export function getReport(ticker: string): Promise<Record<string, unknown>> {
  return apiFetch(`/api/reports/${ticker}`);
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
