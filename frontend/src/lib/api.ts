/**
 * API client for the PyFinAgent FastAPI backend.
 */

import type {
  AnalysisResponse,
  AnalysisStatusResponse,
  PerformanceStats,
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
