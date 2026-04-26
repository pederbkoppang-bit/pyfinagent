/**
 * phase-16.44: pure KPI math derived from the redLineSeries NAV history
 * the home page already fetches. No backend changes; no new endpoints.
 *
 * All helpers return `null` on insufficient data (length < 2), zero
 * variance, or NaN — the calling tile then renders "—" honestly
 * instead of fabricating a value.
 *
 * Sharpe formula: Lo (2002) "The Statistics of Sharpe Ratios" --
 *   sqrt(periodsPerYear) * mean / stddev. Default periodsPerYear=252
 *   for daily series (matches backend/services/perf_metrics.py).
 *
 * Sortino formula: Sortino & Price (1994) -- same shape but stddev is
 *   computed only over downside (negative) returns.
 *
 * Max Drawdown: max((running_max - current) / running_max). Returns
 *   negative percent matching the screenshot's "-3.12%" format.
 */
import type { PaperPosition } from "@/lib/types";

export type NavPoint = { date: string; nav: number };

export type DailyDelta = { dollars: number; pct: number };

export function dailyDelta(series: NavPoint[]): DailyDelta | null {
  if (!series || series.length < 2) return null;
  const last = series[series.length - 1].nav;
  const prev = series[series.length - 2].nav;
  if (!Number.isFinite(last) || !Number.isFinite(prev) || prev === 0) return null;
  const dollars = last - prev;
  const pct = (dollars / prev) * 100;
  return { dollars, pct };
}

function dailyReturns(series: NavPoint[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1].nav;
    const cur = series[i].nav;
    if (Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0) {
      out.push((cur - prev) / prev);
    }
  }
  return out;
}

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function stdDev(xs: number[], mu: number): number {
  if (xs.length < 2) return 0;
  const variance = xs.reduce((s, x) => s + (x - mu) * (x - mu), 0) / (xs.length - 1);
  return Math.sqrt(variance);
}

export function sharpe(series: NavPoint[], periodsPerYear = 252): number | null {
  const rets = dailyReturns(series);
  if (rets.length < 2) return null;
  const mu = mean(rets);
  const sd = stdDev(rets, mu);
  if (sd === 0 || !Number.isFinite(sd)) return null;
  const annualized = (mu / sd) * Math.sqrt(periodsPerYear);
  return Number.isFinite(annualized) ? annualized : null;
}

export function sortino(series: NavPoint[], periodsPerYear = 252): number | null {
  const rets = dailyReturns(series);
  if (rets.length < 2) return null;
  const mu = mean(rets);
  const downside = rets.filter((r) => r < 0);
  if (downside.length === 0) return null; // no downside variance to penalize
  // Downside deviation uses 0 as the target (Sortino & Price 1994)
  const dsd = Math.sqrt(downside.reduce((s, r) => s + r * r, 0) / downside.length);
  if (dsd === 0 || !Number.isFinite(dsd)) return null;
  const annualized = (mu / dsd) * Math.sqrt(periodsPerYear);
  return Number.isFinite(annualized) ? annualized : null;
}

/** Returns max drawdown as a NEGATIVE percent (e.g. -3.12 for a 3.12% drop). */
export function maxDrawdownPct(series: NavPoint[]): number | null {
  if (!series || series.length < 2) return null;
  let peak = -Infinity;
  let worst = 0;
  for (const p of series) {
    if (!Number.isFinite(p.nav)) continue;
    if (p.nav > peak) peak = p.nav;
    if (peak > 0) {
      const dd = (p.nav - peak) / peak;
      if (dd < worst) worst = dd;
    }
  }
  if (worst === 0 || !Number.isFinite(worst)) return null;
  return worst * 100;
}

export type PositionsBreakdown = { long: number; short: number; total: number };

export function categorizePositions(positions: PaperPosition[]): PositionsBreakdown {
  let long = 0;
  let short = 0;
  for (const p of positions || []) {
    if (p.quantity > 0) long += 1;
    else if (p.quantity < 0) short += 1;
  }
  return { long, short, total: long + short };
}
