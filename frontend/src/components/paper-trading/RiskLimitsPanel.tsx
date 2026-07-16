"use client";

// phase-70.1: surfaces the runtime risk_overrides shadow on the Manage tab.
// An override set via /api/paper-trading/risk-limits is read at decide-time by
// portfolio_manager and SHADOWS the .env/settings value -- previously invisible
// from the app, so an operator who raised a cap in Settings could not see or
// clear a shadowing override (finding #12). This panel shows configured-vs-
// effective per adjustable cap, an active-override warning, and Set/Clear.
// It is ALSO the editor for paper_max_per_sector_nav_pct (an ALLOWED_KEYS cap
// that is NOT in the .env settings form). No emoji; Phosphor icons only.

import { useEffect, useState } from "react";
import {
  clearRiskLimit,
  getRiskLimits,
  setRiskLimit,
  type RiskLimitEntry,
  type RiskLimitsResponse,
} from "@/lib/api";
import { IconWarning } from "@/lib/icons";

const KEY_LABELS: Record<string, string> = {
  paper_max_per_sector: "Max positions per sector",
  paper_max_per_sector_nav_pct: "Max NAV % per sector",
  paper_min_cash_reserve_pct: "Min cash reserve (%)",
  paper_max_positions: "Max simultaneous positions",
};

function fmt(v: number | null): string {
  if (v == null) return "--";
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

export function RiskLimitsPanel({ onChanged }: { onChanged?: () => void | Promise<void> }) {
  const [data, setData] = useState<RiskLimitsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);

  const load = async () => {
    try {
      const r = await getRiskLimits();
      setData(r);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load risk limits");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const refresh = async () => {
    await load();
    await onChanged?.();
  };

  const doSet = async (key: string, entry: RiskLimitEntry) => {
    const raw = drafts[key];
    const n = raw != null && raw.trim() !== "" ? Number(raw) : NaN;
    if (!Number.isFinite(n) || n < entry.min || n > entry.max) {
      setError(`${KEY_LABELS[key] ?? key}: must be between ${entry.min} and ${entry.max}.`);
      return;
    }
    setBusy(key);
    try {
      await setRiskLimit(key, n);
      setDrafts((d) => {
        const m = { ...d };
        delete m[key];
        return m;
      });
      setError(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Set override failed");
    } finally {
      setBusy(null);
    }
  };

  const doClear = async (key: string) => {
    setBusy(key);
    try {
      await clearRiskLimit(key);
      setError(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Clear override failed");
    } finally {
      setBusy(null);
    }
  };

  const entries = data ? Object.entries(data.risk_limits) : [];
  const activeKeys = entries.filter(([, e]) => e.overridden).map(([k]) => k);

  return (
    <section className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
      <div className="mb-1">
        <h3 className="text-lg font-semibold text-slate-100">Risk limits (live overrides)</h3>
        <p className="text-sm text-slate-500">
          Runtime deployment/concentration caps read at decide-time. An override shadows the saved
          settings value until cleared. Kill-switch loss limits are intentionally not adjustable here.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-3 py-8 text-slate-400">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
          Loading risk limits...
        </div>
      )}

      {error && (
        <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
          <p className="text-sm text-rose-300">{error}</p>
        </div>
      )}

      {!loading && !error && entries.length === 0 && (
        <div className="py-8 text-center text-sm text-slate-500">
          No adjustable risk limits are exposed.
        </div>
      )}

      {!loading && activeKeys.length > 0 && (
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-950/30 p-3">
          <IconWarning size={18} weight="fill" className="mt-0.5 shrink-0 text-amber-400" />
          <p className="text-sm text-amber-200">
            {activeKeys.length} active override{activeKeys.length > 1 ? "s" : ""} shadowing your saved
            settings: {activeKeys.map((k) => KEY_LABELS[k] ?? k).join(", ")}. The engine enforces the
            effective value, not the settings value, until cleared.
          </p>
        </div>
      )}

      {!loading && !error && entries.length > 0 && (
        <div className="mt-4 overflow-hidden rounded-xl border border-navy-700">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-navy-700 bg-navy-800/80">
              <tr>
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-slate-200">Limit</th>
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-slate-200">Configured</th>
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-slate-200">Effective</th>
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-slate-200">Set override</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/50">
              {entries.map(([key, e]) => (
                <tr key={key} className="align-top transition-colors hover:bg-navy-700/40">
                  <td className="px-4 py-3">
                    <div className="font-medium text-slate-100">{KEY_LABELS[key] ?? key}</div>
                    <div className="text-xs text-slate-500">{e.description}</div>
                    {e.overridden && (
                      <span className="mt-1 inline-block rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                        Override active
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-300">{fmt(e.settings_default)}</td>
                  <td className={`px-4 py-3 font-mono ${e.overridden ? "text-amber-300" : "text-slate-200"}`}>
                    {fmt(e.effective_value)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min={e.min}
                        max={e.max}
                        step={e.type === "int" ? 1 : 0.5}
                        value={drafts[key] ?? ""}
                        placeholder={`${e.min}-${e.max}`}
                        onChange={(ev) =>
                          setDrafts((d) => ({ ...d, [key]: ev.target.value }))
                        }
                        className="w-24 rounded-md border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm text-slate-100 focus:border-sky-500/50 focus:outline-none"
                      />
                      <button
                        type="button"
                        disabled={busy === key || (drafts[key] ?? "").trim() === ""}
                        onClick={() => void doSet(key, e)}
                        className="rounded-md border border-sky-500/40 bg-sky-500/10 px-3 py-1.5 text-sm text-sky-300 transition-colors hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {busy === key ? "..." : "Set"}
                      </button>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      type="button"
                      disabled={busy === key || !e.overridden}
                      onClick={() => void doClear(key)}
                      className="rounded-md border border-navy-600 px-3 py-1.5 text-sm text-slate-300 transition-colors hover:bg-navy-700/60 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Clear
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && !error && entries.length > 0 && activeKeys.length === 0 && (
        <p className="mt-3 text-xs text-slate-500">
          No active overrides -- the engine uses your saved settings values.
        </p>
      )}
    </section>
  );
}
