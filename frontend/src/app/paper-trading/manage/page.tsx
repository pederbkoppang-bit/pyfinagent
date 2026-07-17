"use client";

// phase-44.2 -- Manage sub-route.
//
// MANAGE_REMOVAL_DEFERRED -- this sub-route exists to migrate the monolith
// verbatim. Per research brief topic 5 + risk flag P-4, removing the Manage
// tab requires operator_approval_44.2.md. Once approval lands, deletion is
// (a) remove the entry from the layout's TABS array, (b) delete this
// directory, (c) add a <Drawer/> trigger to the page header. Operator
// runbook documented in handoff/current/live_check_44.2.md.

import { useEffect, useState } from "react";
import {
  PaperSettingNum,
  PaperMarketsField,
  ReadOnlyField,
} from "@/components/paper-trading/cockpit-helpers";
import { RiskLimitsPanel } from "@/components/paper-trading/RiskLimitsPanel";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import {
  depositPaperFunds,
  getFullSettings,
  getPaperPortfolio,
  getPaperTradingStatus,
  updateSettings,
} from "@/lib/api";
import type { FullSettings } from "@/lib/types";

export default function ManagePage() {
  const { refresh, portfolio } = usePaperTradingData();

  const [manageSettings, setManageSettings] = useState<FullSettings | null>(null);
  const [manageDirty, setManageDirty] = useState<Partial<FullSettings>>({});
  const [depositAmount, setDepositAmount] = useState("");
  const [depositLoading, setDepositLoading] = useState(false);
  const [depositError, setDepositError] = useState<string | null>(null);
  const [depositSuccess, setDepositSuccess] = useState<string | null>(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsSuccess, setSettingsSuccess] = useState<string | null>(null);
  // phase-70.1: per-field client-side range errors lifted from PaperSettingNum
  // so Save is disabled (and a summary shown) while any field is out of range,
  // instead of the old silent save-time 422.
  const [fieldErrors, setFieldErrors] = useState<Record<string, string | undefined>>({});
  const handleFieldValidity = (field: string, err: string | undefined) =>
    setFieldErrors((fe) => ({ ...fe, [field]: err }));
  const hasFieldError = Object.values(fieldErrors).some(Boolean);

  useEffect(() => {
    if (manageSettings) return;
    let cancelled = false;
    getFullSettings()
      .then((s) => {
        if (!cancelled) setManageSettings(s);
      })
      .catch((e) => {
        if (!cancelled) {
          setSettingsError(e instanceof Error ? e.message : "Failed to load settings");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [manageSettings]);

  const handleDeposit = async () => {
    setDepositError(null);
    setDepositSuccess(null);
    const amt = parseFloat(depositAmount);
    if (!Number.isFinite(amt) || amt <= 0 || amt > 1_000_000) {
      setDepositError("Enter an amount between $1 and $1,000,000.");
      return;
    }
    setDepositLoading(true);
    try {
      const r = await depositPaperFunds(amt);
      setDepositSuccess(
        `Deposited $${r.amount.toLocaleString()} — new NAV $${r.new_nav.toLocaleString()} ` +
          `(starting capital now $${r.new_starting_capital.toLocaleString()})`,
      );
      setDepositAmount("");
      // Touch layer-level state by replaying the shared refresh + per-tab
      // status/portfolio reads (covers the case where the layout context
      // hasn't been re-mounted across nav).
      await Promise.all([getPaperTradingStatus(), getPaperPortfolio()]);
      await refresh();
    } catch (e) {
      setDepositError(e instanceof Error ? e.message : "Deposit failed");
    } finally {
      setDepositLoading(false);
    }
  };

  const handleSettingsSave = async () => {
    if (Object.keys(manageDirty).length === 0) {
      setSettingsError("No changes to save.");
      return;
    }
    setSettingsError(null);
    setSettingsSuccess(null);
    setSettingsSaving(true);
    try {
      const updated = await updateSettings(manageDirty);
      setManageSettings(updated);
      setManageDirty({});
      setSettingsSuccess("Settings saved.");
    } catch (e) {
      setSettingsError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSettingsSaving(false);
    }
  };

  return (
    <div
      role="tabpanel"
      id="panel-manage"
      aria-labelledby="tab-manage"
      tabIndex={0}
      className="space-y-6"
    >
      {/* Top up Fund */}
      <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
        <h3 className="mb-1 text-lg font-semibold text-slate-100">Top up fund</h3>
        <p className="mb-4 text-sm text-slate-500">
          Deposit additional virtual capital. The amount is added to BOTH the cash
          balance AND the starting capital so total P&amp;L % stays meaningful (a deposit
          is P&amp;L-neutral by definition).
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[12rem]">
            <label htmlFor="deposit-amount" className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
              Amount (USD)
            </label>
            <div className="flex items-center rounded-md border border-navy-600 bg-navy-900 px-3 py-2 focus-within:border-sky-500/50">
              <span className="mr-2 text-slate-500">$</span>
              <input
                id="deposit-amount"
                type="number"
                min={1}
                max={1_000_000}
                step={100}
                value={depositAmount}
                onChange={(e) => setDepositAmount(e.target.value)}
                placeholder="5000"
                className="w-full bg-transparent text-slate-100 placeholder:text-slate-600 focus:outline-none"
                disabled={depositLoading}
              />
            </div>
          </div>
          <button
            type="button"
            onClick={handleDeposit}
            disabled={depositLoading || !depositAmount}
            className="rounded-md bg-emerald-600/30 px-5 py-2.5 text-sm font-medium text-emerald-200 hover:bg-emerald-600/50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {depositLoading ? "Depositing..." : "Deposit"}
          </button>
        </div>
        {depositError && (
          <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-sm text-rose-300">
            {depositError}
          </div>
        )}
        {depositSuccess && (
          <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-950/30 p-3 text-sm text-emerald-300">
            {depositSuccess}
          </div>
        )}
      </div>

      {/* Trading Settings */}
      <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-100">Trading settings</h3>
            <p className="text-sm text-slate-500">
              Paper-trading-specific knobs. Changes persist via the .env file.
            </p>
          </div>
          <button
            type="button"
            onClick={handleSettingsSave}
            disabled={settingsSaving || hasFieldError || Object.keys(manageDirty).length === 0}
            className="rounded-md bg-sky-600/30 px-5 py-2.5 text-sm font-medium text-sky-200 hover:bg-sky-600/50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {settingsSaving ? "Saving..." : "Save"}
          </button>
        </div>

        {!manageSettings && !settingsError && (
          <p className="text-sm text-slate-500">Loading settings…</p>
        )}
        {settingsError && (
          <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-sm text-rose-300">
            {settingsError}
          </div>
        )}
        {settingsSuccess && (
          <div className="mb-3 rounded-lg border border-emerald-500/30 bg-emerald-950/30 p-3 text-sm text-emerald-300">
            {settingsSuccess}
          </div>
        )}
        {hasFieldError && (
          <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-sm text-rose-300">
            Fix the highlighted fields before saving.
          </div>
        )}

        {manageSettings && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ReadOnlyField
              label="Starting capital"
              // phase-70.5: show the LIVE BQ paper_portfolio.starting_capital (which a
              // Top-up updates), not the immutable .env config value -- the two diverge
              // after a deposit. refresh() updates portfolio live after a deposit.
              value={`$${(portfolio?.starting_capital ?? manageSettings.paper_starting_capital ?? 10000).toLocaleString()}`}
              hint="Reflects deposits (Top up fund). Total P&L % is measured against this."
            />
            <div className="md:col-span-2 flex items-start gap-3 rounded-lg border border-navy-700 bg-navy-800/40 p-3">
              <input
                type="checkbox"
                id="paper-lite-mode"
                checked={
                  manageDirty.lite_mode ?? manageSettings.lite_mode ?? false
                }
                onChange={(e) => {
                  const next = e.target.checked;
                  setManageDirty((d) => {
                    const merged = { ...d };
                    if (next === manageSettings.lite_mode) {
                      delete merged.lite_mode;
                    } else {
                      merged.lite_mode = next;
                    }
                    return merged;
                  });
                }}
                className="mt-1 h-4 w-4 cursor-pointer rounded border-navy-600 bg-navy-900 text-sky-500 focus:ring-2 focus:ring-sky-500/50"
              />
              <label htmlFor="paper-lite-mode" className="cursor-pointer text-sm">
                <span className="font-medium text-slate-200">Lite mode (cheap fast analysis)</span>
                <p className="mt-1 text-xs text-slate-500">
                  When ON, paper trading uses a single 4-field Claude call (~$0.01/ticker)
                  instead of the full 15-step orchestrator with debate / risk-judge / bias-audit
                  ($0.50-2.00/ticker but much richer rationale and reports). The
                  `paper_max_daily_cost_usd` cap below is the safety circuit-breaker either way.
                </p>
              </label>
            </div>
            {/* phase-50.6: live-loop markets multi-select (writes paper_markets). */}
            <PaperMarketsField settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} />
            <PaperSettingNum label="Max simultaneous positions" field="paper_max_positions" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={1} max={50} step={1} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Max positions per sector" field="paper_max_per_sector" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0} max={20} step={1} hint="Default 2 = at least 5 distinct sectors for a 10-position portfolio. 0 disables (legacy)." onValidity={handleFieldValidity} />
            <PaperSettingNum label="Daily LLM cost cap (USD)" field="paper_max_daily_cost_usd" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0.1} max={50} step={0.1} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Default stop-loss (%)" field="paper_default_stop_loss_pct" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={1} max={50} step={0.5} hint="O'Neil canonical: 7-8%." onValidity={handleFieldValidity} />
            <PaperSettingNum label="Screen top-N candidates" field="paper_screen_top_n" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={1} max={100} step={1} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Analyze top-K with LLM" field="paper_analyze_top_n" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={1} max={50} step={1} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Transaction cost (%)" field="paper_transaction_cost_pct" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0} max={5} step={0.05} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Daily loss limit (%)" field="paper_daily_loss_limit_pct" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0.5} max={25} step={0.5} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Trailing drawdown limit (%)" field="paper_trailing_dd_limit_pct" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={1} max={50} step={0.5} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Min cash reserve (%)" field="paper_min_cash_reserve_pct" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0} max={50} step={0.5} onValidity={handleFieldValidity} />
            <PaperSettingNum label="Daily run hour (ET, 0-23)" field="paper_trading_hour" settings={manageSettings} dirty={manageDirty} setDirty={setManageDirty} min={0} max={23} step={1} hint="Reschedules the daily cron on save (no restart needed)." onValidity={handleFieldValidity} />
          </div>
        )}
      </div>

      {/* phase-70.1: risk-override transparency + editor (incl. the NAV%/sector
          cap, which is a risk_overrides ALLOWED_KEY, not a .env settings field). */}
      <RiskLimitsPanel onChanged={refresh} />
    </div>
  );
}
