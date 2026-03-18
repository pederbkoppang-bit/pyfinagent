"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import {
  getFullSettings,
  getAvailableModels,
  updateSettings,
  getLatestCostSummary,
} from "@/lib/api";
import type { FullSettings, ModelPricing, LatestCostSummary, AgentCostEntry } from "@/lib/types";

// Which agents run in each mode — used for cost estimation
const LITE_SKIP_AGENTS = new Set([
  "Devil's Advocate",
  "Deep Dive",
  "Aggressive",
  "Conservative",
  "Neutral",
  "Risk Judge",
]);

function isDebateAgent(name: string): boolean {
  return /^(Bull|Bear) Agent/i.test(name);
}

function isRiskAgent(name: string): boolean {
  return /^(Aggressive|Conservative|Neutral|Risk Judge)/i.test(name);
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<FullSettings | null>(null);
  const [models, setModels] = useState<ModelPricing[]>([]);
  const [costData, setCostData] = useState<LatestCostSummary | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // Local form state (mirrors settings for unsaved changes)
  const [form, setForm] = useState<Partial<FullSettings>>({});

  useEffect(() => {
    getFullSettings()
      .then((s) => {
        setSettings(s);
        setForm(s);
      })
      .catch(() => {});
    getAvailableModels().then(setModels).catch(() => {});
    getLatestCostSummary().then(setCostData).catch(() => {});
  }, []);

  const updateForm = useCallback(
    <K extends keyof FullSettings>(key: K, value: FullSettings[K]) => {
      setForm((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const hasChanges = useMemo(() => {
    if (!settings) return false;
    return (Object.keys(form) as (keyof FullSettings)[]).some(
      (k) => form[k] !== settings[k]
    );
  }, [form, settings]);

  const handleSave = async () => {
    if (!settings) return;
    setSaving(true);
    setSaveMsg(null);
    // Build diff — only send changed fields
    const diff: Partial<FullSettings> = {};
    for (const k of Object.keys(form) as (keyof FullSettings)[]) {
      if (form[k] !== settings[k]) {
        (diff as Record<string, unknown>)[k] = form[k];
      }
    }
    try {
      const updated = await updateSettings(diff);
      setSettings(updated);
      setForm(updated);
      setSaveMsg("Settings saved");
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (e: unknown) {
      setSaveMsg(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  // ── Live cost estimator ──────────────────────────────────────
  const estimatedCost = useMemo(() => {
    if (!costData?.agents?.length || !models.length) return null;

    const liteMode = form.lite_mode ?? false;
    const debateRounds = form.max_debate_rounds ?? 2;
    const riskRounds = form.max_risk_debate_rounds ?? 1;
    const synthIter = form.max_synthesis_iterations ?? 2;
    const stdModel = form.gemini_model ?? "gemini-2.0-flash";
    const dtModel = form.deep_think_model ?? stdModel;

    const stdPricing = models.find((m) => m.model === stdModel) ?? models[0];
    const dtPricing = models.find((m) => m.model === dtModel) ?? stdPricing;

    // Use the real per-agent token counts from last analysis as baseline
    const lastDebateRounds = costData.agents.filter((a) =>
      isDebateAgent(a.agent_name)
    ).length / 2; // Bull+Bear per round
    const baseDebateRounds = Math.max(lastDebateRounds, 1);
    const lastRiskRounds = costData.agents.filter(
      (a) => isRiskAgent(a.agent_name) && a.agent_name !== "Risk Judge"
    ).length / 3;
    const baseRiskRounds = Math.max(lastRiskRounds, 1);

    let totalCost = 0;
    let totalTokens = 0;
    let calls = 0;

    for (const agent of costData.agents) {
      // Skip agents that lite mode removes
      if (liteMode && LITE_SKIP_AGENTS.has(agent.agent_name)) continue;

      // Scale debate agents by round ratio
      let scale = 1;
      if (isDebateAgent(agent.agent_name)) {
        const effectiveRounds = liteMode ? 1 : debateRounds;
        scale = effectiveRounds / baseDebateRounds;
      } else if (isRiskAgent(agent.agent_name) && agent.agent_name !== "Risk Judge") {
        scale = riskRounds / baseRiskRounds;
      } else if (agent.agent_name.startsWith("Synthesis") || agent.agent_name === "Critic") {
        // Scale synthesis iterations
        const baseSynthIter = costData.agents.filter((a) =>
          a.agent_name.startsWith("Synthesis") || a.agent_name === "Critic"
        ).length / 2; // Each iteration = 1 synthesis + 1 critic
        if (baseSynthIter > 0) {
          const effectiveIter = liteMode ? 1 : synthIter;
          scale = effectiveIter / baseSynthIter;
        }
      }

      const pricing = agent.is_deep_think ? dtPricing : stdPricing;
      const inTokens = Math.round(agent.input_tokens * scale);
      const outTokens = Math.round(agent.output_tokens * scale);
      const cost =
        (inTokens * pricing.input_per_1m + outTokens * pricing.output_per_1m) /
        1_000_000;

      totalCost += cost;
      totalTokens += inTokens + outTokens;
      calls += Math.round(scale);
    }

    return { totalCost, totalTokens, calls };
  }, [costData, models, form]);

  const weightTotal = useMemo(() => {
    return (
      (form.weight_corporate ?? 0.35) +
      (form.weight_industry ?? 0.2) +
      (form.weight_valuation ?? 0.2) +
      (form.weight_sentiment ?? 0.15) +
      (form.weight_governance ?? 0.1)
    );
  }, [form]);

  if (!settings) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 p-8">
          <p className="text-slate-400">Loading settings...</p>
        </main>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Settings</h2>
            <p className="text-sm text-slate-500">
              Configure models, scoring weights, cost controls, and debate depth
            </p>
          </div>
          <div className="flex items-center gap-3">
            {saveMsg && (
              <span
                className={`text-sm ${
                  saveMsg.includes("saved") ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {saveMsg}
              </span>
            )}
            <button
              onClick={handleSave}
              disabled={!hasChanges || saving}
              className="rounded-lg bg-sky-600 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {saving ? "Saving..." : "Save All Settings"}
            </button>
          </div>
        </div>

        <div className="grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2">
          {/* ── Analysis Mode ───────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              ⚡ Analysis Mode
            </h3>
            <div className="flex items-center gap-4">
              <button
                onClick={() => updateForm("lite_mode", false)}
                className={`flex-1 rounded-lg border px-4 py-3 text-center text-sm transition-colors ${
                  !form.lite_mode
                    ? "border-sky-500 bg-sky-500/10 text-sky-300"
                    : "border-slate-700 text-slate-400 hover:border-slate-600"
                }`}
              >
                <div className="font-semibold">Full Analysis</div>
                <div className="mt-1 text-xs opacity-70">
                  All agents • {settings.max_debate_rounds} debate rounds • reflection
                </div>
              </button>
              <button
                onClick={() => updateForm("lite_mode", true)}
                className={`flex-1 rounded-lg border px-4 py-3 text-center text-sm transition-colors ${
                  form.lite_mode
                    ? "border-emerald-500 bg-emerald-500/10 text-emerald-300"
                    : "border-slate-700 text-slate-400 hover:border-slate-600"
                }`}
              >
                <div className="font-semibold">Lite Mode</div>
                <div className="mt-1 text-xs opacity-70">
                  ~50% fewer calls • 1 debate round • no reflection
                </div>
              </button>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Lite mode skips: Deep Dive, Devil&apos;s Advocate, Risk Assessment,
              and limits debate to 1 round with no synthesis reflection loop.
            </p>
          </BentoCard>

          {/* ── Live Cost Estimator ─────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              💰 Cost Estimator
            </h3>
            {estimatedCost ? (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-sky-300">
                      ${estimatedCost.totalCost.toFixed(4)}
                    </div>
                    <div className="text-xs text-slate-500">est. per analysis</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-slate-200">
                      {(estimatedCost.totalTokens / 1000).toFixed(0)}k
                    </div>
                    <div className="text-xs text-slate-500">tokens</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-slate-200">
                      {estimatedCost.calls}
                    </div>
                    <div className="text-xs text-slate-500">LLM calls</div>
                  </div>
                </div>
                <p className="text-xs text-slate-600">
                  Based on real token usage from last analysis
                  {costData?.ticker ? ` (${costData.ticker})` : ""}.
                  Updates live as you change settings above.
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                Run at least one analysis to enable real token-based cost
                estimation.
              </p>
            )}
          </BentoCard>

          {/* ── Model Configuration ─────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              🧠 Model Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="mb-1 block text-sm text-slate-300">
                  Standard Model (all agents)
                </label>
                <select
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 focus:border-sky-500 focus:outline-none"
                  value={form.gemini_model ?? ""}
                  onChange={(e) => updateForm("gemini_model", e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m.model} value={m.model}>
                      {m.model} — ${m.input_per_1m}/${m.output_per_1m} per 1M
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm text-slate-300">
                  Deep Think Model (Moderator, Synthesis, Critic, Risk Judge)
                </label>
                <select
                  className="w-full rounded-lg border border-violet-700/50 bg-slate-800 px-3 py-2 text-sm text-slate-200 focus:border-violet-500 focus:outline-none"
                  value={form.deep_think_model ?? ""}
                  onChange={(e) => updateForm("deep_think_model", e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m.model} value={m.model}>
                      {m.model} — ${m.input_per_1m}/${m.output_per_1m} per 1M
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </BentoCard>

          {/* ── Cost Controls ───────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              🛡️ Cost Controls
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Budget per Analysis (USD)
                  </label>
                  <span className="font-mono text-sm text-sky-300">
                    ${form.max_analysis_cost_usd?.toFixed(2) ?? "0.50"}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={5.0}
                  step={0.05}
                  value={form.max_analysis_cost_usd ?? 0.5}
                  onChange={(e) =>
                    updateForm("max_analysis_cost_usd", parseFloat(e.target.value))
                  }
                  className="mt-1 w-full accent-sky-500"
                />
                <p className="mt-1 text-xs text-slate-600">
                  Soft limit — logs a warning when exceeded, does not abort.
                </p>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Synthesis Iterations
                  </label>
                  <span className="font-mono text-sm text-sky-300">
                    {form.max_synthesis_iterations ?? 2}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={3}
                  step={1}
                  value={form.max_synthesis_iterations ?? 2}
                  onChange={(e) =>
                    updateForm(
                      "max_synthesis_iterations",
                      parseInt(e.target.value, 10)
                    )
                  }
                  className="mt-1 w-full accent-sky-500"
                />
                <p className="mt-1 text-xs text-slate-600">
                  1 = no reflection loop. 2 = one Synthesis↔Critic revision pass.
                </p>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Min Data Quality
                  </label>
                  <span className="font-mono text-sm text-sky-300">
                    {((form.data_quality_min ?? 0.5) * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={form.data_quality_min ?? 0.5}
                  onChange={(e) =>
                    updateForm("data_quality_min", parseFloat(e.target.value))
                  }
                  className="mt-1 w-full accent-sky-500"
                />
                <p className="mt-1 text-xs text-slate-600">
                  Below this threshold, debate and risk assessment are skipped.
                </p>
              </div>
            </div>
          </BentoCard>

          {/* ── Debate Depth ────────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              🗣️ Debate Depth
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Bull↔Bear Rounds
                  </label>
                  <span className="font-mono text-sm text-violet-300">
                    {form.max_debate_rounds ?? 2}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={form.max_debate_rounds ?? 2}
                  onChange={(e) =>
                    updateForm("max_debate_rounds", parseInt(e.target.value, 10))
                  }
                  className="mt-1 w-full accent-violet-500"
                />
                <p className="mt-1 text-xs text-slate-600">
                  Iterative Bull vs Bear rebuttal exchanges
                </p>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Risk Debate Rounds
                  </label>
                  <span className="font-mono text-sm text-violet-300">
                    {form.max_risk_debate_rounds ?? 1}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={3}
                  step={1}
                  value={form.max_risk_debate_rounds ?? 1}
                  onChange={(e) =>
                    updateForm(
                      "max_risk_debate_rounds",
                      parseInt(e.target.value, 10)
                    )
                  }
                  className="mt-1 w-full accent-violet-500"
                />
                <p className="mt-1 text-xs text-slate-600">
                  Aggressive / Conservative / Neutral exchanges
                </p>
              </div>
            </div>
            {form.lite_mode && (
              <p className="mt-3 text-xs text-amber-400/80">
                ⚡ Lite mode overrides: 1 debate round, no Devil&apos;s Advocate,
                no risk assessment.
              </p>
            )}
          </BentoCard>

          {/* ── Pillar Weights ──────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              📊 Pillar Weights
            </h3>
            <div className="space-y-3">
              {([
                { label: "Corporate", key: "weight_corporate" as const },
                { label: "Industry", key: "weight_industry" as const },
                { label: "Valuation", key: "weight_valuation" as const },
                { label: "Sentiment", key: "weight_sentiment" as const },
                { label: "Governance", key: "weight_governance" as const },
              ] as const).map((w) => (
                <div key={w.key} className="flex items-center gap-3">
                  <label className="w-24 text-sm text-slate-300">
                    {w.label}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.05}
                    value={form[w.key] ?? 0}
                    onChange={(e) =>
                      updateForm(w.key, parseFloat(e.target.value))
                    }
                    className="flex-1 accent-sky-500"
                  />
                  <span className="w-12 text-right font-mono text-sm text-sky-300">
                    {((form[w.key] ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
            <p
              className={`mt-3 text-xs ${
                Math.abs(weightTotal - 1.0) > 0.01
                  ? "text-rose-400"
                  : "text-slate-600"
              }`}
            >
              Total: {(weightTotal * 100).toFixed(0)}%
              {Math.abs(weightTotal - 1.0) > 0.01 && " — must equal 100%"}
            </p>
          </BentoCard>
        </div>
      </main>
    </div>
  );
}
