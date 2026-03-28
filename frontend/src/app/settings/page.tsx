"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { PageSkeleton } from "@/components/Skeleton";
import {
  getFullSettings,
  getAvailableModels,
  updateSettings,
  getLatestCostSummary,
  getPerfSummary,
  getCacheStats,
  clearCache,
  startPerfOptimizer,
  stopPerfOptimizer,
  getPerfOptimizerStatus,
  getPerfOptimizerExperiments,
} from "@/lib/api";
import type {
  FullSettings,
  ModelPricing,
  LatestCostSummary,
  PerfSummary,
  CacheStats,
  PerfOptimizerStatus,
  PerfExperiment,
} from "@/lib/types";
import { PerfProgressChart } from "@/components/PerfProgressChart";
import {
  SettingsMode,
  SettingsDebate,
  SettingsModel,
  SettingsCostControls,
  SettingsEstimator,
  SettingsPillars,
  SettingsCache,
  SettingsOptimizer,
  SettingsLatency,
  SettingsRefresh,
  IconWarning,
  IconCheck,
} from "@/lib/icons";

// ── Sub-navigation tabs ───────────────────────────────────────────
type SettingsTab = "models" | "cost" | "performance";
const SETTINGS_TABS: { id: SettingsTab; label: string }[] = [
  { id: "models", label: "Models & Analysis" },
  { id: "cost", label: "Cost & Weights" },
  { id: "performance", label: "Performance" },
];

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

// ── VS Code-style Model Picker ──────────────────────────────────────────

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "gemini-2.0-flash": "Gemini 2.0 Flash",
  "gemini-2.5-flash": "Gemini 2.5 Flash",
  "gemini-2.5-pro": "Gemini 2.5 Pro",
  // OpenAI
  "gpt-4o": "GPT-4o",
  "gpt-4o-mini": "GPT-4o mini",
  "gpt-4.1": "GPT-4.1",
  "gpt-4.1-mini": "GPT-4.1 mini",
  "gpt-4.1-nano": "GPT-4.1 nano",
  "gpt-5": "GPT-5",
  "gpt-5-chat": "GPT-5 chat",
  "gpt-5-mini": "GPT-5 mini",
  "gpt-5-nano": "GPT-5 nano",
  "o1": "o1",
  "o1-mini": "o1-mini",
  "o1-preview": "o1-preview",
  "o3": "o3",
  "o3-mini": "o3-mini",
  "o4-mini": "o4-mini",
  // Anthropic
  "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
  "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
  "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
  "claude-sonnet-4": "Claude Sonnet 4",
  "claude-opus-4": "Claude Opus 4",
  "claude-sonnet-4-6": "Claude Sonnet 4.6",
  // Meta
  "meta-llama-3.1-405b-instruct": "Llama 3.1 405B",
  "meta-llama-3.1-8b-instruct": "Llama 3.1 8B",
  "llama-3.3-70b-instruct": "Llama 3.3 70B",
  "llama-4-maverick": "Llama 4 Maverick",
  "llama-4-scout": "Llama 4 Scout",
  // DeepSeek
  "deepseek-r1": "DeepSeek R1",
  "deepseek-r1-0528": "DeepSeek R1 (0528)",
  "deepseek-v3-0324": "DeepSeek V3",
  // xAI
  "grok-3": "Grok 3",
  "grok-3-mini": "Grok 3 mini",
  // Microsoft
  "phi-4": "Phi-4",
  "mai-ds-r1": "MAI DS R1",
  "phi-4-mini-instruct": "Phi-4 mini",
  "phi-4-mini-reasoning": "Phi-4 mini Reasoning",
  "phi-4-reasoning": "Phi-4 Reasoning",
  // Mistral
  "ministral-3b": "Ministral 3B",
  "codestral-2501": "Codestral 2501",
  "mistral-medium-2505": "Mistral Medium",
  "mistral-small-2503": "Mistral Small",
};

const PRIMARY_MODEL_NAMES = new Set([
  "gemini-2.0-flash",
  "gemini-2.5-flash",
  "gpt-4.1",
  "gpt-4o",
  "gpt-5",
  "claude-sonnet-4",
  "claude-sonnet-4-6",
  "claude-3-7-sonnet-20250219",
  "deepseek-r1",
  "llama-4-maverick",
  "grok-3",
  "o4-mini",
  "o3-mini",
]);

function CostBadge({
  model,
  githubConfigured,
}: {
  model: ModelPricing;
  githubConfigured: boolean;
}) {
  if (
    model.provider === "GitHub Models" &&
    githubConfigured &&
    model.copilot_multiplier !== undefined
  ) {
    const mult = model.copilot_multiplier;
    const colorClass =
      mult <= 0.33
        ? "bg-emerald-900/50 text-emerald-300"
        : mult >= 3
        ? "bg-amber-900/50 text-amber-300"
        : "bg-slate-700/60 text-slate-300";
    return (
      <span className={`rounded px-1.5 py-0.5 font-mono text-xs ${colorClass}`}>
        {mult}x
      </span>
    );
  }
  return (
    <span className="font-mono text-xs text-slate-500">
      ${model.input_per_1m}/{model.output_per_1m}
    </span>
  );
}

function ModelPicker({
  label,
  value,
  models,
  githubConfigured,
  onChange,
  accentColor = "sky",
}: {
  label: string;
  value: string;
  models: ModelPricing[];
  githubConfigured: boolean;
  onChange: (v: string) => void;
  accentColor?: "sky" | "violet";
}) {
  const [search, setSearch] = useState("");
  const [showOther, setShowOther] = useState(false);

  const selected = models.find((m) => m.model === value);

  const filtered = models.filter((m) => {
    const display = (MODEL_DISPLAY_NAMES[m.model] ?? m.model).toLowerCase();
    const q = search.toLowerCase();
    return display.includes(q) || m.model.toLowerCase().includes(q);
  });

  // When searching, don't pin — show natural filtered order
  // When not searching, pin selected to top and exclude from list body
  const isSearching = search.trim().length > 0;
  const listModels = isSearching ? filtered : filtered.filter((m) => m.model !== value);
  const primary = listModels.filter((m) => PRIMARY_MODEL_NAMES.has(m.model));
  const other = listModels.filter((m) => !PRIMARY_MODEL_NAMES.has(m.model));

  const borderClass =
    accentColor === "violet" ? "border-violet-600" : "border-sky-600";
  const checkClass =
    accentColor === "violet" ? "text-violet-400" : "text-sky-400";

  const ModelRow = ({ m }: { m: ModelPricing }) => (
    <button
      key={m.model}
      onClick={() => onChange(m.model)}
      className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors hover:bg-slate-800 ${
        value === m.model ? "bg-slate-800/80" : ""
      }`}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className={`w-3 shrink-0 ${value === m.model ? checkClass : "text-transparent"}`}>
          <IconCheck size={12} weight="bold" />
        </span>
        <span
          className={`truncate ${
            value === m.model ? "font-medium text-slate-100" : "text-slate-300"
          }`}
        >
          {MODEL_DISPLAY_NAMES[m.model] ?? m.model}
        </span>
        <span className="shrink-0 text-xs text-slate-600">
          {m.provider ?? "Gemini"}
        </span>
        {m.context_limited && (
          <span className="shrink-0 rounded px-1 py-0.5 text-xs bg-amber-900/40 text-amber-400">
            ctx limit
          </span>
        )}
      </div>
      <CostBadge model={m} githubConfigured={githubConfigured} />
    </button>
  );

  return (
    <div>
      <label className="mb-1.5 block text-sm text-slate-300">{label}</label>
      <div
        className={`overflow-hidden rounded-lg border bg-slate-900 ${
          value ? borderClass : "border-slate-700"
        }`}
      >
        {/* Search bar */}
        <div className="border-b border-slate-800 px-3 py-2">
          <input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-transparent text-sm text-slate-300 placeholder-slate-600 focus:outline-none"
          />
        </div>
        {/* Model rows */}
        <div className="max-h-52 overflow-y-auto scrollbar-thin">
          {/* Pinned selected model at top (when not searching) */}
          {!isSearching && selected && (
            <>
              <ModelRow m={selected} />
              <div className="mx-3 border-t border-slate-800" />
            </>
          )}
          {primary.map((m) => (
            <ModelRow key={m.model} m={m} />
          ))}
          {other.length > 0 && (
            <>
              <button
                onClick={() => setShowOther((x) => !x)}
                className="flex w-full items-center justify-between px-4 py-1.5 text-xs text-slate-500 transition-colors hover:text-slate-400"
              >
                <span>Other models ({other.length})</span>
                <span>{showOther ? "▲" : "▼"}</span>
              </button>
              {showOther && other.map((m) => <ModelRow key={m.model} m={m} />)}
            </>
          )}
          {primary.length === 0 && other.length === 0 && (
            <p className="px-3 py-4 text-center text-sm text-slate-600">
              No models match
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<FullSettings | null>(null);
  const [models, setModels] = useState<ModelPricing[]>([]);
  const [costData, setCostData] = useState<LatestCostSummary | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Local form state (mirrors settings for unsaved changes)
  const [form, setForm] = useState<Partial<FullSettings>>({});

  // Sub-navigation
  const [activeTab, setActiveTab] = useState<SettingsTab>("models");

  // Performance tab state
  const [perfSummary, setPerfSummary] = useState<PerfSummary | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [optimizerStatus, setOptimizerStatus] = useState<PerfOptimizerStatus | null>(null);
  const [perfLoading, setPerfLoading] = useState(false);
  const [cacheClearMsg, setCacheClearMsg] = useState<string | null>(null);
  const [perfExperiments, setPerfExperiments] = useState<PerfExperiment[]>([]);

  const loadSettings = useCallback(() => {
    setLoadError(null);
    Promise.all([
      getFullSettings().then((s) => { setSettings(s); setForm(s); }),
      getAvailableModels().then(setModels),
      getLatestCostSummary().then(setCostData),
    ]).catch((e) => {
      setLoadError(e instanceof Error ? e.message : "Failed to load settings — backend may be down.");
    });
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  // Fetch performance data when Performance tab is active
  const refreshPerfData = useCallback(async () => {
    setPerfLoading(true);
    try {
      const [cache, summary, optStatus, expData] = await Promise.all([
        getCacheStats(),
        getPerfSummary(),
        getPerfOptimizerStatus(),
        getPerfOptimizerExperiments(),
      ]);
      setCacheStats(cache);
      setPerfSummary(summary);
      setOptimizerStatus(optStatus);
      setPerfExperiments(expData.experiments);
    } catch {
      // silently ignore — tab will show empty state
    } finally {
      setPerfLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "performance") refreshPerfData();
  }, [activeTab, refreshPerfData]);

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

  // Estimated Copilot premium requests consumed for this analysis
  const premiumRequests = useMemo(() => {
    if (!estimatedCost || !settings?.github_token_configured) return null;
    const stdModel = models.find((m) => m.model === (form.gemini_model ?? ""));
    const dtModel = models.find(
      (m) => m.model === (form.deep_think_model ?? form.gemini_model ?? "")
    );
    if (
      stdModel?.provider !== "GitHub Models" &&
      dtModel?.provider !== "GitHub Models"
    )
      return null;
    const stdMult =
      stdModel?.provider === "GitHub Models"
        ? (stdModel.copilot_multiplier ?? 1)
        : 0;
    const dtMult =
      dtModel?.provider === "GitHub Models"
        ? (dtModel.copilot_multiplier ?? 1)
        : 0;
    const dtCalls = costData?.deep_think_calls ?? 0;
    const stdCalls = estimatedCost.calls - dtCalls;
    return Math.round(stdCalls * stdMult + dtCalls * dtMult);
  }, [estimatedCost, models, form, settings, costData]);

  if (!settings) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto scrollbar-thin p-8">
          {loadError ? (
            <div className="rounded-lg border border-rose-900 bg-rose-950/50 p-4">
              <p className="text-sm font-medium text-rose-200">{loadError}</p>
              {loadError.includes("Cannot reach") && (
                <p className="mt-1 text-xs text-rose-300/60">
                  Make sure the backend is running: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
                </p>
              )}
              <button onClick={loadSettings} className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60">
                Retry
              </button>
            </div>
          ) : (
            <PageSkeleton />
          )}
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Settings</h2>
            <p className="text-sm text-slate-500">
              Configure models, scoring weights, cost controls, and performance
            </p>
          </div>
          {activeTab !== "performance" && (
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
          )}
        </div>

        {/* Sub-navigation tabs */}
        <div className="mb-6 flex gap-1 rounded-lg bg-slate-800/50 p-1 max-w-fit">
          {SETTINGS_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-slate-700 text-slate-100 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* ── Models & Analysis Tab ─────────────────────── */}
        {activeTab === "models" && (
        <div className="grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2">
          {/* ── Analysis Mode ───────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsMode size={20} weight="duotone" className="inline -mt-0.5" /> Analysis Mode
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

          {/* ── Debate Depth ────────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsDebate size={20} weight="duotone" className="inline -mt-0.5" /> Debate Depth
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
                <SettingsMode size={14} className="inline -mt-0.5" /> Lite mode overrides: 1 debate round, no Devil&apos;s Advocate,
                no risk assessment.
              </p>
            )}
          </BentoCard>

          {/* ── Model Configuration ─────────────────────────── */}
          <BentoCard className="lg:col-span-2">
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsModel size={20} weight="duotone" className="inline -mt-0.5" /> Model Configuration
            </h3>
            {/* Provider key status */}
            <div className="mb-4 flex flex-wrap gap-2">
              <span className="rounded-full bg-sky-900/40 px-2 py-0.5 text-xs text-sky-300">
                Gemini: Always available
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${
                  settings?.github_token_configured
                    ? "bg-emerald-900/40 text-emerald-300"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                GitHub Models:{" "}
                {settings?.github_token_configured ? "Configured" : "No token"}
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${
                  settings?.anthropic_key_configured
                    ? "bg-emerald-900/40 text-emerald-300"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                Anthropic:{" "}
                {settings?.anthropic_key_configured ? "Configured" : "No key"}
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${
                  settings?.openai_key_configured
                    ? "bg-emerald-900/40 text-emerald-300"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                OpenAI:{" "}
                {settings?.openai_key_configured ? "Configured" : "No key"}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              <ModelPicker
                label="Standard Model (all agents)"
                value={form.gemini_model ?? ""}
                models={models}
                githubConfigured={!!settings?.github_token_configured}
                onChange={(v) => updateForm("gemini_model", v)}
                accentColor="sky"
              />
              <ModelPicker
                label="Deep Think Model (Moderator, Synthesis, Critic, Risk Judge)"
                value={form.deep_think_model ?? ""}
                models={models}
                githubConfigured={!!settings?.github_token_configured}
                onChange={(v) => updateForm("deep_think_model", v)}
                accentColor="violet"
              />
            </div>
            {/* Context-limited warning banner */}
            {models.find((m) => m.model === form.gemini_model)?.context_limited && (
              <div className="mt-3 flex items-start gap-2.5 rounded-lg border border-amber-700/50 bg-amber-900/20 px-3.5 py-2.5 text-sm text-amber-300">
                <IconWarning size={18} weight="fill" className="mt-0.5 shrink-0" />
                <div>
                  <span className="font-medium">{MODEL_DISPLAY_NAMES[form.gemini_model ?? ""] ?? form.gemini_model}</span> has a small context window on GitHub Models (~4K–8K tokens).
                  {" "}Agent Debate prompts will be automatically compacted (summaries only, no full analysis text).
                  {" "}For full-quality debate, use <span className="font-medium">GPT-4.1</span>, <span className="font-medium">GPT-4o</span>, or any Gemini/Claude model.
                </div>
              </div>
            )}
          </BentoCard>
        </div>
        )}

        {/* ── Cost & Weights Tab ────────────────────────── */}
        {activeTab === "cost" && (
        <div className="grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2">
          {/* ── Live Cost Estimator ─────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsEstimator size={20} weight="duotone" className="inline -mt-0.5" /> Cost Estimator
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
                {premiumRequests !== null && (
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-emerald-300">
                      ~{premiumRequests}
                    </div>
                    <div className="text-xs text-slate-500">Copilot premium req.</div>
                  </div>
                )}
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

          {/* ── Cost Controls ───────────────────────────────── */}
          <BentoCard>
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsCostControls size={20} weight="duotone" className="inline -mt-0.5" /> Cost Controls
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

          {/* ── Pillar Weights ──────────────────────────────── */}
          <BentoCard className="lg:col-span-2">
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              <SettingsPillars size={20} weight="duotone" className="inline -mt-0.5" /> Pillar Weights
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
        )}

        {/* ── Performance Tab ───────────────────────────── */}
        {activeTab === "performance" && (
        <div className="grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2">
          {/* ── Cache Health ────────────────────────────────── */}
          <BentoCard>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-300">
                <SettingsCache size={20} weight="duotone" className="inline -mt-0.5" /> Cache Health
              </h3>
              <button
                onClick={refreshPerfData}
                disabled={perfLoading}
                className="rounded-md px-2 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
                title="Refresh"
              >
                {perfLoading ? <SettingsRefresh size={14} className="animate-spin" /> : <SettingsRefresh size={14} />}
              </button>
            </div>
            {cacheStats ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-slate-200">
                      {cacheStats.entries}
                    </div>
                    <div className="text-xs text-slate-500">entries</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-emerald-300">
                      {cacheStats.hit_rate_pct}%
                    </div>
                    <div className="text-xs text-slate-500">hit rate</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-slate-200">
                      {cacheStats.total_gets}
                    </div>
                    <div className="text-xs text-slate-500">total gets</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                    <div className="text-lg font-bold text-slate-200">
                      {cacheStats.total_hits}
                    </div>
                    <div className="text-xs text-slate-500">total hits</div>
                  </div>
                </div>
                <button
                  onClick={async () => {
                    try {
                      const res = await clearCache();
                      setCacheClearMsg(`Cleared ${res.cleared} entries`);
                      setTimeout(() => setCacheClearMsg(null), 3000);
                      refreshPerfData();
                    } catch {
                      setCacheClearMsg("Failed to clear cache");
                      setTimeout(() => setCacheClearMsg(null), 3000);
                    }
                  }}
                  className="w-full rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 transition-colors hover:border-rose-600 hover:text-rose-300"
                >
                  Clear Cache
                </button>
                {cacheClearMsg && (
                  <p className={`text-center text-xs ${
                    cacheClearMsg.includes("Failed") ? "text-rose-400" : "text-emerald-400"
                  }`}>
                    {cacheClearMsg}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-500">Loading cache stats...</p>
            )}
          </BentoCard>

          {/* ── TTL Optimizer ───────────────────────────────── */}
          <BentoCard>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-300">
                <SettingsOptimizer size={20} weight="duotone" className="inline -mt-0.5" /> TTL Optimizer
              </h3>
              <button
                onClick={async () => {
                  const s = await getPerfOptimizerStatus();
                  setOptimizerStatus(s);
                }}
                className="rounded-md px-2 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
                title="Refresh status"
              >
                <SettingsRefresh size={14} />
              </button>
            </div>
            {optimizerStatus ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      optimizerStatus.status === "running"
                        ? "bg-emerald-400 animate-pulse"
                        : optimizerStatus.status === "error"
                        ? "bg-rose-400"
                        : "bg-slate-500"
                    }`}
                  />
                  <span className="text-sm font-medium capitalize text-slate-300">
                    {optimizerStatus.status}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                    <div className="text-base font-bold text-slate-200">
                      {optimizerStatus.iterations}
                    </div>
                    <div className="text-xs text-slate-500">iterations</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                    <div className="text-base font-bold text-slate-200">
                      {optimizerStatus.best_p95_ms?.toFixed(0) ?? "—"}
                    </div>
                    <div className="text-xs text-slate-500">best p95 ms</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                    <div className="text-base font-bold text-emerald-300">
                      {optimizerStatus.kept}
                    </div>
                    <div className="text-xs text-slate-500">kept</div>
                  </div>
                  <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                    <div className="text-base font-bold text-slate-400">
                      {optimizerStatus.discarded}
                    </div>
                    <div className="text-xs text-slate-500">discarded</div>
                  </div>
                </div>
                {optimizerStatus.status === "running" ? (
                  <button
                    onClick={async () => {
                      await stopPerfOptimizer();
                      const s = await getPerfOptimizerStatus();
                      setOptimizerStatus(s);
                    }}
                    className="w-full rounded-lg border border-amber-700/50 bg-amber-900/20 px-3 py-2 text-sm text-amber-300 transition-colors hover:bg-amber-900/40"
                  >
                    Stop Optimizer
                  </button>
                ) : (
                  <button
                    onClick={async () => {
                      await startPerfOptimizer();
                      const s = await getPerfOptimizerStatus();
                      setOptimizerStatus(s);
                    }}
                    className="w-full rounded-lg border border-emerald-700/50 bg-emerald-900/20 px-3 py-2 text-sm text-emerald-300 transition-colors hover:bg-emerald-900/40"
                  >
                    Start Optimizer
                  </button>
                )}
                <p className="text-xs text-slate-600">
                  Autoresearch loop that tunes per-endpoint cache TTL values.
                  Each experiment takes ~60s.
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Loading optimizer status...</p>
            )}
          </BentoCard>

          {/* ── Optimization Progress Chart ──────────────────────── */}
          <BentoCard className="lg:col-span-2">
            <PerfProgressChart experiments={perfExperiments} />
          </BentoCard>

          {/* ── API Latency ─────────────────────────────────── */}
          <BentoCard className="lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-300">
                  <SettingsLatency size={20} weight="duotone" className="inline -mt-0.5" /> API Latency
                </h3>
                <p className="text-xs text-slate-500">Last 5 minutes</p>
              </div>
              <button
                onClick={refreshPerfData}
                disabled={perfLoading}
                className="rounded-md px-2 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
                title="Refresh"
              >
                {perfLoading ? <SettingsRefresh size={14} className="animate-spin" /> : <SettingsRefresh size={14} />}
              </button>
            </div>
            {perfSummary ? (
              perfSummary.total_requests > 0 ? (
                <div className="space-y-4">
                  {/* Overall stats */}
                  <div className="grid grid-cols-4 gap-3">
                    <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                      <div className="text-base font-bold text-slate-200">
                        {perfSummary.total_requests}
                      </div>
                      <div className="text-xs text-slate-500">requests</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                      <div className="text-base font-bold text-slate-200">
                        {perfSummary.p50_ms}
                      </div>
                      <div className="text-xs text-slate-500">p50 ms</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                      <div className="text-base font-bold text-amber-300">
                        {perfSummary.p95_ms}
                      </div>
                      <div className="text-xs text-slate-500">p95 ms</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/50 p-2.5 text-center">
                      <div className="text-base font-bold text-rose-300">
                        {perfSummary.p99_ms}
                      </div>
                      <div className="text-xs text-slate-500">p99 ms</div>
                    </div>
                  </div>
                  {/* Per-endpoint table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-slate-700 text-xs text-slate-500">
                          <th className="pb-2 pr-4 font-medium">Endpoint</th>
                          <th className="pb-2 pr-4 text-right font-medium">Requests</th>
                          <th className="pb-2 pr-4 text-right font-medium">p50 ms</th>
                          <th className="pb-2 text-right font-medium">p95 ms</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(perfSummary.per_endpoint)
                          .sort(([, a], [, b]) => b.p95_ms - a.p95_ms)
                          .slice(0, 20)
                          .map(([ep, data]) => (
                            <tr
                              key={ep}
                              className="border-b border-slate-800/50 text-slate-300"
                            >
                              <td className="py-1.5 pr-4 font-mono text-xs">
                                {ep}
                              </td>
                              <td className="py-1.5 pr-4 text-right tabular-nums">
                                {data.count}
                              </td>
                              <td className="py-1.5 pr-4 text-right tabular-nums">
                                {data.p50_ms}
                              </td>
                              <td
                                className={`py-1.5 text-right tabular-nums ${
                                  data.p95_ms > 1000
                                    ? "text-rose-400"
                                    : data.p95_ms > 500
                                    ? "text-amber-400"
                                    : "text-slate-300"
                                }`}
                              >
                                {data.p95_ms}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-slate-500">
                  No requests recorded yet — load some pages first to generate
                  traffic, then refresh.
                </p>
              )
            ) : (
              <p className="text-sm text-slate-500">Loading latency data...</p>
            )}
          </BentoCard>
        </div>
        )}
      </main>
    </div>
  );
}
