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
        <span className={`w-3 shrink-0 text-xs ${value === m.model ? checkClass : "text-transparent"}`}>
          ✓
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

          {/* ── Model Configuration ─────────────────────────── */}
          <BentoCard className="lg:col-span-2">
            <h3 className="mb-3 text-lg font-semibold text-slate-300">
              🧠 Model Configuration
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
                <span className="mt-0.5 shrink-0 text-base">⚠</span>
                <div>
                  <span className="font-medium">{MODEL_DISPLAY_NAMES[form.gemini_model ?? ""] ?? form.gemini_model}</span> has a small context window on GitHub Models (~4K–8K tokens).
                  {" "}Agent Debate prompts will be automatically compacted (summaries only, no full analysis text).
                  {" "}For full-quality debate, use <span className="font-medium">GPT-4.1</span>, <span className="font-medium">GPT-4o</span>, or any Gemini/Claude model.
                </div>
              </div>
            )}
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
