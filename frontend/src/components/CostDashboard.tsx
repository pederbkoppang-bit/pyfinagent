"use client";

import { BentoCard } from "@/components/BentoCard";
import type { CostSummary, AgentCostEntry } from "@/lib/types";
import { IconDeepThink } from "@/lib/icons";

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

function ModelBadge({ model, isDeepThink }: { model: string; isDeepThink: boolean }) {
  const color = isDeepThink
    ? "bg-violet-500/20 text-violet-300 border-violet-500/30"
    : "bg-sky-500/20 text-sky-300 border-sky-500/30";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${color}`}>
      {isDeepThink && <IconDeepThink size={14} weight="fill" className="mr-1" />}
      {model}
    </span>
  );
}

function AgentRow({ entry }: { entry: AgentCostEntry }) {
  return (
    <tr className="border-t border-slate-800">
      <td className="py-2 pr-4 text-sm text-slate-300">{entry.agent_name}</td>
      <td className="py-2 pr-4">
        <ModelBadge model={entry.model} isDeepThink={entry.is_deep_think} />
      </td>
      <td className="py-2 pr-4 text-right font-mono text-sm text-slate-400">
        {formatTokens(entry.input_tokens)}
      </td>
      <td className="py-2 pr-4 text-right font-mono text-sm text-slate-400">
        {formatTokens(entry.output_tokens)}
      </td>
      <td className="py-2 text-right font-mono text-sm text-emerald-400">
        {formatCost(entry.cost_usd)}
      </td>
    </tr>
  );
}

export function CostDashboard({ costSummary }: { costSummary: CostSummary | undefined }) {
  if (!costSummary || !costSummary.agents?.length) {
    return (
      <BentoCard>
        <p className="text-sm text-slate-500">
          No cost data available. Cost tracking is recorded after the analysis completes.
        </p>
      </BentoCard>
    );
  }

  const { total_tokens, total_input_tokens, total_output_tokens, total_cost_usd, total_calls, deep_think_calls, model_breakdown, agents } = costSummary;

  // Sort agents by cost descending
  const sortedAgents = [...agents].sort((a, b) => b.cost_usd - a.cost_usd);

  // Compute top-level bar chart data (input vs output tokens)
  const inputPct = total_tokens > 0 ? (total_input_tokens / total_tokens) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-6 md:col-span-3">
          <BentoCard glow>
            <p className="text-xs text-slate-400">Total Cost</p>
            <p className="mt-1 font-mono text-3xl font-bold text-emerald-400">
              {formatCost(total_cost_usd)}
            </p>
          </BentoCard>
        </div>
        <div className="col-span-6 md:col-span-3">
          <BentoCard>
            <p className="text-xs text-slate-400">Total Tokens</p>
            <p className="mt-1 font-mono text-3xl font-bold text-sky-400">
              {formatTokens(total_tokens)}
            </p>
          </BentoCard>
        </div>
        <div className="col-span-6 md:col-span-3">
          <BentoCard>
            <p className="text-xs text-slate-400">LLM Calls</p>
            <p className="mt-1 font-mono text-3xl font-bold text-slate-200">
              {total_calls}
            </p>
          </BentoCard>
        </div>
        <div className="col-span-6 md:col-span-3">
          <BentoCard>
            <p className="text-xs text-slate-400">Deep Think Calls</p>
            <p className="mt-1 font-mono text-3xl font-bold text-violet-400">
              {deep_think_calls}
            </p>
            <p className="mt-0.5 flex items-center gap-1 text-xs text-slate-500">
              <IconDeepThink size={12} weight="fill" /> Moderator, Risk Judge, Synthesis, Critic
            </p>
          </BentoCard>
        </div>
      </div>

      {/* Token split bar */}
      <BentoCard>
        <h3 className="mb-3 text-sm font-semibold text-slate-300">Token Distribution</h3>
        <div className="flex h-4 overflow-hidden rounded-full bg-slate-800">
          <div
            className="bg-sky-500 transition-all"
            style={{ width: `${inputPct}%` }}
            title={`Input: ${formatTokens(total_input_tokens)}`}
          />
          <div
            className="bg-amber-500 transition-all"
            style={{ width: `${100 - inputPct}%` }}
            title={`Output: ${formatTokens(total_output_tokens)}`}
          />
        </div>
        <div className="mt-2 flex justify-between text-xs text-slate-400">
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-full bg-sky-500" />
            Input: {formatTokens(total_input_tokens)}
          </span>
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-500" />
            Output: {formatTokens(total_output_tokens)}
          </span>
        </div>
      </BentoCard>

      {/* Model breakdown */}
      {Object.keys(model_breakdown).length > 0 && (
        <BentoCard>
          <h3 className="mb-3 text-sm font-semibold text-slate-300">Cost by Model</h3>
          <div className="space-y-3">
            {Object.entries(model_breakdown)
              .sort(([, a], [, b]) => b.cost_usd - a.cost_usd)
              .map(([model, data]) => {
                const pct = total_cost_usd > 0 ? (data.cost_usd / total_cost_usd) * 100 : 0;
                return (
                  <div key={model}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span className="text-slate-300">{model}</span>
                      <span className="font-mono text-emerald-400">
                        {formatCost(data.cost_usd)} ({data.calls} calls)
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-emerald-500 transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </BentoCard>
      )}

      {/* Per-agent breakdown table */}
      <BentoCard>
        <h3 className="mb-3 text-sm font-semibold text-slate-300">
          Per-Agent Breakdown ({sortedAgents.length} agents)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-xs uppercase text-slate-500">
                <th className="pb-2 pr-4">Agent</th>
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2 pr-4 text-right">Input</th>
                <th className="pb-2 pr-4 text-right">Output</th>
                <th className="pb-2 text-right">Cost</th>
              </tr>
            </thead>
            <tbody>
              {sortedAgents.map((entry, i) => (
                <AgentRow key={`${entry.agent_name}-${i}`} entry={entry} />
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-slate-700 font-semibold">
                <td className="pt-2 text-sm text-slate-200" colSpan={2}>Total</td>
                <td className="pt-2 text-right font-mono text-sm text-slate-300">
                  {formatTokens(total_input_tokens)}
                </td>
                <td className="pt-2 text-right font-mono text-sm text-slate-300">
                  {formatTokens(total_output_tokens)}
                </td>
                <td className="pt-2 text-right font-mono text-sm text-emerald-400">
                  {formatCost(total_cost_usd)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </BentoCard>
    </div>
  );
}
