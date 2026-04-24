/**
 * phase-10.5.6 StrategyDetail.
 *
 * Per-strategy detail panel for `/sovereign/strategy/[id]`. Three
 * scoped sections, each with deterministic empty-state copy:
 *
 *   1. Equity curve   (data-testid="equity-curve")
 *   2. Override timeline (data-testid="override-timeline")
 *   3. Kill-switch events (data-testid="kill-switch-events")
 *
 * Today: equity + overrides arrive empty (no live source); events
 * are sourced from `handoff/demotion_audit.jsonl` filtered by
 * `challenger_id`. Component is props-driven so the test can mount
 * with deterministic fixtures.
 */
"use client";

import { BentoCard } from "@/components/BentoCard";
import { TrendUp, ListBullets, ShieldCheck } from "@phosphor-icons/react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  StrategyEquityPoint,
  StrategyKillSwitchEvent,
  StrategyOverride,
} from "@/lib/api";

export interface StrategyDetailProps {
  strategyId: string;
  equity: StrategyEquityPoint[];
  overrides: StrategyOverride[];
  events: StrategyKillSwitchEvent[];
  note?: string | null;
}

function EmptyState({ message }: { message: string }) {
  return (
    <p className="py-6 text-center text-xs text-slate-500">{message}</p>
  );
}

export function StrategyDetail({
  strategyId,
  equity,
  overrides,
  events,
  note,
}: StrategyDetailProps) {
  return (
    <div data-testid="strategy-detail" className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-100">{strategyId}</h2>
        <p className="text-xs text-slate-500">
          Per-strategy detail (equity / overrides / kill-switch events)
        </p>
      </div>

      {/* Equity curve */}
      <BentoCard>
        <div className="mb-3 flex items-center gap-2">
          <TrendUp size={18} className="text-emerald-400" weight="fill" />
          <h3 className="text-sm font-semibold text-slate-300">
            Equity curve ({equity.length})
          </h3>
        </div>
        <div data-testid="equity-curve">
          {equity.length === 0 ? (
            <EmptyState message="No per-strategy NAV recorded yet (paper_portfolio_snapshots is global today)" />
          ) : (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equity} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      color: "#e2e8f0",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="nav"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.15}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </BentoCard>

      {/* Override timeline */}
      <BentoCard>
        <div className="mb-3 flex items-center gap-2">
          <ListBullets size={18} className="text-sky-400" weight="fill" />
          <h3 className="text-sm font-semibold text-slate-300">
            Parameter overrides ({overrides.length})
          </h3>
        </div>
        <div data-testid="override-timeline">
          {overrides.length === 0 ? (
            <EmptyState message="No parameter overrides recorded yet" />
          ) : (
            <ol className="space-y-2">
              {overrides.map((o, i) => (
                <li
                  key={`${o.date}-${o.param}-${i}`}
                  data-override={o.param}
                  className="flex flex-wrap items-baseline gap-2 border-l-2 border-sky-500/40 pl-3 text-xs"
                >
                  <span className="font-mono text-slate-500">{o.date}</span>
                  <span className="font-mono text-slate-300">{o.param}</span>
                  <span className="text-slate-500">
                    {o.from_value ?? "--"} {"->"} {o.to_value ?? "--"}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </BentoCard>

      {/* Kill-switch events */}
      <BentoCard>
        <div className="mb-3 flex items-center gap-2">
          <ShieldCheck size={18} className="text-rose-400" weight="fill" />
          <h3 className="text-sm font-semibold text-slate-300">
            Kill-switch events ({events.length})
          </h3>
        </div>
        <div data-testid="kill-switch-events">
          {events.length === 0 ? (
            <EmptyState message="No kill-switch events recorded for this strategy" />
          ) : (
            <ol className="space-y-2">
              {events.map((e, i) => (
                <li
                  key={`${e.date}-${i}`}
                  data-event={e.label}
                  className="flex flex-wrap items-baseline gap-2 border-l-2 border-rose-500/40 pl-3 text-xs"
                >
                  <span className="font-mono text-slate-500">{e.date}</span>
                  <span className="rounded-full bg-rose-500/15 px-2 py-0.5 font-medium text-rose-300">
                    {e.label}
                  </span>
                  {e.detail && (
                    <span className="font-mono text-slate-500">{e.detail}</span>
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>
      </BentoCard>

      {note && (
        <p className="text-center text-[11px] italic text-slate-600">{note}</p>
      )}
    </div>
  );
}
