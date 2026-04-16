"use client";

import { useEffect, useState } from "react";
import { IconX } from "@/lib/icons";

interface Signal {
  agent: string;
  role: string;
  rationale: string;
  weight: number;
}

interface Rationale {
  trade_id: string;
  ticker: string | null;
  action: string | null;
  created_at: string | null;
  reason: string | null;
  signals: Signal[];
  tree: {
    analyst: Signal[];
    debate: { bull: Signal[]; bear: Signal[] };
    trader: Signal[];
    risk: Signal[];
  };
}

interface Props {
  tradeId: string | null;
  onClose: () => void;
}

/**
 * Progressive-disclosure drawer. Layers expand one at a time (not shown inline
 * on the trades table). TradingAgents pattern: Analyst -> Bull/Bear -> Trader
 * -> Risk. Each layer is collapsible so operator reviews in order.
 */
export function AgentRationaleDrawer({ tradeId, onClose }: Props) {
  const [data, setData] = useState<Rationale | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!tradeId) return;
    setLoading(true);
    setError(null);
    setData(null);
    fetch(`/api/paper-trading/trades/${encodeURIComponent(tradeId)}/rationale`, {
      credentials: "include",
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message || "failed to load rationale"))
      .finally(() => setLoading(false));
  }, [tradeId]);

  if (!tradeId) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/40"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <aside
        onClick={(e) => e.stopPropagation()}
        className="scrollbar-thin h-full w-full max-w-xl overflow-y-auto border-l border-navy-700 bg-navy-900 p-6"
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wider text-slate-500">
              Agent rationale
            </p>
            <h3 className="text-lg font-semibold text-slate-100">
              {data?.ticker ?? tradeId}{" "}
              {data?.action && (
                <span
                  className={
                    "ml-2 rounded px-2 py-0.5 text-xs font-medium " +
                    (data.action === "BUY"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-rose-500/20 text-rose-300")
                  }
                >
                  {data.action}
                </span>
              )}
            </h3>
            {data?.created_at && (
              <p className="mt-1 text-xs text-slate-500">{data.created_at}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-navy-800 hover:text-slate-200"
            aria-label="Close"
          >
            <IconX size={18} />
          </button>
        </div>

        {loading && (
          <p className="text-sm text-slate-400">Loading rationale...</p>
        )}
        {error && (
          <p className="rounded border border-rose-500/30 bg-rose-950/30 p-2 text-sm text-rose-300">
            {error}
          </p>
        )}

        {data && data.signals.length === 0 && (
          <p className="text-sm text-slate-500">
            No signal attribution recorded for this trade.
          </p>
        )}

        {data && data.signals.length > 0 && (
          <div className="space-y-3">
            <Layer title="Analyst" items={data.tree.analyst} />
            <DebateLayer bull={data.tree.debate.bull} bear={data.tree.debate.bear} />
            <Layer title="Trader" items={data.tree.trader} />
            <Layer title="Risk Judge" items={data.tree.risk} emphasize />
          </div>
        )}
      </aside>
    </div>
  );
}

function Layer({
  title,
  items,
  emphasize = false,
}: {
  title: string;
  items: Signal[];
  emphasize?: boolean;
}) {
  if (!items || items.length === 0) return null;
  return (
    <details className={"rounded-lg border p-3 " + (emphasize ? "border-sky-500/30" : "border-navy-700")} open>
      <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-slate-200">
        <span
          className={
            "h-2 w-2 rounded-full " +
            (emphasize ? "bg-sky-400" : "bg-emerald-500")
          }
        />
        {title}
        <span className="ml-auto text-[10px] text-slate-500">{items.length}</span>
      </summary>
      <div className="mt-2 space-y-2">
        {items.map((s, i) => (
          <div key={i} className="rounded bg-navy-800/60 p-2">
            <p className="text-xs text-slate-400">
              <span className="font-medium text-slate-300">{s.agent}</span>
              {s.role ? <span className="ml-1 text-slate-500">({s.role})</span> : null}
              {typeof s.weight === "number" ? (
                <span className="float-right font-mono text-[11px] text-slate-500">
                  weight {s.weight.toFixed(2)}
                </span>
              ) : null}
            </p>
            <p className="mt-1 whitespace-pre-wrap text-sm text-slate-200">
              {s.rationale}
            </p>
          </div>
        ))}
      </div>
    </details>
  );
}

function DebateLayer({ bull, bear }: { bull: Signal[]; bear: Signal[] }) {
  if ((!bull || bull.length === 0) && (!bear || bear.length === 0)) return null;
  return (
    <details className="rounded-lg border border-navy-700 p-3" open>
      <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-slate-200">
        <span className="h-2 w-2 rounded-full bg-amber-400" />
        Debate
        <span className="ml-auto text-[10px] text-slate-500">
          {(bull?.length || 0) + (bear?.length || 0)}
        </span>
      </summary>
      <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-emerald-400">
            Bull
          </p>
          {bull.length === 0 && <p className="text-xs text-slate-500">No bull case.</p>}
          {bull.map((s, i) => (
            <p key={i} className="mt-1 whitespace-pre-wrap rounded bg-emerald-950/30 p-2 text-sm text-slate-200">
              {s.rationale}
            </p>
          ))}
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-rose-400">
            Bear
          </p>
          {bear.length === 0 && <p className="text-xs text-slate-500">No bear case.</p>}
          {bear.map((s, i) => (
            <p key={i} className="mt-1 whitespace-pre-wrap rounded bg-rose-950/30 p-2 text-sm text-slate-200">
              {s.rationale}
            </p>
          ))}
        </div>
      </div>
    </details>
  );
}
