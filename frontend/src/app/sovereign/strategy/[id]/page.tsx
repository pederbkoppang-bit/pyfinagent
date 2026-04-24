/**
 * phase-10.5.6 /sovereign/strategy/[id] route shell.
 *
 * Thin client-side shell that resolves `id` via `useParams`, fetches
 * `getSovereignStrategy(id)`, and renders `<StrategyDetail .../>`.
 */
"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Sidebar } from "@/components/Sidebar";
import { StrategyDetail } from "@/components/StrategyDetail";
import { getSovereignStrategy } from "@/lib/api";
import type { StrategyDetailResponse } from "@/lib/api";
import { CaretLeft } from "@phosphor-icons/react";

export default function StrategyDetailPage() {
  const params = useParams<{ id: string }>();
  const strategyId = params?.id || "";
  const [data, setData] = useState<StrategyDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!strategyId) {
      setLoading(false);
      return;
    }
    let ignore = false;
    setLoading(true);
    getSovereignStrategy(strategyId)
      .then((d) => {
        if (ignore) return;
        setData(d);
        setError(null);
      })
      .catch((e: Error) => {
        if (ignore) return;
        setData(null);
        setError(e.message || "Failed to load strategy detail");
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, [strategyId]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <Link
            href="/sovereign"
            className="mb-4 inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300"
          >
            <CaretLeft size={12} weight="bold" />
            Back to Sovereign
          </Link>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {loading && (
            <p className="py-12 text-center text-sm text-slate-500">Loading...</p>
          )}
          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-rose-400">
              {error}
            </div>
          )}
          {data && (
            <StrategyDetail
              strategyId={data.strategy_id}
              equity={data.equity}
              overrides={data.overrides}
              events={data.events}
              note={data.note}
            />
          )}
        </div>
      </main>
    </div>
  );
}
