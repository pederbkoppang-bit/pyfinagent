"use client";

// phase-73 fix (2026-05-26): moved from `app/paper-trading/learnings/`
// to root `app/learnings/` because the cycle-63 paper-trading route-split
// (cycle 63 layout.tsx) wrapped this page in cockpit chrome (Sidebar +
// KPI hero + cockpit tab bar) that doesn't apply to a peer-level
// "learnings" destination. Sidebar's Learnings link is a sibling of
// Paper Trading (Sidebar.tsx:44 group "Trading"), so the URL belongs
// at root, not under /paper-trading. Operator-flagged 2026-05-26:
// double-Sidebar rendering on /paper-trading/learnings.

import { useEffect, useState } from "react";

import { Sidebar } from "@/components/Sidebar";
import { VirtualFundLearnings } from "@/components/VirtualFundLearnings";
import { getPaperLearnings } from "@/lib/api";
import type { VirtualFundLearningsData } from "@/lib/types";

const WINDOW_DAYS = 30;

export default function LearningsPage() {
  const [data, setData] = useState<VirtualFundLearningsData | undefined>(
    undefined,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getPaperLearnings(WINDOW_DAYS)
      .then((d) => {
        if (cancelled) return;
        setData(d);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(`Failed to load virtual-fund learnings: ${msg}`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          <VirtualFundLearnings
            data={data}
            loading={loading}
            error={error}
          />
        </div>
      </main>
    </div>
  );
}
