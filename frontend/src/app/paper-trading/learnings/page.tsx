"use client";

import { useEffect, useState } from "react";

import { Sidebar } from "@/components/Sidebar";
import { VirtualFundLearnings } from "@/components/VirtualFundLearnings";
import { getPaperLearnings } from "@/lib/api";
import type { VirtualFundLearningsData } from "@/lib/types";

const WINDOW_DAYS = 30;

export default function PaperTradingLearningsPage() {
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
