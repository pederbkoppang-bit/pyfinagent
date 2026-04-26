"use client";

/**
 * phase-16.42: Quick Actions panel for the authenticated home page.
 *
 * Sections:
 *   1. Ticker input + Analyze button (mirrors the existing page.tsx logic;
 *      parent owns the ticker state).
 *   2. Three action rows with keyboard-shortcut badges:
 *      - Run morning cycle      [Ctrl+Shift+R]  -> triggerPaperTradingCycle()
 *      - Open backtest console  [Ctrl+B]        -> router.push("/backtest")
 *      - Halt all trading       [Ctrl/Cmd+Shift+H] -> FLATTEN_ALL + PAUSE
 *
 * Halt sequence is two API calls in order, matching KillSwitchShortcut.tsx
 * lines 18-21. We do NOT register a new Ctrl/Cmd+Shift+H listener -- that
 * shortcut is owned globally by KillSwitchShortcut. The Halt button here
 * is click-only; the kbd badge is a label, not a re-binding.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { triggerPaperTradingCycle, postPaperKillSwitchAction } from "@/lib/api";
import { NavBacktest, Warning, ChartLineUp } from "@/lib/icons";

type Props = {
  ticker: string;
  onTickerChange: (t: string) => void;
  onAnalyze: () => void;
};

type ActionState = "idle" | "pending" | "success" | "error";

type ActionRow = {
  label: string;
  shortcut: string;
  Icon: typeof NavBacktest;
  onClick: () => void;
  state: ActionState;
  message: string | null;
};

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="shrink-0 whitespace-nowrap rounded border border-navy-700 bg-navy-900 px-1.5 py-0.5 font-mono text-[10px] text-slate-500">
      {children}
    </kbd>
  );
}

export function HomeQuickActionsPanel({ ticker, onTickerChange, onAnalyze }: Props) {
  const router = useRouter();
  const [cycleState, setCycleState] = useState<ActionState>("idle");
  const [cycleMsg, setCycleMsg] = useState<string | null>(null);
  const [haltState, setHaltState] = useState<ActionState>("idle");
  const [haltMsg, setHaltMsg] = useState<string | null>(null);

  const runMorningCycle = useCallback(async () => {
    if (cycleState === "pending") return;
    setCycleState("pending");
    setCycleMsg(null);
    try {
      const res = await triggerPaperTradingCycle();
      setCycleState("success");
      setCycleMsg(res?.message ?? "Cycle started.");
    } catch (e) {
      setCycleState("error");
      setCycleMsg(e instanceof Error ? e.message : "Cycle failed");
    }
    window.setTimeout(() => {
      setCycleState("idle");
      setCycleMsg(null);
    }, 4000);
  }, [cycleState]);

  const openBacktest = useCallback(() => {
    router.push("/backtest");
  }, [router]);

  const haltAll = useCallback(async () => {
    if (haltState === "pending") return;
    const ok = window.confirm(
      "Emergency halt: flatten all paper-trading positions and pause new orders?\n\nShortcut: Ctrl/Cmd+Shift+H",
    );
    if (!ok) return;
    setHaltState("pending");
    setHaltMsg("Halting...");
    try {
      // Two-step sequence required (FLATTEN_ALL then PAUSE) -- mirrors
      // KillSwitchShortcut.tsx:18-21.
      await postPaperKillSwitchAction("FLATTEN_ALL");
      await postPaperKillSwitchAction("PAUSE");
      setHaltState("success");
      setHaltMsg("Halted: positions flattened, trading paused.");
    } catch (e) {
      setHaltState("error");
      setHaltMsg(e instanceof Error ? e.message : "Halt failed");
    }
    window.setTimeout(() => {
      setHaltState("idle");
      setHaltMsg(null);
    }, 5000);
  }, [haltState]);

  // Keyboard shortcuts: ONLY Ctrl+Shift+R and Ctrl+B here.
  // Ctrl/Cmd+Shift+H stays owned by KillSwitchShortcut globally to avoid
  // double-fire (it would prompt twice if both listeners ran).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.shiftKey && (e.key === "R" || e.key === "r")) {
        e.preventDefault();
        runMorningCycle();
        return;
      }
      // Ctrl+B / Cmd+B (no shift) -> backtest. Don't override Cmd+B in
      // text inputs (browser bookmark shortcut on some platforms);
      // skip if focused element is an input or textarea.
      if (mod && !e.shiftKey && (e.key === "B" || e.key === "b")) {
        const target = e.target as HTMLElement | null;
        if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
        e.preventDefault();
        openBacktest();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runMorningCycle, openBacktest]);

  const actions: ActionRow[] = [
    {
      label: "Run morning cycle",
      shortcut: "Ctrl+Shift+R",
      Icon: ChartLineUp,
      onClick: runMorningCycle,
      state: cycleState,
      message: cycleMsg,
    },
    {
      label: "Open backtest console",
      shortcut: "Ctrl+B",
      Icon: NavBacktest,
      onClick: openBacktest,
      state: "idle",
      message: null,
    },
    {
      label: "Halt all trading",
      shortcut: "Ctrl/Cmd+Shift+H",
      Icon: Warning,
      onClick: haltAll,
      state: haltState,
      message: haltMsg,
    },
  ];

  return (
    <div className="h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/40">
      <div className="border-b border-navy-700 px-4 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Quick Actions</h3>
      </div>

      {/* Section A: ticker input + Analyze. phase-16.47: input wrapper
          gets min-w-0 so flex-1 shrinks below content min-width without
          overflow; button gets shrink-0 so it never gets cropped. */}
      <div className="border-b border-navy-700 p-4">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="TICKER"
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
            onKeyDown={(e) => {
              if (e.key === "Enter" && ticker.trim()) onAnalyze();
            }}
            className="min-w-0 flex-1 rounded-lg border border-navy-700 bg-navy-900 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
            aria-label="Ticker symbol"
          />
          <button
            onClick={onAnalyze}
            disabled={!ticker.trim()}
            className="shrink-0 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-navy-700 disabled:text-slate-500"
          >
            Analyze
          </button>
        </div>
      </div>

      {/* Section B: action rows */}
      <ul className="divide-y divide-navy-700/50">
        {actions.map((a) => {
          const isPending = a.state === "pending";
          const isError = a.state === "error";
          return (
            <li key={a.label}>
              <button
                onClick={a.onClick}
                disabled={isPending}
                className="flex w-full items-center gap-2 px-4 py-3 text-left transition-colors hover:bg-navy-700/40 focus:bg-navy-700/40 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
                aria-label={`${a.label} (${a.shortcut})`}
              >
                <a.Icon size={16} weight="duotone" className={`shrink-0 ${isPending ? "animate-pulse text-sky-400" : "text-slate-400"}`} />
                <span className="min-w-0 flex-1 truncate text-sm text-slate-200">{a.label}</span>
                <Kbd>{a.shortcut}</Kbd>
              </button>
              {a.message && (
                <p className={`px-4 pb-3 text-xs ${isError ? "text-rose-400" : "text-sky-400"}`}>
                  {a.message}
                </p>
              )}
            </li>
          );
        })}
      </ul>

    </div>
  );
}
