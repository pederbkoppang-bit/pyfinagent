/**
 * phase-44.1: <CommandPalette/> -- Cmd-K command palette (cmdk by Vercel/Pacos).
 *
 * Operator-approved 2026-05-22 (see handoff/current/operator_approval_44.1.md).
 *
 * Mounted at app/layout.tsx root so Cmd+K (mac) / Ctrl+K (win/linux) opens
 * from any route. Keyboard-first per uxpatterns.dev 2026 + Linear / Vercel /
 * Stripe / Raycast conventions. Behind featureFlags.ts::command_palette
 * (default ON per /goal gate 3 carve-out: feature is fully additive + has no
 * destructive actions; flag exists so operator can disable if needed).
 *
 * Initial command set: navigate to any of the 15 routes + "Analyze ticker
 * {ticker}" jump-to-signals. Settings, kill-switch, manual-run, etc., wired
 * in later phases (44.7 / 44.8).
 */
"use client";

import { Command } from "cmdk";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { clsx } from "clsx";
import { isFeatureEnabled } from "@/lib/featureFlags";
import { useKeyboardShortcut } from "@/lib/hooks";
import {
  NavHome,
  NavSignals,
  NavReports,
  NavPerformance,
  NavPaperTrading,
  NavBacktest,
  NavSettings,
  NavSovereign,
  Robot,
  Graph,
  Clock,
  Database,
  MagnifyingGlass,
} from "@/lib/icons";
import type { Icon } from "@/lib/icons";

interface RouteEntry {
  href: string;
  label: string;
  group: "Analyze" | "Reports" | "Trading" | "System" | "Settings";
  // Phosphor Icon type (ForwardRefExoticComponent<IconProps>)
  icon: Icon;
}

const ROUTES: RouteEntry[] = [
  { href: "/", label: "Home", group: "Analyze", icon: NavHome },
  { href: "/signals", label: "Signals", group: "Analyze", icon: NavSignals },
  { href: "/reports", label: "Reports", group: "Reports", icon: NavReports },
  { href: "/performance", label: "Performance", group: "Reports", icon: NavPerformance },
  { href: "/paper-trading", label: "Paper Trading", group: "Trading", icon: NavPaperTrading },
  { href: "/paper-trading/learnings", label: "Learnings", group: "Trading", icon: NavPerformance },
  { href: "/backtest", label: "Backtest", group: "Trading", icon: NavBacktest },
  { href: "/sovereign", label: "Sovereign", group: "Trading", icon: NavSovereign },
  { href: "/agents", label: "MAS Dashboard", group: "System", icon: Robot },
  { href: "/agent-map", label: "Agent Map", group: "System", icon: Graph },
  { href: "/cron", label: "Cron / Logs", group: "System", icon: Clock },
  { href: "/observability", label: "Data Freshness", group: "System", icon: Database },
  { href: "/settings", label: "Settings", group: "Settings", icon: NavSettings },
];

const GROUPS: Array<RouteEntry["group"]> = ["Analyze", "Reports", "Trading", "System", "Settings"];

export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [tickerQuery, setTickerQuery] = useState("");

  const enabled = isFeatureEnabled("command_palette");

  useKeyboardShortcut("mod+k", () => {
    if (!enabled) return;
    setOpen((o) => !o);
  }, { enabled });

  useEffect(() => {
    if (!open) setTickerQuery("");
  }, [open]);

  if (!enabled) return null;

  const handleSelect = (href: string) => {
    setOpen(false);
    router.push(href);
  };

  const handleAnalyzeTicker = (raw: string) => {
    const ticker = raw.trim().toUpperCase().replace(/[^A-Z0-9.]/g, "");
    if (!ticker) return;
    setOpen(false);
    router.push(`/signals?ticker=${encodeURIComponent(ticker)}`);
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command palette"
      className={clsx(
        "fixed inset-0 z-50 flex items-start justify-center",
        "bg-black/40 backdrop-blur-sm",
        "pt-[15vh] px-4",
      )}
    >
      <div
        className={clsx(
          "w-full max-w-xl",
          "bg-navy-900 border border-navy-700 rounded-2xl shadow-2xl",
          "overflow-hidden",
        )}
      >
        <div className="flex items-center gap-2 px-4 py-3 border-b border-navy-700">
          <MagnifyingGlass size={16} weight="bold" className="text-slate-400" aria-hidden="true" />
          <Command.Input
            placeholder="Type a command, route, or ticker..."
            className={clsx(
              "flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500",
              "outline-none border-0",
            )}
            value={tickerQuery}
            onValueChange={setTickerQuery}
          />
          <kbd className="hidden sm:inline-flex text-[10px] font-mono text-slate-500 bg-navy-800 border border-navy-700 rounded px-1.5 py-0.5">
            ESC
          </kbd>
        </div>
        <Command.List className="max-h-[60vh] overflow-y-auto scrollbar-thin py-2">
          <Command.Empty className="px-4 py-6 text-sm text-slate-500 text-center">
            No commands match. Press Enter on a ticker (e.g. NVDA) to analyze.
          </Command.Empty>
          {GROUPS.map((group) => {
            const items = ROUTES.filter((r) => r.group === group);
            if (items.length === 0) return null;
            return (
              <Command.Group key={group} heading={group} className="text-[10px] font-medium text-slate-500 uppercase tracking-wider px-3 pt-3 pb-1">
                {items.map((entry) => {
                  const IconCmp = entry.icon;
                  return (
                    <Command.Item
                      key={entry.href}
                      value={`${group} ${entry.label} ${entry.href}`}
                      onSelect={() => handleSelect(entry.href)}
                      className={clsx(
                        "flex items-center gap-2 px-3 py-2 mx-1 rounded-lg",
                        "text-sm text-slate-200 cursor-pointer",
                        "data-[selected=true]:bg-sky-950/50 data-[selected=true]:text-sky-100",
                        "min-h-[24px]",
                      )}
                    >
                      <IconCmp size={14} weight="bold" aria-hidden="true" />
                      <span className="flex-1">{entry.label}</span>
                      <span className="text-[10px] font-mono text-slate-500">{entry.href}</span>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            );
          })}
          {/[A-Za-z]/.test(tickerQuery) && tickerQuery.length <= 5 ? (
            <Command.Group heading="Analyze" className="text-[10px] font-medium text-slate-500 uppercase tracking-wider px-3 pt-3 pb-1">
              <Command.Item
                value={`analyze-${tickerQuery}`}
                onSelect={() => handleAnalyzeTicker(tickerQuery)}
                className={clsx(
                  "flex items-center gap-2 px-3 py-2 mx-1 rounded-lg",
                  "text-sm text-slate-200 cursor-pointer",
                  "data-[selected=true]:bg-sky-950/50 data-[selected=true]:text-sky-100",
                  "min-h-[24px]",
                )}
              >
                <NavSignals size={14} weight="bold" aria-hidden="true" />
                <span>Analyze ticker {tickerQuery.toUpperCase()}</span>
                <span className="ml-auto text-[10px] font-mono text-slate-500">/signals</span>
              </Command.Item>
            </Command.Group>
          ) : null}
        </Command.List>
        <div className="px-4 py-2 text-[10px] text-slate-500 border-t border-navy-700 flex items-center justify-between">
          <span>Cmd-K opens this palette from anywhere</span>
          <span>
            <kbd className="font-mono">↑↓</kbd> navigate · <kbd className="font-mono">↵</kbd> select · <kbd className="font-mono">ESC</kbd> close
          </span>
        </div>
      </div>
    </Command.Dialog>
  );
}
