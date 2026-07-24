"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { IconCheckCircle, IconInfo, IconWarning } from "@/lib/icons";
import { formatRelativeTime } from "@/lib/formatRelativeTime";
import {
  getPaperCyclesHistory,
  getPaperFreshness,
  getPaperGate,
  getPaperKillSwitchState,
  postPaperKillSwitchAction,
} from "@/lib/api";
import { MarketFilter } from "@/components/paper-trading/MarketFilter";
import { isMarketOpen } from "@/lib/format";
import type { GoLiveGate } from "@/components/GoLiveGateWidget";

/**
 * One-row operator status bar. Four dense segments: Go-Live Gate, Kill
 * Switch, Cycle Health, Scheduler. Each segment fits on one line and
 * surfaces only the single fact an operator needs at a glance.
 *
 * Pattern comes from Stripe / Linear / Vercel / Grafana 12 dashboards:
 * dense status strip above the primary content area, NOT stacked cards.
 */

type KillSwitchState = {
  paused: boolean;
  pause_reason: string | null;
  current_nav: number;
  breach: {
    any_breached: boolean;
    daily_loss_pct: number;
    daily_loss_limit_pct: number;
    trailing_dd_pct: number;
    trailing_dd_limit_pct: number;
  };
};

type Freshness = {
  sources: Record<string, { band: string }>;
  heartbeat: { band: string };
  thresholds: { warn_ratio: number; critical_ratio: number };
};

type CycleRow = {
  cycle_id: string;
  started_at: string;
  status: string;
};

interface Props {
  nextRunAt?: string | null;
  // goal-market-filter-in-gate-bar: optional market-filter segment. Present
  // ONLY on the paper-trading cockpit (layout.tsx). All three must be supplied
  // together for the Market segment to render; the homepage (page.tsx) passes
  // none, so its bar is unchanged.
  markets?: string[];
  activeMarket?: string;
  onMarketChange?: (market: string) => void;
}

const MAX_CONSECUTIVE_FAILURES = 5;

export function OpsStatusBar({
  nextRunAt,
  markets,
  activeMarket,
  onMarketChange,
}: Props) {
  const [gate, setGate] = useState<GoLiveGate | null>(null);
  const [kill, setKill] = useState<KillSwitchState | null>(null);
  const [fresh, setFresh] = useState<Freshness | null>(null);
  const [latestCycle, setLatestCycle] = useState<CycleRow | null>(null);
  const [actionBusy, setActionBusy] = useState<"PAUSE" | "FLATTEN_ALL" | "RESUME" | null>(null);
  // phase-75.12 (frontend-05): failRef was DEAD CODE -- every fetcher below
  // already `.catch(() => null)`s individually, so Promise.all never
  // rejects and the outer try/catch's `catch` branch (where failRef used
  // to be incremented) was unreachable. failRef now increments on the
  // ALL-FOUR-null outcome instead, which IS reachable, and after 5
  // consecutive rounds renders a visible stale segment + stops the
  // interval-driven poll (cron-page failuresRef/stoppedRef template).
  const failRef = useRef(0);
  const stoppedRef = useRef(false);
  const [stale, setStale] = useState(false);

  const refresh = useCallback(async () => {
    const [g, k, f, c] = await Promise.all([
      getPaperGate().catch(() => null),
      getPaperKillSwitchState().catch(() => null) as Promise<KillSwitchState | null>,
      getPaperFreshness().catch(() => null) as Promise<Freshness | null>,
      getPaperCyclesHistory(1).catch(() => null),
    ]);
    const allNull = g == null && k == null && f == null && c == null;
    if (allNull) {
      failRef.current += 1;
      if (failRef.current >= MAX_CONSECUTIVE_FAILURES) {
        stoppedRef.current = true;
        setStale(true);
      }
      return;
    }
    if (g) setGate(g);
    if (k) setKill(k);
    if (f) setFresh(f);
    if (c?.cycles?.length) setLatestCycle(c.cycles[0] as CycleRow);
    failRef.current = 0;
    stoppedRef.current = false;
    setStale(false);
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden && !stoppedRef.current) void refresh();
    }, 60_000);
    // Regaining tab focus always retries, even after the circuit has
    // tripped -- the recovery path (mirrors the pre-existing visibility
    // refetch semantics; a success here clears `stale` above).
    const onVis = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [refresh]);

  const handleAction = async (action: "PAUSE" | "RESUME" | "FLATTEN_ALL") => {
    const confirmMsg =
      action === "FLATTEN_ALL"
        ? "Flatten all open positions and pause trading?"
        : action === "PAUSE"
          ? "Pause new-order generation?"
          : "Resume new-order generation?";
    if (!window.confirm(confirmMsg)) return;
    setActionBusy(action);
    try {
      await postPaperKillSwitchAction(action);
      await refresh();
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
    } finally {
      setActionBusy(null);
    }
  };

  return (
    <section
      aria-label="Paper-trading operator status"
      className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-navy-700 bg-navy-800/60 px-4 py-3"
    >
      {/* goal-market-filter-in-gate-bar: market filter folded into the bar as
          the left-most ("scope before status") segment. Conditional on all
          three props so the homepage instance renders the original 5 segments.
          Keep this a plain <section> (NOT role="toolbar") -- a toolbar would
          hijack the radiogroup's arrow keys (W3C APG). */}
      {markets && activeMarket && onMarketChange && (
        <>
          <MarketSegment
            markets={markets}
            activeMarket={activeMarket}
            onMarketChange={onMarketChange}
          />
          <Divider />
        </>
      )}
      <GateSegment gate={gate} />
      <Divider />
      <KillSegment
        kill={kill}
        busy={actionBusy}
        onAction={handleAction}
      />
      <Divider />
      <CycleSegment fresh={fresh} latestCycle={latestCycle} />
      {stale && (
        <>
          <Divider />
          <div className="flex items-center gap-2" data-testid="ops-stale-segment">
            <IconWarning size={14} weight="fill" className="text-amber-400" />
            <span className="text-xs text-amber-300">Stale (polling paused)</span>
          </div>
        </>
      )}
      <Divider />
      <LastSegment lastStartedAt={latestCycle?.started_at ?? null} />
      <Divider />
      <NextSegment nextRunAt={nextRunAt} />
    </section>
  );
}

// ── Segments ──────────────────────────────────────────────────────

function Divider() {
  return <span className="hidden h-6 w-px bg-navy-700 sm:block" aria-hidden />;
}

function SegmentLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
      {children}
    </span>
  );
}

// goal-market-filter-in-gate-bar: the market-filter segment. Owns a mount-
// guarded clock so each pill's open/closed dot doesn't trigger a hydration
// mismatch -- `now` is null on the server / first client paint (pills show the
// neutral per-market dot), then set on mount and refreshed each minute so the
// emerald/slate session colour stays current. Mirrors MarketSessionStrip's
// retired two-pass pattern; the session signal now lives on the pills.
function MarketSegment({
  markets,
  activeMarket,
  onMarketChange,
}: {
  markets: string[];
  activeMarket: string;
  onMarketChange: (market: string) => void;
}) {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(id);
  }, []);
  const sessionOpen = useMemo(() => {
    if (now == null) return undefined;
    const map: Record<string, boolean> = {};
    for (const m of markets) map[m] = isMarketOpen(m, now);
    return map;
  }, [markets, now]);
  return (
    <div className="flex items-center gap-2">
      <SegmentLabel>Market</SegmentLabel>
      <MarketFilter
        value={activeMarket}
        onChange={onMarketChange}
        markets={markets}
        sessionOpen={sessionOpen}
      />
    </div>
  );
}

function GateSegment({ gate }: { gate: GoLiveGate | null }) {
  if (!gate) {
    return (
      <div className="flex items-center gap-2">
        <SegmentLabel>Gate</SegmentLabel>
        <span className="text-xs text-slate-500">—</span>
      </div>
    );
  }
  const passes = Object.values(gate.booleans).filter(Boolean).length;
  const total = Object.values(gate.booleans).length;
  const eligible = gate.promote_eligible;
  // phase-23.2.19: per-criterion tooltip mirroring GoLiveGateWidget labels.
  // Native multi-line title= is sufficient under WCAG 1.4.13 native-attr
  // exemption (operator-only UI). Full breakdown still lives on the paper-
  // trading page's GoLiveGateWidget; this is the at-a-glance summary.
  const { booleans: b, details: d, thresholds: t } = gate;
  const tooltipLines: string[] = [
    `GATE CHECKS (${passes}/${total} passing)`,
    `${b.trades_ge_100 ? "PASS" : "FAIL"}: >=${t.trades} trades (${d.n_round_trips} closed round trips)`,
    `${b.psr_ge_95_sustained_30d ? "PASS" : "FAIL"}: PSR >= ${t.psr.toFixed(2)} (${t.psr_sustained_days}d) (${d.psr != null ? `${d.psr.toFixed(3)}, n_obs=${d.n_obs}` : "insufficient data"})`,
    `${b.dsr_ge_95 ? "PASS" : "FAIL"}: DSR >= ${t.dsr.toFixed(2)} (${d.dsr != null ? d.dsr.toFixed(3) : "n/a"})`,
    `${b.sr_gap_le_30pct ? "PASS" : "FAIL"}: Reality gap <= ${(t.sr_gap * 100).toFixed(0)}% (${d.latest_reconciliation_divergence_pct.toFixed(2)}%)`,
    `${b.max_dd_within_tolerance ? "PASS" : "FAIL"}: Max DD <= ${t.max_dd_pct.toFixed(0)}% (${d.realized_max_dd_pct.toFixed(2)}%)`,
  ];
  const tooltip = tooltipLines.join("\n");
  return (
    <div className="flex items-center gap-2" title={tooltip}>
      <SegmentLabel>Gate</SegmentLabel>
      <span
        className={clsx(
          "rounded-full px-2 py-0.5 text-[10px] font-semibold",
          eligible ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/15 text-rose-300",
        )}
      >
        {eligible ? "ELIGIBLE" : "NOT ELIGIBLE"}
      </span>
      <span className="font-mono text-xs text-slate-400">{passes}/{total}</span>
      <IconInfo size={12} className="text-slate-600" />
    </div>
  );
}

function KillSegment({
  kill,
  busy,
  onAction,
}: {
  kill: KillSwitchState | null;
  busy: "PAUSE" | "RESUME" | "FLATTEN_ALL" | null;
  onAction: (a: "PAUSE" | "RESUME" | "FLATTEN_ALL") => void;
}) {
  if (!kill) {
    return (
      <div className="flex items-center gap-2">
        <SegmentLabel>Kill</SegmentLabel>
        <span className="text-xs text-slate-500">—</span>
      </div>
    );
  }
  const alarm = kill.paused || kill.breach.any_breached;
  return (
    <div className="flex items-center gap-2">
      <SegmentLabel>Kill</SegmentLabel>
      <IconWarning
        size={14}
        weight={alarm ? "fill" : "regular"}
        className={alarm ? "text-rose-400" : "text-slate-500"}
      />
      <span
        className={clsx(
          "rounded-full px-2 py-0.5 text-[10px] font-semibold",
          kill.paused ? "bg-rose-500/15 text-rose-300" : "bg-emerald-500/15 text-emerald-300",
        )}
      >
        {kill.paused ? "PAUSED" : "ACTIVE"}
      </span>
      <span
        className="font-mono text-[10px] text-slate-500"
        title={`Daily: ${kill.breach.daily_loss_pct.toFixed(2)}% of ${kill.breach.daily_loss_limit_pct}% | Trailing: ${kill.breach.trailing_dd_pct.toFixed(2)}% of ${kill.breach.trailing_dd_limit_pct}%`}
      >
        {kill.breach.daily_loss_pct.toFixed(1)}% / {kill.breach.trailing_dd_pct.toFixed(1)}%
      </span>
      {kill.paused ? (
        <button
          type="button"
          onClick={() => onAction("RESUME")}
          disabled={busy !== null || kill.breach.any_breached}
          aria-label="Resume paper trading"
          className="rounded-md border border-emerald-500/30 px-2 py-1 text-[10px] font-medium text-emerald-300 hover:bg-emerald-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-40 min-h-[24px] min-w-[24px]"
        >
          Resume
        </button>
      ) : (
        <button
          type="button"
          onClick={() => onAction("PAUSE")}
          disabled={busy !== null}
          aria-label="Pause paper trading"
          className="rounded-md border border-amber-500/30 px-2 py-1 text-[10px] font-medium text-amber-300 hover:bg-amber-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-40 min-h-[24px] min-w-[24px]"
        >
          Pause
        </button>
      )}
      <button
        type="button"
        onClick={() => onAction("FLATTEN_ALL")}
        disabled={busy !== null}
        aria-label="Flatten all paper positions and pause trading"
        className="rounded-md border border-rose-500/30 px-2 py-1 text-[10px] font-medium text-rose-300 hover:bg-rose-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-40 min-h-[24px] min-w-[24px]"
      >
        Flatten
      </button>
    </div>
  );
}

function CycleSegment({
  fresh,
  latestCycle,
}: {
  fresh: Freshness | null;
  latestCycle: CycleRow | null;
}) {
  if (!fresh) {
    return (
      <div className="flex items-center gap-2">
        <SegmentLabel>Cycle</SegmentLabel>
        <span className="text-xs text-slate-500">—</span>
      </div>
    );
  }
  const bands: Array<{ name: string; band: string }> = [
    { name: "heartbeat", band: fresh.heartbeat.band },
    ...Object.entries(fresh.sources).map(([n, s]) => ({ name: n, band: s.band })),
  ];
  // phase-23.1.12: collapse `unknown` into amber per Google SRE / Azure WAF
  // worst-of-N aggregation convention. Previously a green heartbeat with two
  // unknown sources rendered as green — masking degraded state.
  const worst = bands.some((b) => b.band === "red")
    ? "red"
    : bands.some((b) => b.band === "amber" || b.band === "unknown")
      ? "amber"
      : bands.every((b) => b.band === "green")
        ? "green"
        : "amber";
  const statusLabel = latestCycle?.status ?? "idle";
  return (
    <div
      className="flex items-center gap-2"
      title={bands.map((b) => `${b.name}: ${b.band}`).join(" | ")}
    >
      <SegmentLabel>Cycle</SegmentLabel>
      <div className="flex gap-0.5">
        {bands.map((b) => (
          <span
            key={b.name}
            className={clsx(
              "inline-block h-2 w-2 rounded-full",
              b.band === "green" && "bg-emerald-500",
              b.band === "amber" && "bg-amber-500",
              b.band === "red" && "bg-rose-500",
              (b.band === "unknown" || !b.band) && "bg-slate-600",
            )}
          />
        ))}
      </div>
      <span
        className={clsx(
          "text-xs",
          worst === "red"
            ? "text-rose-300"
            : worst === "amber"
              ? "text-amber-300"
              : worst === "green"
                ? "text-emerald-300"
                : "text-slate-400",
        )}
      >
        {statusLabel}
      </span>
    </div>
  );
}

// phase-16.44: split scheduler into Last + Next segments per user request.
function LastSegment({ lastStartedAt }: { lastStartedAt: string | null }) {
  return (
    <div className="ml-auto flex items-center gap-2">
      <SegmentLabel>Last</SegmentLabel>
      <span className="font-mono text-xs text-slate-300" suppressHydrationWarning>
        {lastStartedAt ? formatRelativeTime(lastStartedAt) : "—"}
      </span>
    </div>
  );
}

function NextSegment({ nextRunAt }: { nextRunAt?: string | null }) {
  if (!nextRunAt) {
    return (
      <div className="flex items-center gap-2">
        <SegmentLabel>Next</SegmentLabel>
        <span className="text-xs text-slate-500">—</span>
      </div>
    );
  }
  const dt = new Date(nextRunAt);
  const label = isNaN(dt.getTime())
    ? nextRunAt
    : dt.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
  return (
    <div className="flex items-center gap-2">
      <SegmentLabel>Next</SegmentLabel>
      <span className="font-mono text-xs text-slate-300">{label}</span>
      <IconCheckCircle size={12} className="text-emerald-500/60" />
    </div>
  );
}

