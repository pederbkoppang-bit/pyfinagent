"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { clsx } from "clsx";
import { IconCheckCircle, IconInfo, IconWarning } from "@/lib/icons";
import {
  getPaperCyclesHistory,
  getPaperFreshness,
  getPaperGate,
  getPaperKillSwitchState,
  postPaperKillSwitchAction,
} from "@/lib/api";
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
}

export function OpsStatusBar({ nextRunAt }: Props) {
  const [gate, setGate] = useState<GoLiveGate | null>(null);
  const [kill, setKill] = useState<KillSwitchState | null>(null);
  const [fresh, setFresh] = useState<Freshness | null>(null);
  const [latestCycle, setLatestCycle] = useState<CycleRow | null>(null);
  const [actionBusy, setActionBusy] = useState<"PAUSE" | "FLATTEN_ALL" | "RESUME" | null>(null);
  const failRef = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const [g, k, f, c] = await Promise.all([
        getPaperGate().catch(() => null),
        getPaperKillSwitchState().catch(() => null) as Promise<KillSwitchState | null>,
        getPaperFreshness().catch(() => null) as Promise<Freshness | null>,
        getPaperCyclesHistory(1).catch(() => null),
      ]);
      if (g) setGate(g);
      if (k) setKill(k);
      if (f) setFresh(f);
      if (c?.cycles?.length) setLatestCycle(c.cycles[0] as CycleRow);
      failRef.current = 0;
    } catch {
      failRef.current += 1;
    }
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, 60_000);
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
      <GateSegment gate={gate} />
      <Divider />
      <KillSegment
        kill={kill}
        busy={actionBusy}
        onAction={handleAction}
      />
      <Divider />
      <CycleSegment fresh={fresh} latestCycle={latestCycle} />
      <Divider />
      <SchedulerSegment nextRunAt={nextRunAt} />
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
  return (
    <div className="flex items-center gap-2" title={`${passes}/${total} checks passing`}>
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
          className="rounded-md border border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-300 hover:bg-emerald-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Resume
        </button>
      ) : (
        <button
          type="button"
          onClick={() => onAction("PAUSE")}
          disabled={busy !== null}
          aria-label="Pause paper trading"
          className="rounded-md border border-amber-500/30 px-2 py-0.5 text-[10px] font-medium text-amber-300 hover:bg-amber-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-40"
        >
          Pause
        </button>
      )}
      <button
        type="button"
        onClick={() => onAction("FLATTEN_ALL")}
        disabled={busy !== null}
        aria-label="Flatten all paper positions and pause trading"
        className="rounded-md border border-rose-500/30 px-2 py-0.5 text-[10px] font-medium text-rose-300 hover:bg-rose-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:opacity-40"
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
  const worst = bands.some((b) => b.band === "red")
    ? "red"
    : bands.some((b) => b.band === "amber")
      ? "amber"
      : bands.every((b) => b.band === "green")
        ? "green"
        : "unknown";
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

function SchedulerSegment({ nextRunAt }: { nextRunAt?: string | null }) {
  if (!nextRunAt) {
    return (
      <div className="flex items-center gap-2">
        <SegmentLabel>Scheduler</SegmentLabel>
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
    <div className="ml-auto flex items-center gap-2">
      <SegmentLabel>Next run</SegmentLabel>
      <span className="font-mono text-xs text-slate-300">{label}</span>
      <IconCheckCircle size={12} className="text-emerald-500/60" />
    </div>
  );
}

