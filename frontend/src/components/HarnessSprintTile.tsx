/**
 * phase-10.9 Harness-tab sprint-state tile.
 *
 * Read-only surface over the phase-10 autoresearch sprint state:
 *   - Thursday batch kicks (phase-10.3)
 *   - Friday promotion gate (phase-10.4)
 *   - Monthly Champion/Challenger Sortino delta + HITL status (phase-10.6)
 *
 * The tile itself fetches nothing; the parent (HarnessDashboard) passes a
 * HarnessSprintWeekState snapshot. Contract: NO mutation controls.
 * See frontend-layout.md BentoCard + empty-state patterns.
 */
import {
  IconCheckCircle,
  IconWarning,
  IconTimer,
  IconChart,
} from "@/lib/icons";
import type { HarnessSprintWeekState } from "@/lib/types";

export interface HarnessSprintTileProps {
  data: HarnessSprintWeekState | null;
}

function formatDelta(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(3)}`;
}

export function HarnessSprintTile({ data }: HarnessSprintTileProps) {
  if (data === null) {
    return (
      <section
        aria-label="Sprint state"
        className="rounded-xl border border-navy-700 bg-navy-800/60 p-5"
      >
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <IconTimer
            size={32}
            weight="duotone"
            className="text-slate-600"
            aria-hidden="true"
          />
          <p className="mt-3 text-sm text-slate-400">
            No sprint activity yet
          </p>
          <p className="mt-1 text-xs text-slate-600">
            Thursday batch and Friday promotion will appear here once the
            first sprint runs.
          </p>
        </div>
      </section>
    );
  }

  const { weekIso, thu, fri, monthly } = data;

  const monthlyColorClass = monthly?.approved
    ? "text-emerald-400"
    : monthly?.approvalPending
      ? "text-amber-400"
      : "text-slate-400";

  return (
    <section
      aria-label="Sprint state"
      data-week-iso={weekIso}
      className="rounded-xl border border-navy-700 bg-navy-800/60 p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Sprint
        </h3>
        <span className="font-mono text-xs text-slate-500">{weekIso}</span>
      </div>

      {/* Weekly state (Thu + Fri) */}
      <div
        data-section="weekly-state"
        className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2"
      >
        <div className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
            Thursday batch
          </p>
          {thu === null ? (
            <p className="mt-2 text-sm text-slate-500">Not fired yet</p>
          ) : (
            <>
              <p
                data-cell="thu-candidates"
                className="mt-1 text-2xl font-bold text-slate-100"
              >
                {thu.candidatesKicked.toLocaleString()}
              </p>
              <p className="mt-1 font-mono text-xs text-slate-500">
                {thu.batchId.slice(0, 8)}
              </p>
            </>
          )}
        </div>

        <div className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
            Friday promotion
          </p>
          {fri === null ? (
            <p className="mt-2 text-sm text-slate-500">Not fired yet</p>
          ) : (
            <>
              <p
                data-cell="fri-promoted-count"
                className="mt-1 text-2xl font-bold text-emerald-400"
              >
                {fri.promotedIds.length}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {fri.promotedIds.length} promoted · {fri.rejectedIds.length}{" "}
                rejected
              </p>
            </>
          )}
        </div>
      </div>

      {/* Monthly Sortino delta */}
      <div
        data-section="monthly-state"
        className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4"
      >
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
            Monthly Champion/Challenger
          </p>
          {monthly === null ? (
            <IconTimer
              size={16}
              weight="duotone"
              className="text-slate-600"
              aria-hidden="true"
            />
          ) : monthly.approved ? (
            <IconCheckCircle
              size={16}
              weight="fill"
              className="text-emerald-400"
              aria-hidden="true"
            />
          ) : monthly.approvalPending ? (
            <IconWarning
              size={16}
              weight="fill"
              className="text-amber-400"
              aria-hidden="true"
            />
          ) : (
            <IconChart
              size={16}
              weight="duotone"
              className="text-slate-500"
              aria-hidden="true"
            />
          )}
        </div>
        {monthly === null ? (
          <p className="mt-2 text-sm text-slate-500">
            Awaiting last trading Friday of month
          </p>
        ) : (
          <div className="mt-2 flex items-baseline gap-3">
            <span
              data-cell="sortino-delta"
              className={`text-2xl font-bold ${monthlyColorClass}`}
            >
              {formatDelta(monthly.sortinoDelta)}
            </span>
            <span className="text-xs text-slate-500">
              {monthly.approved
                ? "approved"
                : monthly.approvalPending
                  ? "awaiting approval"
                  : "rejected"}
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
