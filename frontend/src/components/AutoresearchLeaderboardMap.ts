import type { LeaderboardCandidate } from "./AutoresearchLeaderboard";

// The shape the optimizer experiments endpoint returns (subset).
// Kept intentionally loose so we don't couple to the exact TSV schema.
export interface OptimizerExperimentLike {
  run_id: string;
  param_changed: string | null;
  dsr: number | string | null;
  pbo?: number | null;
  metric_after: number | string | null;
  status: string | null;
}

const STARTING_CAPITAL_DEFAULT = 100_000;

function numOrNull(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

// Translate optimizer experiments into leaderboard candidates. Must use the
// backend-provided `pbo` field -- a production regression that hardcodes null
// (instead of passing through real values when available) is the failure mode
// qa-evaluator flagged in Cycle 73.
export function mapExperimentsToCandidates(
  experiments: OptimizerExperimentLike[],
  opts?: { startingCapital?: number },
): LeaderboardCandidate[] {
  const cap = opts?.startingCapital ?? STARTING_CAPITAL_DEFAULT;
  return experiments.map((exp, index) => {
    const sharpe = numOrNull(exp.metric_after);
    return {
      index,
      run_id: exp.run_id,
      param_changed: exp.param_changed ?? null,
      dsr: numOrNull(exp.dsr),
      pbo: exp.pbo === undefined ? null : numOrNull(exp.pbo),
      realized_pnl_if_promoted: sharpe == null ? null : sharpe * cap,
      starting_capital: cap,
      status: (exp.status ?? "").toString().toLowerCase() || "pending",
    };
  });
}
