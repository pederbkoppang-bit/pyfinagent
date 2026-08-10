"""phase-9.6 Nightly outcome-tracker rebuild from trade ledger."""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat
from backend.services.recommendation_vocab import resolve_outcome_recommendation  # phase-86.25

logger = logging.getLogger(__name__)
JOB_NAME = "nightly_outcome_rebuild"


def run(
    *,
    ledger_fetch_fn: Callable[[], list[dict]] | None = None,
    outcome_write_fn: Callable[[list[dict]], int] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    key = IdempotencyKey.daily(JOB_NAME, day=day)
    result: dict[str, Any] = {"rebuilt": 0, "key": key, "skipped": False}
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        trades = (ledger_fetch_fn or _default_fetch)()
        outcomes = _compute_outcomes(trades)
        try:
            n = (outcome_write_fn or _default_write)(outcomes)
        except Exception as exc:
            logger.warning("outcome_rebuild: write fail-open: %r", exc)
            n = 0
        result["rebuilt"] = int(n)
    return result


def _compute_outcomes(trades: list[dict]) -> list[dict]:
    """Per-trade outcome: pnl > 0 -> win, <= 0 -> loss.

    phase-82.48 NULL-SAFETY. `t.get("pnl", 0.0)` returns the default only when
    the KEY IS ABSENT -- for a key present with value `None` it returns `None`,
    and `None > 0` raises TypeError. This function is called OUTSIDE the `try`
    that wraps the write, so the raise escapes into `heartbeat`.

    HONEST SCOPE, because the fix should not be read as a live-bug repair: this
    was NOT reachable in production. The fetch keeps NULLs out with
    `realized_pnl_pct IS NOT NULL`, and `heartbeat` is a @contextmanager whose
    `except` does not re-raise, so contextlib would have suppressed it anyway.
    It was latent, guarded only by an un-asserted SQL predicate.

    ADJACENT AND NOT FIXED HERE: a FLOAT `NaN` passes `IS NOT NULL`, and
    `nan > 0` is False, so a NaN is silently graded "loss".

    Also carries through the fields the 9-column destination needs; the write
    function owns the schema mapping.
    """
    out: list[dict] = []
    for t in (trades or []):
        pnl = t.get("pnl")
        if pnl is None:
            pnl = 0.0
        # `analysis_date` and `recommendation` are REQUIRED at the destination,
        # so neither may fall through as None: analysis_id is the recommendation
        # identity and created_at is the SELL timestamp.
        #
        # CORRECTED phase-86.25: this comment used to continue "...risk_judge_decision
        # is the decision that was acted on, and the trade action is the fallback",
        # which described the line below as it was AND asserted that the fallback was
        # legitimate. It was not -- the trade action is not a recommendation -- and a
        # comment blessing the defect is how it survived review. Left as a correction
        # rather than a deletion so the next reader can see what was believed.
        analysis_date = t.get("analysis_id") or t.get("created_at")
        # phase-86.25 (S2). WAS: `t.get("risk_judge_decision") or t.get("action")`.
        # THIS is the seam that wrote the three live 'SELL'-spelled
        # outcome_tracking rows: `risk_judge_decision` is empty on 32/32 SELL
        # trades (MEASURED), so the `or` fell through to the trade ACTION and a
        # literal "SELL" was persisted where an ANALYST RECOMMENDATION was
        # expected. An action is not a recommendation -- the trade going out is
        # not a call that it should. Unlike S1 this job is NOT behind
        # `paper_learn_loop_enabled`; it runs on cron at 04:00 UTC, so this line
        # has been producing mislabelled rows in production.
        #
        # Resolved at the boundary now: a real analyst recommendation or an
        # explicit UNKNOWN. Row COUNT is unchanged -- UNKNOWN is truthy, so the
        # skip below still does not fire and no close is silently dropped; only
        # the label changes, from a fabricated direction to an honest absence.
        # WHY THE (A) BRANCH IS DEAD (added cycle 2, Q/A finding W1).
        # CORRECTED AGAIN cycle 3: the first version of this block was pasted
        # verbatim from autonomous_loop.py and opened "An earlier version of this
        # comment blamed the unreachable ANCHOR". That was FALSE OF THIS FILE --
        # this file's comment never mentioned the anchor. A copy-pasted claim
        # about a file's own history is a fresh false statement, introduced by
        # the very commit that was fixing a false-statement finding.
        #
        # The substance, which does apply here: the anchor measurements
        # (analysis_id empty 32/32, round_trip_id one-sided) are real but are NOT
        # why this branch is dead, and citing them would tell a future reader
        # that fixing round_trip_id makes this path resolve. IT WOULD NOT.
        #
        # MEASURED: `analyst_recommendation` is not a column of paper_trades at all (18
        # columns), and `_production_fns.LEDGER_FETCH_SQL` selects ten named columns,
        # none of them this one. The lookup below therefore reads a dict key that NO
        # PRODUCER EMITS: the branch is dead BY CONSTRUCTION, not by a missing join.
        #
        # CONSEQUENCE, stated so nobody credits it as coverage: making this resolve
        # needs a PRODUCER change -- some writer must start emitting an analyst
        # recommendation onto the trade -- which is a separate step. The call is kept
        # because it is the correct boundary SHAPE (a caller hands over a real
        # recommendation or nothing), not because it currently does anything.
        recommendation = resolve_outcome_recommendation(t.get("analyst_recommendation"))
        if not analysis_date or not recommendation:
            # SKIP rather than fabricate. BigQuery's REQUIRED mode rejects NULL
            # but ACCEPTS an empty string, so a `or ""` fallback would happily
            # insert an outcome with no identity -- polluting the table that
            # get_performance_stats serves to four consumers. Caught by this
            # step's own guard (test_required_columns_are_never_None); the first
            # draft did exactly that.
            logger.warning(
                "outcome_rebuild: skipping trade %r -- no usable %s",
                t.get("trade_id"),
                "analysis anchor" if not analysis_date else "recommendation",
            )
            continue
        out.append({
            "trade_id": t.get("trade_id"),
            "ticker": t.get("ticker"),
            "pnl": pnl,
            "outcome": "win" if pnl > 0 else "loss",
            "analysis_date": str(analysis_date),
            "recommendation": str(recommendation),
            "price": t.get("price"),
            "holding_days": t.get("holding_days"),
        })
    return out


def _default_fetch() -> list[dict]:
    return []  # production reads pyfinagent_pms.paper_trades


def _default_write(outcomes: list[dict]) -> int:
    return len(outcomes)


__all__ = ["run", "JOB_NAME"]
