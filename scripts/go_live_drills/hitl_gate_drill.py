"""HITL Champion/Challenger gate drill (BLOCKER-3 verification).

Exercises the full monthly gate pipeline end-to-end, hermetically:
  1. Gate fires on a past last-trading-Friday with synthetic passing metrics.
  2. Pending row persists to a temp state file (NOT real state).
  3. Slack notification is captured (injected slack_fn).
  4. Approval is recorded.
  5. BQ log row is emitted (captured via injected bq_fn -- no real BQ write).

Prints PASS / FAIL. Exits 0 / 1.
"""
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.autoresearch.monthly_champion_challenger import (
    record_approval,
    run_monthly_sortino_gate,
)


def _synth_returns(n: int, mean: float, seed: int, sigma: float = 0.003) -> list[float]:
    """Deterministic pseudo-Gaussian returns (no numpy dep if unavailable)."""
    import random
    rng = random.Random(seed)
    return [rng.gauss(mean, sigma) for _ in range(n)]


def main() -> int:
    failures: list[str] = []

    # Eval date: 2026-03-27 is the last Friday of March 2026 (a regular NYSE
    # trading Friday; no holiday). Using a PAST date ensures
    # is_last_trading_friday evaluates deterministically regardless of today.
    eval_d = date(2026, 3, 27)
    month_key = "2026-03"
    challenger_id = f"DRILL-{month_key}"

    tmpdir = Path(tempfile.mkdtemp(prefix="hitl_drill_"))
    state_path = tmpdir / "drill_state.json"

    slack_calls: list[tuple[str, dict]] = []
    bq_calls: list[dict] = []

    def slack_capture(msg: str, payload: dict) -> None:
        slack_calls.append((msg, payload))

    try:
        # Synthesize champion vs challenger return series that clear all 3 gates.
        # Low sigma + big mean gap ensures a positive Sortino delta >= 0.3
        # deterministically under seed 1 / 2.
        champ = _synth_returns(30, mean=0.0002, seed=1, sigma=0.002)
        chall = _synth_returns(30, mean=0.004, seed=2, sigma=0.002)
        pbo = 0.10        # < 0.2 threshold
        champ_dd = 0.15   # 15%
        chall_dd = 0.12   # 12% (ratio 0.80 <= 1.2)

        # Fix "now" so the created timestamp + expiry are deterministic for
        # downstream checks; use a point 1h into 2026-03-27.
        fixed_now = datetime(2026, 3, 27, 1, 0, 0, tzinfo=timezone.utc)

        result = run_monthly_sortino_gate(
            eval_d,
            champion_returns=champ,
            challenger_returns=chall,
            champion_max_dd=champ_dd,
            challenger_max_dd=chall_dd,
            challenger_pbo=pbo,
            challenger_id=challenger_id,
            slack_fn=slack_capture,
            state_path=state_path,
            now=fixed_now,
        )

        # Step 1: gate fired
        if not (result.get("fired") and result.get("gate_pass") and result.get("approval_pending")):
            failures.append(f"step1_gate_fired: {result}")
        else:
            print(f"step1_gate_fired: sortino_delta={result['sortino_delta']:.3f} "
                  f"dd_ratio={result['dd_ratio']:.3f} pbo={result['pbo']:.3f}")

        # Step 2: pending row in JSON state
        import json
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pending_row = state.get(month_key) or {}
        if pending_row.get("status") != "pending":
            failures.append(f"step2_pending: expected status=pending, got {pending_row}")
        else:
            print(f"step2_pending: status=pending challenger_id={pending_row.get('challenger_id')} "
                  f"expires_at_iso={pending_row.get('expires_at_iso')}")

        # Slack was captured exactly once
        if len(slack_calls) != 1:
            failures.append(f"slack_calls: expected 1, got {len(slack_calls)}")

        # Step 3: record approval -> terminal transition
        approved_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)  # within 48h window
        updated = record_approval(
            month_key,
            status="approved",
            state_path=state_path,
            now=approved_now,
            bq_fn=bq_calls.append,
        )
        if updated.get("status") != "approved" or not updated.get("resolved_at_iso"):
            failures.append(f"step3_approved: {updated}")
        else:
            print(f"step3_approved: status=approved resolved_at_iso={updated['resolved_at_iso']}")

        # Verify JSON state reflects the transition on disk
        state2 = json.loads(state_path.read_text(encoding="utf-8"))
        if state2.get(month_key, {}).get("status") != "approved":
            failures.append(f"step3_state_on_disk: {state2}")

        # Step 4: BQ row emitted
        if len(bq_calls) != 1:
            failures.append(f"step4_bq_row_written: expected 1 bq_fn call, got {len(bq_calls)}")
        else:
            row = bq_calls[0]
            missing = [k for k in ("strategy_id", "status", "pbo", "deployed_at", "allocation_pct", "notes") if k not in row]
            if missing:
                failures.append(f"step4_bq_row_written: missing keys {missing}: {row}")
            elif row["strategy_id"] != challenger_id or row["status"] != "approved":
                failures.append(f"step4_bq_row_written: bad row: {row}")
            else:
                print(f"step4_bq_row_written: strategy_id={row['strategy_id']} "
                      f"status={row['status']} pbo={row['pbo']} deployed_at={row['deployed_at']}")

    finally:
        # Cleanup temp state
        try:
            if state_path.exists():
                state_path.unlink()
            tmpdir.rmdir()
        except Exception:
            pass

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
