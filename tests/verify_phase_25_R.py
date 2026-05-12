"""phase-25.R verifier -- strategy auto-switching policy (closes red-line goal-c).

Closes phase-24.13 audit F-3 (strategy-switching mechanism does not exist).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_R.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
PROMOTER = REPO / "backend" / "autoresearch" / "promoter.py"
FORMATTERS = REPO / "backend" / "slack_bot" / "formatters.py"
AUTOLOOP = REPO / "backend" / "services" / "autonomous_loop.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PROMOTER, FORMATTERS, AUTOLOOP):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    promoter_text = PROMOTER.read_text(encoding="utf-8")
    formatters_text = FORMATTERS.read_text(encoding="utf-8")
    autoloop_text = AUTOLOOP.read_text(encoding="utf-8")

    # ---- Claim 1: Promoter.write_to_registry signature exists.
    sig = re.search(
        r"def\s+write_to_registry\s*\(\s*self\s*,\s*bq_client:\s*Any\s*,\s*trial:\s*dict\[str\s*,\s*Any\]\s*,\s*\*\s*,\s*week_iso:\s*str\s*,\s*slack_fn:.*?\)\s*->\s*dict\[str\s*,\s*Any\]\s*:",
        promoter_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if sig else "FAIL",
        "promoter_write_to_registry_signature",
        "Promoter must declare write_to_registry(self, bq_client, trial, *, week_iso, slack_fn=None) -> dict[str, Any]",
    ))

    # ---- Claim 2: format_strategy_switch exists.
    fmt_sig = re.search(
        r"def\s+format_strategy_switch\s*\(\s*event:\s*dict\s*\)\s*->\s*list\[dict\]\s*:",
        formatters_text,
    )
    results.append((
        "PASS" if fmt_sig else "FAIL",
        "format_strategy_switch_slack_notification_implemented",
        "formatters.py must declare format_strategy_switch(event: dict) -> list[dict]",
    ))

    # ---- Claim 3: BEHAVIORAL happy path (gate passes, prior active exists).
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.autoresearch.promoter", None)
    from backend.autoresearch.promoter import Promoter  # type: ignore

    happy_ok = False
    happy_err = ""
    try:
        p = Promoter()
        trial = {
            "trial_id": "trial_42",
            "shadow_trading_days": 7,
            "dsr": 1.10,
            "pbo": 0.10,
            "params": {"lookback": 20, "tp_pct": 0.10},
            "sortino_monthly": 0.42,
        }
        fake_bq = MagicMock()
        fake_bq.get_latest_promoted_strategy.return_value = {
            "strategy_id": "trial_31",
            "week_iso": "2026-W18",
            "params": {},
            "dsr": 0.95, "pbo": 0.20, "status": "active",
        }
        fake_slack = MagicMock()

        result = p.write_to_registry(fake_bq, trial, week_iso="2026-W20", slack_fn=fake_slack)

        if not result.get("promoted"):
            happy_err = f"result.promoted is {result.get('promoted')!r}, expected True"
        elif fake_bq.save_promoted_strategy.call_count != 1:
            happy_err = f"save_promoted_strategy called {fake_bq.save_promoted_strategy.call_count} times, expected 1"
        elif fake_bq.update_promoted_strategy_status.call_count != 1:
            happy_err = f"update_promoted_strategy_status called {fake_bq.update_promoted_strategy_status.call_count} times, expected 1 (prior supersede)"
        elif fake_slack.call_count != 1:
            happy_err = f"slack_fn called {fake_slack.call_count} times, expected 1"
        elif result.get("alert_sent") is not True:
            happy_err = f"alert_sent={result.get('alert_sent')!r}, expected True"
        elif result.get("prior_strategy_id") != "trial_31":
            happy_err = f"prior_strategy_id={result.get('prior_strategy_id')!r}"
        elif result.get("new_strategy_id") != "trial_42":
            happy_err = f"new_strategy_id={result.get('new_strategy_id')!r}"
        else:
            written = fake_bq.save_promoted_strategy.call_args.args[0]
            if written.get("status") != "active":
                happy_err = f"row.status was {written.get('status')!r}, expected 'active'"
            else:
                superseded_call = fake_bq.update_promoted_strategy_status.call_args
                if superseded_call.args[1] != "superseded":
                    happy_err = f"update args[1] was {superseded_call.args[1]!r}, expected 'superseded'"
                else:
                    happy_ok = True
    except Exception as e:
        happy_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if happy_ok else "FAIL",
        "promoter_writes_registry_with_status_active_on_gate_clear",
        f"happy-path behavioral: save status=active + supersede prior + slack ({happy_err})",
    ))

    # ---- Claim 4: BEHAVIORAL gate-fail (no writes, no slack).
    gate_fail_ok = False
    gate_fail_err = ""
    try:
        p2 = Promoter()
        trial_fail = {"trial_id": "trial_99", "shadow_trading_days": 3, "dsr": 0.50}
        fake_bq2 = MagicMock()
        fake_slack2 = MagicMock()
        result2 = p2.write_to_registry(fake_bq2, trial_fail, week_iso="2026-W20", slack_fn=fake_slack2)
        if result2.get("promoted") is not False:
            gate_fail_err = f"promoted={result2.get('promoted')!r}, expected False"
        elif fake_bq2.save_promoted_strategy.call_count != 0:
            gate_fail_err = f"save_promoted_strategy called on gate-fail"
        elif fake_slack2.call_count != 0:
            gate_fail_err = f"slack_fn called on gate-fail"
        elif not result2.get("reason"):
            gate_fail_err = "reason missing on gate-fail"
        else:
            gate_fail_ok = True
    except Exception as e:
        gate_fail_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if gate_fail_ok else "FAIL",
        "gate_fail_skips_registry_and_slack",
        f"gate-fail must skip save_promoted_strategy + slack_fn ({gate_fail_err})",
    ))

    # ---- Claim 5: BEHAVIORAL first-promotion (no prior active).
    first_ok = False
    first_err = ""
    try:
        p3 = Promoter()
        trial3 = {
            "trial_id": "trial_first",
            "shadow_trading_days": 6,
            "dsr": 1.0,
            "pbo": 0.15,
            "params": {},
            "sortino_monthly": 0.3,
        }
        fake_bq3 = MagicMock()
        fake_bq3.get_latest_promoted_strategy.return_value = None
        fake_slack3 = MagicMock()
        result3 = p3.write_to_registry(fake_bq3, trial3, week_iso="2026-W20", slack_fn=fake_slack3)
        if not result3.get("promoted"):
            first_err = f"promoted={result3.get('promoted')!r}, expected True"
        elif fake_bq3.save_promoted_strategy.call_count != 1:
            first_err = "save_promoted_strategy not called once"
        elif fake_bq3.update_promoted_strategy_status.call_count != 0:
            first_err = f"update_promoted_strategy_status called {fake_bq3.update_promoted_strategy_status.call_count} times on first promotion (must be 0)"
        elif fake_slack3.call_count != 1:
            first_err = "slack_fn not called once"
        elif result3.get("prior_strategy_id") is not None:
            first_err = f"prior_strategy_id={result3.get('prior_strategy_id')!r}, expected None"
        else:
            first_ok = True
    except Exception as e:
        first_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if first_ok else "FAIL",
        "first_promotion_skips_supersession_and_still_fires_slack",
        f"first-promotion behavioral: no supersede call, save+slack called ({first_err})",
    ))

    # ---- Claim 6: BEHAVIORAL fail-open BQ (save raises).
    fail_open_ok = False
    fail_open_err = ""
    try:
        p4 = Promoter()
        trial4 = {
            "trial_id": "trial_boom",
            "shadow_trading_days": 6,
            "dsr": 1.0,
            "pbo": 0.15,
            "params": {},
        }
        fake_bq4 = MagicMock()
        fake_bq4.get_latest_promoted_strategy.return_value = None
        fake_bq4.save_promoted_strategy.side_effect = RuntimeError("BQ blew up")
        fake_slack4 = MagicMock()
        result4 = p4.write_to_registry(fake_bq4, trial4, week_iso="2026-W20", slack_fn=fake_slack4)
        # Should NOT raise; should return dict.
        if not isinstance(result4, dict):
            fail_open_err = f"result4 not dict: {type(result4)}"
        elif fake_slack4.call_count != 0:
            fail_open_err = f"slack_fn called {fake_slack4.call_count} times AFTER BQ failure (must be 0 to avoid lying about state)"
        elif result4.get("alert_sent") is not False:
            fail_open_err = f"alert_sent={result4.get('alert_sent')!r} after BQ failure, expected False"
        else:
            fail_open_ok = True
    except Exception as e:
        fail_open_err = f"{type(e).__name__}: {e} -- function should fail-open, not raise"

    results.append((
        "PASS" if fail_open_ok else "FAIL",
        "bq_failure_does_not_crash_and_does_not_lie_via_slack",
        f"BQ save failure must fail-open + NOT fire slack ({fail_open_err})",
    ))

    # ---- Claim 7: format_strategy_switch shape.
    fmt_shape_ok = False
    fmt_shape_err = ""
    try:
        sys.modules.pop("backend.slack_bot.formatters", None)
        from backend.slack_bot.formatters import format_strategy_switch  # type: ignore
        blocks = format_strategy_switch({
            "new_strategy_id": "trial_42",
            "prior_strategy_id": "trial_31",
            "dsr": 1.10,
            "pbo": 0.10,
            "allocation_pct": 0.05,
            "switched_at": "2026-05-12T22:00:00+00:00",
            "week_iso": "2026-W20",
        })
        if not isinstance(blocks, list) or len(blocks) < 3:
            fmt_shape_err = f"blocks not a list of >=3 dicts: {type(blocks)}, len={len(blocks) if hasattr(blocks, '__len__') else 'n/a'}"
        elif not any(isinstance(b, dict) and b.get("type") == "header" for b in blocks):
            fmt_shape_err = "no header block found"
        elif "Strategy Auto-Switch" not in str(blocks):
            fmt_shape_err = "'Strategy Auto-Switch' not in output"
        elif "trial_42" not in str(blocks):
            fmt_shape_err = "new_strategy_id not in output"
        elif not any("phase-25.R" in str(b) or "goal-c" in str(b) for b in blocks):
            fmt_shape_err = "phase-25.R / goal-c attribution missing from context footer"
        else:
            fmt_shape_ok = True
    except Exception as e:
        fmt_shape_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if fmt_shape_ok else "FAIL",
        "format_strategy_switch_block_kit_shape",
        f"format_strategy_switch must return >=3-block list with header + new_id + phase-25.R attribution ({fmt_shape_err})",
    ))

    # ---- Claim 8: format_strategy_switch handles None prior gracefully.
    none_prior_ok = False
    none_prior_err = ""
    try:
        blocks2 = format_strategy_switch({
            "new_strategy_id": "trial_alpha",
            "prior_strategy_id": None,
            "dsr": 1.0,
            "pbo": 0.15,
            "allocation_pct": 0.05,
            "switched_at": "2026-05-12T22:00:00+00:00",
            "week_iso": "2026-W20",
        })
        text = str(blocks2)
        if "None" in text:
            none_prior_err = f"literal 'None' leaked into output: {text[:300]!r}"
        elif "first promotion" not in text and "(none" not in text:
            none_prior_err = "no graceful first-promotion sentinel rendered"
        else:
            none_prior_ok = True
    except Exception as e:
        none_prior_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if none_prior_ok else "FAIL",
        "format_strategy_switch_handles_none_prior",
        f"None prior_strategy_id must render gracefully (no literal 'None') ({none_prior_err})",
    ))

    # ---- Claim 9: autonomous_loop.py uses load_promoted_params(bq) as registry source.
    registry_wire = re.search(
        r"best_params\s*=\s*load_promoted_params\(\s*bq\s*\)",
        autoloop_text,
    )
    results.append((
        "PASS" if registry_wire else "FAIL",
        "autonomous_loop_uses_registry_as_primary_strategy_source",
        "autonomous_loop.py must call load_promoted_params(bq) (criterion 2; 25.B3 wire preserved)",
    ))

    # ---- Claim 10: Promoter remains a frozen dataclass (no self-mutation).
    frozen_ok = False
    frozen_err = ""
    try:
        src = inspect.getsource(Promoter.write_to_registry)
        if re.search(r"self\.\w+\s*=", src):
            frozen_err = "self.* assignment detected in write_to_registry (would crash frozen dataclass)"
        else:
            # Also confirm dataclass still frozen=True.
            cls_src = inspect.getsource(Promoter)
            if "frozen=True" not in cls_src:
                frozen_err = "@dataclass(frozen=True) decorator missing from Promoter"
            else:
                frozen_ok = True
    except Exception as e:
        frozen_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if frozen_ok else "FAIL",
        "promoter_remains_frozen_dataclass",
        f"@dataclass(frozen=True) must be preserved and write_to_registry must not mutate self.* ({frozen_err})",
    ))

    # ---- Claim 11: supersession call uses 'superseded' literal.
    superseded_literal = re.search(
        r"update_promoted_strategy_status\(\s*prior_id\s*,\s*[\"']superseded[\"']",
        promoter_text,
    )
    results.append((
        "PASS" if superseded_literal else "FAIL",
        "supersession_uses_superseded_literal",
        "supersession call must pass 'superseded' as the new_status literal",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
