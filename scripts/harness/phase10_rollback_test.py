"""phase-10.7 verification CLI: rollback kill-switch.

Three cases matching the masterplan success_criteria:
  1. challenger_dd_breach_auto_demotes
  2. demotion_logged_with_auto_demoted_decision
  3. no_human_approval_required_for_demotion

Each case runs in a tempfile.TemporaryDirectory with injectable now.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.autoresearch.rollback import auto_demote_on_dd_breach


def case_challenger_dd_breach_auto_demotes() -> bool:
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        apath = Path(td) / "audit.jsonl"
        now = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)

        # Breach: dd=-0.11 > threshold 0.10
        r_breach = auto_demote_on_dd_breach(
            challenger_id="strategy_alpha",
            challenger_current_dd=-0.11,
            state_path=spath,
            audit_path=apath,
            now=now,
        )
        # No breach: dd=-0.05 < threshold
        r_ok = auto_demote_on_dd_breach(
            challenger_id="strategy_beta",
            challenger_current_dd=-0.05,
            state_path=spath,
            audit_path=apath,
            now=now,
        )

    ok = (
        r_breach["demoted"] is True
        and r_breach["decision"] == "auto_demoted"
        and r_ok["demoted"] is False
        and r_ok["decision"] == "no_breach"
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] challenger_dd_breach_auto_demotes  "
        f"(breach={r_breach['demoted']}/{r_breach['decision']}, "
        f"ok={r_ok['demoted']}/{r_ok['decision']})"
    )
    return ok


def case_demotion_logged_with_auto_demoted_decision() -> bool:
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        apath = Path(td) / "audit.jsonl"
        now = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)

        auto_demote_on_dd_breach(
            challenger_id="strategy_gamma",
            challenger_current_dd=-0.15,
            state_path=spath,
            audit_path=apath,
            now=now,
        )

        audit_lines = apath.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in audit_lines]
        state = json.loads(spath.read_text(encoding="utf-8"))

    ok = (
        len(records) == 1
        and records[0]["decision"] == "auto_demoted"
        and records[0]["challenger_id"] == "strategy_gamma"
        and state.get("demotions", {}).get("strategy_gamma", {}).get("status") == "auto_demoted"
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] demotion_logged_with_auto_demoted_decision  "
        f"(audit_records={len(records)}, decision={records[0]['decision'] if records else None})"
    )
    return ok


def case_no_human_approval_required_for_demotion() -> bool:
    """The function signature must expose no slack_fn / approver kwargs,
    and the call must complete without any HITL interaction."""
    sig = inspect.signature(auto_demote_on_dd_breach)
    params = sig.parameters

    has_slack = any("slack" in name.lower() for name in params)
    has_approver = any("approv" in name.lower() for name in params)

    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        apath = Path(td) / "audit.jsonl"
        now = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)

        # No slack_fn / approver_id kwargs are needed or accepted -- a bare call succeeds.
        r = auto_demote_on_dd_breach(
            challenger_id="strategy_delta",
            challenger_current_dd=-0.20,
            state_path=spath,
            audit_path=apath,
            now=now,
        )

    ok = (
        not has_slack
        and not has_approver
        and r["demoted"] is True
        and r["decision"] == "auto_demoted"
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] no_human_approval_required_for_demotion  "
        f"(has_slack_kwarg={has_slack}, has_approver_kwarg={has_approver})"
    )
    return ok


def main() -> int:
    results = [
        case_challenger_dd_breach_auto_demotes(),
        case_demotion_logged_with_auto_demoted_decision(),
        case_no_human_approval_required_for_demotion(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
