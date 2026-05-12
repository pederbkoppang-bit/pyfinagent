"""phase-25.C3 verifier -- strategy registry status field + actual_replacement.

Closes phase-24.3 audit F-4 (monthly_champion_challenger.py:75 + 263 hard-coded
`actual_replacement=False`; no BQ status flip on monthly HITL approval).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_C3.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "scripts" / "migrations" / "create_promoted_strategies_table.py"
SETTINGS = REPO / "backend" / "config" / "settings.py"
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"
MONTHLY = REPO / "backend" / "autoresearch" / "monthly_champion_challenger.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (MIGRATION, SETTINGS, BQ_CLIENT, MONTHLY):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    mig_text = MIGRATION.read_text(encoding="utf-8")
    settings_text = SETTINGS.read_text(encoding="utf-8")
    bq_text = BQ_CLIENT.read_text(encoding="utf-8")
    monthly_text = MONTHLY.read_text(encoding="utf-8")

    # ---- Claim 1: Settings.real_capital_enabled default False.
    flag = re.search(
        r"real_capital_enabled\s*:\s*bool\s*=\s*Field\(\s*False\s*,",
        settings_text,
    )
    results.append((
        "PASS" if flag else "FAIL",
        "settings_real_capital_enabled_defaults_false",
        "Settings.real_capital_enabled must be declared with Field(False, ...)",
    ))

    # ---- Claim 2: BigQueryClient.update_promoted_strategy_status signature.
    bq_sig = re.search(
        r"def\s+update_promoted_strategy_status\s*\(\s*self\s*,\s*strategy_id:\s*str\s*,\s*new_status:\s*str\s*,\s*\*\s*,\s*week_iso:\s*str\s*\|\s*None\s*=\s*None\s*,?\s*\)\s*->\s*None\s*:",
        bq_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if bq_sig else "FAIL",
        "bq_update_promoted_strategy_status_signature",
        "BigQueryClient must declare update_promoted_strategy_status(strategy_id, new_status, *, week_iso=None) -> None",
    ))

    # ---- Claim 3: UPDATE SQL parameterized + result(timeout=30) inside helper.
    fn_match = re.search(
        r"def\s+update_promoted_strategy_status\s*\(.*?\)(.*?)(?=\n    def\s|\nclass\s|\Z)",
        bq_text,
        re.DOTALL,
    )
    parameterized = False
    has_timeout = False
    if fn_match:
        body = fn_match.group(1)
        parameterized = (
            "@new_status" in body
            and "@strategy_id" in body
            and "UPDATE" in body
            and "SET status = @new_status" in body
        )
        has_timeout = "result(timeout=30)" in body
    results.append((
        "PASS" if parameterized and has_timeout else "FAIL",
        "bq_update_uses_parameterized_sql_and_timeout_30",
        "update_promoted_strategy_status must use @new_status/@strategy_id named params + result(timeout=30)",
    ))

    # ---- Claim 4: run_monthly_sortino_gate accepts real_capital_enabled kwarg.
    sig_has_flag = re.search(
        r"def\s+run_monthly_sortino_gate\(.*?real_capital_enabled:\s*bool\s*=\s*False.*?\)\s*->\s*dict",
        monthly_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if sig_has_flag else "FAIL",
        "run_monthly_sortino_gate_accepts_real_capital_enabled_kwarg",
        "run_monthly_sortino_gate must accept real_capital_enabled: bool = False",
    ))

    # ---- Claim 5: no hardcoded {"actual_replacement": False} in result dict.
    hardcoded_result = re.search(r'"actual_replacement":\s*False\s*,', monthly_text)
    derived_assignment = re.search(
        r'actual_replacement\s*=\s*bool\(real_capital_enabled\)',
        monthly_text,
    )
    results.append((
        "PASS" if not hardcoded_result and derived_assignment else "FAIL",
        "actual_replacement_no_longer_hardcoded_false",
        "result dict must no longer hardcode \"actual_replacement\": False; derivation from real_capital_enabled required",
    ))

    # ---- Claim 6: no hardcoded actual_replacement=False literal in notes f-string.
    hardcoded_notes = "actual_replacement=False" in monthly_text and re.search(
        r'f"month=.*?actual_replacement=False"', monthly_text, re.DOTALL,
    )
    # The notes string should reference the variable (curly braces) instead.
    derived_notes = re.search(
        r"f['\"][^'\"]*actual_replacement=\{actual_replacement\}[^'\"]*['\"]",
        monthly_text,
    )
    results.append((
        "PASS" if not hardcoded_notes and derived_notes else "FAIL",
        "deployment_log_notes_uses_derived_actual_replacement",
        "_emit_deployment_log_row notes must use f-string interpolation of actual_replacement (not literal False)",
    ))

    # ---- Claim 7: record_approval signature accepts status_update_fn.
    rec_sig = re.search(
        r"def\s+record_approval\(.*?status_update_fn:\s*Callable\[\[str\s*,\s*str\]\s*,\s*None\]\s*\|\s*None\s*=\s*None.*?\)\s*->\s*dict",
        monthly_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if rec_sig else "FAIL",
        "record_approval_accepts_status_update_fn_kwarg",
        "record_approval must accept status_update_fn: Callable[[str, str], None] | None = None",
    ))

    # ---- Behavioral fixtures: import the module and run record_approval + run_monthly_sortino_gate
    # with monkey-patched state path.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.autoresearch.monthly_champion_challenger", None)
    from backend.autoresearch import monthly_champion_challenger as mcc  # type: ignore

    # Helper: seed a state file with a pending row.
    def _seed_state_with_pending(month_key: str = "2026-05") -> Path:
        td = Path(tempfile.mkdtemp(prefix="phase25c3_"))
        state_path = td / "state.json"
        now_iso = datetime.now(timezone.utc).isoformat()
        future_iso = (datetime.now(timezone.utc).replace(year=2099)).isoformat()
        state = {
            month_key: {
                "month": month_key,
                "created_at_iso": now_iso,
                "expires_at_iso": future_iso,
                "status": "pending",
                "sortino_delta": 0.5,
                "dd_ratio": 1.0,
                "pbo": 0.15,
                "challenger_id": "challenger_42",
                "actual_replacement": False,
            }
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    # ---- Claim 8: BEHAVIORAL approval flip -- status_update_fn called once with (challenger_id, "active").
    approval_ok = False
    approval_err = ""
    try:
        state_path = _seed_state_with_pending("2026-05")
        fake_fn = MagicMock()
        result_row = mcc.record_approval(
            "2026-05",
            status="approved",
            state_path=state_path,
            status_update_fn=fake_fn,
        )
        if result_row.get("status") != "approved":
            approval_err = f"state row status is {result_row.get('status')!r}, expected approved"
        elif fake_fn.call_count != 1:
            approval_err = f"status_update_fn called {fake_fn.call_count} times, expected 1"
        else:
            args = fake_fn.call_args.args
            if args != ("challenger_42", "active"):
                approval_err = f"status_update_fn args were {args!r}, expected ('challenger_42', 'active')"
            else:
                approval_ok = True
    except Exception as e:
        approval_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if approval_ok else "FAIL",
        "monthly_approval_flips_status_from_shadow_to_active",
        f"record_approval(approved) must invoke status_update_fn(challenger_id, 'active') exactly once ({approval_err})",
    ))

    # ---- Claim 9: BEHAVIORAL rejection -- status_update_fn NOT called.
    rejection_ok = False
    rejection_err = ""
    try:
        state_path2 = _seed_state_with_pending("2026-06")
        fake_fn2 = MagicMock()
        result_row2 = mcc.record_approval(
            "2026-06",
            status="rejected",
            state_path=state_path2,
            status_update_fn=fake_fn2,
        )
        if result_row2.get("status") != "rejected":
            rejection_err = f"state row status is {result_row2.get('status')!r}, expected rejected"
        elif fake_fn2.call_count != 0:
            rejection_err = f"status_update_fn called {fake_fn2.call_count} times on rejection (must be 0)"
        else:
            rejection_ok = True
    except Exception as e:
        rejection_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if rejection_ok else "FAIL",
        "record_approval_rejection_does_not_flip_status",
        f"record_approval(rejected) must NOT invoke status_update_fn ({rejection_err})",
    ))

    # ---- Claim 10: BEHAVIORAL derived actual_replacement (flag True -> True; flag False -> False).
    derived_ok = False
    derived_err = ""
    try:
        # The gate uses calendar checks; we need an eval_date that is the
        # last trading Friday of its month so the path proceeds past
        # criterion 1. Use the module's own helper to find one.
        def _find_last_trading_friday(year: int) -> date:
            for month in range(12, 0, -1):
                # search backwards from end of year
                for day in range(31, 0, -1):
                    try:
                        d = date(year, month, day)
                    except ValueError:
                        continue
                    if mcc.is_last_trading_friday(d):
                        return d
            raise RuntimeError("no last trading Friday in year")

        eval_date = _find_last_trading_friday(2026)
        td = Path(tempfile.mkdtemp(prefix="phase25c3_gate_"))
        # Returns large enough for sortino + the gate to fire.
        chall_returns = [0.01] * 22 + [0.015] * 5
        champ_returns = [0.005] * 27

        # Real-capital flag TRUE.
        r_on = mcc.run_monthly_sortino_gate(
            eval_date,
            champion_returns=champ_returns,
            challenger_returns=chall_returns,
            champion_max_dd=0.10,
            challenger_max_dd=0.10,
            challenger_pbo=0.10,
            state_path=td / "on.json",
            real_capital_enabled=True,
        )
        # Real-capital flag FALSE (default).
        r_off = mcc.run_monthly_sortino_gate(
            eval_date,
            champion_returns=champ_returns,
            challenger_returns=chall_returns,
            champion_max_dd=0.10,
            challenger_max_dd=0.10,
            challenger_pbo=0.10,
            state_path=td / "off.json",
            real_capital_enabled=False,
        )
        if r_on.get("actual_replacement") is not True:
            derived_err = f"flag=True case got actual_replacement={r_on.get('actual_replacement')!r}"
        elif r_off.get("actual_replacement") is not False:
            derived_err = f"flag=False case got actual_replacement={r_off.get('actual_replacement')!r}"
        else:
            derived_ok = True
    except Exception as e:
        derived_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if derived_ok else "FAIL",
        "behavioral_actual_replacement_derived_from_real_capital_flag",
        f"actual_replacement must track real_capital_enabled (True->True, False->False) ({derived_err})",
    ))

    # ---- Claim 11: BEHAVIORAL BQ UPDATE round-trip.
    rt_ok = False
    rt_err = ""
    try:
        sys.modules.pop("backend.db.bigquery_client", None)
        from backend.db import bigquery_client as bqc  # type: ignore

        fake_client = MagicMock()
        fake_query_job = MagicMock()
        fake_client.query.return_value = fake_query_job

        bq_instance = bqc.BigQueryClient.__new__(bqc.BigQueryClient)
        bq_instance.client = fake_client
        fake_settings = MagicMock()
        fake_settings.gcp_project_id = "test-project"
        bq_instance.settings = fake_settings

        bq_instance.update_promoted_strategy_status(
            "trial_42", "active", week_iso="2026-W20",
        )
        if fake_client.query.call_count != 1:
            rt_err = f"client.query called {fake_client.query.call_count} times, expected 1"
        else:
            call_args = fake_client.query.call_args
            sql = call_args.args[0] if call_args.args else ""
            if "SET status = @new_status" not in sql:
                rt_err = "SQL did not contain SET status = @new_status"
            elif "WHERE strategy_id = @strategy_id" not in sql:
                rt_err = "SQL did not contain WHERE strategy_id = @strategy_id"
            elif "AND week_iso = @week_iso" not in sql:
                rt_err = "SQL did not contain AND week_iso = @week_iso (week_iso pinned path)"
            elif fake_query_job.result.call_count != 1:
                rt_err = f"job.result called {fake_query_job.result.call_count} times"
            elif fake_query_job.result.call_args.kwargs.get("timeout") != 30:
                rt_err = f"result(timeout=) was {fake_query_job.result.call_args.kwargs}"
            else:
                rt_ok = True
    except Exception as e:
        rt_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if rt_ok else "FAIL",
        "behavioral_bq_update_status_round_trip",
        f"update_promoted_strategy_status must execute parameterized UPDATE + result(timeout=30) ({rt_err})",
    ))

    # ---- Claim 12: migration schema mentions active + at least one retired-equivalent.
    has_active = "active" in mig_text
    has_retired_eq = any(
        v in mig_text for v in ("superseded", "retired", "paused", "rolled_back")
    )
    results.append((
        "PASS" if has_active and has_retired_eq else "FAIL",
        "strategy_registry_table_has_status_field_active_shadow_retired",
        "migration must document status enum including 'active' + a retired-equivalent (superseded/retired/paused/rolled_back)",
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
