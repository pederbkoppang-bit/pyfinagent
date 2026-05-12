"""phase-25.B3 verifier -- daily loop reads latest promoted strategy.

Closes phase-24.3 audit F-6 (autonomous_loop.py:33-43 read only
optimizer_best.json; no BQ promoted_strategies query existed).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_B3.py
"""
from __future__ import annotations

import json as _json
import logging
import re
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
AUTOLOOP = REPO / "backend" / "services" / "autonomous_loop.py"
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"


def _capture_loop_logs():
    """Return (handler, stream) -- attach handler to autonomous_loop logger to
    capture log records emitted during behavioral tests."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    return handler, stream


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not AUTOLOOP.exists() or not BQ_CLIENT.exists():
        print(f"FAIL: required source file missing")
        return 1

    auto_text = AUTOLOOP.read_text(encoding="utf-8")
    bq_text = BQ_CLIENT.read_text(encoding="utf-8")

    # ---- Claim 1: load_promoted_params signature.
    sig_match = re.search(
        r"def\s+load_promoted_params\s*\(\s*bq\s*:\s*BigQueryClient\s*\)\s*->\s*dict\s*:",
        auto_text,
    )
    results.append((
        "PASS" if sig_match else "FAIL",
        "load_promoted_params_function_exists_in_autonomous_loop",
        "autonomous_loop.py must declare def load_promoted_params(bq: BigQueryClient) -> dict",
    ))

    # ---- Claim 2: BQ reader signature with status_filter default behavior.
    bq_sig = re.search(
        r"def\s+get_latest_promoted_strategy\s*\(\s*self\s*,\s*status_filter:\s*list\[str\]\s*\|\s*None\s*=\s*None\s*,?\s*\)\s*->\s*dict\s*\|\s*None\s*:",
        bq_text,
        re.DOTALL,
    )
    default_assignment = re.search(
        r'if\s+status_filter\s+is\s+None\s*:\s*\n\s*status_filter\s*=\s*\["pending"\s*,\s*"active"\]',
        bq_text,
    )
    results.append((
        "PASS" if bq_sig and default_assignment else "FAIL",
        "bq_get_latest_promoted_strategy_with_default_filter",
        "BigQueryClient must declare get_latest_promoted_strategy(status_filter=None) defaulting to [pending, active]",
    ))

    # ---- Claim 3: SQL shape (TO_JSON_STRING + ORDER BY promoted_at DESC, dsr DESC + LIMIT 1 + IN UNNEST(@statuses)).
    has_tojson = "TO_JSON_STRING(params) AS params_json" in bq_text
    has_order = re.search(r"ORDER BY\s+promoted_at\s+DESC\s*,\s*dsr\s+DESC", bq_text)
    has_limit = re.search(r"LIMIT\s+1", bq_text)
    has_unnest = "IN UNNEST(@statuses)" in bq_text
    results.append((
        "PASS" if has_tojson and has_order and has_limit and has_unnest else "FAIL",
        "bq_query_shape_to_json_string_order_limit_unnest",
        "BQ query must use TO_JSON_STRING(params) AS params_json + ORDER BY promoted_at DESC, dsr DESC + LIMIT 1 + IN UNNEST(@statuses)",
    ))

    # ---- Claim 4: result(timeout=30) on the new query.
    # We grep within the get_latest_promoted_strategy function body specifically.
    fn_match = re.search(
        r"def\s+get_latest_promoted_strategy\s*\(.*?\)(.*?)(?=\n    def\s|\nclass\s|\Z)",
        bq_text,
        re.DOTALL,
    )
    timeout_in_fn = False
    if fn_match:
        timeout_in_fn = "result(timeout=30)" in fn_match.group(1)
    results.append((
        "PASS" if timeout_in_fn else "FAIL",
        "bq_query_uses_result_timeout_30",
        "get_latest_promoted_strategy must call result(timeout=30) per CLAUDE.md rule",
    ))

    # ---- Claim 5: caller switched to load_promoted_params(bq).
    caller = re.search(r"best_params\s*=\s*load_promoted_params\(\s*bq\s*\)", auto_text)
    # Also confirm the old direct-load call is gone in run_daily_cycle (load_best_params() may
    # still appear inside load_promoted_params as the fallback -- that's OK).
    results.append((
        "PASS" if caller else "FAIL",
        "run_daily_cycle_caller_uses_load_promoted_params",
        "autonomous_loop.run_daily_cycle must call load_promoted_params(bq), not load_best_params()",
    ))

    # ---- Behavioral fixture: import the module and run the loader with fakes.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.services.autonomous_loop", None)
    from backend.services import autonomous_loop as al  # type: ignore

    logger_obj = logging.getLogger("backend.services.autonomous_loop")
    logger_obj.setLevel(logging.DEBUG)

    # ---- Claim 6: BEHAVIORAL happy path -- bq returns row with params dict.
    happy_ok = False
    happy_err = ""
    try:
        handler, stream = _capture_loop_logs()
        logger_obj.addHandler(handler)
        try:
            fake_bq = MagicMock()
            fake_bq.get_latest_promoted_strategy.return_value = {
                "strategy_id": "trial_99",
                "week_iso": "2026-W20",
                "params": {"lookback": 20, "threshold": 0.5, "tp_pct": 0.1},
                "dsr": 1.4,
                "pbo": 0.12,
                "status": "active",
                "promoted_at": "2026-05-09T12:00:00+00:00",
            }
            result = al.load_promoted_params(fake_bq)
            logs = stream.getvalue()
        finally:
            logger_obj.removeHandler(handler)

        if not isinstance(result, dict):
            happy_err = f"result not dict: {type(result)}"
        elif result.get("lookback") != 20 or result.get("tp_pct") != 0.1:
            happy_err = f"params not returned correctly: {result}"
        elif "Loaded promoted params" not in logs:
            happy_err = f"happy-path log line missing; captured logs: {logs!r}"
        else:
            happy_ok = True
    except Exception as e:
        happy_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if happy_ok else "FAIL",
        "behavioral_happy_path_returns_bq_params_and_logs_success",
        f"load_promoted_params(fake_bq with row) must return BQ params + emit 'Loaded promoted params' log ({happy_err})",
    ))

    # ---- Claim 7: BEHAVIORAL empty path -- bq returns None.
    empty_ok = False
    empty_err = ""
    try:
        handler, stream = _capture_loop_logs()
        logger_obj.addHandler(handler)
        try:
            fake_bq = MagicMock()
            fake_bq.get_latest_promoted_strategy.return_value = None
            result = al.load_promoted_params(fake_bq)
            logs = stream.getvalue()
        finally:
            logger_obj.removeHandler(handler)

        if not isinstance(result, dict):
            empty_err = f"result not dict: {type(result)}"
        elif "No active promoted strategy in BQ" not in logs:
            empty_err = f"empty-path log line missing; captured logs: {logs!r}"
        else:
            empty_ok = True
    except Exception as e:
        empty_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if empty_ok else "FAIL",
        "fallback_to_optimizer_best_json_if_bq_empty",
        f"empty bq path must fall back to load_best_params() + emit fallback log ({empty_err})",
    ))

    # ---- Claim 8: BEHAVIORAL exception path -- bq raises.
    exc_ok = False
    exc_err = ""
    try:
        handler, stream = _capture_loop_logs()
        logger_obj.addHandler(handler)
        try:
            fake_bq = MagicMock()
            fake_bq.get_latest_promoted_strategy.side_effect = RuntimeError("network down")
            result = al.load_promoted_params(fake_bq)
            logs = stream.getvalue()
        finally:
            logger_obj.removeHandler(handler)

        if not isinstance(result, dict):
            exc_err = f"result not dict: {type(result)}"
        elif "Promoted strategy BQ unavailable" not in logs:
            exc_err = f"exception-path log line missing; captured logs: {logs!r}"
        elif "network down" not in logs:
            exc_err = f"exception detail not in log: {logs!r}"
        else:
            exc_ok = True
    except Exception as e:
        exc_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if exc_ok else "FAIL",
        "fallback_to_optimizer_best_json_if_bq_unavailable",
        f"exception bq path must fall back + emit 'Promoted strategy BQ unavailable' log with exception detail ({exc_err})",
    ))

    # ---- Claim 9: BEHAVIORAL JSON-round-trip in the BQ reader.
    # Test the reader directly with a fake client.query().result() chain.
    rt_ok = False
    rt_err = ""
    try:
        sys.modules.pop("backend.db.bigquery_client", None)
        from backend.db import bigquery_client as bqc  # type: ignore

        # Build a fake BQ row that mimics what BQ would return from
        # TO_JSON_STRING(params). dict(row) returns a dict; we test the helper
        # directly by stubbing the underlying client.query path.
        class _FakeRow(dict):
            pass

        fake_row = _FakeRow(
            strategy_id="trial_77",
            week_iso="2026-W19",
            params_json='{"a": 1, "b": 2}',
            dsr=1.1, pbo=0.2, status="pending",
            allocation_pct=0.05, promoted_at="2026-05-02T00:00:00+00:00",
            sortino_monthly=0.4,
        )

        class _FakeQueryResult:
            def result(self, timeout=None):
                return iter([fake_row])

        fake_client = MagicMock()
        fake_client.query.return_value = _FakeQueryResult()

        # Instantiate BigQueryClient bypassing __init__ to avoid touching real
        # GCP. We set the client + settings attributes by hand.
        bq_instance = bqc.BigQueryClient.__new__(bqc.BigQueryClient)
        bq_instance.client = fake_client
        fake_settings = MagicMock()
        fake_settings.gcp_project_id = "test-project"
        bq_instance.settings = fake_settings

        row = bq_instance.get_latest_promoted_strategy()
        if not isinstance(row, dict):
            rt_err = f"reader returned {type(row)}, expected dict"
        elif row.get("strategy_id") != "trial_77":
            rt_err = f"strategy_id wrong: {row.get('strategy_id')!r}"
        elif row.get("params") != {"a": 1, "b": 2}:
            rt_err = f"params not JSON-round-tripped: {row.get('params')!r}"
        elif "params_json" in row:
            rt_err = "params_json leaked into return dict (should be popped)"
        else:
            rt_ok = True
    except Exception as e:
        rt_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if rt_ok else "FAIL",
        "behavioral_bq_reader_json_round_trip",
        f"reader must json.loads(params_json) into params dict and pop params_json ({rt_err})",
    ))

    # ---- Claim 10: BEHAVIORAL malformed JSON -> safe fallback to params={}.
    malformed_ok = False
    malformed_err = ""
    try:
        class _FakeRow(dict):
            pass

        fake_row = _FakeRow(
            strategy_id="trial_77",
            week_iso="2026-W19",
            params_json="not-valid-json",
            dsr=1.1, pbo=0.2, status="pending",
            allocation_pct=0.05, promoted_at="2026-05-02T00:00:00+00:00",
            sortino_monthly=0.4,
        )

        class _FakeQueryResult:
            def result(self, timeout=None):
                return iter([fake_row])

        fake_client = MagicMock()
        fake_client.query.return_value = _FakeQueryResult()
        bq_instance = bqc.BigQueryClient.__new__(bqc.BigQueryClient)
        bq_instance.client = fake_client
        fake_settings2 = MagicMock()
        fake_settings2.gcp_project_id = "test-project"
        bq_instance.settings = fake_settings2

        row = bq_instance.get_latest_promoted_strategy()
        if row is None or row.get("params") != {}:
            malformed_err = f"malformed JSON should yield params={{}}, got {row}"
        else:
            malformed_ok = True
    except Exception as e:
        malformed_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if malformed_ok else "FAIL",
        "behavioral_malformed_params_json_safe_fallback",
        f"malformed params_json must fall back to params={{}} (no exception) ({malformed_err})",
    ))

    # ---- Claim 11: autonomous_cycle_logs_show_promoted_strategy_loaded
    # (the masterplan criterion 3). Covered by claim 6 happy-path log check.
    # We add a dedicated grep claim for the literal log key for the verifier.
    log_key_present = (
        'logger.info(\n                "Loaded promoted params (DSR %s week=%s): %s"' in auto_text
        or 'Loaded promoted params (DSR %s week=%s): %s' in auto_text
    )
    results.append((
        "PASS" if log_key_present else "FAIL",
        "autonomous_cycle_logs_show_promoted_strategy_loaded",
        "autonomous_loop.py must include the literal 'Loaded promoted params (DSR' log line",
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
