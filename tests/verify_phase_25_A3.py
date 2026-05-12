"""phase-25.A3 verifier -- promoted_strategies BQ table + Friday writer.

Closes phase-24.3 audit F-3 (friday_promotion.py wrote only to flat TSV
ledger; no BQ subscriber existed for the daily autonomous loop).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A3.py
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "scripts" / "migrations" / "create_promoted_strategies_table.py"
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"
FRIDAY = REPO / "backend" / "autoresearch" / "friday_promotion.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (MIGRATION, BQ_CLIENT, FRIDAY):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    mig_text = MIGRATION.read_text(encoding="utf-8")
    bq_text = BQ_CLIENT.read_text(encoding="utf-8")
    fri_text = FRIDAY.read_text(encoding="utf-8")

    # ---- Claim 1: migration script declares correct table FQN.
    has_dataset = "pyfinagent_data" in mig_text
    has_table = "promoted_strategies" in mig_text
    results.append((
        "PASS" if has_dataset and has_table else "FAIL",
        "bigquery_promoted_strategies_table_exists",
        "migration script must declare pyfinagent_data.promoted_strategies as the target table",
    ))

    # ---- Claim 2: CREATE_SQL declares all 10 required columns.
    required_cols = (
        "strategy_id",
        "week_iso",
        "params",
        "dsr",
        "pbo",
        "status",
        "allocation_pct",
        "promoted_at",
        "sortino_monthly",
        "rejection_reason",
    )
    missing = [c for c in required_cols if c not in mig_text]
    results.append((
        "PASS" if not missing else "FAIL",
        "schema_includes_strategy_id_params_json_dsr_pbo_status",
        f"migration must include all 10 columns; missing: {missing}",
    ))

    # ---- Claim 3: idempotent CREATE TABLE IF NOT EXISTS.
    idempotent = "CREATE TABLE IF NOT EXISTS" in mig_text
    results.append((
        "PASS" if idempotent else "FAIL",
        "migration_is_idempotent_create_table_if_not_exists",
        "migration must use CREATE TABLE IF NOT EXISTS",
    ))

    # ---- Claim 4: params column is JSON, not STRING.
    json_col = re.search(r"params\s+JSON\b", mig_text)
    results.append((
        "PASS" if json_col else "FAIL",
        "params_column_is_json_type",
        "params column must be declared as JSON",
    ))

    # ---- Claim 5: PARTITION BY DATE(promoted_at) + CLUSTER BY strategy_id, week_iso.
    partition = "PARTITION BY DATE(promoted_at)" in mig_text
    cluster = re.search(r"CLUSTER BY\s+strategy_id\s*,\s*week_iso", mig_text)
    results.append((
        "PASS" if partition and cluster else "FAIL",
        "migration_partition_and_cluster_correct",
        "must PARTITION BY DATE(promoted_at) AND CLUSTER BY strategy_id, week_iso",
    ))

    # ---- Claim 6: BigQueryClient.save_promoted_strategy method exists.
    save_method = re.search(
        r"def\s+save_promoted_strategy\s*\(\s*self\s*,\s*row:\s*dict\s*\)\s*->\s*None\s*:",
        bq_text,
    )
    results.append((
        "PASS" if save_method else "FAIL",
        "bigquery_client_save_promoted_strategy_method_exists",
        "BigQueryClient must declare save_promoted_strategy(self, row: dict) -> None",
    ))

    # ---- Claim 7: MERGE on (week_iso, strategy_id) + 30s timeout.
    merge_key = re.search(
        r"ON T\.week_iso = S\.week_iso AND T\.strategy_id = S\.strategy_id",
        bq_text,
    )
    timeout = "result(timeout=30)" in bq_text
    results.append((
        "PASS" if merge_key and timeout else "FAIL",
        "save_promoted_strategy_merge_with_timeout_30",
        "save_promoted_strategy must MERGE on (week_iso, strategy_id) AND call result(timeout=30)",
    ))

    # ---- Claim 8: run_friday_promotion signature has bq_client=None kwarg.
    sig_match = re.search(
        r"def\s+run_friday_promotion\(.*?bq_client:\s*Any\s*\|\s*None\s*=\s*None.*?\)\s*->\s*dict",
        fri_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if sig_match else "FAIL",
        "run_friday_promotion_accepts_bq_client_kwarg_default_none",
        "run_friday_promotion signature must include bq_client: Any | None = None",
    ))

    # ---- Claim 9: BEHAVIORAL round-trip -- pass fake bq_client, verify exactly
    # one save_promoted_strategy call per promoted candidate with correct shape.
    behavior_ok = False
    behavior_err = ""
    try:
        sys.path.insert(0, str(REPO))
        # Module-level import; flush any cached module first to ensure we
        # exercise the patched source.
        sys.modules.pop("backend.autoresearch.friday_promotion", None)
        from backend.autoresearch import friday_promotion as fp  # type: ignore

        # Fake gate that always promotes a candidate.
        class _FakeGate:
            def evaluate(self, candidate):
                return {"promoted": True, "decision": "PROMOTE"}

        # Fake ledger module: read_rows returns a row matching week_iso;
        # append_row returns True. We need to monkey-patch the weekly_ledger
        # module reference inside fp.
        fake_ledger_row = {
            "week_iso": "2026-W20",
            "thu_batch_id": "batch_xyz",
            "thu_candidates_kicked": "2",
            "fri_promoted_ids": "",
            "fri_rejected_ids": "",
            "cost_usd": "0.0",
            "sortino_monthly": "0.42",
            "notes": "",
        }
        fake_ledger = MagicMock()
        fake_ledger.read_rows.return_value = [fake_ledger_row]
        fake_ledger.append_row.return_value = True
        fake_ledger.LEDGER_PATH = Path("/tmp/fake_ledger.tsv")
        fp.weekly_ledger = fake_ledger  # type: ignore

        fake_bq = MagicMock()
        # The candidate carries trial_id + params + dsr + pbo.
        candidate = {
            "trial_id": "trial_42",
            "params": {"lookback": 20, "threshold": 0.5},
            "dsr": 1.2,
            "pbo": 0.15,
        }

        result = fp.run_friday_promotion(
            week_iso="2026-W20",
            candidates=[candidate],
            top_n=1,
            gate=_FakeGate(),
            bq_client=fake_bq,
        )

        n_calls = fake_bq.save_promoted_strategy.call_count
        if result.get("error"):
            behavior_err = f"error returned: {result.get('error')}"
        elif n_calls != 1:
            behavior_err = f"save_promoted_strategy call_count={n_calls}, expected 1"
        else:
            called_row = fake_bq.save_promoted_strategy.call_args.args[0]
            missing_keys = [
                k for k in (
                    "strategy_id", "week_iso", "params", "dsr", "pbo",
                    "status", "allocation_pct", "promoted_at", "sortino_monthly",
                )
                if k not in called_row
            ]
            import json as _json
            if missing_keys:
                behavior_err = f"row missing keys: {missing_keys}"
            elif called_row.get("strategy_id") != "trial_42":
                behavior_err = f"strategy_id wrong: {called_row.get('strategy_id')!r}"
            elif called_row.get("week_iso") != "2026-W20":
                behavior_err = f"week_iso wrong: {called_row.get('week_iso')!r}"
            elif called_row.get("status") != "pending":
                behavior_err = f"status wrong: {called_row.get('status')!r}"
            elif _json.loads(called_row.get("params", "{}")) != {"lookback": 20, "threshold": 0.5}:
                behavior_err = f"params not round-tripped: {called_row.get('params')!r}"
            else:
                behavior_ok = True
    except Exception as e:
        behavior_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if behavior_ok else "FAIL",
        "friday_promotion_writes_row_per_promotion",
        f"behavioral round-trip must call save_promoted_strategy once with correct shape ({behavior_err})",
    ))

    # ---- Claim 10: bq_client=None still succeeds and ledger TSV write happens.
    none_ok = False
    none_err = ""
    try:
        # Reuse same fake_ledger; fp module is still imported with the fake.
        fake_ledger.read_rows.return_value = [{
            "week_iso": "2026-W21",
            "thu_batch_id": "batch_abc",
            "thu_candidates_kicked": "1",
            "fri_promoted_ids": "",
            "fri_rejected_ids": "",
            "cost_usd": "0.0",
            "sortino_monthly": "0.1",
            "notes": "",
        }]
        fake_ledger.append_row.reset_mock()
        fake_ledger.append_row.return_value = True
        result2 = fp.run_friday_promotion(
            week_iso="2026-W21",
            candidates=[{
                "trial_id": "trial_99",
                "params": {},
                "dsr": 1.0,
                "pbo": 0.2,
            }],
            top_n=1,
            gate=_FakeGate(),
            bq_client=None,
        )
        if result2.get("error"):
            none_err = f"error returned: {result2.get('error')}"
        elif fake_ledger.append_row.call_count != 1:
            none_err = f"ledger append called {fake_ledger.append_row.call_count} times, expected 1"
        else:
            none_ok = True
    except Exception as e:
        none_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if none_ok else "FAIL",
        "bq_client_none_preserves_existing_ledger_write_no_regression",
        f"with bq_client=None, ledger TSV write still happens, no error ({none_err})",
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
