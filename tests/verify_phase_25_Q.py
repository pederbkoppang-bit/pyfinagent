"""phase-25.Q verifier -- real-time profit_per_llm_dollar metric.

Closes red-line goal-d / phase-24.13 audit F-2 (sovereign_api.py:386-390
hardcoded anthropic/vertex/openai LLM cost = 0; no profit_per_llm_dollar
metric anywhere).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_Q.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "scripts" / "migrations" / "add_efficiency_snapshots.py"
SOVEREIGN = REPO / "backend" / "api" / "sovereign_api.py"
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (MIGRATION, SOVEREIGN, BQ_CLIENT):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    mig_text = MIGRATION.read_text(encoding="utf-8")
    sov_text = SOVEREIGN.read_text(encoding="utf-8")
    bq_text = BQ_CLIENT.read_text(encoding="utf-8")

    # ---- Claim 1: migration declares efficiency_snapshots with required cols.
    required_cols = (
        "snapshot_date", "window_days", "profit_per_llm_dollar",
        "realized_pnl_usd", "llm_cost_usd",
        "anthropic_cost_usd", "vertex_cost_usd", "openai_cost_usd",
        "computed_at",
    )
    missing = [c for c in required_cols if c not in mig_text]
    has_table = "pyfinagent_data.efficiency_snapshots" in mig_text
    results.append((
        "PASS" if not missing and has_table else "FAIL",
        "migration_declares_efficiency_snapshots_with_required_columns",
        f"migration must declare pyfinagent_data.efficiency_snapshots + all 9 columns; missing: {missing}",
    ))

    # ---- Claim 2: idempotent CREATE + PARTITION + CLUSTER.
    idempotent = "CREATE TABLE IF NOT EXISTS" in mig_text
    partition = "PARTITION BY snapshot_date" in mig_text
    cluster = "CLUSTER BY window_days" in mig_text
    results.append((
        "PASS" if idempotent and partition and cluster else "FAIL",
        "migration_idempotent_with_partition_and_cluster",
        "migration must use CREATE IF NOT EXISTS + PARTITION BY snapshot_date + CLUSTER BY window_days",
    ))

    # ---- Claim 3: BigQueryClient.save_efficiency_snapshot signature + MERGE + timeout.
    save_sig = re.search(
        r"def\s+save_efficiency_snapshot\s*\(\s*self\s*,\s*row:\s*dict\s*\)\s*->\s*None\s*:",
        bq_text,
    )
    merge_key = re.search(
        r"ON T\.snapshot_date = S\.snapshot_date AND T\.window_days = S\.window_days",
        bq_text,
    )
    # Confirm timeout=30 is inside the new function body, not just elsewhere.
    fn_match = re.search(
        r"def\s+save_efficiency_snapshot\s*\(.*?\)(.*?)(?=\n    def\s|\nclass\s|\Z)",
        bq_text,
        re.DOTALL,
    )
    timeout_in_fn = bool(fn_match and "result(timeout=30)" in fn_match.group(1))
    results.append((
        "PASS" if save_sig and merge_key and timeout_in_fn else "FAIL",
        "bq_save_efficiency_snapshot_merge_and_timeout",
        "save_efficiency_snapshot must MERGE on (snapshot_date, window_days) + result(timeout=30)",
    ))

    # ---- Claim 4: _fetch_llm_cost_by_provider exists in sovereign_api.py.
    helper = re.search(
        r"def\s+_fetch_llm_cost_by_provider\s*\(\s*window_days:\s*int\s*\)\s*->\s*dict\s*:",
        sov_text,
    )
    results.append((
        "PASS" if helper else "FAIL",
        "fetch_llm_cost_by_provider_helper_exists",
        "sovereign_api.py must declare _fetch_llm_cost_by_provider(window_days: int) -> dict",
    ))

    # ---- Claim 5: get_compute_cost no longer hardcodes zeros in per-day rows.
    # The literal pattern that was the bug:
    hardcoded_per_day = re.search(
        r"ProviderCostPoint\(\s*date=row\[.d.\]\s*,\s*anthropic=0\.0\s*,\s*vertex=0\.0\s*,\s*openai=0\.0",
        sov_text,
        re.DOTALL,
    )
    # Verify the totals dict is now populated with non-zero defaults for the 3 providers.
    totals_uses_llm = (
        'totals["anthropic"] = round(llm_costs.get("anthropic"' in sov_text
        or "totals['anthropic'] = round(llm_costs.get('anthropic'" in sov_text
    )
    results.append((
        "PASS" if not hardcoded_per_day and totals_uses_llm else "FAIL",
        "sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs",
        "get_compute_cost must no longer hardcode anthropic=0/vertex=0/openai=0 in per-day rows AND totals must use _fetch_llm_cost_by_provider output",
    ))

    # ---- Claim 6: EfficiencyResponse Pydantic model has profit_per_llm_dollar.
    model_match = re.search(
        r"class\s+EfficiencyResponse\(BaseModel\)\s*:[^}]*?profit_per_llm_dollar:\s*Optional\[float\]",
        sov_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if model_match else "FAIL",
        "efficiency_response_pydantic_model_has_profit_per_llm_dollar",
        "EfficiencyResponse must declare profit_per_llm_dollar: Optional[float]",
    ))

    # ---- Claim 7: GET /efficiency route registered.
    route_match = re.search(
        r'@router\.get\(["\']/efficiency["\']\s*,\s*response_model=EfficiencyResponse\s*\)',
        sov_text,
    )
    fn_match2 = re.search(r"def\s+get_efficiency\s*\(", sov_text)
    results.append((
        "PASS" if route_match and fn_match2 else "FAIL",
        "new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar",
        "sovereign_api.py must register @router.get('/efficiency', response_model=EfficiencyResponse) -> def get_efficiency",
    ))

    # ---- Behavioral fixtures: load sovereign_api with mocked dependencies.
    # We patch the heavy collaborators inside get_efficiency so the test
    # doesn't reach BQ / settings.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.api.sovereign_api", None)
    from backend.api import sovereign_api as sov  # type: ignore

    def _build_fake_bq(realized_pnl_total: float):
        """Fake BigQueryClient -- trades list has exactly one matching round-trip
        whose realized_pnl_usd equals the target total."""
        fake = MagicMock()
        # pair_round_trips returns a list of dicts with realized_pnl_usd; we
        # stub trades + pair_round_trips path together below.
        fake.get_paper_trades_in_window.return_value = []
        return fake

    def _call_get_efficiency(
        pnl_total: float, llm_costs: dict, *, persist: bool = False, window: str = "30d",
    ):
        """Invoke the endpoint synchronously with mocks for BQ + helper +
        pair_round_trips. Returns (response, fake_bq_instance)."""
        # Clear cache so each call goes through.
        from backend.services.api_cache import get_api_cache as _g
        cache = _g()
        cache._store.clear() if hasattr(cache, "_store") else None

        fake_bq = _build_fake_bq(pnl_total)

        with patch(
            "backend.db.bigquery_client.BigQueryClient",
            return_value=fake_bq,
        ), patch(
            "backend.config.settings.get_settings",
            return_value=MagicMock(),
        ), patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=[{"realized_pnl_usd": pnl_total}] if pnl_total != 0 else [],
        ), patch.object(
            sov, "_fetch_llm_cost_by_provider", return_value=llm_costs,
        ):
            return sov.get_efficiency(window=window, persist=persist), fake_bq

    # ---- Claim 8: BEHAVIORAL round-trip -- valid P&L + cost -> ratio.
    rt_ok = False
    rt_err = ""
    try:
        resp, fake_bq = _call_get_efficiency(
            pnl_total=1000.0,
            llm_costs={"anthropic": 50.0, "vertex": 30.0, "openai": 20.0},
        )
        if resp.realized_pnl_usd != 1000.0:
            rt_err = f"realized_pnl_usd={resp.realized_pnl_usd}"
        elif resp.llm_cost_usd != 100.0:
            rt_err = f"llm_cost_usd={resp.llm_cost_usd}, expected 100"
        elif resp.profit_per_llm_dollar != 10.0:
            rt_err = f"profit_per_llm_dollar={resp.profit_per_llm_dollar}, expected 10.0"
        elif resp.anthropic_cost_usd != 50.0:
            rt_err = f"anthropic_cost_usd={resp.anthropic_cost_usd}"
        else:
            rt_ok = True
    except Exception as e:
        rt_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if rt_ok else "FAIL",
        "behavioral_efficiency_endpoint_returns_correct_ratio",
        f"happy-path: pnl=1000 + cost=100 -> ratio=10.0 ({rt_err})",
    ))

    # ---- Claim 9: BEHAVIORAL zero-cost -> profit_per_llm_dollar = None.
    zc_ok = False
    zc_err = ""
    try:
        resp2, _ = _call_get_efficiency(
            pnl_total=500.0,
            llm_costs={"anthropic": 0.0, "vertex": 0.0, "openai": 0.0},
            window="7d",
        )
        if resp2.profit_per_llm_dollar is not None:
            zc_err = f"ratio={resp2.profit_per_llm_dollar}, expected None"
        elif resp2.realized_pnl_usd != 500.0:
            zc_err = f"P&L wrong: {resp2.realized_pnl_usd}"
        elif not resp2.note or "undefined" not in resp2.note:
            zc_err = f"note missing the undefined sentinel: {resp2.note!r}"
        else:
            zc_ok = True
    except Exception as e:
        zc_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if zc_ok else "FAIL",
        "behavioral_zero_llm_cost_yields_none_ratio_not_inf",
        f"zero-cost path must yield profit_per_llm_dollar=None + descriptive note ({zc_err})",
    ))

    # ---- Claim 10: BEHAVIORAL persist=True -> save_efficiency_snapshot called.
    persist_ok = False
    persist_err = ""
    try:
        # We need to inspect that fake_bq.save_efficiency_snapshot was called.
        # Re-running with persist=True and a fresh mock chain.
        from backend.services.api_cache import get_api_cache as _g
        cache = _g()
        if hasattr(cache, "_store"):
            cache._store.clear()

        fake_bq = MagicMock()
        fake_bq.get_paper_trades_in_window.return_value = []

        with patch(
            "backend.db.bigquery_client.BigQueryClient",
            return_value=fake_bq,
        ), patch(
            "backend.config.settings.get_settings",
            return_value=MagicMock(),
        ), patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=[{"realized_pnl_usd": 1500.0}],
        ), patch.object(
            sov,
            "_fetch_llm_cost_by_provider",
            return_value={"anthropic": 40.0, "vertex": 20.0, "openai": 40.0},
        ):
            resp3 = sov.get_efficiency(window="30d", persist=True)

        if fake_bq.save_efficiency_snapshot.call_count != 1:
            persist_err = (
                f"save_efficiency_snapshot called {fake_bq.save_efficiency_snapshot.call_count} times, "
                "expected 1"
            )
        else:
            written = fake_bq.save_efficiency_snapshot.call_args.args[0]
            if written.get("window_days") != 30:
                persist_err = f"window_days={written.get('window_days')}, expected 30"
            elif written.get("realized_pnl_usd") != 1500.0:
                persist_err = f"realized_pnl_usd={written.get('realized_pnl_usd')}"
            elif written.get("llm_cost_usd") != 100.0:
                persist_err = f"llm_cost_usd={written.get('llm_cost_usd')}"
            elif written.get("profit_per_llm_dollar") != 15.0:
                persist_err = f"profit_per_llm_dollar={written.get('profit_per_llm_dollar')}"
            else:
                persist_ok = True
    except Exception as e:
        persist_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if persist_ok else "FAIL",
        "metric_persisted_to_bq_for_30d_window",
        f"persist=True must call save_efficiency_snapshot once with the computed row ({persist_err})",
    ))

    # ---- Claim 11: provider mapping gemini -> vertex in helper output.
    # Direct test of _fetch_llm_cost_by_provider with a mocked BQ client.
    mapping_ok = False
    mapping_err = ""
    try:
        sys.modules.pop("backend.api.sovereign_api", None)
        from backend.api import sovereign_api as sov2  # type: ignore

        fake_rows = [
            MagicMock(),
        ]
        # MagicMock get(...) calls.
        def _row_get(key, default=None):
            data = {"provider": "gemini", "model": "gemini-2.5-flash", "in_tok": 1000, "out_tok": 500}
            return data.get(key, default)
        fake_rows[0].get.side_effect = _row_get

        class _FakeResult:
            def result(self, timeout=None):
                return iter(fake_rows)

        fake_client = MagicMock()
        fake_client.query.return_value = _FakeResult()
        with patch.object(sov2, "_bq_client", return_value=fake_client):
            costs = sov2._fetch_llm_cost_by_provider(30)

        if costs.get("vertex", 0.0) <= 0:
            mapping_err = f"vertex bucket empty: {costs}"
        elif costs.get("anthropic", 0.0) != 0.0:
            mapping_err = f"anthropic should be 0 (no rows): {costs}"
        else:
            mapping_ok = True
    except Exception as e:
        mapping_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if mapping_ok else "FAIL",
        "provider_mapping_gemini_to_vertex_enforced",
        f"_fetch_llm_cost_by_provider must map provider='gemini' -> 'vertex' bucket ({mapping_err})",
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
