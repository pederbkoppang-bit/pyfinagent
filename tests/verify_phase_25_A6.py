"""phase-25.A6 verifier -- explicit live-vs-backtest Sharpe reconciliation.

Closes phase-24.6 F-3 (paper_go_live_gate.py:87-94 used NAV-divergence as
a proxy for the Sharpe gap; closes the proxy mismatch in favor of an
explicit Sharpe comparison with a 30% threshold per Jacquier et al.
arxiv 2501.03938).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A6.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
PERF_METRICS = REPO / "backend" / "services" / "perf_metrics.py"
GATE = REPO / "backend" / "services" / "paper_go_live_gate.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PERF_METRICS, GATE):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    perf_text = PERF_METRICS.read_text(encoding="utf-8")
    gate_text = GATE.read_text(encoding="utf-8")

    # ---- Claim 1: compute_sharpe_gap signature exists.
    sig = re.search(
        r"def\s+compute_sharpe_gap\s*\(\s*bq:\s*Any\s*,\s*\*\s*,\s*backtest_sharpe_source:\s*str\s*=\s*[\"']optimizer_best[\"']\s*,\s*risk_free_rate:\s*float\s*=\s*0\.04\s*,\s*min_snapshots:\s*int\s*=\s*6\s*,?\s*\)\s*->\s*dict\s*:",
        perf_text,
        re.DOTALL,
    )
    results.append((
        "PASS" if sig else "FAIL",
        "new_function_compute_live_realized_sharpe_vs_backtest_exists",
        "perf_metrics.py must declare compute_sharpe_gap(bq: Any, *, backtest_sharpe_source='optimizer_best', risk_free_rate=0.04, min_snapshots=6) -> dict",
    ))

    # ---- Claim 2: SR_GAP_THRESHOLD = 0.30 (criterion 3).
    threshold_perf = re.search(r"SR_GAP_THRESHOLD\s*=\s*0\.30", perf_text)
    threshold_gate = re.search(r"SR_GAP_THRESHOLD\s*=\s*0\.30", gate_text)
    results.append((
        "PASS" if threshold_perf and threshold_gate else "FAIL",
        "threshold_at_30pct_per_industry_benchmark",
        "SR_GAP_THRESHOLD must equal 0.30 in both perf_metrics.py and paper_go_live_gate.py",
    ))

    # ---- Claim 3: paper_go_live_gate uses compute_sharpe_gap, NOT the old proxy.
    legacy_proxy = re.search(
        r"sr_gap_proxy\s*=\s*latest_divergence_pct\s*/\s*100\.0",
        gate_text,
    )
    uses_helper = re.search(
        r"sharpe_gap\s*=\s*compute_sharpe_gap\(\s*bq\s*\)",
        gate_text,
    )
    results.append((
        "PASS" if not legacy_proxy and uses_helper else "FAIL",
        "paper_go_live_gate_uses_explicit_sharpe_not_nav_proxy",
        "paper_go_live_gate.py must call compute_sharpe_gap(bq) and remove the legacy sr_gap_proxy = latest_divergence_pct / 100.0",
    ))

    # ---- Claim 4: compute_gate details includes Sharpe diagnostic fields.
    has_live_in_details = '"live_sharpe":' in gate_text
    has_backtest_in_details = '"backtest_sharpe":' in gate_text
    has_gap_rel = '"sharpe_gap_rel":' in gate_text
    has_source = '"sharpe_gap_source":' in gate_text
    results.append((
        "PASS" if all([has_live_in_details, has_backtest_in_details, has_gap_rel, has_source]) else "FAIL",
        "compute_gate_details_includes_sharpe_diagnostics",
        "compute_gate details dict must include live_sharpe, backtest_sharpe, sharpe_gap_rel, sharpe_gap_source",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.services.perf_metrics", None)
    from backend.services import perf_metrics as pm  # type: ignore

    def _fake_bq_with_snapshots(snapshots: list[dict]) -> MagicMock:
        fake = MagicMock()
        fake.get_paper_snapshots.return_value = snapshots
        return fake

    def _build_climbing_snapshots(n: int, start: float = 10000.0, daily_pct: float = 0.005) -> list[dict]:
        """Synthesize n snapshots with upward drift + noise (real variance
        so the Sharpe formula returns a finite value, not clamped 0.0)."""
        import random
        rng = random.Random(42)
        nav = start
        out = []
        for i in range(n):
            # Drift + ~1% std noise so variance > 0.
            noise = rng.gauss(0.0, 0.01)
            nav = nav * (1.0 + daily_pct + noise)
            out.append({"total_nav": round(nav, 2)})
        return out

    # ---- Claim 5: BEHAVIORAL primary-source (optimizer_best.json exists).
    primary_ok = False
    primary_err = ""
    try:
        td = Path(tempfile.mkdtemp(prefix="phase25a6_"))
        opt_path = td / "optimizer_best.json"
        opt_path.write_text(json.dumps({"sharpe": 1.0, "params": {}}), encoding="utf-8")

        snapshots = _build_climbing_snapshots(60, daily_pct=0.003)
        fake_bq = _fake_bq_with_snapshots(snapshots)

        with patch.object(pm, "_OPTIMIZER_BEST_PATH", opt_path):
            r = pm.compute_sharpe_gap(fake_bq)

        if r.get("source") != "optimizer_best":
            primary_err = f"source={r.get('source')!r}, expected 'optimizer_best'"
        elif r.get("backtest_sharpe") != 1.0:
            primary_err = f"backtest_sharpe={r.get('backtest_sharpe')!r}, expected 1.0"
        elif r.get("live_sharpe") is None:
            primary_err = "live_sharpe is None despite 60 snapshots"
        elif r.get("gap_within_threshold") is None:
            primary_err = "gap_within_threshold is None despite both sharpes set"
        elif r.get("gap_abs") is None or r.get("gap_rel") is None:
            primary_err = f"gap_abs/gap_rel are None: {r}"
        elif r.get("threshold") != 0.30:
            primary_err = f"threshold={r.get('threshold')}, expected 0.30"
        else:
            primary_ok = True
    except Exception as e:
        primary_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if primary_ok else "FAIL",
        "behavioral_primary_source_optimizer_best_json",
        f"primary-source happy path must surface source='optimizer_best' + non-None gap fields ({primary_err})",
    ))

    # ---- Claim 6: BEHAVIORAL threshold-failure (gap > 30%).
    fail_ok = False
    fail_err = ""
    try:
        td2 = Path(tempfile.mkdtemp(prefix="phase25a6_fail_"))
        opt_path2 = td2 / "optimizer_best.json"
        # backtest=1.0; we want live around 2.0 so gap_rel = 1.0 > 0.30.
        opt_path2.write_text(json.dumps({"sharpe": 0.50}), encoding="utf-8")

        # Stronger drift -> higher Sharpe.
        snapshots2 = _build_climbing_snapshots(60, daily_pct=0.020)
        fake_bq2 = _fake_bq_with_snapshots(snapshots2)

        with patch.object(pm, "_OPTIMIZER_BEST_PATH", opt_path2):
            r2 = pm.compute_sharpe_gap(fake_bq2)

        if r2.get("gap_within_threshold") is not False:
            fail_err = f"gap_within_threshold={r2.get('gap_within_threshold')}, expected False (got {r2})"
        elif (r2.get("gap_rel") or 0) <= 0.30:
            fail_err = f"gap_rel={r2.get('gap_rel')} not > 0.30 in fail-path test"
        else:
            fail_ok = True
    except Exception as e:
        fail_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if fail_ok else "FAIL",
        "behavioral_threshold_failure_gap_above_30pct",
        f"large gap must yield gap_within_threshold=False ({fail_err})",
    ))

    # ---- Claim 7: BEHAVIORAL no-data path -- live snapshots empty.
    nodata_ok = False
    nodata_err = ""
    try:
        td3 = Path(tempfile.mkdtemp(prefix="phase25a6_nodata_"))
        opt_path3 = td3 / "optimizer_best.json"
        opt_path3.write_text(json.dumps({"sharpe": 1.0}), encoding="utf-8")

        fake_bq3 = _fake_bq_with_snapshots([])

        # Need to also stub compute_reconciliation so the proxy fallback doesn't fire.
        with patch.object(pm, "_OPTIMIZER_BEST_PATH", opt_path3), patch(
            "backend.services.reconciliation.compute_reconciliation",
            return_value={"series": [], "summary": {"latest_divergence_pct": None}},
        ):
            r3 = pm.compute_sharpe_gap(fake_bq3)

        if r3.get("live_sharpe") is not None:
            nodata_err = f"live_sharpe={r3.get('live_sharpe')!r}, expected None"
        elif r3.get("gap_within_threshold") is not None:
            nodata_err = f"gap_within_threshold={r3.get('gap_within_threshold')!r}, expected None"
        else:
            nodata_ok = True
    except Exception as e:
        nodata_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if nodata_ok else "FAIL",
        "behavioral_no_data_gate_stays_red",
        f"no live snapshots must yield live_sharpe=None + gap_within_threshold=None ({nodata_err})",
    ))

    # ---- Claim 8: BEHAVIORAL fallback-2 (shadow curve when optimizer_best absent).
    sc_ok = False
    sc_err = ""
    try:
        td4 = Path(tempfile.mkdtemp(prefix="phase25a6_shadow_"))
        # NO optimizer_best.json -> primary fallback fails.
        missing_path = td4 / "missing.json"

        # Live snapshots + a shadow curve.
        snaps = _build_climbing_snapshots(60, daily_pct=0.003)
        fake_bq4 = _fake_bq_with_snapshots(snaps)

        # Shadow curve needs variance for a finite Sharpe.
        import random as _rnd
        shadow_rng = _rnd.Random(123)
        shadow_navs: list[float] = []
        nav_s = 10000.0
        for i in range(30):
            nav_s = nav_s * (1.0 + 0.004 + shadow_rng.gauss(0.0, 0.008))
            shadow_navs.append(nav_s)
        shadow_series = [
            {"date": f"2026-01-{i+1:02d}", "backtest_nav": shadow_navs[i]}
            for i in range(30)
        ]

        with patch.object(pm, "_OPTIMIZER_BEST_PATH", missing_path), patch(
            "backend.services.reconciliation.compute_reconciliation",
            return_value={"series": shadow_series, "summary": {"latest_divergence_pct": 1.0}},
        ):
            r4 = pm.compute_sharpe_gap(fake_bq4)

        if r4.get("source") != "shadow_curve":
            sc_err = f"source={r4.get('source')!r}, expected 'shadow_curve' (got {r4})"
        elif r4.get("backtest_sharpe") is None:
            sc_err = "backtest_sharpe is None despite shadow curve >=6 points"
        else:
            sc_ok = True
    except Exception as e:
        sc_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if sc_ok else "FAIL",
        "behavioral_fallback_shadow_curve_used_when_optimizer_best_absent",
        f"missing optimizer_best.json must yield source='shadow_curve' ({sc_err})",
    ))

    # ---- Claim 9: BEHAVIORAL fallback-3 (proxy when both primary + shadow absent).
    pf_ok = False
    pf_err = ""
    try:
        td5 = Path(tempfile.mkdtemp(prefix="phase25a6_proxy_"))
        missing_path5 = td5 / "missing.json"
        snaps5 = _build_climbing_snapshots(60, daily_pct=0.003)
        fake_bq5 = _fake_bq_with_snapshots(snaps5)

        # Shadow series empty -> secondary fallback fails too.
        with patch.object(pm, "_OPTIMIZER_BEST_PATH", missing_path5), patch(
            "backend.services.reconciliation.compute_reconciliation",
            return_value={"series": [], "summary": {"latest_divergence_pct": 15.0}},
        ):
            r5 = pm.compute_sharpe_gap(fake_bq5)

        if r5.get("source") != "proxy_fallback":
            pf_err = f"source={r5.get('source')!r}, expected 'proxy_fallback' (got {r5})"
        elif r5.get("proxy_fallback") is not True:
            pf_err = f"proxy_fallback={r5.get('proxy_fallback')!r}, expected True"
        elif r5.get("gap_rel") != 0.15:
            pf_err = f"gap_rel={r5.get('gap_rel')}, expected 0.15 (15/100)"
        else:
            pf_ok = True
    except Exception as e:
        pf_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if pf_ok else "FAIL",
        "behavioral_fallback_proxy_when_both_primary_and_shadow_unavailable",
        f"both primary+shadow unavailable must yield source='proxy_fallback' + proxy_fallback=True ({pf_err})",
    ))

    # ---- Claim 10: compute_gate integrates compute_sharpe_gap correctly.
    int_ok = False
    int_err = ""
    try:
        sys.modules.pop("backend.services.paper_go_live_gate", None)
        from backend.services import paper_go_live_gate as gate  # type: ignore

        # Mock all deps of compute_gate so we only exercise the Sharpe-gap wire.
        fake_bq6 = MagicMock()
        fake_bq6.get_paper_trades.return_value = []
        fake_bq6.get_paper_snapshots.return_value = _build_climbing_snapshots(35, daily_pct=0.002)

        td6 = Path(tempfile.mkdtemp(prefix="phase25a6_gate_"))
        opt_path6 = td6 / "optimizer_best.json"
        opt_path6.write_text(json.dumps({"sharpe": 1.0}), encoding="utf-8")

        with patch.object(pm, "_OPTIMIZER_BEST_PATH", opt_path6), patch(
            "backend.services.paper_go_live_gate.compute_metrics_v2",
            return_value={"psr": 0.9, "dsr": 0.9, "rolling_sharpe": 1.0, "n_obs": 35},
        ), patch(
            "backend.services.paper_go_live_gate.pair_round_trips",
            return_value=[],
        ), patch(
            "backend.services.paper_go_live_gate.compute_reconciliation",
            return_value={"series": [], "summary": {"latest_divergence_pct": 0.0}},
        ), patch(
            "backend.services.reconciliation.compute_reconciliation",
            return_value={"series": [], "summary": {"latest_divergence_pct": 0.0}},
        ):
            result = gate.compute_gate(fake_bq6)

        details = result.get("details", {})
        if "live_sharpe" not in details:
            int_err = "details missing live_sharpe"
        elif "backtest_sharpe" not in details:
            int_err = "details missing backtest_sharpe"
        elif "sharpe_gap_source" not in details:
            int_err = "details missing sharpe_gap_source"
        elif "sr_gap_le_30pct" not in result.get("booleans", {}):
            int_err = "booleans missing sr_gap_le_30pct"
        elif details.get("backtest_sharpe") != 1.0:
            int_err = f"backtest_sharpe in details = {details.get('backtest_sharpe')}, expected 1.0"
        else:
            int_ok = True
    except Exception as e:
        int_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if int_ok else "FAIL",
        "compute_gate_uses_new_helper_and_exposes_details",
        f"compute_gate must wire compute_sharpe_gap and surface diagnostics in details ({int_err})",
    ))

    # ---- Claim 11: industry-benchmark attribution in docstring.
    industry_doc = ("industry benchmark" in perf_text.lower() or "30-50%" in perf_text or "30%" in perf_text)
    results.append((
        "PASS" if industry_doc else "FAIL",
        "industry_benchmark_attribution_in_docstring",
        "compute_sharpe_gap docstring must reference the 30% industry benchmark",
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
