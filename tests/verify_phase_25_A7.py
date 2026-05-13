"""phase-25.A7 verifier -- per-table freshness endpoint covering 5+ data tables.

Closes phase-24.7 F-1 (cycle_health.compute_freshness only queried
paper_trades + paper_portfolio_snapshots; 4 historical/log tables were
unmonitored).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A7.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
CYCLE_HEALTH = REPO / "backend" / "services" / "cycle_health.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not CYCLE_HEALTH.exists():
        print(f"FAIL: required source file missing: {CYCLE_HEALTH}")
        return 1

    src = CYCLE_HEALTH.read_text(encoding="utf-8")

    # ---- Claim 1: _TABLE_MAX_AGE_SEC constant with required keys + values.
    const_match = re.search(
        r"_TABLE_MAX_AGE_SEC\s*:\s*dict\[str\s*,\s*float\]\s*=\s*\{(.*?)\}",
        src,
        re.DOTALL,
    )
    keys_ok = False
    values_ok = False
    if const_match:
        body = const_match.group(1)
        keys_ok = all(
            k in body for k in (
                '"historical_prices"',
                '"historical_fundamentals"',
                '"historical_macro"',
                '"paper_portfolio_snapshots"',
            )
        )
        # Check the documented values are present.
        values_ok = (
            "93_600" in body
            and "8_208_000" in body
            and "3_024_000" in body
        )
    results.append((
        "PASS" if const_match and keys_ok and values_ok else "FAIL",
        "table_max_age_sec_constant_with_documented_intervals",
        "_TABLE_MAX_AGE_SEC must declare 4 keys (3 historical + paper_portfolio_snapshots) with documented intervals",
    ))

    # ---- Claim 2: _worst_band helper exists.
    worst_sig = re.search(
        r"def\s+_worst_band\s*\(\s*bands:\s*list\[str\]\s*\)\s*->\s*str\s*:",
        src,
    )
    results.append((
        "PASS" if worst_sig else "FAIL",
        "worst_band_helper_signature",
        "cycle_health.py must declare _worst_band(bands: list[str]) -> str",
    ))

    # ---- Claim 3: _fire_freshness_alarm helper exists.
    fire_sig = re.search(
        r"def\s+_fire_freshness_alarm\s*\(\s*sources:\s*dict\s*\)\s*->\s*None\s*:",
        src,
    )
    imports_raise = "raise_cron_alert_sync" in src
    results.append((
        "PASS" if fire_sig and imports_raise else "FAIL",
        "fire_freshness_alarm_helper_with_raise_cron_alert_sync",
        "cycle_health.py must declare _fire_freshness_alarm(sources: dict) -> None and import raise_cron_alert_sync",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.services.cycle_health", None)
    from backend.services import cycle_health as ch  # type: ignore

    # ---- Claim 4: _worst_band priority (behavioral).
    wb_ok = False
    wb_err = ""
    try:
        cases = [
            (["green", "amber", "red"], "red"),
            (["green", "amber"], "amber"),
            (["green"], "green"),
            (["unknown"], "unknown"),
            ([], "unknown"),
            (["red", "unknown"], "red"),
        ]
        for inp, expected in cases:
            got = ch._worst_band(inp)
            if got != expected:
                wb_err = f"_worst_band({inp}) -> {got!r}, expected {expected!r}"
                break
        else:
            wb_ok = True
    except Exception as e:
        wb_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if wb_ok else "FAIL",
        "behavioral_worst_band_priority_red_amber_green_unknown",
        f"_worst_band must follow red>amber>green>unknown priority ({wb_err})",
    ))

    # ---- Helper to build a fake bq with custom MAX(time_col) responses
    # per (logical_table, time_col).
    def _build_fake_bq(ages: dict):
        fake = MagicMock()
        fake._pt_table = lambda name: f"test-proj.test_ds.{name}"

        def _fake_query(sql, *args, **kwargs):
            # Detect the table name in the FROM clause to route to the right age.
            for table_name, age_val in ages.items():
                if f".{table_name}`" in sql:
                    row = MagicMock()
                    row.get = lambda key, default=None, _age=age_val: _age if key == "age" else default
                    row.__getitem__ = lambda self, idx, _age=age_val: _age if idx == 0 else None
                    rs = MagicMock()
                    rs.result.return_value = iter([row])
                    return rs
            # Default empty if not found.
            rs = MagicMock()
            rs.result.return_value = iter([])
            return rs

        fake.client.query.side_effect = _fake_query
        return fake

    # ---- Claim 5: compute_freshness returns sources with all 6 required keys.
    keys_ok = False
    keys_err = ""
    try:
        fake_bq = _build_fake_bq({
            "paper_trades": 10.0,
            "paper_portfolio_snapshots": 100.0,
            "historical_prices": 1000.0,
            "historical_fundamentals": 100_000.0,
            "historical_macro": 50_000.0,
            "signals_log": 60.0,
        })
        result = ch.compute_freshness(fake_bq, cycle_interval_sec=86_400.0)
        sources = result.get("sources", {})
        required = {
            "paper_trades", "paper_portfolio_snapshots",
            "historical_prices", "historical_fundamentals",
            "historical_macro", "signals_log",
        }
        missing = required - set(sources.keys())
        if missing:
            keys_err = f"missing source keys: {missing}"
        else:
            keys_ok = True
    except Exception as e:
        keys_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if keys_ok else "FAIL",
        "api_observability_freshness_returns_per_table_ages_for_5_tables",
        f"sources dict must include all 6 monitored tables ({keys_err})",
    ))

    # ---- Claim 6: each source entry carries the right schema (band etc).
    schema_ok = False
    schema_err = ""
    try:
        if sources:
            required_fields = {"last_tick_age_sec", "interval_sec", "ratio", "band"}
            for table_name, info in sources.items():
                if not isinstance(info, dict):
                    schema_err = f"{table_name} entry is not a dict"
                    break
                missing_fields = required_fields - set(info.keys())
                if missing_fields:
                    schema_err = f"{table_name} missing fields: {missing_fields}"
                    break
                if info["band"] not in ("green", "amber", "red", "unknown"):
                    schema_err = f"{table_name}.band={info['band']!r} not in green/amber/red/unknown"
                    break
            else:
                schema_ok = True
        else:
            schema_err = "no sources to inspect"
    except Exception as e:
        schema_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if schema_ok else "FAIL",
        "sla_bands_green_amber_red_implemented_per_table",
        f"each source entry must carry last_tick_age_sec/interval_sec/ratio/band with band in green/amber/red/unknown ({schema_err})",
    ))

    # ---- Claim 7: top-level overall_band key is present in returned dict.
    overall_ok = "overall_band" in (result or {})
    results.append((
        "PASS" if overall_ok else "FAIL",
        "compute_freshness_returns_overall_band_aggregate",
        "compute_freshness must include top-level overall_band key in the response",
    ))

    # ---- Claim 8: BEHAVIORAL happy path -- all green -> NO alert dispatched.
    happy_ok = False
    happy_err = ""
    try:
        sys.modules.pop("backend.services.cycle_health", None)
        from backend.services import cycle_health as ch2  # type: ignore

        # Small ages -> green for every table at cycle_interval=86400s.
        fake_bq2 = _build_fake_bq({
            "paper_trades": 10.0,
            "paper_portfolio_snapshots": 100.0,
            "historical_prices": 1000.0,
            "historical_fundamentals": 100_000.0,
            "historical_macro": 50_000.0,
            "signals_log": 60.0,
        })
        with patch(
            "backend.services.observability.alerting.raise_cron_alert_sync",
        ) as mock_alert:
            res2 = ch2.compute_freshness(fake_bq2, cycle_interval_sec=86_400.0)
        if res2.get("overall_band") != "green":
            happy_err = f"overall_band={res2.get('overall_band')!r}, expected 'green'"
        elif mock_alert.call_count != 0:
            happy_err = f"alert dispatched {mock_alert.call_count} times on all-green (must be 0)"
        else:
            happy_ok = True
    except Exception as e:
        happy_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if happy_ok else "FAIL",
        "behavioral_happy_path_all_green_no_alert",
        f"all-green sources must yield overall_band=green + NO alert dispatch ({happy_err})",
    ))

    # ---- Claim 9: BEHAVIORAL critical -- one red table -> alert dispatched.
    crit_ok = False
    crit_err = ""
    try:
        # Push historical_prices well past 2x its interval (93_600 * 2 = 187_200)
        # to force a "red" band.
        fake_bq3 = _build_fake_bq({
            "paper_trades": 10.0,
            "paper_portfolio_snapshots": 100.0,
            "historical_prices": 500_000.0,  # red
            "historical_fundamentals": 100_000.0,
            "historical_macro": 50_000.0,
            "signals_log": 60.0,
        })
        with patch(
            "backend.services.observability.alerting.raise_cron_alert_sync",
        ) as mock_alert3:
            res3 = ch2.compute_freshness(fake_bq3, cycle_interval_sec=86_400.0)
        if res3.get("overall_band") != "red":
            crit_err = f"overall_band={res3.get('overall_band')!r}, expected 'red'"
        elif mock_alert3.call_count == 0:
            crit_err = "raise_cron_alert_sync NOT called on red band"
        else:
            # Check at least one call had severity P1 + table name in details.
            calls = mock_alert3.call_args_list
            found = False
            for call in calls:
                kwargs = call.kwargs
                if (
                    kwargs.get("severity") == "P1"
                    and (kwargs.get("details") or {}).get("table") == "historical_prices"
                ):
                    found = True
                    break
            if not found:
                crit_err = f"no P1 alert with table=historical_prices in details among {len(calls)} calls"
            else:
                crit_ok = True
    except Exception as e:
        crit_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if crit_ok else "FAIL",
        "slack_alarm_fires_on_critical_band",
        f"red band must trigger raise_cron_alert_sync with severity=P1 + table in details ({crit_err})",
    ))

    # ---- Claim 10: BEHAVIORAL fail-open -- raise_cron_alert_sync raises -> no propagation.
    fo_ok = False
    fo_err = ""
    try:
        fake_bq4 = _build_fake_bq({
            "paper_trades": 10.0,
            "paper_portfolio_snapshots": 100.0,
            "historical_prices": 500_000.0,
            "historical_fundamentals": 100_000.0,
            "historical_macro": 50_000.0,
            "signals_log": 60.0,
        })
        with patch(
            "backend.services.observability.alerting.raise_cron_alert_sync",
            side_effect=RuntimeError("slack down"),
        ):
            res4 = ch2.compute_freshness(fake_bq4, cycle_interval_sec=86_400.0)
        if not isinstance(res4, dict):
            fo_err = f"result not dict: {type(res4)}"
        elif res4.get("overall_band") != "red":
            fo_err = f"overall_band={res4.get('overall_band')!r}, expected 'red'"
        else:
            fo_ok = True
    except Exception as e:
        fo_err = f"caller crashed: {type(e).__name__}: {e}"

    results.append((
        "PASS" if fo_ok else "FAIL",
        "alarm_dispatch_fail_open_on_slack_failure",
        f"raise_cron_alert_sync exception must not propagate ({fo_err})",
    ))

    # ---- Claim 11: source code mentions green/amber/red SLA bands.
    bands_doc = all(b in src for b in ("green", "amber", "red"))
    results.append((
        "PASS" if bands_doc else "FAIL",
        "sla_band_names_green_amber_red_present_in_source",
        "source must reference green/amber/red SLA bands",
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
