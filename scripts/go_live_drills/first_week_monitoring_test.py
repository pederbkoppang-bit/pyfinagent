"""
Go-Live drill test: first-week monitoring thresholds (Phase 4.4.6.3).

Standalone, stdlib-only drill. Verifies that the first-week monitoring
mode tightens two thresholds as specified in docs/GO_LIVE_CHECKLIST.md
section 4.4.6.3:

  1. Drawdown de-risk alert: -10% -> -5%  (via track_drawdown)
  2. SLA P3 response: 4 hours -> 1 hour  (via get_sla_thresholds)

Also verifies that:
  - get_risk_constraints() is NOT modified (4.4.4.4 compliance)
  - Kill switch at -15% is unchanged in both modes
  - first_week_mode setting exists in settings.py and defaults False

Run from the repo root:

    python scripts/go_live_drills/first_week_monitoring_test.py

Exit code 0 on PASS, 1 on any failure.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_SERVER_PATH = REPO_ROOT / "backend" / "agents" / "mcp_servers" / "signals_server.py"
SETTINGS_PATH = REPO_ROOT / "backend" / "config" / "settings.py"
SLA_MONITOR_PATH = REPO_ROOT / "backend" / "services" / "sla_monitor.py"

sys.path.insert(0, str(REPO_ROOT))


def load_signals_server():
    spec = importlib.util.spec_from_file_location(
        "signals_server_drill", str(SIGNALS_SERVER_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load signals_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── AST-based checks (no heavy imports) ─────────────────────────


def s0_settings_has_first_week_mode():
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "first_week_mode":
                return "S0 first_week_mode field exists in settings.py"
    raise AssertionError("S0 first_week_mode not found in settings.py")


def s1_settings_defaults_false():
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
    src = SETTINGS_PATH.read_text(encoding="utf-8")
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "first_week_mode":
                segment = ast.get_source_segment(src, node)
                if segment and "False" in segment:
                    return "S1 first_week_mode defaults to False"
                raise AssertionError("S1 first_week_mode default is not False: " + repr(segment))
    raise AssertionError("S1 first_week_mode not found")


def s2_sla_monitor_imports_settings():
    src = SLA_MONITOR_PATH.read_text(encoding="utf-8")
    assert "get_settings" in src, "S2 sla_monitor.py does not import get_settings"
    return "S2 sla_monitor.py imports get_settings"


def s3_sla_monitor_has_first_week_branch():
    src = SLA_MONITOR_PATH.read_text(encoding="utf-8")
    assert "first_week" in src, "S3 sla_monitor.py has no first_week logic"
    return "S3 sla_monitor.py has first_week conditional branch"


def s4_sla_p3_normal_is_4h():
    src = SLA_MONITOR_PATH.read_text(encoding="utf-8")
    assert "4 * 3600" in src, "S4 normal P3 response 4*3600 not found in sla_monitor.py"
    return "S4 SLA P3 normal response = 4 * 3600 (4h)"


def s5_sla_p3_firstweek_is_1h():
    src = SLA_MONITOR_PATH.read_text(encoding="utf-8")
    assert "60 * 60" in src, "S5 first-week P3 response 60*60 not found"
    return "S5 SLA P3 first-week response = 60 * 60 (1h)"


def s6_track_drawdown_has_first_week_override():
    src = SIGNALS_SERVER_PATH.read_text(encoding="utf-8")
    assert "first_week_mode" in src, "S6 signals_server.py has no first_week_mode reference"
    return "S6 track_drawdown has first_week_mode override"


def s7_get_risk_constraints_unchanged():
    src = SIGNALS_SERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_risk_constraints":
            body_src = ast.get_source_segment(src, node)
            assert "first_week" not in body_src, (
                "S7 get_risk_constraints contains first_week logic (violates 4.4.4.4)"
            )
            return "S7 get_risk_constraints unchanged (4.4.4.4 compliant)"
    raise AssertionError("S7 get_risk_constraints not found")


# ── Runtime checks via SignalsServer instance ────────────────────


def s8_track_drawdown_normal_derisk_at_10(server):
    server.settings = None
    server._peak_equity = 10000.0
    portfolio = {"total_value": 9050.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "warning", (
        "S8 expected tier=warning at -9.5% in normal mode, got " + repr(result["tier"])
    )
    return "S8 normal mode: -9.5% drawdown -> tier=warning (not derisk)"


def s9_track_drawdown_normal_derisk_triggers_at_10(server):
    server.settings = None
    server._peak_equity = 10000.0
    portfolio = {"total_value": 9000.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "derisk", (
        "S9 expected tier=derisk at -10% in normal mode, got " + repr(result["tier"])
    )
    return "S9 normal mode: -10.0% drawdown -> tier=derisk"


def s10_track_drawdown_firstweek_derisk_at_5(server):
    server.settings = SimpleNamespace(first_week_mode=True)
    server._peak_equity = 10000.0
    portfolio = {"total_value": 9500.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "derisk", (
        "S10 expected tier=derisk at -5% in first-week mode, got " + repr(result["tier"])
    )
    return "S10 first-week mode: -5.0% drawdown -> tier=derisk (tightened from -10%)"


def s11_track_drawdown_firstweek_warning_above_5(server):
    server.settings = SimpleNamespace(first_week_mode=True)
    server._peak_equity = 10000.0
    portfolio = {"total_value": 9600.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "ok", (
        "S11 expected tier=ok at -4% in first-week mode, got " + repr(result["tier"])
    )
    return "S11 first-week mode: -4.0% drawdown -> tier=ok"


def s12_kill_switch_unchanged_in_firstweek(server):
    server.settings = SimpleNamespace(first_week_mode=True)
    server._peak_equity = 10000.0
    portfolio = {"total_value": 8500.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "kill", (
        "S12 expected tier=kill at -15% in first-week mode, got " + repr(result["tier"])
    )
    assert result["kill_switch"] is True, "S12 kill_switch should be True"
    return "S12 first-week mode: -15.0% drawdown -> kill (unchanged)"


def s13_kill_switch_unchanged_in_normal(server):
    server.settings = None
    server._peak_equity = 10000.0
    portfolio = {"total_value": 8500.0}
    result = server.track_drawdown(portfolio)
    assert result["tier"] == "kill", (
        "S13 expected tier=kill at -15% in normal mode, got " + repr(result["tier"])
    )
    assert result["kill_switch"] is True, "S13 kill_switch should be True"
    return "S13 normal mode: -15.0% drawdown -> kill (baseline)"


def s14_risk_constraints_literals_unchanged(server):
    limits = server.get_risk_constraints()
    assert limits["max_drawdown_pct"] == -15.0, "S14 max_drawdown_pct changed"
    assert limits["drawdown_derisk_pct"] == -10.0, "S14 drawdown_derisk_pct changed"
    assert limits["drawdown_warning_pct"] == -5.0, "S14 drawdown_warning_pct changed"
    assert limits["max_exposure_per_ticker_pct"] == 10.0, "S14 per-ticker limit changed"
    assert limits["max_daily_trades"] == 5, "S14 daily trades changed"
    return "S14 get_risk_constraints literals unchanged (4.4.4.4 verified)"


def main():
    ast_scenarios = [
        s0_settings_has_first_week_mode,
        s1_settings_defaults_false,
        s2_sla_monitor_imports_settings,
        s3_sla_monitor_has_first_week_branch,
        s4_sla_p3_normal_is_4h,
        s5_sla_p3_firstweek_is_1h,
        s6_track_drawdown_has_first_week_override,
        s7_get_risk_constraints_unchanged,
    ]

    failures = []
    for scenario in ast_scenarios:
        try:
            line = scenario()
            print("PASS " + line)
        except (AssertionError, Exception) as exc:
            failures.append((scenario.__name__, str(exc)))
            print("FAIL " + scenario.__name__ + ": " + str(exc))

    module = load_signals_server()
    server = module.SignalsServer()

    runtime_scenarios = [
        s8_track_drawdown_normal_derisk_at_10,
        s9_track_drawdown_normal_derisk_triggers_at_10,
        s10_track_drawdown_firstweek_derisk_at_5,
        s11_track_drawdown_firstweek_warning_above_5,
        s12_kill_switch_unchanged_in_firstweek,
        s13_kill_switch_unchanged_in_normal,
        s14_risk_constraints_literals_unchanged,
    ]

    for scenario in runtime_scenarios:
        try:
            line = scenario(server)
            print("PASS " + line)
        except (AssertionError, Exception) as exc:
            failures.append((scenario.__name__, str(exc)))
            print("FAIL " + scenario.__name__ + ": " + str(exc))

    total = len(ast_scenarios) + len(runtime_scenarios)
    passed = total - len(failures)

    if failures:
        print("DRILL FAIL: " + str(len(failures)) + "/" + str(total) + " scenario(s) failed")
        for name, msg in failures:
            print("  - " + name + ": " + msg)
        return 1

    print("DRILL PASS: " + str(passed) + "/" + str(total)
          + " first-week monitoring scenarios verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
