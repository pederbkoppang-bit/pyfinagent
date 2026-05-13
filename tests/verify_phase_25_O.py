"""verify_phase_25_O -- Error escalation Slack routing.

Verifies:
  1. `_route_exception_to_p1` exists in scheduler.py with correct kwargs.
  2. Helper builds fingerprint as `f"{type(exc).__name__}:{endpoint}"`.
  3. Helper invokes raise_cron_alert_sync with severity="P1".
  4. At least 4 call sites in scheduler.py invoke `_route_exception_to_p1`.
  5. Behavioral round-trip: patch raise_cron_alert_sync, call the helper,
     assert it was invoked with error_type="ValueError:morning_digest" and
     severity="P1".

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: helper exists with correct signature ───────────────────────
sched_src = (REPO / "backend/slack_bot/scheduler.py").read_text(encoding="utf-8")
tree = ast.parse(sched_src)
fn_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_route_exception_to_p1":
        fn_node = node
        break

if fn_node:
    pos_args = [a.arg for a in fn_node.args.args]
    kw_args = [a.arg for a in fn_node.args.kwonlyargs]
    has_exc = "exc" in pos_args
    has_endpoint = "endpoint" in kw_args
else:
    pos_args = kw_args = []
    has_exc = has_endpoint = False

claim(
    "1. route_exception_to_p1_helper_exists",
    bool(fn_node) and has_exc and has_endpoint,
    f"found={bool(fn_node)} pos={pos_args} kw={kw_args}",
)


# ── Claim 2: fingerprint format ─────────────────────────────────────────
fingerprint_pattern = bool(
    re.search(r'fingerprint\s*=\s*f"\{type\(exc\)\.__name__\}:\{endpoint\}"', sched_src)
)
claim(
    "2. dedup_fingerprint_by_exception_class_plus_endpoint",
    fingerprint_pattern,
    "Fingerprint built as f'{type(exc).__name__}:{endpoint}'" if fingerprint_pattern else "Fingerprint pattern not found",
)


# ── Claim 3: helper passes severity="P1" to raise_cron_alert_sync ──────
if fn_node:
    body_src = ast.unparse(fn_node)
    calls_raise = "raise_cron_alert_sync" in body_src
    # ast.unparse may use either ' or " for the literal; accept either form
    passes_p1 = ('severity="P1"' in body_src) or ("severity='P1'" in body_src)
else:
    calls_raise = passes_p1 = False

claim(
    "3. high_severity_exceptions_route_to_p1_slack",
    calls_raise and passes_p1,
    f"calls_raise_cron_alert_sync={calls_raise} severity_P1={passes_p1}",
)


# ── Claim 4: at least 4 call sites ──────────────────────────────────────
call_sites = re.findall(r"_route_exception_to_p1\(", sched_src)
# Subtract 1 for the def itself
n_calls = len(call_sites) - 1
claim(
    "4. at_least_four_call_sites_wired",
    n_calls >= 4,
    f"call_sites={n_calls} (expected >=4)",
)


# ── Claim 5: behavioral round-trip ─────────────────────────────────────
try:
    with patch(
        "backend.services.observability.alerting.raise_cron_alert_sync"
    ) as mock_raise:
        mock_raise.return_value = True
        # Reimport to ensure patch is applied at module level
        import importlib

        sched_mod = importlib.import_module("backend.slack_bot.scheduler")
        sched_mod._route_exception_to_p1(
            ValueError("test_error"),
            endpoint="morning_digest",
        )
        # Confirm the patch was hit
        assert mock_raise.called, "raise_cron_alert_sync was not called"
        kwargs = mock_raise.call_args.kwargs
        assert kwargs.get("error_type") == "ValueError:morning_digest", (
            f"Expected error_type='ValueError:morning_digest', got {kwargs.get('error_type')!r}"
        )
        assert kwargs.get("severity") == "P1", (
            f"Expected severity='P1', got {kwargs.get('severity')!r}"
        )
        assert kwargs.get("source") == "scheduler", (
            f"Expected source='scheduler', got {kwargs.get('source')!r}"
        )
        details = kwargs.get("details") or {}
        assert details.get("endpoint") == "morning_digest"
        assert details.get("exception_class") == "ValueError"
        rt_ok = True
        rt_detail = "Helper invoked raise_cron_alert_sync with expected fingerprint + P1 severity"
except AssertionError as e:
    rt_ok = False
    rt_detail = f"AssertionError: {e}"
except Exception as e:
    rt_ok = False
    rt_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "5. behavioral_round_trip_helper_fires_p1",
    rt_ok,
    rt_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.O verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
