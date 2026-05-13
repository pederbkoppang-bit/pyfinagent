"""verify_phase_25_P -- Weekly autoresearch summary Slack notification.

Verifies:
  1. `format_autoresearch_summary` exists in formatters with `results: dict` arg + list return.
  2. cron.py emits raise_cron_alert_sync with severity="P3" + dedup key.
  3. Behavioral: call format_autoresearch_summary with sample dict; assert valid Block Kit shape.
  4. Behavioral: patch raise_cron_alert_sync; call run_meta_evolution_cycle; assert P3 alert fired.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results_log: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results_log.append((name, bool(condition), detail))


# ── Claim 1: format_autoresearch_summary exists ────────────────────────
formatters_src = (REPO / "backend/slack_bot/formatters.py").read_text(encoding="utf-8")
tree = ast.parse(formatters_src)
fn_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "format_autoresearch_summary":
        fn_node = node
        break

if fn_node:
    args = [a.arg for a in fn_node.args.args]
    has_results_arg = "results" in args
    returns_list = ast.unparse(fn_node.returns).startswith("list")
else:
    args = []
    has_results_arg = returns_list = False

claim(
    "1. format_autoresearch_summary_in_formatters",
    bool(fn_node) and has_results_arg and returns_list,
    f"found={bool(fn_node)} args={args} returns_list={returns_list}",
)


# ── Claim 2: cron.py emits the P3 alert ────────────────────────────────
cron_src = (REPO / "backend/meta_evolution/cron.py").read_text(encoding="utf-8")
imports_raise = "raise_cron_alert_sync" in cron_src
has_p3 = 'severity="P3"' in cron_src or "severity='P3'" in cron_src
has_summary_dedup = '"meta_evolution_weekly_summary"' in cron_src or "'meta_evolution_weekly_summary'" in cron_src
claim(
    "2. meta_evolution_cron_emits_slack_on_sunday_completion",
    imports_raise and has_p3 and has_summary_dedup,
    f"import={imports_raise} severity_P3={has_p3} dedup_key={has_summary_dedup}",
)


# ── Claim 3: behavioral -- formatter returns Block Kit shape ───────────
try:
    from backend.slack_bot.formatters import format_autoresearch_summary  # noqa: E402
    sample = {
        "started_at": "2026-05-10T02:00:00+00:00",
        "finished_at": "2026-05-10T02:01:30+00:00",
        "duration_seconds": 90.5,
        "cron_allocations": {"a": 1, "b": 2},
        "provider_allocations": {"openai": 0.4, "anthropic": 0.6},
        "archetype_count": 7,
        "errors": [],
    }
    blocks = format_autoresearch_summary(sample)
    rt3_ok = (
        isinstance(blocks, list)
        and any(b.get("type") == "header" for b in blocks)
        and any(b.get("type") == "section" for b in blocks)
        and any(b.get("type") == "context" for b in blocks)
    )
    rt3_detail = f"blocks={len(blocks)} types={[b.get('type') for b in blocks]}"
except Exception as e:
    rt3_ok = False
    rt3_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "3. format_autoresearch_summary_returns_block_kit_shape",
    rt3_ok,
    rt3_detail,
)


# ── Claim 4: behavioral -- run_meta_evolution_cycle fires P3 alert ────
# Patch raise_cron_alert_sync and call run_meta_evolution_cycle. Sub-modules
# will fail-open into results["errors"] (already the existing behavior), but
# the new P3 summary alert at end of function should still fire.
try:
    with patch(
        "backend.services.observability.alerting.raise_cron_alert_sync"
    ) as mock_raise:
        mock_raise.return_value = True
        from backend.meta_evolution.cron import run_meta_evolution_cycle  # noqa: E402
        run_meta_evolution_cycle()
        rt4_ok = mock_raise.called and any(
            c.kwargs.get("severity") == "P3"
            and c.kwargs.get("error_type") == "meta_evolution_weekly_summary"
            for c in mock_raise.call_args_list
        )
        kwargs_seen = [
            (c.kwargs.get("error_type"), c.kwargs.get("severity"))
            for c in mock_raise.call_args_list
        ]
        rt4_detail = f"called={mock_raise.called} error_types_seen={kwargs_seen}"
except Exception as e:
    rt4_ok = False
    rt4_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "4. behavioral_run_cycle_fires_p3_summary_alert",
    rt4_ok,
    rt4_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.P verification ===\n")
all_pass = True
for name, ok, detail in results_log:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results_log)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results_log if not ok)
    print(f"{failed} of {len(results_log)} claims FAILED")
    sys.exit(1)
