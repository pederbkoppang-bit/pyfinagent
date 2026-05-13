"""verify_phase_25_N -- Cycle-completion Slack summary.

Verifies:
  1. `format_cycle_summary(summary: dict) -> list[dict]` exists in
     `backend/slack_bot/formatters.py`.
  2. `autonomous_loop.py` has a `cycle_completed_summary` dedup-key branch
     that calls `raise_cron_alert_sync` on the happy path.
  3. (structural) formatter returns Block Kit blocks with header + section.
  4. behavioral round-trip: import + call `format_cycle_summary` with a
     sample summary and confirm the block structure is valid.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: format_cycle_summary exists with correct signature ────────
formatters_src = (REPO / "backend/slack_bot/formatters.py").read_text(encoding="utf-8")
tree = ast.parse(formatters_src)
fn_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "format_cycle_summary":
        fn_node = node
        break
sig_args = [a.arg for a in fn_node.args.args] if fn_node else []
has_summary_arg = "summary" in sig_args
returns_list = bool(fn_node) and ast.unparse(fn_node.returns or ast.Name("None")).startswith("list")
claim(
    "1. format_cycle_summary_function_in_formatters",
    bool(fn_node) and has_summary_arg and returns_list,
    f"found={bool(fn_node)} args={sig_args} returns_list={returns_list}",
)


# ── Claim 2: autonomous_loop emits Slack at cycle completion ───────────
loop_src = (REPO / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
has_summary_branch = (
    'error_type="cycle_completed_summary"' in loop_src
    and 'severity="P3"' in loop_src
)
has_completed_branch = bool(
    re.search(r'elif\s+_final_status\s*==\s*["\']completed["\']', loop_src)
)
imports_raise_cron_alert = "raise_cron_alert_sync" in loop_src
claim(
    "2. autonomous_loop_emits_slack_at_cycle_completion",
    has_summary_branch and has_completed_branch and imports_raise_cron_alert,
    f"branch={has_completed_branch} summary_dedup={has_summary_branch} import={imports_raise_cron_alert}",
)


# ── Claim 3: formatter returns Block Kit blocks ─────────────────────────
# Inspect the function body for header + section types
if fn_node:
    body_src = ast.unparse(fn_node)
    has_header = '"type": "header"' in body_src or "'type': 'header'" in body_src
    has_section = '"type": "section"' in body_src or "'type': 'section'" in body_src
    has_return_blocks = "return blocks" in body_src or "return [" in body_src
else:
    has_header = has_section = has_return_blocks = False

claim(
    "3. format_cycle_summary_returns_block_kit_shape",
    has_header and has_section and has_return_blocks,
    f"header={has_header} section={has_section} return_blocks={has_return_blocks}",
)


# ── Claim 4: behavioral round-trip ─────────────────────────────────────
try:
    from backend.slack_bot.formatters import format_cycle_summary  # noqa: E402

    sample = {
        "cycle_id": "abc12345",
        "started_at": "2026-05-13T01:00:00Z",
        "status": "completed",
        "duration_sec": 372.5,
        "trades_executed": 4,
        "stops_executed": 0,
        "mode": "full",
        "recommendations_count": 7,
    }
    blocks = format_cycle_summary(sample)
    rt_ok = (
        isinstance(blocks, list)
        and len(blocks) >= 3
        and any(b.get("type") == "header" for b in blocks)
        and any(b.get("type") == "section" for b in blocks)
        and any(b.get("type") == "context" for b in blocks)
    )
    detail4 = f"blocks_count={len(blocks)} types={[b.get('type') for b in blocks]}"
except Exception as e:
    rt_ok = False
    detail4 = f"Exception: {type(e).__name__}: {e}"

claim(
    "4. behavioral_round_trip_returns_valid_blocks",
    rt_ok,
    detail4,
)


# ── Claim 5: dedup key distinct from failure path ──────────────────────
# Failure path uses "cycle_<status>" (e.g., cycle_error). Summary path uses
# "cycle_completed_summary". They must not collide.
failure_pattern_present = 'error_type=f"cycle_{_final_status}"' in loop_src
summary_pattern_present = 'error_type="cycle_completed_summary"' in loop_src
distinct_keys = failure_pattern_present and summary_pattern_present
claim(
    "5. dedup_keys_distinct_between_failure_and_summary_paths",
    distinct_keys,
    f"failure_key={failure_pattern_present} summary_key={summary_pattern_present}",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.N verification ===\n")
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
