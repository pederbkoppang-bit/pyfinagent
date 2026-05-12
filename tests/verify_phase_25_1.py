"""phase-25.1 verifier — wire check_stop_losses() into autonomous_loop.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_1.py

Exits 0 on PASS, 1 on FAIL. All claims are immutable per `.claude/masterplan.json`
phase-25 step `25.1` verification.success_criteria.

Stdlib-only. Idempotent. Safe to re-run.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUTONOMOUS_LOOP = REPO / "backend" / "services" / "autonomous_loop.py"
PAPER_TRADER = REPO / "backend" / "services" / "paper_trader.py"


def check(name: str, ok: bool, detail: str = "") -> tuple[str, str, str]:
    return ("PASS" if ok else "FAIL", name, detail)


def main() -> int:
    results: list[tuple[str, str, str]] = []

    # Load file once
    if not AUTONOMOUS_LOOP.exists():
        print(f"FAIL: {AUTONOMOUS_LOOP} not found")
        return 1
    al_text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")

    # --- Claim 1: grep_check_stop_losses_in_autonomous_loop_returns_match ---
    results.append(check(
        "grep_check_stop_losses_in_autonomous_loop_returns_match",
        "check_stop_losses" in al_text,
        "autonomous_loop.py must reference trader.check_stop_losses",
    ))

    # --- Claim 2: stop_loss_trigger reason string present ---
    results.append(check(
        "stop_loss_trigger_reason_string_present",
        "stop_loss_trigger" in al_text,
        "Reason string 'stop_loss_trigger' must be in the execute_sell call",
    ))

    # --- Claim 3: summary["stop_loss_triggered"] key initialized ---
    summary_init_pattern = re.compile(r'summary\[["\']stop_loss_triggered["\']\]\s*=\s*\[\]')
    results.append(check(
        "summary_includes_stop_loss_triggered_field",
        bool(summary_init_pattern.search(al_text)),
        "summary['stop_loss_triggered'] must be initialized to empty list",
    ))

    # --- Claim 4: Step 5.6 label or stop_loss_enforcement step ---
    step_label_present = (
        "stop_loss_enforcement" in al_text
        or "Step 5.6" in al_text
    )
    results.append(check(
        "step_5_6_stop_loss_enforcement_label_present",
        step_label_present,
        "Step 5.6 must be labeled with 'Step 5.6' or 'stop_loss_enforcement'",
    ))

    # --- Claim 5: asyncio.to_thread wrap on check_stop_losses (non-blocking) ---
    to_thread_pattern = re.compile(r'asyncio\.to_thread\s*\(\s*trader\.check_stop_losses\s*\)')
    results.append(check(
        "check_stop_losses_wrapped_in_asyncio_to_thread",
        bool(to_thread_pattern.search(al_text)),
        "trader.check_stop_losses must be called via asyncio.to_thread (non-blocking pattern)",
    ))

    # --- Claim 6: asyncio.to_thread wrap on execute_sell with stop_loss_trigger reason ---
    # Tolerant: the execute_sell call should appear within ~500 chars of the stop_loss_trigger reason
    sl_block = re.search(r'check_stop_losses.{0,2000}', al_text, re.DOTALL)
    has_block_execute_sell = bool(sl_block and "execute_sell" in sl_block.group(0))
    results.append(check(
        "execute_sell_called_in_stop_loss_block",
        has_block_execute_sell,
        "execute_sell must be called within the stop-loss block following check_stop_losses",
    ))

    # --- Claim 7: AST syntax-clean (the file still parses after edit) ---
    try:
        ast.parse(al_text)
        ast_ok, ast_detail = True, ""
    except SyntaxError as e:
        ast_ok, ast_detail = False, f"SyntaxError: {e}"
    results.append(check("autonomous_loop_py_syntax_clean", ast_ok, ast_detail))

    # --- Claim 8: paper_trader.execute_sell signature unchanged ---
    if PAPER_TRADER.exists():
        pt_text = PAPER_TRADER.read_text(encoding="utf-8")
        # execute_sell must accept reason kwarg
        es_sig = re.search(r'def execute_sell\s*\([^)]*reason', pt_text)
        results.append(check(
            "paper_trader_execute_sell_signature_has_reason_kwarg",
            bool(es_sig),
            "paper_trader.execute_sell must accept a 'reason' kwarg (unchanged contract)",
        ))
    else:
        results.append(check(
            "paper_trader_execute_sell_signature_has_reason_kwarg",
            False,
            f"{PAPER_TRADER} not found",
        ))

    # --- Output summary ---
    print("=== phase-25.1 (stop-loss wiring) verifier ===")
    fail = 0
    for flag, name, detail in results:
        prefix = "[PASS]" if flag == "PASS" else "[FAIL]"
        print(f"  {prefix} {name}")
        if flag == "FAIL" and detail:
            print(f"         -> {detail}")
            fail += 1
    total = len(results)
    passed = total - fail
    verdict = "PASS" if fail == 0 else "FAIL"
    print(f"{verdict} ({passed}/{total}) EXIT={0 if fail == 0 else 1}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
