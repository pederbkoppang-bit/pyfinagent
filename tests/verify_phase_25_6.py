"""phase-25.6 verifier — no-stop-on-entry hard block in execute_buy().

Closes phase-24.1 audit F-4 (_extract_stop_loss fallback only applied
to NEW buys; positions pre-dating phase-23.1.8 had None forever).
Defense-in-depth alongside 25.2 backfill: prevents regression.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_6.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
PAPER_TRADER = REPO / "backend" / "services" / "paper_trader.py"
sys.path.insert(0, str(REPO))


def main() -> int:
    if not PAPER_TRADER.exists():
        print(f"FAIL: {PAPER_TRADER} not found")
        return 1
    text = PAPER_TRADER.read_text(encoding="utf-8")
    results: list[tuple[str, str, str]] = []

    # Claim 1: execute_buy starts with stop_loss_price None-check
    none_check_pattern = re.search(
        r'def execute_buy.*?if stop_loss_price is None',
        text,
        re.DOTALL,
    )
    results.append(("PASS" if none_check_pattern else "FAIL",
                    "execute_buy_checks_stop_loss_price_is_none_at_entry",
                    "execute_buy must include `if stop_loss_price is None:` check"))

    # Claim 2: synthesizes stop from settings.paper_default_stop_loss_pct
    default_pct_pattern = re.search(
        r'paper_default_stop_loss_pct',
        text,
    )
    results.append(("PASS" if default_pct_pattern else "FAIL",
                    "execute_buy_uses_paper_default_stop_loss_pct_for_synthesis",
                    "settings.paper_default_stop_loss_pct must be used to synthesize the stop"))

    # Claim 3: phase-25.6 attribution
    results.append(("PASS" if "phase-25.6" in text else "FAIL",
                    "phase_25_6_attribution_comment_present",
                    "Comment must reference phase-25.6 closure of phase-24.1 F-4"))

    # Claim 4: logger.warning fires when default applied
    log_pattern = re.search(
        r'logger\.warning\s*\([^)]*phase-25\.6[^)]*\)',
        text,
    )
    results.append(("PASS" if log_pattern else "FAIL",
                    "execute_buy_logs_warning_when_default_stop_applied",
                    "logger.warning must fire (operator visibility) when default stop synthesized"))

    # Claim 5: AST clean
    try:
        ast.parse(text)
        results.append(("PASS", "paper_trader_py_syntax_clean", ""))
    except SyntaxError as e:
        results.append(("FAIL", "paper_trader_py_syntax_clean", f"SyntaxError: {e}"))

    # Claim 6: behavioral round-trip — call execute_buy with stop_loss_price=None,
    # confirm a stop is computed before persistence. Use ast walk rather than
    # full execution (too many side effects to mock cleanly).
    # We check that within the execute_buy function body, after the None-check,
    # stop_loss_price is reassigned to a non-None value.
    # Heuristic: find the substring "if stop_loss_price is None:" and confirm
    # the next ~10 lines reassign stop_loss_price.
    none_block = re.search(
        r'if stop_loss_price is None:\s*\n(.*?)\n\s+portfolio\s*=',
        text,
        re.DOTALL,
    )
    if none_block:
        block_body = none_block.group(1)
        reassigns = re.search(r'stop_loss_price\s*=\s*round', block_body)
        results.append(("PASS" if reassigns else "FAIL",
                        "stop_loss_price_reassigned_to_computed_value_within_none_branch",
                        "Within the `if stop_loss_price is None:` branch, stop_loss_price must be reassigned via round()"))
    else:
        results.append(("FAIL",
                        "stop_loss_price_reassigned_to_computed_value_within_none_branch",
                        "Could not locate the if-None block before portfolio fetch"))

    # Claim 7: the formula is `price * (1 - default_pct / 100.0)`
    formula = re.search(
        r'price\s*\*\s*\(\s*1\.0\s*-\s*default_pct\s*/\s*100\.0\s*\)',
        text,
    )
    results.append(("PASS" if formula else "FAIL",
                    "stop_synthesized_via_canonical_formula_price_times_one_minus_pct_over_100",
                    "Formula must be price * (1.0 - default_pct / 100.0)"))

    # Claim 8: guard against price=0 (don't compute stop when price unavailable)
    price_guard = re.search(r'if price > 0:', text)
    results.append(("PASS" if price_guard else "FAIL",
                    "execute_buy_guards_against_zero_price_before_computing_default_stop",
                    "Must guard `if price > 0:` to avoid computing 0 stop on missing price"))

    # --- Output ---
    print("=== phase-25.6 (no-stop-on-entry hard block) verifier ===")
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
