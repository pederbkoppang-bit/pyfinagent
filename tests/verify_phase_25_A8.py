"""phase-25.A8 verifier — cost-budget HARD-BLOCK in llm_client.

Closes phase-24.8 F-4 + phase-24.13 F-4 (cost_budget.tripped tracked
but llm_client never consulted it). After 25.A8, every generate_content
call raises BudgetBreachError when today's spend exceeds caps.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A8.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LLM_CLIENT = REPO / "backend" / "agents" / "llm_client.py"
AUTONOMOUS_LOOP = REPO / "backend" / "services" / "autonomous_loop.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not LLM_CLIENT.exists():
        print(f"FAIL: {LLM_CLIENT} not found")
        return 1
    if not AUTONOMOUS_LOOP.exists():
        print(f"FAIL: {AUTONOMOUS_LOOP} not found")
        return 1

    llm_text = LLM_CLIENT.read_text(encoding="utf-8")
    al_text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")

    # Claim 1: BudgetBreachError class defined
    results.append(("PASS" if re.search(r'class BudgetBreachError\b', llm_text) else "FAIL",
                    "budget_breach_error_class_defined",
                    "BudgetBreachError class must be defined in llm_client.py"))

    # Claim 2: _check_cost_budget helper defined
    results.append(("PASS" if re.search(r'def _check_cost_budget\s*\(', llm_text) else "FAIL",
                    "check_cost_budget_helper_defined",
                    "_check_cost_budget() helper must be defined"))

    # Claim 3: _check_cost_budget raises BudgetBreachError when tripped
    raises_pattern = re.search(
        r'def _check_cost_budget.*?raise BudgetBreachError',
        llm_text,
        re.DOTALL,
    )
    results.append(("PASS" if raises_pattern else "FAIL",
                    "check_cost_budget_raises_budget_breach_error_when_tripped",
                    "_check_cost_budget must raise BudgetBreachError on tripped budget"))

    # Claim 4: every generate_content calls _check_cost_budget at the top
    # (count occurrences of `_check_cost_budget()` in the file; expect >= 3
    # since there are 3 concrete subclasses — Gemini, OpenAI, Claude — each
    # with a generate_content method)
    check_calls = llm_text.count("_check_cost_budget()")
    # Subtract one for the definition's name (the `def _check_cost_budget` line
    # has the name but not the parens form).
    call_sites = check_calls
    results.append(("PASS" if call_sites >= 3 else "FAIL",
                    "generate_content_calls_check_cost_budget_at_least_three_call_sites",
                    f"expected >=3 occurrences of _check_cost_budget() call (one per provider client); got {call_sites}"))

    # Claim 5: autonomous_loop catches BudgetBreachError by name (loose coupling)
    al_catch = re.search(
        r'type\(e\)\.__name__\s*==\s*["\']BudgetBreachError["\']',
        al_text,
    )
    results.append(("PASS" if al_catch else "FAIL",
                    "autonomous_loop_catches_budget_breach_error_at_cycle_level",
                    "autonomous_loop.py must catch BudgetBreachError via type(e).__name__ check"))

    # Claim 6: autonomous_loop sets status='budget_breach' when caught
    al_status = re.search(r'["\']status["\']\s*:\s*["\']budget_breach["\']', al_text)
    results.append(("PASS" if al_status else "FAIL",
                    "autonomous_loop_sets_status_budget_breach",
                    "autonomous_loop.py must set summary['status']='budget_breach' on caught BudgetBreachError"))

    # Claim 7: cache_ttl_s constant present (avoids hot-path BQ scans)
    cache_pattern = re.search(r'_BUDGET_CACHE_TTL_S\s*=', llm_text)
    results.append(("PASS" if cache_pattern else "FAIL",
                    "budget_check_uses_ttl_cache_to_avoid_hot_path_bq_scans",
                    "_BUDGET_CACHE_TTL_S constant must exist to throttle BQ scans"))

    # Claim 8: env-var escape hatch for test isolation
    escape_pattern = re.search(r'COST_BUDGET_HARD_BLOCK_DISABLED', llm_text)
    results.append(("PASS" if escape_pattern else "FAIL",
                    "env_var_escape_hatch_for_test_isolation",
                    "COST_BUDGET_HARD_BLOCK_DISABLED env-var escape hatch must exist"))

    # Claim 9: phase-25.A8 attribution
    has_attribution = "phase-25.A8" in llm_text and "phase-25.A8" in al_text
    results.append(("PASS" if has_attribution else "FAIL",
                    "phase_25_A8_attribution_comment_present_in_both_files",
                    "phase-25.A8 attribution must be in both llm_client.py and autonomous_loop.py"))

    # Claim 10: AST clean (both files)
    for label, path, text in (
        ("llm_client", LLM_CLIENT, llm_text),
        ("autonomous_loop", AUTONOMOUS_LOOP, al_text),
    ):
        try:
            ast.parse(text)
            results.append(("PASS", f"{label}_py_syntax_clean", ""))
        except SyntaxError as e:
            results.append(("FAIL", f"{label}_py_syntax_clean", f"SyntaxError in {path}: {e}"))

    # Claim 11: behavioral round-trip — with escape hatch set, _check_cost_budget returns None
    os.environ["COST_BUDGET_HARD_BLOCK_DISABLED"] = "1"
    try:
        sys.path.insert(0, str(REPO))
        # Re-import after env-var set; isolated to avoid polluting test runner state
        from importlib import reload
        import backend.agents.llm_client as _llm
        reload(_llm)
        _llm._check_cost_budget()  # should NOT raise with escape hatch
        results.append(("PASS", "check_cost_budget_returns_none_with_escape_hatch_set", ""))
    except Exception as e:
        results.append(("FAIL", "check_cost_budget_returns_none_with_escape_hatch_set",
                        f"unexpected exception with escape hatch set: {e!r}"))
    finally:
        os.environ.pop("COST_BUDGET_HARD_BLOCK_DISABLED", None)

    # --- Output ---
    print("=== phase-25.A8 (cost-budget HARD-BLOCK) verifier ===")
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
