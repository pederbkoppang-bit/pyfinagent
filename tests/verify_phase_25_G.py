"""phase-25.G verifier — Slack digest P&L data source + field key fix.

Closes phase-24.5 audit F-1 + F-2 (wrong endpoint + wrong field key).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_G.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEDULER = REPO / "backend" / "slack_bot" / "scheduler.py"
FORMATTERS = REPO / "backend" / "slack_bot" / "formatters.py"
COMMANDS = REPO / "backend" / "slack_bot" / "commands.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for path in (SCHEDULER, FORMATTERS, COMMANDS):
        if not path.exists():
            print(f"FAIL: {path} not found")
            return 1

    sched_text = SCHEDULER.read_text(encoding="utf-8")
    fmt_text = FORMATTERS.read_text(encoding="utf-8")
    cmd_text = COMMANDS.read_text(encoding="utf-8")

    # Claim 1: scheduler.py morning digest uses /api/paper-trading/portfolio
    # (Tolerant: should have at least 2 references to paper-trading/portfolio for morning + evening)
    pp_refs = sched_text.count("/api/paper-trading/portfolio")
    results.append(("PASS" if pp_refs >= 2 else "FAIL",
                    "scheduler_uses_paper_trading_portfolio_endpoint_twice_for_morning_and_evening_digest",
                    f"expected >=2 occurrences of /api/paper-trading/portfolio in scheduler.py; got {pp_refs}"))

    # Claim 2: scheduler.py no longer references the legacy /api/portfolio/performance
    legacy_refs_sched = sched_text.count("/api/portfolio/performance")
    results.append(("PASS" if legacy_refs_sched == 0 else "FAIL",
                    "scheduler_no_longer_references_legacy_in_memory_portfolio_endpoint",
                    f"found {legacy_refs_sched} legacy /api/portfolio/performance references; should be 0"))

    # Claim 3: /portfolio slash command uses paper-trading endpoint
    results.append(("PASS" if "/api/paper-trading/portfolio" in cmd_text else "FAIL",
                    "portfolio_slash_command_uses_paper_trading_portfolio_endpoint",
                    "commands.py handle_portfolio must use /api/paper-trading/portfolio"))

    # Claim 4: /portfolio slash command no longer uses legacy endpoint
    legacy_refs_cmd = cmd_text.count("/api/portfolio/performance")
    results.append(("PASS" if legacy_refs_cmd == 0 else "FAIL",
                    "portfolio_slash_command_no_longer_references_legacy_endpoint",
                    f"found {legacy_refs_cmd} legacy refs in commands.py; should be 0"))

    # Claim 5: formatters.py reads total_pnl_pct field (the canonical key from paper_trading endpoint)
    # Tolerant: at least one site must use total_pnl_pct
    pnl_pct_refs = fmt_text.count("total_pnl_pct")
    results.append(("PASS" if pnl_pct_refs >= 1 else "FAIL",
                    "formatters_reads_total_pnl_pct_field_at_least_once",
                    f"formatters.py must read total_pnl_pct (paper-trading endpoint's key); got {pnl_pct_refs} occurrences"))

    # Claim 6: AST clean for all three files
    for label, path, text in (
        ("scheduler", SCHEDULER, sched_text),
        ("formatters", FORMATTERS, fmt_text),
        ("commands", COMMANDS, cmd_text),
    ):
        try:
            ast.parse(text)
            results.append(("PASS", f"{label}_py_syntax_clean", ""))
        except SyntaxError as e:
            results.append(("FAIL", f"{label}_py_syntax_clean", f"SyntaxError in {path}: {e}"))

    # Claim 7: phase-25.G attribution comment present in at least one file
    attribution_present = (
        "phase-25.G" in sched_text
        or "phase-25.G" in fmt_text
        or "phase-25.G" in cmd_text
    )
    results.append(("PASS" if attribution_present else "FAIL",
                    "phase_25_G_attribution_comment_present",
                    "Comment must reference phase-25.G closure of phase-24.5 F-1 + F-2"))

    # --- Output ---
    print("=== phase-25.G (Slack digest P&L fix) verifier ===")
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
