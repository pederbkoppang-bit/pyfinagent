"""phase-25.H verifier — recent-analyses ticker dedup (5x SNDK fix).

Closes phase-24.5 audit finding F-3: bigquery_client.py:258-268
ORDER BY analysis_date DESC LIMIT 5 returned 5x same ticker when one
ticker had recent analyses.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_H.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BQ_CLIENT = REPO / "backend" / "db" / "bigquery_client.py"


def main() -> int:
    if not BQ_CLIENT.exists():
        print(f"FAIL: {BQ_CLIENT} not found")
        return 1
    text = BQ_CLIENT.read_text(encoding="utf-8")
    results: list[tuple[str, str, str]] = []

    # Claim 1: ROW_NUMBER() OVER (PARTITION BY ticker ...) present
    row_num_pattern = re.compile(
        r'ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*PARTITION\s+BY\s+ticker\s+ORDER\s+BY\s+analysis_date\s+DESC\s*\)',
        re.IGNORECASE,
    )
    results.append(("PASS" if row_num_pattern.search(text) else "FAIL",
                    "get_recent_reports_uses_row_number_partition_by_ticker",
                    "ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) must appear in get_recent_reports"))

    # Claim 2: WHERE rk = 1 (dedup filter)
    rk_filter = re.search(r'WHERE\s+rk\s*=\s*1', text, re.IGNORECASE)
    results.append(("PASS" if rk_filter else "FAIL",
                    "get_recent_reports_filters_to_rk_eq_1",
                    "WHERE rk = 1 must filter to top-1 per ticker"))

    # Claim 3: phase-25.H attribution
    results.append(("PASS" if "phase-25.H" in text else "FAIL",
                    "phase_25_H_attribution_comment_present",
                    "Comment must reference phase-25.H closure of phase-24.5 F-3"))

    # Claim 4: get_recent_reports signature unchanged (still accepts limit)
    sig = re.search(r'def get_recent_reports\s*\(\s*self,\s*limit\s*:\s*int\s*=\s*\d+\s*\)', text)
    results.append(("PASS" if sig else "FAIL",
                    "get_recent_reports_signature_preserved",
                    "get_recent_reports(self, limit: int = N) signature must be unchanged"))

    # Claim 5: AST clean
    try:
        ast.parse(text)
        ast_ok, ast_detail = True, ""
    except SyntaxError as e:
        ast_ok, ast_detail = False, f"SyntaxError: {e}"
    results.append(("PASS" if ast_ok else "FAIL",
                    "bigquery_client_py_syntax_clean", ast_detail))

    # Claim 6: ScalarQueryParameter for limit still used (no SQL injection)
    param_pattern = re.search(r'ScalarQueryParameter\s*\(\s*["\']limit["\']\s*,\s*["\']INT64["\']', text)
    results.append(("PASS" if param_pattern else "FAIL",
                    "get_recent_reports_uses_parameterized_query_for_limit",
                    "ScalarQueryParameter for limit must be preserved (no SQL injection)"))

    # --- Output ---
    print("=== phase-25.H (recent-analyses ticker dedup) verifier ===")
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
