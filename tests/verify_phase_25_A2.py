"""phase-25.A2 verifier — wire bq.save_report into full pipeline.

Closes phase-24.2 audit F-2: orchestrator.py had zero save_report calls;
/reports page was empty because full-pipeline runs evaporated without
persistence. Fix: full path now returns `_path: "full"` and the
persist guard at L277/295 calls `_persist_analysis` for both paths.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_A2.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUTONOMOUS_LOOP = REPO / "backend" / "services" / "autonomous_loop.py"


def main() -> int:
    if not AUTONOMOUS_LOOP.exists():
        print(f"FAIL: {AUTONOMOUS_LOOP} not found")
        return 1
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    results: list[tuple[str, str, str]] = []

    # Claim 1: full-pipeline return dict includes `_path: "full"`
    full_path_marker = re.search(r'["\']_path["\']\s*:\s*["\']full["\']', text)
    results.append(("PASS" if full_path_marker else "FAIL",
                    "full_pipeline_return_dict_includes_path_full_marker",
                    "_run_single_analysis full-path return must include `_path: 'full'`"))

    # Claim 2: persist guard accepts both 'lite' and 'full'
    both_guard = re.findall(r'analysis\.get\(["\']_path["\']\)\s+in\s+\(["\']lite["\'],\s*["\']full["\']\)', text)
    results.append(("PASS" if len(both_guard) >= 2 else "FAIL",
                    "persist_guards_accept_both_lite_and_full_paths",
                    f"both Step 3 and Step 4 persist guards must accept both paths; found {len(both_guard)}/2"))

    # Claim 3: `_persist_analysis` function defined (renamed from _persist_lite_analysis)
    new_def = re.search(r'async def _persist_analysis\s*\(', text)
    results.append(("PASS" if new_def else "FAIL",
                    "persist_analysis_function_defined",
                    "async _persist_analysis must be defined (renamed from _persist_lite_analysis)"))

    # Claim 4: persist guards call `_persist_analysis` (not legacy _persist_lite_analysis)
    new_callsites = text.count("await _persist_analysis(analysis, bq)")
    legacy_callsites = text.count("await _persist_lite_analysis(analysis, bq)")
    results.append(("PASS" if new_callsites >= 2 and legacy_callsites == 0 else "FAIL",
                    "all_persist_callsites_use_renamed_function_no_legacy_left",
                    f"expected >=2 new callsites and 0 legacy; got new={new_callsites}, legacy={legacy_callsites}"))

    # Claim 5: phase-25.A2 attribution
    results.append(("PASS" if "phase-25.A2" in text else "FAIL",
                    "phase_25_A2_attribution_comment_present",
                    "Comment must reference phase-25.A2 closure of phase-24.2 F-2"))

    # Claim 6: stale comment about run_full_analysis self-persisting is gone OR corrected
    # Tolerant: the old stale comment said "the full orchestrator path writes its own row via bq.save_report".
    # Post-25.A2, this stale claim must NOT appear with positive framing. If it appears, it must be
    # framed as historical (e.g., "stale doc comment").
    stale_claim_active = re.search(
        r'full orchestrator path writes its own row via bq\.save_report inside run_full_analysis',
        text,
    )
    # If present, it must be in the context of phase-25.A2 correction
    if stale_claim_active:
        context_start = max(0, stale_claim_active.start() - 200)
        context = text[context_start:stale_claim_active.end() + 100]
        is_corrected = "phase-25.A2" in context or "stale" in context.lower() or "did NOT" in context
        results.append(("PASS" if is_corrected else "FAIL",
                        "stale_comment_about_run_full_analysis_self_persisting_corrected_or_removed",
                        "Old stale comment about full path self-persisting must be framed as historical/incorrect"))
    else:
        results.append(("PASS", "stale_comment_about_run_full_analysis_self_persisting_corrected_or_removed", ""))

    # Claim 7: AST clean
    try:
        ast.parse(text)
        results.append(("PASS", "autonomous_loop_py_syntax_clean", ""))
    except SyntaxError as e:
        results.append(("FAIL", "autonomous_loop_py_syntax_clean", f"SyntaxError: {e}"))

    # Claim 8: persist function still calls bq.save_report (the actual persistence)
    func_block = re.search(
        r'async def _persist_analysis\s*\(.*?(?=\n(?:async )?def \w|\Z)',
        text,
        re.DOTALL,
    )
    has_save_report = func_block and "bq.save_report" in func_block.group(0)
    results.append(("PASS" if has_save_report else "FAIL",
                    "persist_analysis_calls_bq_save_report",
                    "_persist_analysis must call bq.save_report (actual BQ persistence)"))

    # --- Output ---
    print("=== phase-25.A2 (full-pipeline persistence) verifier ===")
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
