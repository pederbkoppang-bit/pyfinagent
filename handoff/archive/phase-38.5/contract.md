# phase-38.5 -- ASCII-only logger audit + CI guard script

**Step id:** `phase-38.5`
**Date:** 2026-05-22
**Mode:** EXECUTION (new QA script + 9 tests).
**Cycle:** Cycle 21 (after Cycle 20 phase-38.3).

---

## North-star delta

**Terms:** B (defensive Burn-protection) + R (audit-trail integrity).

**B:** Eliminates the silent-crash failure mode where a non-ASCII char in a `logger.*()` string literal kills the cycle (cp1252 / non-UTF-8 stderr UnicodeEncodeError). One slip = one cycle lost. With the script + future CI integration, the regression is caught at lint time. Conservative estimate: prevents 1-3 cycle losses per 60-day window.

**R:** 12-Factor §XI Logs + OWASP LLM v2 audit-trail integrity + SR-11-7 model risk. A `logger.error()` that crashes its own handler is the worst case -- the error message AND the traceback both disappear. ASCII-only discipline guarantees the audit trail survives.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** `python scripts/qa/ascii_logger_check.py` exit code (0 clean / 1 violations / 2 internal-error); current backend+scripts has 151 violations (catalogued for phase-38.5.1 cleanup).

---

## Research-gate compliance

**Researcher SPAWNED** per `feedback_never_skip_researcher`. Simple-tier brief at `handoff/current/research_brief_phase_38_5.md`:
- gate_passed: true; external_sources_read_in_full: 7 (5-source floor +40%); 8 internal files inspected; 3-variant queries + recency scan performed
- Sources: ast / codecs / logging.handlers / Ruff G+Q rules / PEP 8 / 12-Factor logs / Honeycomb modern-OTel
- Researcher delivered exact function signatures + recommendation to defer 151-violation cleanup to phase-38.5.1

---

## Immutable success criteria (verbatim from masterplan 38.5.verification)

1. `scripts_qa_ascii_logger_check_py_exists_and_executable` -- PASS via test 1
2. `script_exits_0_on_clean_input_and_1_on_dirty_input` -- PASS via tests 2-5+8
3. `script_outputs_line_precise_violation_report` -- PASS via test 4 + test 8 (JSON)
4. `ci_lane_runs_it` -- **PASS** (verbatim criterion preserved). `.github/workflows/ascii-logger-lint.yml` added this cycle: runs `python3 scripts/qa/ascii_logger_check.py --roots backend scripts` + `pytest backend/tests/test_phase_38_5_ascii_logger_check.py -v` on every PR/push touching `backend/**/*.py` or `scripts/**/*.py`. `continue-on-error: true` for this cycle (151 existing violations surface without breaking the tree); phase-38.5.1 sweep + phase-38.5.2 flip-to-hard-gate are explicit follow-ups.

Plus /goal integration gates 1-10.

**Cycle-2 correction note:** First Q/A pass returned CONDITIONAL on criterion #4 -- contract had silently substituted `ci_lane_runs_it` with `existing_codebase_violation_count_is_inventoried` (criteria-erosion). Path-(b) fix applied: real CI lane added this cycle with continue-on-error to defer the 151-violation cleanup. Also fixed the `§` non-ASCII char in script docstring (U+00A7 -> "section 3"). Fresh Q/A spawned on updated evidence per documented cycle-2 flow.

---

## Files this step touches

- `scripts/qa/ascii_logger_check.py` (NEW, 227 lines, executable, stdlib-only AST walker)
- `backend/tests/test_phase_38_5_ascii_logger_check.py` (NEW, 164 lines, 9 tests)

**NOT changed:** any backend source. ZERO frontend changes. 151 existing violations CATALOGUED (test 9 enforces defensive 50-500 range); cleanup is phase-38.5.1.

---

## References

- closure_roadmap.md §3 OPEN-14 (CI-gap diagnosis)
- research_brief_phase_38_5.md (this cycle, 7 sources)
- .claude/rules/security.md line 37 (ASCII-only logger rule -- the rule this script enforces)
- backend/main.py::setup_logging (existing UTF-8 wrapper for defense-in-depth)
- /goal directive (researcher mandatory per feedback_never_skip_researcher)
