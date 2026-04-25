---
step: phase-16.21
cycle_date: 2026-04-25
verdict: CONDITIONAL
reviewer: qa
---

# Q/A Critique -- phase-16.21

## Harness-compliance audit (5 items)

1. **Research gate** -- PASS-WITH-NIT. `handoff/current/phase-16.21-research-brief.md`
   exists (139 lines). Hard-blocker checklist asserts 5 in-full WebFetch
   sources, 15 URLs collected, recency scan present, three-variant
   query discipline shown (current-frontier 2026, last-2-year, year-less
   canonical). Source mix is appropriately authoritative: Google Cloud
   official docs + community-tier forum hits in the snippet table only.
   **Nit (non-blocking):** the canonical JSON envelope from
   `.claude/rules/research-gate.md` is missing -- the brief uses a
   prose checklist instead. Same nit was tolerated in 16.20; not
   surfacing as blocker, but should be standardized in next housekeeping.

2. **Contract-before-GENERATE** -- PASS. `contract.md` mtime
   1777093977 vs `experiment_results.md` mtime 1777094007. Delta = 30s.
   Contract step header reads `step: phase-16.21` and matches the
   experiment-results step header. No back-dating detected.

3. **Experiment results completeness** -- PASS. `experiment_results.md`
   step=phase-16.21, contains verbatim ImportError stdout for ALL 3
   probes (run_analysis_pipeline, evaluate_recent, retrieve_memories),
   tabulates the underlying-class probe (OutcomeTracker present,
   FinancialSituationMemory present), and records the
   daily-cycle-dependency grep result (0 references in
   autonomous_loop.py and paper_trading.py). `expected_verdict:
   CONDITIONAL` is declared in the front-matter, signaling honest
   forward-cycle expectations rather than over-claim.

4. **Log-last invariant** -- PASS. `grep -c "phase-16.21"
   handoff/harness_log.md` = 0. Main has not pre-logged.

5. **No verdict-shopping** -- PASS. Existing critique on disk was
   for phase-16.20 (also CONDITIONAL). No prior 16.21 verdict
   exists to overturn. This is a first-pass evaluation on
   updated evidence, not a re-spin of unchanged inputs.

## Deterministic checks (independent re-runs)

- **import_errors_reproduced: 3/3 yes**
  - `from backend.tasks.analysis import run_analysis_pipeline` ->
    `ImportError: cannot import name 'run_analysis_pipeline'`
  - `from backend.services.outcome_tracker import evaluate_recent` ->
    `ImportError: cannot import name 'evaluate_recent'`
  - `from backend.agents.memory import retrieve_memories` ->
    `ImportError: cannot import name 'retrieve_memories'`
- **underlying_classes_callable: yes**
  - `OutcomeTracker` imports cleanly from
    `backend.services.outcome_tracker`.
  - `FinancialSituationMemory` imports cleanly from
    `backend.agents.memory`.
  - Confirms Main's claim: only the module-level wrappers are
    missing; the underlying functionality (Monday-critical) is
    intact and salvageable with shim functions.
- **daily_cycle_grep_count: 0** (across `autonomous_loop.py` and
  `paper_trading.py`). The missing wrappers are NOT referenced by
  the daily paper-trading cycle. Production code paths use the
  classes / `/api/analyze` endpoint directly.
- **masterplan_16_2_status: in-progress** (NOT silently flipped to
  done). 16.21 status: pending. Status hygiene preserved.

## LLM judgment

- **second_conditional_systemic_concern**: Yes, this IS a pattern
  worth naming. 16.20 and 16.21 both surfaced as CONDITIONAL with
  the same root cause shape: masterplan immutable verification
  commands target documented module-level entry points
  (`run_analysis_pipeline`, `evaluate_recent`, `retrieve_memories`,
  and 16.20's analogous probes) that don't exist in the code. This
  is doc/code drift -- the masterplan was authored against an
  intended public API surface that was never finalized at the
  module level, while the underlying classes WERE built. The
  harness is surfacing it cleanly (3-of-3 reproducible
  ImportErrors, classes confirmed present), which is the desired
  behavior. But two consecutive CONDITIONALs on the same structural
  pattern means the cost of papering over with "follow-up tickets"
  is rising. A third one without a doc/code reconciliation pass
  should auto-escalate to FAIL.

- **monday_blocker_assessment**: NOT a Monday blocker. The
  daily-cycle grep returns 0 -- the missing wrappers are unreferenced
  by `autonomous_loop.py` and `paper_trading.py`. Layer-1 analysis
  is reachable in production via `/api/analyze` (orchestrator
  invocation) and the underlying classes are importable. Monday
  paper-trading and signal generation are not on the critical path
  of these wrappers. Pre-Monday risk: low.

- **conditional_vs_fail_decision**: CONDITIONAL with explicit
  follow-up tickets, BUT with a pattern-flag escalation clause:
  if a third structurally-identical "missing wrapper" CONDITIONAL
  arrives in this UAT sweep before the doc/code reconciliation
  ticket is closed, that next one MUST be FAIL. Reasoning for
  not FAILing now:
  1. Underlying functionality intact (proven by class imports).
  2. Production code paths unaffected (grep=0).
  3. Plan is verification-only -- it was designed to *discover*
     the gap, not to *deliver* the wrappers. Mechanical 0/3 is
     the diagnostic output, not a delivery failure.
  4. The handoff is honest (`expected_verdict: CONDITIONAL` in
     front-matter; no over-claim).
  Reasoning for the escalation clause:
  - Two consecutive same-pattern CONDITIONALs without a parent
     fix is exactly the "silent doc/code drift" failure mode
     the harness is supposed to prevent. A third without
     remediation means the harness is being used as a logger
     rather than a corrector.

## Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "violated_criteria": [
    "import_run_analysis_pipeline_succeeds",
    "import_evaluate_recent_succeeds",
    "import_retrieve_memories_succeeds"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "from backend.tasks.analysis import run_analysis_pipeline",
      "state": "module exists; symbol absent; OutcomeTracker/FinancialSituationMemory classes present elsewhere",
      "constraint": "masterplan-16.21 verification probes require module-level wrapper functions"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "from backend.services.outcome_tracker import evaluate_recent",
      "state": "module exists; symbol absent; class OutcomeTracker importable",
      "constraint": "masterplan-16.21 verification probe #2"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "from backend.agents.memory import retrieve_memories",
      "state": "module exists; symbol absent; class FinancialSituationMemory importable",
      "constraint": "masterplan-16.21 verification probe #3"
    }
  ],
  "follow_up_tickets": [
    "phase-16.22-FOLLOWUP: add 3 module-level wrapper shims (run_analysis_pipeline, evaluate_recent, retrieve_memories) thin-wrapping the existing classes; verification = re-run the three import probes",
    "phase-16.X-DOC-RECONCILIATION: audit masterplan immutable verification commands phase-16.* against actual module surface; reconcile drift before closing 16.2; if a THIRD same-pattern CONDITIONAL arrives before this ticket closes, auto-escalate to FAIL",
    "research-gate housekeeping: standardize on the JSON envelope from .claude/rules/research-gate.md across all briefs (16.20 and 16.21 both used prose-checklist instead)"
  ],
  "certified_fallback": false,
  "checks_run": [
    "research_brief_presence",
    "contract_mtime_before_results",
    "step_id_consistency",
    "verbatim_importerror_capture",
    "underlying_class_probe",
    "daily_cycle_grep",
    "masterplan_16_2_status",
    "log_last_invariant",
    "verdict_shopping_check",
    "three_import_probes_independent_rerun"
  ],
  "pattern_flag": "SECOND_CONSECUTIVE_SAME_SHAPE_CONDITIONAL -- third instance without doc/code reconciliation must FAIL"
}
```
