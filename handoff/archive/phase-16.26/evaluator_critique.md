---
step: phase-16.26
cycle_date: 2026-04-24
verdict: CONDITIONAL
auditor: qa
---

# Q/A Critique -- phase-16.26

## Harness-compliance (5 items)

1. **Research gate**: PASS. `handoff/current/phase-16.26-research-brief.md`
   exists; tier=simple, 6 sources read in full (FastAPI async, asyncio
   event-loop, Vertex retry, bobbyhadz event-loop, rank-bm25, CoALA
   arXiv:2309.02427), 9 snippet-only URLs, recency scan present,
   `gate_passed: true` in JSON envelope.
2. **Contract-before-GENERATE**: PASS. `contract.md` frontmatter shows
   `step: phase-16.26`, `cycle_date: 2026-04-25`, `harness_required:
   true`. Contract precedes experiment_results.md (timestamps consistent
   with cycle).
3. **Experiment results**: PASS. Verbatim probe stdout for all 3
   probes is included; framing is honest ("FAIL on assertion (final_score
   is None). The wrapper itself works..."); the 16.21 escalation clause
   is explicitly self-flagged in the "Honest disclosures" section #1
   and follow-up #3 -- Main asks Q/A to make the call rather than
   silently assuming CONDITIONAL.
4. **Log-last**: PASS. `grep -c "phase-16.26" handoff/harness_log.md`
   = 0 (no premature log append before Q/A verdict).
5. **No verdict-shopping**: PASS. Prior `evaluator_critique.md` was
   for phase-16.25 (PASS). This is the first Q/A on 16.26; no prior
   16.26 critique to over-write.

## Deterministic checks

- **imports**: all 3 wrappers import cleanly --
  `run_analysis_pipeline`, `evaluate_recent`, `retrieve_memories`.
- **probe_1_assertion**: FAIL. `final_score=None`,
  `status=failed_init`,
  `error=orchestrator_init_failed: ValueError: Model
  'claude-sonnet-4-6' requires a GitHub Token (GITHUB_TOKEN) but none
  is set.`
- **probe_2**: type=`dict`, status=`empty`, reason=`fromisoformat:
  argument must be str`, outcomes=`[]`. Wrapper graceful-degraded.
- **probe_3**: memories=3 (seed archetypes returned for tech query).
- **ast**: `syntax_ok` for all 3 wrapper files.
- **pytest_regression**: 177 passed, 1 skipped, 1 warning in 14.18s.
  No regressions from the wrapper additions.

## Root-cause verification

- **gemini_model**: `claude-sonnet-4-6` (observed via `Settings()`).
- **anthropic_key_starts**: `sk-ant-oat` (OAT/OAuth token; not a
  standard API key -- consistent with the standing "OAT-broken"
  flag).
- **github_token**: `EMPTY`.
- **llm_client.py L1110-L1127**: confirms the gate -- when model is
  in `GITHUB_MODELS_CATALOG` and `github_token` is empty, raises
  `ValueError("Model '<name>' requires a GitHub Token (GITHUB_TOKEN)
  but none is set.")`. Exactly the error surfaced.
- **credentials_blocker_real**: yes. The blocker is structural and
  user-action-only. Wrapper code is not at fault.

## Escalation-clause judgment

- **pattern_classification**: `credentials-blocker-new`
  (relative to the 16.20/16.21 missing-function pattern).
- **corrector_path_taken**: yes. Main implemented all 3 wrappers
  in 16.26 -- `run_analysis_pipeline` (analysis.py +47),
  `evaluate_recent` (outcome_tracker.py +20),
  `retrieve_memories` (memory.py +12). The 16.21 structural gap
  (functions missing) is closed; functions exist, are importable,
  return dicts, and surface errors visibly.
- **verdict**: **CONDITIONAL**.
- **reasoning**: My 16.21 escalation clause was about pattern
  *recurrence without correction* -- "harness used as a logger
  instead of a corrector". Main DID correct: 16.22 closed the alias
  pattern, 16.25 closed `run_orchestrated_round`, and 16.26 closes
  the 3-wrapper structural gap. The remaining failure mode is a
  credentials blocker that is genuinely user-action-only (key swap
  or `GITHUB_TOKEN`). Per CLAUDE.md cycle-2 flow, user-action-
  required is a documented terminal state, not a structural breach.
  Auto-FAIL would be punitive given Main is on the corrector path,
  not the logger path. **Precedent for next Q/A**: if a 4th cycle
  produces a *different* missing-function CONDITIONAL (i.e. a new
  alias gap discovered), that is the auto-FAIL trigger; a
  recurring credentials block on the SAME upstream gate is not.

## LLM judgment

- **wrapper_purity**: PASS. All 3 are short and pure delegation.
  `run_analysis_pipeline` (~50 lines incl. docstring + try/except)
  builds an `AnalysisOrchestrator`, drives `run_full_analysis` on a
  fresh event loop, flattens `final_weighted_score` from
  `final_synthesis`. `evaluate_recent` (~20 lines) defers to
  `OutcomeTracker.evaluate_pending`. `retrieve_memories` (~10 lines)
  defers to `FinancialSituationMemory.get_memories`. No scope creep.
- **graceful_degradation_honest**: PASS. Probe 1's wrapper returns
  `{"final_score": None, "error": "orchestrator_init_failed:
  ValueError: Model 'claude-sonnet-4-6' requires a GitHub Token...",
  "status": "failed_init"}`. The error string contains the exact
  upstream cause and the remediation path. This is the right
  surfacing pattern -- not silent.
- **probe_2_separate_bug_real**: yes, real. `evaluate_recent`
  returns `{"status": "empty", "reason": "fromisoformat: argument
  must be str", ...}`. The downstream `OutcomeTracker.evaluate_pending`
  is calling `datetime.fromisoformat(...)` on a non-str (likely a
  `datetime`/`date` object or `None` from a BQ row -- BQ Python
  client returns native datetime objects, not ISO strings). Worth a
  follow-up ticket; not Monday-blocking.
- **closes_16_21_followup_24**: yes. All 3 wrappers exist,
  importable, return dicts. The 16.21 follow-up #24 (the structural
  gap) is closed.
- **16_2_explicitly_held_open**: yes. Main flagged in "Honest
  disclosures #7" that 16.2 stays in-progress until live pipeline
  runs cleanly + fresh Q/A returns PASS. Confirmed.

## Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 3 wrappers exist, are importable, are pure delegation, and degrade gracefully. Probe 1 fails the final_score assertion because of an upstream credentials gate (GITHUB_TOKEN empty AND ANTHROPIC key is an OAT). The wrapper code itself is correct and honest about the cause. Per the 16.21 escalation clause, this is NOT structurally identical to the missing-function pattern -- Main took the corrector path, the new blocker is user-action-only.",
  "violated_criteria": ["analysis_pipeline_returns_final_score"],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "AnalysisOrchestrator(settings) -> make_client(settings.gemini_model='claude-sonnet-4-6', ...)",
      "state": "anthropic_api_key starts 'sk-ant-oat' (OAT, not API), github_token EMPTY, gemini_model='claude-sonnet-4-6'",
      "constraint": "make_client requires either a working ANTHROPIC_API_KEY or GITHUB_TOKEN for catalog models (llm_client.py L1110-L1127)"
    }
  ],
  "follow_up_tickets": [
    "Set GITHUB_TOKEN=ghp_... in backend/.env (alternative: swap ANTHROPIC_API_KEY off OAT to a real API key) -- unblocks live pipeline run for 16.2 close",
    "evaluate_recent: fromisoformat bug -- OutcomeTracker.evaluate_pending is calling datetime.fromisoformat on a non-str (BQ client returns native datetime). Locate the offending column/row and convert via str() or branch on type. NEW bug, surfaced by 16.26 wrapper, not Monday-blocking.",
    "16.2 stays in-progress -- closes only on PASS of live pipeline run after credentials unblock"
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5",
    "imports_3_wrappers",
    "probe_1_run_analysis_pipeline",
    "probe_2_evaluate_recent",
    "probe_3_retrieve_memories",
    "syntax_ast_3_files",
    "pytest_regression_177_passed",
    "credentials_observation",
    "llm_client_gate_inspection",
    "wrapper_purity_review"
  ]
}
```

**Precedent for the next Q/A**: the 16.21 "third structurally-
identical CONDITIONAL must FAIL" clause is satisfied for the
missing-function pattern (16.20 -> 16.25 corrector; 16.21 -> 16.26
corrector). Going forward, the auto-FAIL trigger fires only on a
*new* missing-function CONDITIONAL (Main re-introducing the alias
gap or surfacing a 4th distinct missing-symbol). A recurring
credentials gate on the SAME upstream (`make_client`) is not in
scope -- it is the documented "user-action-required" terminal
state.
