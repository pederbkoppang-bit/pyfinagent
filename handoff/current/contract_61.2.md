# Contract — Step 61.2: decision-input integrity (dark build → evidence + promotion ask)

- **Step id:** 61.2 (P0, phase-61)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 173

## Research-gate summary

`handoff/current/research_brief_61.2.md` — gate_passed: **true** (7 external sources read in full / 38 URLs / recency scan / 13 internal files). The gate REFUTED both spawn-prompt premises and reshaped the step:

1. **61.2 has SIX immutable criteria** (under `success_criteria`) + a live_check string — copied verbatim below. No self-imposed criteria.
2. **61.2 was BUILT 2026-07-08** (commit 6186784c: 20 source files + a 459-line test module `test_phase_61_2_decision_integrity.py`), dark behind flags; its prior Q/A returned **CONDITIONAL (count: 1)** on live evidence that could not exist yet.
3. **Six-row triage**: A (0.00/HOLD) LIVE DEFECT firing daily — **142/156 full-path rows in 40d (91%)** are `final_score=0.0 + HOLD + $.final_synthesis.error='Failed to parse final report.'`, last 2026-08-06; the laundering line is `orchestrator.py:2280` (stamps `compute_weighted_score({})=0.0` onto the error dict); fix built-dark behind `paper_synthesis_integrity_enabled` (default False). B (timeout≥150s) DONE LIVE UNGATED (`settings.py:186-191` cites the criterion; deliberately INVERTS SRE deadline-propagation so the inner timeout fires first — a considered deviation, stated). C (company_name) FIXED + LIVE-PROVEN (NULLs 5/5/2/4 per day 07-03..07-08 → exactly 0 every day 07-09..08-06 — **this retires prior-Q/A blocker #1**). D (meta-scorer PIT rank-norm + WARN streak) BUILT-DARK; **72.0.1's premise is DEAD** (phase-78.1 rewired `make_client` at meta_scorer.py:242; no ClaudeClient construction remains). E (signal_downgrade) still dead in prod, strictly downstream of A (promoting E alone would SELL healthy positions on fabricated HOLDs — the :114-121 WARN already guards the combination). F (RiskJudge ctx) DARK, defect live (judge receives `''` today).
4. **The immutable -k command is RED today for reasons outside 61.2's blast radius**: 4 failed / 67 passed — two foreign tests matched by the bare word 'persist' (`test_phase_50_2_multicurrency`, `test_phase_83_0_news_corpus_persistence::test_c5`), and **two 61.2 tests broken by phase-36.13's kill-switch gate** (paper_trader.py:276-288 now consults the REAL on-disk audit — currently `paused=True, reason='manual'` — so the tests' outcomes depend on the operator's live pause state).
5. **Promotion risk to state, not bury** (AWS REL05-BP01): flipping `paper_synthesis_integrity_enabled` makes the ~9%-traffic lite fallback carry ~91%. The live_check must measure the post-promotion share. External consensus 7/7 against fabricated neutrals (arXiv:2606.01416 0.0% silent-failure with explicit degraded markers; the PIT-vs-min-max choice backed by arXiv:2603.28886; PIT-vs-RRF openly stated as an un-compared alternative, defensible for a homogeneous composite).

## Immutable success criteria (verbatim from `.claude/masterplan.json` 61.2 `success_criteria`)

1. "a synthesis result carrying final_synthesis.error (or missing scoring_matrix) is never persisted as a 0.0 final_score with a default HOLD: it is either routed to the existing lite fallback or persisted with NULL score plus an explicit degraded marker; a regression test simulates the timeout and asserts no 0.0/HOLD row is written and the same-cycle trade-decision input is not silently neutralized"
2. "claude_code synthesis/critic-class calls run with timeout >= 150s (per the file's own recommended_step_timeout) and the value is configurable"
3. "_persist_analysis falls back to the quant company_name when market_data.name is absent; live_check shows BQ rows from a post-fix autonomous full-path cycle with non-null company_name"
4. "the meta-scorer fallback no longer emits a constant saturated conviction: composite scores are rank/percentile-normalized into the 1-10 scale, and a WARN-level alert fires after 2 consecutive all-fallback cycles; the root cause of the 06-03..06-10 LLM unavailability is diagnosed and documented in experiment_results.md"
5. "positions persist the analysis recommendation (not the trade reason) so the signal_downgrade rule at portfolio_manager.py:127 can match; covered by a unit test"
6. "RiskJudge receives portfolio sector-breakdown context regardless of paper_risk_judge_reject_binding"

**Verification command (immutable; disclosed red-for-foreign-reasons):** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q && test -f handoff/current/live_check_61.2.md` — run verbatim and recorded; the scope-bound green signal is additionally `pytest backend/tests/test_phase_61_2_decision_integrity.py -q`. Per `feedback_immutable_criteria_must_be_green_able` the command is NOT amended; the two foreign failures are proven pre-existing on HEAD.

**live_check (immutable):** BQ rows from ≥1 post-fix autonomous cycle: non-null company_name (capturable NOW — the fix deployed 07-09), zero new 0.0+error rows AND non-constant conviction (require the flag promotion → operator).

## Explicit decisions

- **D1 — scope of THIS cycle**: (i) repair the two kill-switch-coupled 61.2 tests via the `_injected_ks_state` seam with BOTH named mutations (flip injected state → both fail; revert paper_trader.py:443-450 → the criterion-5 test fails — proving the test guards the criterion, not just the kill switch) plus the paused/unpaused invariance run; (ii) bind the criterion-1 regression test to the OBSERVED live error string (`Failed to parse final report.`); (iii) capture live_check Sections A (criterion-3 BQ proof) and B (the 40-day fabrication baseline + lite/full split) — available with no deploy; (iv) publish the promotion ask with the measured evidence. **No flag flips** — promotion changes live trading behaviour and is the operator's (phase-69 doctrine).
- **D2 — expected honest outcome**: if the operator has not decided promotion intra-day, the correct verdict is a **second CONDITIONAL** (prior count 1) with A+B captured — not PASS, not drop. A third would auto-FAIL, which is the argument for the operator deciding on THIS evidence.
- **D3 — 72.0.1 disposition queued separately**: its premise (direct ClaudeClient construction) is dead since phase-78.1; handled as its own disposition after this cycle, not silently inside it.
- **D4 — registered test-debt restated, not quietly fixed**: three source-grep-style 61.2 tests and the untested make_client→timeout threading are carried forward as disclosed debt (freeze-the-tree discipline).
- **D5 — root-cause half of criterion 4**: the 06-03..06-10 unavailability diagnosis is documented in experiment_results from the researcher's evidence (phase-72 record: Anthropic credits dead since 05-17 + rail bypass — now fixed by 78.1), satisfying the "diagnosed and documented" clause.

## Plan

1. Test repair (D1-i) + error-string binding (D1-ii); run the 61.2 module + the mutations; run the immutable command verbatim and record.
2. Capture live_check Sections A + B (BQ, read-only).
3. experiment_results with the six-row triage, the root-cause doc (D5), disclosures (foreign-red proof on HEAD, test-debt, promotion risk).
4. Update ask list: promotion decision item with the 142/156 evidence + the AWS fallback-share warning.
5. qa-verdict → transcribe → (likely CONDITIONAL #2 per D2) → harness_log → status stays pending unless PASS.

## References

`research_brief_61.2.md` (BQ measurements verbatim; arXiv:2606.01416, arXiv:2603.28886, AWS REL05-BP01, Google SRE deadline-propagation, aipatternbook silent-failure, OpenSearch RRF; commits 6186784c, 354eb6b4, 3227347a, phase-78.1).
