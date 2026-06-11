# Contract — Step 57.1

**Step id:** 57.1 — Binding RiskJudge gate + concentration-aware prompt context (phase-57 FEATURE; operator reply verbatim: `PHASE-57: FEATURE`, 2026-06-11; install commit `af4aa8d6`)
**Date:** 2026-06-11
**Phase:** phase-57 (config-gated default-OFF; NO live flag flip; do-no-harm)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=complex, 6 external sources read in full incl. SEC 15c3-5 CFR primary + ESMA Feb-2026 + GuardAgent, 16 URLs, recency scan; envelope `gate_passed: true`)

## Research-gate summary

Decisive topology finding: ALL 3 executed-REJECT BUYs (HPE 06-02, DELL 06-03, 066570.KS 06-09 — all `swap_buy`) flowed through `_compute_swap_candidates` (portfolio_manager.py:405-585, swap BUY emit :553-568 carrying `risk_judge_decision` at :559) — a gate at the main BUY-emit loop would have missed every real execution. The gate goes at the **candidate-build loop (:148-198)**, the common ancestor feeding BOTH paths: a flag-gated `continue` before `buy_candidates.append` (:180) when `decision == "REJECT"`, with a structured warning + `blocked_buys` accumulator surfaced as `summary["risk_judge_blocked"]`. Budget-reallocation is by construction (the next-ranked survivor draws the freed cash). REJECT-only binding (APPROVE_REDUCED/HEDGED stay advisory sizing — ESMA hard-vs-soft tiering; do not over-bind the momentum core). Settings flag `paper_risk_judge_reject_binding` (default False, F-3-citing description). Prompt fixes (F-8) behind the SAME flag — binding on a blind judge is the incoherent half-state (GuardAgent: a guard must be GIVEN its inputs): builder functions return the verbatim `_LITE_RISK_JUDGE_SYSTEM`/`_LITE_RISK_JUDGE_TEMPLATE` constants when OFF (byte-identity incl. prompts) and, when ON, replace the phantom "10% of portfolio in one sector" with the configured `paper_max_per_sector_nav_pct` and append a sector-breakdown context line. Sector summary computed ONCE per cycle (positions read at autonomous_loop.py:774, BEFORE the concurrent fan-out) and threaded as a `portfolio_context` kwarg — never fetched per ticker (concurrency + N-reads pitfall). Event study BQ-confirmed: the 3 trades realized −$0.81 / +$0.54 / −$23.18, net **−$23.45** (dominated by the LG stop-out) — to be presented descriptively with the mandatory selection-bias caveat (Ahern 2006; arXiv:2511.15123), no EV/Sharpe extrapolation. External consensus: risk verdicts must be enforced by a deterministic, non-bypassable pre-execution gate (SEC 15c3-5(c)(1)/(d) "by rejecting orders"/"direct and exclusive control"; ESMA Feb 2026 hard blocks; arXiv:2604.01483; GuardAgent 98%/83% when context-given).

## Hypothesis

A single flag-gated drop at the candidate-build chokepoint binds the REJECT verdict across both BUY paths with zero behavior change when OFF (including prompts), provable by unit tests; the judge becomes trustworthy to bind on because the same flag feeds it the real cap + live sector state.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 57.1)

1. "Binding-gate regression fixture (both BUY paths): a unit test instantiates decide_trades with a candidate whose risk_assessment.decision == 'REJECT' and a screen that routes it through BOTH the main BUY path AND the swap path; with paper_risk_judge_reject_binding=False (default) the REJECT candidate's BUY order IS emitted (advisory, unchanged); with the flag True the REJECT candidate's BUY order is ABSENT from decide_trades output on both paths; test passes"

2. "Default-OFF byte-identity of the US momentum core: a unit test asserts that with paper_risk_judge_reject_binding=False, decide_trades returns an order list byte-identical (same tickers, actions, amounts, order) to the pre-57.1 behavior on a candidate set containing NO REJECT verdicts; AND the rendered lite-RiskJudge system prompt + template with the flag OFF are byte-identical to the pre-57.1 _LITE_RISK_JUDGE_SYSTEM / _LITE_RISK_JUDGE_TEMPLATE constants; no live flag flip occurs in 57.1"

3. "Prompt-context correctness with the flag ON (F-8): a unit test asserts that with paper_risk_judge_reject_binding=True the RiskJudge system prompt contains the configured sector cap value and does NOT contain the phantom '10% of portfolio in one sector', AND the RiskJudge user prompt/template includes a live portfolio sector-breakdown line derived from the current positions (asserted via an injected/fake positions fixture)"

4. "Per-cycle single-compute of the sector summary (concurrency-correct): the sector-breakdown context passed to the per-ticker RiskJudge is computed ONCE per cycle (not once per ticker), verified by a test asserting the position-reading helper is invoked at most once across an N-ticker analysis fan-out (mock/spy on the positions accessor) OR by code-structural assertion that the per-ticker analyzer receives a precomputed portfolio_context argument rather than fetching positions itself"

5. "Event-study evidence artifact (stored-data, $0, honestly framed): handoff/current/live_check_57.1.md exists and contains the BQ query + result enumerating the executed-REJECT BUYs (3 rows: HPE, DELL, 066570.KS), their realized round-trip P&L (-$0.81 / +$0.54 / -$23.18; net -$23.45), AND an explicit selection-bias caveat stating that n=3 is descriptive of those specific trades, not the gate's expected value, with no annualized/Sharpe extrapolation"

6. "Verification command green: the immutable command exits 0 (all new tests pass; live_check artifact present); the flag ships default-OFF and is NOT flipped live inside phase-57"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests -k 'reject_binding or risk_judge_binding' -q && test -f handoff/current/live_check_57.1.md`

## Plan

1. Settings flag `paper_risk_judge_reject_binding: bool = Field(False, ...)` (settings.py, F-3-citing description, near the other paper_* default-OFF flags).
2. Gate at the candidate-build loop (portfolio_manager.py ~:180): flag-gated `continue` on REJECT + `blocked_buys` accumulator + structured warning; surface as a decide_trades-level summary mechanism the cycle can read (mirror existing patterns); document budget-reallocation-by-construction in the diff comment.
3. Prompt builders in autonomous_loop.py: `_build_risk_judge_system(settings)` / `_build_risk_judge_template(settings)` — verbatim constants when OFF; cap-corrected + `{portfolio_context}`-augmented when ON. Call sites: the Claude path (:1796/:1814 area) and the Gemini twin (:1979 area).
4. `portfolio_context` threading: compute the compact sector-weight string ONCE per cycle (after the positions read at :774, with the `quantity*(current_price or avg_entry_price)` fallback idiom), pass via `_run_and_persist_one` → `_run_single_analysis` → both analyzers as an optional kwarg (default None → ""), used only when the flag is ON.
5. Tests (`backend/tests/test_phase_57_1_reject_binding.py`): criterion-1 fixture (both paths), criterion-2 byte-identity (orders + prompts), criterion-3 prompt-content ON, criterion-4 single-compute (precomputed-arg structural assertion + spy).
6. live_check_57.1.md: BQ query + 3-row event-study table + net −$23.45 + selection-bias caveat + test outputs + default-OFF confirmation (settings grep; no flip).
7. experiment_results.md → ONE fresh Q/A → harness_log append → masterplan flip.

## Constraints

- Default-OFF; flag-OFF byte-identical INCLUDING prompts; NO live flag flip in 57.1 (operator flips later, validates OOS).
- REJECT-only binding; APPROVE_REDUCED/HEDGED sizing untouched.
- $0 (stored-data event study; unit tests with fakes/mocks; no LLM trading-cycle spend).
- Every change cites F-3/F-8 + the brief's sources; minimal diffs at the audited sites.
- ASCII-only logger strings; no emojis.

## References

- handoff/current/research_brief.md (researcher 57.1, gate_passed: true; sources: 17 CFR 240.15c3-5, ESMA74-1505669079-10311 Feb-2026, arXiv:2604.01483, GuardAgent arXiv:2406.09187, Ahern 2006, arXiv:2511.15123)
- handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md §2.6 (spec-of-record); findings F-3, F-8
- Code anchors: portfolio_manager.py:148-198,:180,:185,:193-198,:256-365,:336,:405-585,:553-568; autonomous_loop.py:774,:826-870,:1094,:1515-1540,:1773-1816,:1978-1991; settings.py paper_* flag idiom
