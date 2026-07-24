# Contract — phase-75.5.2: the remaining gemini-2.5 behavioural pins → constants

- **Step id:** 75.5.2 (phase-75 follow-up queue, **P1** — deadline-relevant: gemini-2.5 family retires 2026-10-16, and the 2.0-flash retirement under a hardcoded pin cost 9 silent days; executor: **sonnet-tagged → delegated Sonnet executor GENERATE**; Main review + mutation matrix; gates opus/max via Workflow)
- **Date:** 2026-07-24
- **Boundary (from step text):** literals become constants only, **NO tier pin VALUE changed**. Scope decision (researcher-recommended strict reading of C1's "outside model_tiers.py"): the census's 9 behavioural pins INCLUDING the newly-discovered `scripts/harness/run_autonomous_loop.py:74`, plus co-located non-behavioural literals in in-scope files (the 75.5 precedent scan is a strict per-file substring check). Do NOT touch: `model_tiers.py` VALUES, `cost_tracker.py` pricing keys, `settings_api.py`, `backend/tests/**` history, `scripts/migrations/add_news_sentiment_schema.py` (persisted DB enum), `harness_memory.py:48` (gemini-2.0-flash row).

## Research-gate summary (gate PASSED — wf_c5355afe-01b, AUDIT-CLASS, coverage.dry=true)

Envelope: `tier=moderate, external_sources_read_in_full=6, snippet_only_sources=12, urls_collected=20, recency_scan_performed=true, internal_files_inspected=15, gate_passed=true`; census swept until rounds 4 AND 5 surfaced zero new behavioural findings (2 consecutive dry rounds). Brief: `handoff/current/research_brief_75.5.2.md` (full census table with per-site target constants + pre-change resolved values).

Load-bearing findings:

1. **Census (re-derived, not trusted):** exactly the 8 named backend pins survive — sites 1-5 & 8 at stable lines; the two `services/autonomous_loop.py` fallbacks moved +22 (:2670, :2685); none deleted/fixed since 75.5. **A 9th behavioural pin discovered:** `scripts/harness/run_autonomous_loop.py:74` (`evaluator_model="gemini-2.5-flash"`), a runtime-capable harness entrypoint — C1 says "outside model_tiers.py", so the strict reading includes it. ALL 9 resolve to `gemini-2.5-flash` → `GEMINI_WORKHORSE`; NO behavioural gemini-2.5-pro pin exists outside model_tiers.py.
2. **Classification rule (checked into the test):** PIN = literal SELECTS a model for a call (`model=`/`model_name=`/`evaluator_model=` or a var/const/or-fallback feeding one). Non-behavioural = pricing/capability KEY, docstring/comment, roster/display metadata (`_inventory.json` confirmed display-only), sample record/env, test fixture.
3. **`llm_client.py:985` `startswith("gemini-2.5")` is a FAMILY GUARD** — behavioural but not a pin: route via a NEW `GEMINI_2_5_FAMILY_PREFIX = "gemini-2.5"` constant in model_tiers.py (no value change); it is NOT a resolved-value pin for C2/C3.
4. **The 75.5 scan precedent is strict** (test_phase_75_llm_rail.py:396-399: per-file raw `'gemini-2.5' not in text`, docstrings included) → co-located non-behavioural literals in in-scope files must ALSO be cleaned: `harness_memory.py:49-50` MODEL_CONTEXT_WINDOWS keys → the constants; prose rewords at sentiment:30, agent_map:120, autonomous_loop:2174, orchestrator:384, llm_client:797/921/976 (drop the exact lowercase token; "Gemini 2.5" prose form is fine).
5. **`gemini_retirement_warning()` has ZERO runtime callers** — C3's "coverage" is the test-time resolved-model→warning relationship; no startup wiring is in scope (the missing runtime tripwire is QUEUED, see below).
6. Import precedent settled (75.5 added the same import to 6 files; no cycle risk — model_tiers resolves settings lazily). For the 167KB autonomous_loop.py an in-function local import is acceptable (agent_map.py:142 precedent).
7. Retirement date 2026-10-16 triple-confirmed official for 2.5-flash AND 2.5-pro (matches `GEMINI_2_5_RETIREMENT_DATE`). Recency: the official successor is now gemini-3.6-flash (model_tiers comments say 3.5) — migration context only, NOT actionable here (no VALUE changes).
8. The `GEMINI_WORKHORSE == 'gemini-2.5-flash'` value-pin test is an INTENTIONAL Oct-2026 migration tripwire — do not loosen it.

## Hypothesis

Routing all 9 behavioural pins through `GEMINI_WORKHORSE` (+ the family guard through `GEMINI_2_5_FAMILY_PREFIX`) and cleaning co-located literals makes the strict tree-wide scan pass with zero behavioural change (every site provably resolves to the same string as before), so the October retirement becomes a ONE-FILE migration instead of a 10-site hunt — with the resolved-value pins acting as the deliberate migration tripwire.

## Plan (delegated Sonnet executor; Main reviews + runs mutations)

1. `model_tiers.py`: add `GEMINI_2_5_FAMILY_PREFIX = "gemini-2.5"` (ONLY addition; no value changes).
2. Route the 9 pins → `GEMINI_WORKHORSE` (75.5 import line; in-function import acceptable for autonomous_loop.py); `llm_client:985` → `startswith(GEMINI_2_5_FAMILY_PREFIX)`.
3. Clean co-located literals per finding 4 (prose uses "Gemini 2.5" form or the constant NAME; `harness_memory:48` 2.0-flash row untouched).
4. New `backend/tests/test_phase_75_5_2_model_pins.py`:
   - C1: parametrized per-file strict scan over `(REPO/backend).rglob('*.py')` + `scripts/harness/run_autonomous_loop.py`, EXCLUDE = {model_tiers.py, cost_tracker.py, settings_api.py} + backend/tests/** — plus the NON-VACUOUS self-test (in-scope list non-empty AND a superset of the known pin files; the 75.5.8 "a scan that can't find its own members fails" doctrine);
   - C2: value-pins `GEMINI_WORKHORSE == 'gemini-2.5-flash'`, `GEMINI_DEEP_THINK == 'gemini-2.5-pro'` (the migration tripwire); per-site resolution checks (module-const/param-default introspection for sentiment/masker; behavioural capture for agent_map + directive_review/rewriter via patched genai with Anthropic forced to fail; AST Name-reference + MISROUTE guard — references WORKHORSE, not DEEP_THINK — for the deep autonomous_loop fallbacks + the script);
   - C3: `gemini_retirement_warning(m, date(2026,9,15))` truthy + contains '2026-10-16' for BOTH constants; negatives: date(2026,9,14) → None, off-family 'gemini-3.6-flash' at the frozen date → None.
5. Mutation matrix (Main; C4 + §4c): M1 restore a literal at ONE site → its C1 case fails; M2 change the WORKHORSE VALUE → value-pin fails; M3 over-broaden the exclusion (fixture/scan mutation) → the self-test fails; M4 route one site to DEEP_THINK → the C2 misroute guard fails while C1 still passes (C2 independent of C1).
6. QUEUE as follow-up step 75.5.2.1 (do NOT fix here, per feedback_queue_discovered_defects_in_masterplan): (i) the retirement tripwire is never emitted at RUNTIME (zero callers); (ii) `_inventory.json` (~20 nodes) + `.env.example` stale-able display/sample strings; (iii) `_BUILD_TIER['gemini_deep_think']` uses the literal inside model_tiers.py itself.
7. live_check_75.5.2.md: verification output (exit 0) + git diff --stat + the census table reference + mutation evidence; Q/A via qa-verdict Workflow; log; flip; push. This step is the pre-/clear checkpoint (operator decision this session).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.5.2)

> command: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py -q`

1. "New backend/tests/test_phase_75_5_2_model_pins.py passes offline and proves, by a tree-wide scan, that ZERO behavioural gemini-2.5 literals remain outside backend/config/model_tiers.py (docstring prose included -- 75.5 set the precedent that the scan is read strictly, not reinterpreted)"
2. "Test asserts each newly-routed site resolves to the same model string it resolved to before the change (no tier pin VALUE changed -- prove by pinning the resolved values)"
3. "Test asserts gemini_retirement_warning fires for every routed site's resolved model under a frozen >=2026-09-15 date, and is silent both before that date and for a non-2.5 model (two negative controls)"
4. "Mutation matrix in experiment_results.md: restoring a literal at ONE of the 8 sites, and changing one resolved VALUE, each fail at least one test"

## References

- `handoff/current/research_brief_75.5.2.md` (census table + classification rule + coverage log; 6 read-in-full incl. ai.google.dev deprecations + model catalog)
- model_tiers.py (GEMINI_WORKHORSE / GEMINI_DEEP_THINK / GEMINI_2_5_RETIREMENT_DATE / gemini_retirement_warning), test_phase_75_llm_rail.py:396-399 (strict-scan precedent), the 9-site census with pre-change resolved values
- 75.5 doctrine + feedback_queue_discovered_defects_in_masterplan
