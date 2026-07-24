# Experiment results — Step 75.5.2 (the remaining gemini-2.5 behavioural pins → constants)

Date: 2026-07-24. Execution model: **sonnet-tagged → delegated Sonnet executor
GENERATE**; Main review + mutation matrix (which FOUND AND FIXED a guard
weakness); Researcher gate opus/max, **AUDIT-CLASS with coverage.dry**
(wf_c5355afe-01b — census swept until 2 consecutive dry rounds).

## What was built

1. **Census executed (9 sites, not the step text's 8):** the audit-class research
   gate re-derived the 8 named backend pins (2 had moved +22 lines) AND discovered
   a 9th behavioural pin at `scripts/harness/run_autonomous_loop.py:74` — included
   under C1's strict "outside model_tiers.py" reading. All 9 routed to
   `GEMINI_WORKHORSE` (none to deep-think — no behavioural 2.5-pro pin exists
   outside model_tiers.py).
2. **`model_tiers.py`:** ONE addition — `GEMINI_2_5_FAMILY_PREFIX = "gemini-2.5"`
   for the llm_client:985 thinking-budget family guard (behavioural but not a
   pin). NO value changed.
3. **Co-located literal cleanup** per the 75.5 strict-scan precedent (per-file raw
   substring, docstrings included): harness_memory MODEL_CONTEXT_WINDOWS keys →
   constants (the 2.0-flash row untouched); prose rewords at 7 sites using the
   "Gemini 2.5" form.
4. **New `backend/tests/test_phase_75_5_2_model_pins.py`** (306 collected):
   parametrized per-file strict scan + the NON-VACUOUS self-test (in-scope list
   must be a superset of the known pin files — a scan that can't find its own
   members fails); the deliberate migration-tripwire value-pins; per-site
   resolution proofs (const/default introspection + behavioural capture with
   patched genai for the meta_evolution fallbacks + AST reference/misroute guards
   for the deep sites); retirement-warning firing + two negative controls.

## The mutation-matrix finding (this cycle's headline)

M4 (misroute one site to GEMINI_DEEP_THINK) initially **SURVIVED**: an aliased
import (`GEMINI_DEEP_THINK as GEMINI_WORKHORSE`) defeats a call-site name check
while binding the wrong value. Main strengthened the script's guard with an
import-level AST assertion (un-aliased GEMINI_WORKHORSE import required;
GEMINI_DEEP_THINK import forbidden), documented the finding in the test comment,
and re-ran the FULL matrix: **4/4 killed, post-restore green**. This is §4c
working as designed — the matrix found a guard that could not fail under a
realistic sneaky mutation, and the guard got fixed before Q/A.

## Files changed

`backend/config/model_tiers.py` (+1 constant), the 9 pin files
(`directive_review.py`, `directive_rewriter.py`, `sentiment.py`,
`harness_memory.py`, `autonomous_loop.py` ×2 sites, `agent_map.py`,
`llm_client.py`, `orchestrator.py` prose, `scripts/harness/run_autonomous_loop.py`),
`backend/tests/test_phase_75_5_2_model_pins.py` (new),
`.claude/masterplan.json` (75.5.2 → in_progress; +75.5.2.1 queued), handoff artifacts.

## Verbatim verification output

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py -q
306 passed, 1 warning in 4.01s
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py backend/tests/test_phase_75_llm_rail.py -q
348 passed, 1 warning in 7.05s
$ .venv/bin/python -c "import backend.config.model_tiers, backend.news.sentiment, ... ; print('imports OK')"
imports OK
$ grep -rn "gemini-2.5" backend/ --include='*.py' | <exclusions> | wc -l
0
```

Lint: new test clean; all 10 edited files at exact finding-class parity with
their HEAD baselines.

## Mutation matrix (C4 + qa.md §4c) — final 4/4 killed (1 survivor found + fixed)

Full table + the survivor narrative in live_check_75.5.2.md §3; runner + both
verbatim logs in the session scratchpad. The two criterion-4-required mutations:
M1 (restore a literal → that file's scan case fails) and M2 (change a resolved
VALUE → 7 failures across the value-pin + every resolution test).

## Queued (feedback_queue_discovered_defects_in_masterplan)

**75.5.2.1** (P2, sonnet-tagged, research-gated): (i) the retirement tripwire has
ZERO runtime callers — nothing will ever page before 2026-10-16; (ii)
_inventory.json + .env.example stale-able display/sample strings; (iii)
_BUILD_TIER's in-file literal.

## Delegation record

Executor followed the census/brief precisely (its report pending flush at close;
Main verified everything first-hand: both suites, independent zero-literal grep,
per-file lint parity, key-diff review, and the mutation matrix incl. the M4
survivor fix authored by Main).

## CYCLE 2 (post-CONDITIONAL fix)

The cycle-1 Q/A CONDITIONAL named exactly one blocker: the alias-misroute shape
was closed at 1 of 3 AST-guarded sites (Main's site-9 fix), still open at
autonomous_loop sites 6,7. Fix: a new import-level alias-proof test scoped to
the module's model_tiers imports (un-aliased bind required; GEMINI_DEEP_THINK
import forbidden), comment citing the Q/A finding. Matrix extended with M5 (the
exact surviving mutation) — final: **5 mutations, 5 killed, survivors NONE**,
post-restore green, combined suites 349 passed. The cycle-1 headline
("aliased shape closed") is now true at ALL guarded sites, and the
over-generalisation is corrected here per the critique.
