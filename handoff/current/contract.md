# Contract — phase-73.2: D2b learn-loop v2 design

**Step id:** 73.2 (phase-73, depends_on 73.1 = done/PASS @5c058428)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_a195f7b3-5d8` (opus/max, tier=moderate): 6 sources read in full (FinMem Eq 1-6 via ar5iv; agent survey; FinCon CVRF; Generative-Agents memory scoring; Reflexion; 2026 Agentic-Trading survey), 20 URLs, recency scan, 6 internal files re-verified line-by-line. Brief: `research_brief_73.2.md`. Returned five structured `design_inputs` + an exhaustive 6-item `deadness_causes` stack — all transcribed verbatim into the design doc.

Load-bearing findings:
1. **The deadness is a STACK, not one crash** (corrects 73.0's "~2-line keystone" framing): DC1 the crash with TWO legs (TYPE: `evaluate_all_pending:137` passes raw native datetime; TZ: the live path's `created_at.isoformat()` is tz-aware — both die in the shared method at `:47/:50`); DC2 the flag default gating the ENTIRE fan-out at `autonomous_loop.py:2964` (independent of the crash); DC3 the `logger.debug` swallow at `:3050` that hid 36+ days of empty tables while Step 9 reported "learning" healthy; DC4 the `model=None` branch killing reflections on all periodic/manual paths; DC5 rolling-mark P&L (learns from the wrong number even when it works).
2. **Two things are ALREADY FIXED — do not re-fix**: the close-event seam (phase-30.3) and live model-injection (phase-31.1); base BQ tables exist.
3. **Design decisions**: single Q=90d multiplicative decay (`bm25_norm × exp(-d/90) × imp_mult`) over FinMem's filing-typed 14/90/365 layers (our-scale justified; matches the holding horizon); trim reflections 4→1-2 per close for BM25 corpus hygiene (Reflexion/FinCon one-lesson-per-episode); ONE additive nullable BQ migration serves importance-bump + idempotency + forward-compatible evidence-source attribution (73.5 PiT-RAG ready, zero backfill); two-stage injection gate (relevance floor FIRST, decay rerank within survivors); realized-exit-primary reflection satisfying the survey's Outcome-Embargo by construction.

## Hypothesis

Clearing the enumerated stack (crash + swallow + realized-P&L inversion) and shipping the decay/injection upgrades turns the dead loop into the compounding-edge substrate (#2) that #1's calibration and 73.5's evidence weighting both consume — at $0, on existing tables, with the flag flip remaining the operator's dark-until-token decision.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.2)

- "b_learn_loop_v2.md enumerates every independent deadness cause with file:line and its fix, plus the decay-tier memory design mapped onto the existing FinancialSituationMemory/BM25 substrate (upgrade, not greenfield)"
- "Reflection write/retrieve seams specified end-to-end (closed trade -> reflection -> retrieval into future analysis prompts) with token-cost bounds"
- "Executor-tagged build steps appended pending with live_checks (a BQ reflection row from a real closed trade; a retrieval hit in a live analysis prompt); no code edited this session"

verification.command: `bash -c 'test -f handoff/current/design_pack_73/b_learn_loop_v2.md && grep -Eqi "decay|tier" handoff/current/design_pack_73/b_learn_loop_v2.md'`

## Plan

1. GENERATE: design doc finalized verbatim from the gate (done, 15,331 chars — deadness stack §top, five component specs §1-5, decisions-of-record); append executor build steps 73.2.1-73.2.3 (pending, tagged, immutable live_checks matching the criteria's two artifacts).
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 120) → flip 73.2 done.

## References

- `handoff/current/research_brief_73.2.md`; `frontier_map_73.md` (#2 verdict); `design_pack_73/b_learn_loop_v2.md`
- FinMem ar5iv 2311.13743 (Eq 1-6); Generative Agents 2304.03442; FinCon 2407.06567; Reflexion 2303.11366; agent survey 2408.06361; 2026 Agentic-Trading survey (Outcome Embargo)
