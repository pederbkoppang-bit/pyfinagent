# Experiment Results — phase-73.2: D2b learn-loop v2 design

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered).

## What was built

1. **Research gate** (`wf_a195f7b3-5d8`, opus/max, tier=moderate): gate_passed=true; 6 sources read in full (FinMem decay math Eq 1-6 via ar5iv, Generative-Agents scoring canon, FinCon realized-outcome self-critique, Reflexion, agent survey, 2026 Agentic-Trading survey with its Outcome-Embargo prescription); 6 internal files re-verified line-by-line; returned five `design_inputs` + a 6-item exhaustive `deadness_causes` stack. Brief: `research_brief_73.2.md`.
2. **`design_pack_73/b_learn_loop_v2.md` finalized (15,331 chars)** — deadness stack transcribed verbatim (DC1 two-leg crash, DC2 flag gate independent of the crash, DC3 the DEBUG swallow that hid 36+ empty days, DC4 model=None on periodic paths, DC5 rolling-mark P&L, plus a verified NOT-DEAD do-not-re-fix list correcting the frontier map's framing), five component specs verbatim, and five design decisions of record (single Q=90d multiplicative decay; 4→1-2 reflection trim for corpus hygiene; one additive migration serving three components; two-stage injection gate; realized-exit-primary).
3. **Executor build steps appended pending**: 73.2.1 crash-fix + observability [sonnet-4.6/high], 73.2.2 reflection-on-close + additive BQ migration [sonnet-4.6/high], 73.2.3 decay re-rank + two-stage injection [sonnet-4.6/high] — each with an immutable live_check; the criteria's two named artifacts map to 73.2.2's live_check (BQ reflection row from a real closed trade) and 73.2.3's (retrieval hit in a live analysis prompt). The DC2 flag flip stays the operator's dark-until-token decision — no step self-promotes it.
4. Token-cost bounds recorded end-to-end: reflections 1-2 flat-fee Gemini calls per close; injection ~250-350 tok/agent at k=2; decay re-rank pure O(N) arithmetic, no embeddings.

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/design_pack_73/b_learn_loop_v2.md && grep -Eqi "decay|tier" handoff/current/design_pack_73/b_learn_loop_v2.md'
73.2 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## File list

- `handoff/current/contract.md` (73.2; gate → contract → GENERATE; write-first skeleton disclosed as in prior steps)
- `handoff/current/research_brief_73.2.md`
- `handoff/current/design_pack_73/b_learn_loop_v2.md`
- `.claude/masterplan.json` (73.2 in-progress; 73.2.1-73.2.3 appended pending)

## Scope honesty

No product code, no .env, no flags, no optimizer runs, no metered spend. The design CORRECTS the frontier map's own "~2-line keystone" simplification (stack of 5 + 1 not-dead) and explicitly fences the already-fixed seams against re-work.
