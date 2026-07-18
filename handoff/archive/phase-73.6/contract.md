# Contract — phase-73.6: D3 money runway (recommend-only)

**Step id:** 73.6 (phase-73, depends_on 73.5 = done/PASS @47936199)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No spend, no flags, no code, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_9b114107-a7e` (opus/max, tier=simple, floor held: 6 sources read in full — 2 broker [Alpaca paper-fill fidelity], 2 governance [SR 11-7/SR 26-2], 2 practitioner [paper-to-live transition rigor]; 32 URLs; recency scan; 8 internal files incl. `paper_go_live_gate.py` in full). Brief: `research_brief_73.6.md`. Returned 3 ordered `runway_stages` with prerequisites/evidence-anchors/operator-decisions — transcribed verbatim.

Load-bearing findings:
1. **The phase-69 register note is VERIFIED FIXED**: the two under-spec go-live booleans were tightened to their documented definitions in 69.2 (sustained-PSR true 30-day min; dd_tolerance = backtest_max_dd + 5.0) — the gate measures what it claims.
2. **Honest go-live answer: NOT eligible, clock not started** — trades_ge_100 counts REAL round trips (0 today; ~30 synthetic whole-table; '59' was raw fills); the clock starts only at the Stage-2 alpaca_paper cutover.
3. **`real_capital_enabled=False` is a hard gate beyond the 5 booleans**; its SR 11-7 citation superseded by SR 26-2 (2026-04-17) — principles carry forward, stale-citation doc-drift noted recommend-only.
4. **Stage 2 strengthens three phase-73 designs**: measured slippage replaces 73.4's estimate; 68.4 activates the write path 73.2 repairs; real fills are the clean substrate 73.3's calibration needs.
5. External consensus: live degrades vs sim; even Alpaca paper is optimistically biased; our 100-round-trip + human-token bar is deliberately stricter than the retail base rate.

## Hypothesis

A one-page, honestly-scored runway (restore → real fills → go-live) with every operator decision as a verbatim line turns "make more absolute $" from an ambition into a checklist — without duplicating phase-68/58.1 or spending anything.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.6)

- "money_runway_73.md is one page, sequences paper-restoration -> real-fill -> go-live with prerequisites and evidence anchors, and enumerates every operator decision as one actionable line"
- "Consistent with (not duplicating) phase-68/58.1 masterplan entries and the phase-72 ACT-NOW block"
- "Recommend-only; no spend, no flags, no code"

verification.command: `bash -c 'test -f handoff/current/money_runway_73.md && grep -Eqi "real.?fill|go.?live" handoff/current/money_runway_73.md'`

## Plan

1. GENERATE: runway finalized verbatim from the gate (done, 6,742 chars, one page, 3 stages, 13 operator-decision lines). No build steps to append — the runway references the EXISTING phase-68/58.1 queue rather than duplicating it.
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 124) → flip 73.6 done.

## References

- `handoff/current/research_brief_73.6.md`; `money_diagnosis_72.md` P0; `operator_decision_sheet_72.md` ACT-NOW/P3; `frontier_map_73.md` #10; masterplan phase-68 + 58.1
- Alpaca paper-trading docs; SR 11-7 / SR 26-2 (sia-partners); paper-to-live practitioner treatments (in brief)
