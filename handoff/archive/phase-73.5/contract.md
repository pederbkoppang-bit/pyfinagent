# Contract — phase-73.5: D2e judged pilots

**Step id:** 73.5 (phase-73, depends_on 73.4 = done/PASS @10dea413)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_b9308ff4-315` (opus/max, tier=simple — floor held: 6 sources read in full incl. champion/challenger deployment canon [Databricks, Wallaroo, PDx arXiv:2512.22305] and the QuantAgent adversarial contrast; recency scan; **15 internal files** — the heaviest internal leg of the phase). Brief: `research_brief_73.5.md`. Returned 3 structured `pilot_verdicts` + the `unfreeze_token` package — transcribed verbatim.

Load-bearing findings:
1. **Champion-bridge scope DECISIVELY NARROWED**: exhaustive grep proved `tp_pct`/`holding_days`/`target_annual_vol` have ZERO live consumers (the live exit is an R-multiple model, structurally unlike the backtest's fixed barriers) and `backtest_*` settings are unread by services. v1 bridge = `max_positions→paper_max_positions` (the one clean live-consumed key), sl/trailing as explicit opt-in sub-flags (they move the R unit / HWM base), everything else NEVER-BRIDGE. The bridge must DEFER to the operator's `risk_overrides` runtime lever, never clobber it.
2. **The gap re-verified**: best_params reaches only the heartbeat (autonomous_loop.py:404-408); decide_trades takes settings only (:1406, portfolio_manager.py:66); the flag is fresh.
3. **No code-level freeze flag exists** — historical_macro freeze is a doctrinal operator boundary; the bridge consumer reads no historical_macro (freeze-safe build-dark); only the live-flip needs the validation batch.
4. **#6/#7 DEFER re-verified post-73.2/73.3**: bars moved (73.2's attribution field advances one of #6's three prerequisites) but do NOT clear; #7 unchanged (73.1 guards designed-not-built; mining against unbuilt guards industrializes overfitting).
5. **Adversarial anchor**: QuantAgent auto-adopts refinements with no promotion gate — the standing justification for deferring self-evolution; external champion-challenger consensus = our dark-until-validated pattern.

## Hypothesis

One narrow, deferential, dark champion-bridge build plus two honestly-deferred pilots (each with a concrete flip bar) captures the available pilot value without a single leakage-suspect return claim or premature build.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.5)

- "e_pilot_verdicts.md gives each pilot an explicit BUILD or HOLD verdict grounded in D1's mechanism notes + our-scale constraints, never in leakage-suspect return claims"
- "The champion-bridge verdict includes the un-freeze validation plan + verbatim proposed operator token; recommend-only discipline held"
- "BUILD verdicts appended as executor-tagged pending steps with live_checks; HOLD verdicts state the evidence that would flip them; no code edited"

verification.command: `bash -c 'test -f handoff/current/design_pack_73/e_pilot_verdicts.md && grep -Eqi "BUILD|HOLD" handoff/current/design_pack_73/e_pilot_verdicts.md'`

## Plan

1. GENERATE: design doc finalized verbatim (done, 10,706 chars — three verdicts + the `HISTORICAL MACRO UNFREEZE: CHAMPION-VALIDATION-BATCH` token proposal with validation plan + scope limits); append build step 73.5.1 (champion-bridge, the sole BUILD).
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 123) → flip 73.5 done.

## References

- `handoff/current/research_brief_73.5.md`; `frontier_map_73.md` (#5/#6/#7); `design_pack_73/{b,c}` (substrate cross-refs)
- Databricks champion-challenger docs; Wallaroo shadow deployment; PDx arXiv:2512.22305; QuantAgent 2402.03755 (adversarial); Profit Mirage (leakage discipline)
