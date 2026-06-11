# Experiment Results — Step 57.1 (GENERATE)

**Step:** 57.1 — Binding RiskJudge gate + concentration-aware prompt context (phase-57 FEATURE; operator pick verbatim `PHASE-57: FEATURE`). **Date:** 2026-06-11. **Mode:** config-gated default-OFF feature; NO live flag flip; do-no-harm.

## What was built (4 files)

| File | Change | Finding |
|---|---|---|
| `backend/config/settings.py` | NEW flag `paper_risk_judge_reject_binding: bool = Field(False, ...)` (F-3/F-8-citing description; SEC 15c3-5 rationale) | F-3 |
| `backend/services/portfolio_manager.py` | The binding gate at the candidate-build chokepoint: flag-gated `continue` on `decision == "REJECT"` BEFORE `buy_candidates.append` — the common ancestor of the main BUY loop AND the swap path (which carried all 3 real-world REJECT executions); structured warning + new backward-compatible `blocked_out` out-channel kwarg; budget reallocates by construction (dropped candidates never consume emit-loop cash) | F-3 |
| `backend/services/autonomous_loop.py` | Prompt builders `_build_risk_judge_system/_build_risk_judge_template` (flag-OFF returns the VERBATIM constants — `is`-identity; flag-ON corrects the phantom 10% → configured 30% cap and injects the sector-context line); `_build_portfolio_sector_context` (compact sector weights, `current_price or avg_entry_price` fallback); per-cycle single compute wired at the positions read, threaded as `portfolio_context` kwarg through `_run_single_analysis` → both lite analyzers (Claude + Gemini twins, incl. the full-path lite fallback); `summary["risk_judge_blocked"]` surfacing | F-8, F-3 |
| `backend/tests/test_phase_57_1_reject_binding.py` | NEW: 7 tests — main-path + swap-path regression fixtures (OFF emits / ON blocks), OFF order-identity, OFF prompt verbatim-constant identity, ON prompt content (cap + sector line), sector-context edges, structural single-compute assertions | C1-C4 |

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests -k 'reject_binding or risk_judge_binding' -q
7 passed, 767 deselected, 1 warning in 2.37s
$ test -f handoff/current/live_check_57.1.md && echo PASS
PASS
```

Full-suite regression: `756 passed, 12 skipped, 6 xfailed` (exit 0 — +7 from the new file, zero breakage; the flag-OFF default leaves every existing test untouched).

## Key outcomes

1. **The away-week vulnerability is reproduced in a fixture and closed by the flag**: the swap-path test shows a REJECT candidate swap-buying with `risk_judge_decision == "REJECT"` on the emitted order (exactly the HPE/DELL/LG pattern) when OFF, and being blocked — with the next-ranked survivor taking the freed slot — when ON.
2. **Event study confirmed live** (BQ, 2026-06-11): the 3 executed-REJECT trades realized −$0.81 / +$0.54 / −$23.18, net **−$23.45**, presented with the mandatory n=3 selection-bias caveat and no EV extrapolation.
3. **Byte-identity proof is structural**: flag-OFF prompt builders return the constant OBJECTS (`is` assertions), and flag-ON == flag-OFF on REJECT-free order sets.
4. **The judge is no longer blind when binding**: ON-mode prompts carry the real 30% cap + live sector breakdown computed once per cycle (never per-ticker — structural source-scan assertions).
5. Effective runtime flag verified `False` post-deploy-restart context (settings loader) — **no live flip**; the operator flips after OOS observation.

## Honest limitations

- The n=3 event study is descriptive only (selection-conditioned); the gate's EV is unknowable until post-flip OOS observation — stated in the live_check and the chapter.
- The full-pipeline (non-lite) RiskJudge path is the orchestrator's own risk debate — out of 57.1 scope (the lite path is what trades autonomously today).
- Blocked-BUY evidence is log + cycle-summary only (no BQ table); a durable `paper_blocked_trades` table is a possible follow-on if DoD-7 wants per-block attribution.
