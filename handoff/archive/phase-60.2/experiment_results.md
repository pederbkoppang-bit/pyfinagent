# Experiment Results -- Step 60.2 (GENERATE)

**Step:** 60.2 -- Churn-engine fix: swap sentinel + re-eval/stamp mismatch + delta scale (AW-5). **Date:** 2026-06-11.

## What was built

### 1. The flag + sentinel elimination (criteria 1+2)

- NEW `settings.paper_swap_churn_fix_enabled` (default **OFF**, do-no-harm). Description records the full design rationale + the operator-gated promotion rule.
- `backend/services/portfolio_manager.py::_compute_swap_candidates`: flag ON -> a holding absent from the same-cycle `holding_lookup` is **EXCLUDED from swap displacement entirely** (criterion-1 option B). Flag OFF -> the historical sentinel 0.0 verbatim (byte-identical), now commented as a KNOWN DEFECT pointing at the flag.
- **Design choice recorded (researcher-grounded, departs from the brief's first preference WITH reasons):** the researcher's §1 preferred LOCF valuation with exclusion fallback; GENERATE chose pure exclusion because (a) criterion 2 demands the away-week pattern be impossible BY CONSTRUCTION -- under LOCF a day-old holding can still be displaced when day-over-day score noise (mean |delta| 1.10 on the 1-10 scale, 59.3 stability table) crosses the 25% relative bar at low scores; exclusion is structural and noise-robust; (b) the holding keeps its slot only 3-4 days (re-eval cadence) and stop-losses still protect it; (c) Alpha Decay literature (brief §A) says a day-old score is evidence FOR holding, not for displacement urgency. The LOCF BQ plumbing was prototyped and REMOVED (no dead code).
- Criterion-2 mechanism: the BUY-time `last_analysis_date` stamp is semantically true (the buy-day analysis exists); the defect was the swap path never consuming it. With exclusion, a holding lacking same-cycle analysis can never be displaced -- 'N new + 0 re-evals' followed by same-cycle swap-out is impossible by construction (parametrized test: candidate scores 7.0/9.0/10.0 all fail to displace). Plus a flag-ON hours-precise re-eval age comparison in autonomous_loop (HONESTY NOTE: behavior-identical for integer day thresholds -- floor(x)>=n iff x>=n -- it only matters for fractional cadences; NOT claimed as the criterion-2 mechanism).

### 2. Delta-scale correction (criterion 3)

- The false "[0,1] final_score" comment at the denominator REWRITTEN with the true facts (scores are 1-10; the 0.01 epsilon made sentinel comparisons ~70,000% vs the 25% bar).
- Flag ON: denominator `max(abs(holding_score), 1.0)` -- the clamp `settings.paper_swap_min_delta_pct`'s own description has always documented. Flag OFF: the historical 0.01 epsilon verbatim.
- The 25.0 bar itself UNTOUCHED (widening it would be the 53.1/55.3-rejected band family -- boundary recorded in the contract's criterion-3 interpretation). Boundary tests: 7-vs-5 = 40% fires (true evidence), 6-vs-5 = 20% does not; for real scores >= 1.0 the clamp is inert (OFF/ON identical decisions).

### 3. Replay event study (criterion 4) -- NEW `scripts/replay/replay_60_2_swap_fix.py`

- Replays every recorded swap decision in 2026-05-29..06-10 through the PRODUCTION `_compute_swap_candidates` (both arms). Fidelity arm 12/13 reproduced (1 away-week persistence gap, disclosed). ON arm: 11/13 suppressed (10 sentinel-driven), 2 survive on true evidence.
- Counterfactual ledger currency-neutral (pct moves x USD notionals -- the first run had a KRW/USD unit bug in KR hold-through legs, caught and fixed before reporting; AW-9's own lesson).
- Full results: handoff/current/replay_60_2_results.md + summarized in live_check_60.2.md §C with the honest both-sides reading (net one-step counterfactual -270.86 USD in this falling window vs the structural chain-suppression benefit the one-step method cannot capture).

## Files changed

backend/config/settings.py (flag), backend/services/portfolio_manager.py (exclusion + clamp + corrected comments), backend/services/autonomous_loop.py (hours-precise re-eval age, flag-gated), backend/tests/test_phase_60_2_churn_fix.py (NEW, 8 tests), scripts/replay/replay_60_2_swap_fix.py (NEW), handoff/current/replay_60_2_results.md (NEW, generated).

## Verification command output (verbatim)

```
$ python -m pytest backend/tests -k 'swap or sentinel or reeval or 60_2' -q
17 passed, 793 deselected, 1 warning in 2.33s          (exit 0)
$ test -f handoff/current/live_check_60.2.md && echo OK  -> OK
```
FULL suite: `792 passed, 12 skipped, 6 xfailed` exit 0 (was 784 post-60.1).

## Artifact shape

- Flag-ON swap exclusions log: `"Swap path: <ticker> has no same-cycle analysis -- excluded from displacement this cycle (churn fix ON)"`.
- Replay artifacts: per-swap table (day/sold/bought/OFF/ON/basis), named-round-trips table, counterfactual ledger lines, metrics table OFF-vs-ON.
- Operator decision line in live_check_60.2.md §D: PENDING (the flag is never auto-applied).
