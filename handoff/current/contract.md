# Contract -- 60.2 Churn-engine fix: swap sentinel + re-eval/stamp mismatch + delta scale (AW-5)

**Step:** 60.2 (phase-60, P0, harness_required, depends_on 60.1 done). **Date:** 2026-06-11.
**AW basis:** AW-5 (handoff/archive/phase-59.3/59.3-harness-free-output.md §5-6; draft cites = snapshot 70a8242b, CURRENT lines re-verified by the researcher below).

## Research-gate summary (researcher ad82d911, tier=complex, gate_passed:true)

6 external sources read in full (KDD-2026 FINSABER overtrading; arXiv:2603.27539 eval standards; Boyd et al. multi-period trading; Alpha Decay (Di Mascio/Lines/Naik); Balch et al. JPM replay-vs-simulation; Maven alpha-decay), 35 URLs, recency scan. Brief: handoff/current/research_brief.md.

- **Sentinel (HEAD):** `backend/services/portfolio_manager.py:476-483` -- holding absent from same-cycle `holding_lookup` -> conviction 0.0 "Treat as worst". With the `/max(|h|, 0.01)` epsilon (:531), candidate 7.0 vs sentinel = **70,000% delta vs the 25% bar** -- the load-bearing bug behind the MU/SNDK/DELL churn.
- **Scale-comment drift:** settings.py:293 DOCUMENTS a `max(|h|, 1.0)` clamp; the code replaced it with 0.01 on the false "[0,1] final_score" premise (:526-531). The "scale accident" is the DENOMINATOR (sentinel 0.0 + epsilon), not the 25.0 bar.
- **Stamp is semantically true:** `paper_trader.py` BUY-time `last_analysis_date` reflects a REAL same-day candidate analysis; the defect is the swap path never CONSUMING that analysis (holding_lookup = this-cycle only). Re-eval gate `.days` truncation makes 3 days effectively 3-4 (DELL sold at 2d23h unanalyzed).
- **Design (researcher-grounded):** LOCF age-capped valuation from `financial_reports.analysis_results` via existing `bigquery_client.get_report` (:303-358); holdings with NO score inside the cap are EXCLUDED from swap displacement (alpha decays over months, not 1 day -- Alpha Decay paper + Maven: staleness cost is progressive, no 1-day cliff; fabricated-0.0 evidence is a correctness defect).
- **Binding-ruling boundary (53.1/55.3 anti-band rulings):** sentinel/stamp/scale fixes are CORRECTNESS repairs (fabricated evidence, invalid comparison, spec drift). FORBIDDEN here: raising paper_swap_min_delta_pct, new absolute floors, tenure shields ("no swap if held < N days" = time-domain band). LOCF equal-score no-swap is evidence symmetry, not stickiness.
- **Replay reality check:** NO existing tool replays the swap path (`decide_trades` consumers: autonomous_loop:1155 + tests + 2 drill scripts; strategy_backtest_adapter.py:43 confirms non-wiring; rebalance_band.py is the 53.1-rejected monthly lever, not this). Criterion-4 is delivered as a **decision-replay event study** (57.1 precedent): per away-week cycle, rebuild decide_trades inputs from persisted BQ analyses + position state; flag-OFF arm must REPRODUCE the recorded orders (fidelity proof); flag-ON arm reports one-step order diffs (primary), compounded NAV/Sharpe/turnover/maxDD (secondary, candidate-stream divergence disclosed per Balch et al.; Sharpe via analytics.py:239 sharpe_diff_test, reported as underpowered at T~12).
- **Fixtures BQ-confirmed (exact rows in brief):** MU 06-08->06-09 -$44.95 in 1d00h00m03s; SNDK 06-08->06-09 -$2.46 then re-bought HIGHER 06-10; DELL 06-05->06-08 +$11.17 and 06-09->06-10 +$14.73 -- the DELL legs were PROFITABLE; suppressing them is a COST of the fix and must be reported as such.
- **Test-net trap:** `-k 'swap or sentinel or reeval or 60_2'` currently collects 9 tests; `test_portfolio_swap.py` TECH1 fixture uses [0,1] scores and would STOP FIRING under a flag-OFF clamp change -> the corrected formula must be flag-ON only; ON tests need 1-10 integer fixtures; must compose with the 57.1 binding-REJECT flag (disjoint regions; test both ON).

## Hypothesis

Valuing unanalyzed holdings at their last real, age-capped score (instead of fabricated 0.0) and excluding unvaluable holdings from displacement removes the away-week churn mechanism by construction (no sentinel -> no swap-out bait), with the US momentum core byte-identical while the flag is OFF; the replay event study quantifies suppressed round trips (incl. the profitable DELL legs) so the operator can decide promotion on evidence.

## Immutable success criteria (verbatim from .claude/masterplan.json step 60.2)

**Command:** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'swap or sentinel or reeval or 60_2' -q && test -f handoff/current/live_check_60.2.md`

1. "the conviction-0.0 sentinel is eliminated behind a config flag (default OFF, do-no-harm): with the flag ON, a holding absent from the same-cycle holding_lookup is valued at its last persisted final_score (age-capped, source: financial_reports.analysis_results) or is excluded from swap displacement entirely -- design choice researcher-grounded; a regression test reproduces the 06-09 scenario (position bought the prior cycle, no same-cycle analysis, equal-score candidate) and asserts the swap fires with the flag OFF (current behavior) and does NOT fire with the flag ON"
2. "the re-eval/stamp mismatch is closed: with the flag ON, any holding that is a candidate for swap displacement is re-analyzed in the same cycle before the swap decision, OR last_analysis_date semantics are corrected so a BUY-time stamp cannot suppress re-evaluation of swap-eligible holdings; a test asserts the away-week pattern ('N new + 0 re-evals' followed by same-cycle swap-out of a day-old holding) is impossible by construction with the flag ON"
3. "the swap delta threshold is re-derived for the actual 1-10 integer score scale (the [0,1] assumption at portfolio_manager.py:495-496 is corrected in code + comment): a 2-point integer difference (7-vs-5) no longer auto-clears the bar by scale accident; the chosen threshold semantics are documented against paper_swap_min_delta_pct and covered by tests at the boundary"
4. "ON-vs-OFF is measured on the $0 replay over the production universe (incl. the away-week window) reporting Sharpe/return/turnover/maxDD: the report must show the away-week one-day round trips (MU 06-08->06-09, SNDK 06-08->06-09, DELL 06-05->06-08->06-09) suppressed or surviving with stated reasons; promotion (flag flip) is an OPERATOR DECISION recorded in the live_check, never auto-applied; US momentum core byte-identical with the flag OFF (test)"

**live_check:** "REQUIRED -- regression-test output, the ON-vs-OFF replay table, BQ MCP rows reproducing the 06-08/06-09 sentinel swaps as the fixture basis, and the operator's promotion decision line (or 'pending')."

### Criterion-3 interpretation (recorded BEFORE GENERATE so Q/A can hold us to it)

The "scale accident" in the away week was the DENOMINATOR: a real-scored holding (e.g. 5) entered the comparison as sentinel 0.0, so ANY candidate cleared the 25% bar at ~70,000%. The re-derivation = correct denominator semantics on the true 1-10 scale (flag ON: `max(|h|, 1.0)` clamp per settings.py:293's own documented spec + LOCF holding scores), comment corrected, semantics documented against `paper_swap_min_delta_pct`, boundary tests proving 7-vs-5 computes 40% (fires iff 40 > bar -- true evidence, not accident) and 6-vs-5 computes 20% (does NOT fire at the default 25 bar). WIDENING the bar so 7-vs-5 never fires would be the forbidden 53.1/55.3 band family -- explicitly NOT done.

## Plan

1. **Settings (2 new fields):** `paper_swap_churn_fix_enabled: bool = False` (the flag; do-no-harm) + `paper_swap_locf_max_age_days: int` (LOCF age cap, default 5 calendar days -- inside the 3-4-day effective re-eval cadence so a holding on normal cadence always has a valid LOCF score; researcher-grounded: alpha horizon >> days).
2. **portfolio_manager (flag ON path only):** holdings absent from same-cycle holding_lookup get LOCF score from `analysis_results` (existing `get_report`-shaped query, age-capped, batched once per decide call, fail-open to exclusion on BQ error); holdings with no in-cap score are EXCLUDED from displacement (cannot displace what cannot be valued); delta denominator `max(|h|, 1.0)`; comment corrected at the [0,1] site (comment fix is unconditional; code change flag-gated). Compose with the 57.1 binding-REJECT gate (disjoint code regions; both-ON test).
3. **Re-eval/stamp (criterion-2, chosen branch = semantics corrected):** the swap decision now CONSUMES the same analysis the BUY stamp refers to (LOCF), so a day-old holding can never be sentinel-bait; plus flag-ON fix of the `.days` truncation in the re-eval gate (hours-precise comparison so "3 days" means 72h, not 3.0-3.99 days). Test: construct the away-week scenario (bought prior cycle, no same-cycle analysis, equal-score candidate) -> flag OFF swap fires (regression-locks current behavior), flag ON it cannot (LOCF equal-score = no delta > bar; with LOCF removed/aged-out -> excluded from displacement) -- impossible by construction either way.
4. **Replay event study** `scripts/replay/replay_60_2_swap_fix.py`: away-week window (2026-05-29..06-10) + full production stream; per cycle rebuild decide_trades inputs from BQ analysis_results rows + reconstructed position state from paper_trades; ARM A flag OFF must reproduce recorded orders (fidelity gate -- mismatches reported verbatim); ARM B flag ON one-step order diffs (primary) + compounded Sharpe/return/turnover/maxDD (secondary, divergence disclosed) + the 3 named round trips suppressed-or-survived WITH the profitable-DELL cost stated; sharpe_diff_test T~12 power caveat.
5. **Tests** (named into the immutable -k net): the 06-09 regression scenario OFF-fires/ON-not; away-week-pattern-impossible; boundary 7-vs-5=40%/6-vs-5=20%; LOCF age-cap edges (in-cap used, out-of-cap excluded); flag-OFF byte-identity (existing TECH1 [0,1] fixture untouched + an explicit OFF-path identity test); 57.1-compose both-ON.
6. **live_check_60.2.md:** regression output, replay table, BQ MCP fixture rows, operator promotion line ("pending" until the operator replies), burn note ($0 -- replay reads persisted rows only).
7. Fresh Q/A -> harness_log -> flip.

## Do-no-harm

Flag default OFF; OFF path byte-identical (locked by the existing [0,1]-fixture tests + explicit identity test); comment-only edits on the OFF path; replay is read-only BQ. NO live flag flips in this step; promotion is the operator's line in the live_check.

## References

handoff/current/research_brief.md (full source table + BQ fixture rows); analytics.py:239 sharpe_diff_test (52.3); 57.1 event-study precedent; 53.1/55.3 binding rulings (anti-band); Anthropic harness-design (file-based handoffs).
