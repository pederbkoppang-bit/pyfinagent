# Contract — phase-53.1 (Algorithm/quant elevation)

**Date:** 2026-06-01. **Tier:** complex. **Step:** phase-53.1 (P2). $0 offline
replay (free yfinance, no LLM, no live cycles). Measure-first; NO live flag flip.

## N* delta (N* = Profit − Risk − Burn)

**Profit↑ (net-of-cost, speculative + measured):** a transaction-cost-aware no-trade band
cuts turnover on the monthly momentum reconstitution → lifts NET-of-cost Sharpe without
touching the (intact) gross momentum alpha. Honest prognosis: gross-flat, net-positive;
a net REJECT is a valid outcome. Risk/Burn neutral (offline measurement; default-OFF).

## Research-gate summary

`researcher` ran FIRST (gate **PASSED**: 7 sources read in full, 21 URLs, recency scan,
9 internal files). Brief: `handoff/current/research_brief.md`. **Recommended lever:
transaction-cost-aware no-trade / rebalance buffer band** — on each monthly top-N
reconstitution, retain a held name unless its rank drops below `TOP_N×(1+b)`, and add a
new name only when it clears the entry rank (Garleanu-Pedersen "trade partially toward
the aim" + Kitces/Daryanani tolerance band, specialized to the long-only equal-weight US
momentum basket). Justification: momentum alpha is intact net of costs, so the
highest-EV remaining lever is TURNOVER REDUCTION (arXiv:2412.11575: cost-aware lifts OOS
Sharpe 1.04→1.30, turnover 18.2→2.65). The other four candidates REJECTED from the
literature: vol-targeting (already in the replay + rejected; Barroso-Detzel net-of-cost
fail), covariance-shrinkage/min-variance (DeMiguel-Garlappi-Uppal RFS 2009: ~6000 months
needed to beat 1/N at 50 assets — contraindicated for ~10-25 names), PBO/DSR (already in
the promotion gate, not a construction lever), TSMOM regime filter (multi-asset futures
result, not single-equity).

**Machinery to reuse (file:line):** SR-diff gate `backend/backtest/analytics.py:239-289`
`sharpe_diff_test(...) -> {delta,p_one_sided,ci_low,ci_high,...}` (52.3 calls it via
`scripts/ablation/dsr_52wh_verdict.py:30`); $0 replay `scripts/ablation/sector_neutral_replay.py`
(main :141-282; basket_fwd_return :101-113; ann_sharpe :116-120; turnover :201; rebalance
loop :183-228; paired dump :269-279); config-gate pattern `backend/tools/screener.py:249-264`
+ gated re-score :445-483; sizing `paper_trader.py:203-211` (equal-weight); cost model
`backtest_engine.py:162-168` (round-trip 2×0.1%). GAP: the replay reports Sharpe/return/
turnover but NOT maxDD → add a `max_drawdown` helper.

## Immutable success criteria — VERBATIM from masterplan phase-53.1 (do NOT edit)

1. the research gate passed (researcher brief cited in the contract: >=5 sources read in
   full + recency scan) and the chosen lever is justified from the literature, not assumed
2. the candidate is measured ON-vs-OFF against the live baseline via the $0 replay/backtest
   on the production universe, reporting Sharpe / return / turnover / maxDD
3. any +improvement is subjected to the SAME Ledoit-Wolf SR-difference robustness gate as
   52.3/52.4 (paired stationary-bootstrap; a-priori rule p<0.05 AND delta>=+0.05 &
   CI_low>0); a 'not robust' REJECT is a VALID, honestly-reported outcome
4. the change is config-gated and does NOT regress the working US momentum core (default
   behavior preserved/byte-identical unless the flag is explicitly enabled); NO live flag
   flip in this step; live_check_53.1.md records the comparison + robustness stats + a
   keep/reject recommendation

## Plan steps

1. **Replay arm** — extend the replay (`scripts/ablation/no_trade_band_replay.py`, reusing
   `sector_neutral_replay.py`'s loader/basket/sharpe/turnover) to add a no-trade-band arm
   (param `b`, e.g. 0.2) alongside the OFF baseline; add a `max_drawdown(monthly)` helper.
   Report per arm: ann Sharpe / ann return / turnover / maxDD, GROSS and NET-of-cost (net
   applies the `backtest_engine` round-trip cost × turnover).
2. **Robustness gate (dual, pre-registered)** — call `analytics.sharpe_diff_test` identically
   on (a) GROSS returns = the **do-no-harm leg** (require `ci_low > -0.05`: the band must
   not significantly HURT gross), and (b) NET-of-cost returns = the **promote leg**
   (a-priori rule: `p_one_sided<0.05 AND delta>=+0.05 AND ci_low>0`). Honest keep/reject.
3. **Config-gate (measure-only; NO live flip)** — add a default-OFF gate for the band in
   the construction path (mirror the 52.2 `momentum_52wh_tilt` flag pattern in
   `screener.py`), so the measured lever is production-ready + reversible. Default OFF ⇒
   byte-identical. Do NOT enable it live.
4. **Tests** — mirror `test_phase_52_3_dsr.py` (SR-diff reuse) + `test_phase_52_2_live_tilt.py`
   (OFF byte-identity) + a band-logic unit test + the maxDD helper.
5. **Verify** — run the $0 replay; `ast.parse`; `pytest` the new tests + the 52.x regression;
   no emoji/ASCII. Write `live_check_53.1.md` (ON-vs-OFF Sharpe/return/turnover/maxDD gross
   +net + the SR-diff stats for both legs + the cited basis + keep/reject recommendation).
6. **Fresh qa → log → flip → commit.**

## Guardrails / DO-NO-HARM

- $0 (free yfinance prices, no LLM, no live cycles, no BQ writes). Default-OFF gate ⇒ the
  +20% US momentum core byte-identical; NO live flag flip. Reuse `sharpe_diff_test` + the
  replay verbatim (don't fork the stats). Honest reporting: a net REJECT is valid + must be
  stated plainly (no p-hacking; the a-priori rule + dual legs are fixed BEFORE the run).
  Same a-priori rule + n_boot as 52.3/52.4. No emoji; ASCII loggers.

## References

`handoff/current/research_brief.md`; `backend/backtest/analytics.py:239-289`;
`scripts/ablation/sector_neutral_replay.py`; `scripts/ablation/dsr_52wh_verdict.py`;
`backend/tools/screener.py`; `backend/backtest/backtest_engine.py`;
`backend/tests/test_phase_52_3_dsr.py` / `test_phase_52_2_live_tilt.py` /
`test_phase_52_4_residual_momentum.py`. External: Garleanu-Pedersen, arXiv:2412.11575,
Kitces tolerance-band, DeMiguel-Garlappi-Uppal (RFS 2009).
