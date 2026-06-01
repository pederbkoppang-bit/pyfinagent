# phase-52.1 EVALUATE -- 2026-06-01

**Agent:** Q/A (merged qa-evaluator + harness-verifier), Layer-3 single independent pass.
**Step:** 52.1 -- 52-week-high momentum tilt (price-only alpha signal), MEASURE-FIRST.
**Verdict:** PASS (pending independent replay re-run confirmation of the dSharpe sign; logic-level A/B fairness already confirmed).

## 1. Harness-compliance audit (5 items -- all PASS)
- **researcher before contract:** PASS. `research_brief.md` header `# research_brief -- phase-52.1`, GATE ENVELOPE `gate_passed: true`, tier complex, `external_sources_read_in_full: 7` (George-Hwang 2004 PDF, Hanauer-Windmuller 2019 PDF, QuantConnect, Quantpedia, Blitz-Huij-Martens 2011 repec, arXiv:2304.03437 Echo-disappears, Aalto thesis), 19 URLs, recency scan 2024-2026 present (52wh-2026 J.Econ.Dyn.&Control, MA-distance Dec-2023, Lin 2020, Echo-disappears). contract.md References section line 43 cites `research_brief.md (52.1 gate)`.
- **contract before GENERATE:** PASS. contract.md success criteria 1-4 are VERBATIM identical to masterplan 52.1 success_criteria (char-for-char compared via masterplan JSON walk). Verification command + live_check field match.
- **experiment_results + live_check present:** PASS. Both files exist; live_check_52.1.md records the ON-vs-OFF table + cited basis + ESCALATE recommendation.
- **log-last:** PASS. `grep -E "phase=52\.1\b" handoff/harness_log.md` returns NO cycle header (the "52.1" greps that matched are "152/1" regression-count substrings + unrelated old phase IDs). masterplan 52.1 status=pending. Log + status-flip correctly deferred to after this verdict.
- **first 52.1 verdict / no shopping:** PASS. 0 `phase=52.1.*result=CONDITIONAL` entries in harness_log. retry_count=0/3. This is the first and only verdict; no prior evidence to shop against. The prior evaluator_critique.md content was phase-51.4 (overwritten per instruction).

## 2. Deterministic checks (reproduced verbatim)
- **pytest** `backend/tests/test_phase_52_1_alpha_signal.py -q`: `5 passed in 0.22s`. (verification.command exit 0.)
- **ast.parse** scripts/ablation/sector_neutral_replay.py -> `replay AST OK`; test file -> `test AST OK`.
- **test -f** handoff/current/live_check_52.1.md -> `live_check present`.
- **git diff --name-only HEAD** (code): `scripts/ablation/sector_neutral_replay.py` (M) + `backend/tests/test_phase_52_1_alpha_signal.py` (new) ONLY. The rest are handoff docs, hook-appended audit JSONL, and masterplan (status still pending). NO live-engine file (screener.py / autonomous_loop / decide_trades / paper_trader / risk_engine / kill_switch / backtest_engine) in the diff -- grep exit=1 (clean).
- **secret scan** on diff: no match.
- **independent replay re-run:** IN PROGRESS (background); see verdict note.

## 3. The 4 immutable criteria -- judged each

**Criterion 1 (price-based signal measured ON-vs-OFF, Sharpe/return/turnover): PASS.**
The 52wh proximity (`pct_to_52w_high = last / 252d-rolling-max`, replay :86-87) is price-only and IS the George-Hwang 2004 formula. Measured ON (hi52_k0.5, hi52_k1.0) vs OFF (baseline) on the S&P-500 replay; the table reports ann_Sharpe + avg_fwd_mo% + avg_turnover for every config. Cited 2025-2026 (J.Econ.Dyn.&Control 2026 recency hit) + canonical (George-Hwang 2004). Computable from daily closes alone (confirmed -- only `c` the causal close window is used).

**Criterion 2 (reuses production rank_candidates, identical screen_data both arms, sole delta = the signal, causal fwd returns, honestly reported): PASS -- the key criterion, verified at the CODE level.**
- **Identical screen_data:** `rows` is built ONCE per rebalance (replay :186-190) and shared by baseline (:194) and the tilt arm (:219). Same dicts, same fields.
- **Production composite reused verbatim:** the tilt arm calls `rank_candidates(rows, top_n=len(rows), strategy="momentum")` (:219) -- the SAME production function as baseline, passing NONE of the overlay signals, so every `if <signal>:` block in screener.py:322-407 is a no-op and the composite is exactly `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25` with the RSI/vol multipliers (screener.py:295-309). `top_n=len(rows)` returns ALL scored rows (screener.py:474-475 `scored.sort; return scored[:top_n]`), and EVERY row carries `composite_score` (screener.py:409 `{**stock, "composite_score": round(score,3)}`). So the tilt reads a real production composite, not 0.0.
- **Sole delta = the tilt:** `hi52_tilt_basket` (:123-138) only post-processes the production-scored rows: `composite_score * (1 + k*(pct_to_52w - mean_pct))`, re-sort, top_n. Replay-side; no live-engine change.
- **Causal forward returns:** the basket is chosen at `t_idx` from `closes[...].iloc[win_lo:t_idx+1]` (inclusive, causal, :188); `basket_fwd_return` measures `s.iloc[t_idx+horizon] / s.iloc[t_idx] - 1` (t+21 vs t, :110). No look-ahead -- the tilt cannot peek at forward data (it reads only `pct_to_52w_high`, a backward feature).
- **Honestly reported:** see Dimension-4 below; the small/noisy edge is disclosed, both k reported.

**Criterion 3 (NO live engine change; US momentum core untouched): PASS.**
Diff = replay script + new test ONLY. No screener.py / autonomous_loop / decide_trades / risk-guard / kill-switch / paper_trader change. No flag flip (`multidim_momentum_enabled` / `momentum_52wh` appear ONLY in markdown prose describing the FUTURE operator-gated step, never in .py). The working US momentum core is literally untouched.

**Criterion 4 (live_check records ON-vs-OFF + cited basis + keep/reject rec): PASS.**
live_check_52.1.md has the verbatim 5-config table, a per-criterion table, cited basis (George-Hwang + Barroso-Wang large-cap mute), and the recommendation: ESCALATE k=0.5 to a live operator gate.

## 4. Adversarial judgment

- **Honesty / no overselling (the key risk for a POSITIVE result): PASS.** Main reports the edge as SMALL and noisy: experiment_results :27 "Honestly small + noisy: a preview run showed +0.057/+0.054; this run +0.051/+0.047 (live-yfinance drift). True edge ~+0.05 ann Sharpe -- modest ... NOT a game-changer; reported without overselling." live_check :38-42 names the Barroso-Wang large-cap-mute, k=1.0 borderline (+0.047 just under bar), and lists survivorship + McLean-Pontiff factor-decay + long-only-no-short caveats (:57-63). The framing is honest, not oversold. A +0.05 Sharpe lift is correctly characterized as near-threshold and modest.
- **Recommendation correctly operator-gated: PASS.** The rec is ESCALATE k=0.5 to a SEPARATE operator-gated live step (a `momentum_52wh` overlay, default-OFF), NOT a silent live-wire. Criterion #3 honored.
- **Over-tuning: PASS.** k in {0.5, 1.0} ONLY -- committed a priori in contract :12/:39 and brief, NOT a sweep. The verdict block (:258-266) reports BOTH k (no cherry-pick). k-monotonicity (k=0.5 > k=1.0) is reported as evidence AGAINST over-tilting -- a sound anti-overfit argument, not a cherry-pick.
- **Fair A/B / no look-ahead: PASS** -- see criterion 2 (identical rows, production composite reused, causal fwd returns, tilt reads only a backward feature).
- **Tests real (non-tautological): PASS.** The 5 tests pin the tilt MECHANISM with constructed inputs + asserted orderings: tie-break toward 52wh (`["NEAR","FAR"]`), centering (all-equal-pct -> composite order preserved), gentle-k-can't-overturn-a-big-gap (`["STRONG"]`), missing-pct no-crash, and the feature math (linspace->1.0, peaked-to-80% -> ~0.80). No `assert x==x`, no mock-and-assert-called, no over-mock. They WOULD fail if the tilt were additive instead of centered, or if the direction were inverted.

## 5. Code-review heuristics (5 dimensions) -- NO findings
- Dim 1 (security): no secret in diff; no LLM->execution path; no subprocess/eval; replay reads yfinance + Wikipedia only. CLEAN.
- Dim 2 (trading-domain): NO live execution path touched -- kill-switch / stop-loss / perf-metrics / max-position all untouched (this is a measure-only replay). `ann_sharpe` is a replay-local Sharpe for the ABLATION harness, not a live perf-metric write (perf-metrics-bypass N/A -- the replay is not the live metrics path and 51.2 established this harness). CLEAN.
- Dim 3 (code quality): the `except Exception: continue` at replay :159 is in the yfinance MultiIndex-flattening download loop (pre-existing from 51.2, not in a risk path) -- acceptable for a research script. CLEAN.
- Dim 4 (anti-rubber-stamp): financial logic (the tilt) HAS behavioral tests (5, pinning the mechanism). NOT a rubber-stamp. CLEAN.
- Dim 5 (LLM-evaluator anti-patterns): first verdict, evidence cited at file:line throughout, no rebuttal context to be swayed by. CLEAN.

## Verdict
**PASS.** The measurement is sound (fair A/B confirmed at the code level: identical screen_data both arms, the production `rank_candidates` momentum composite reused verbatim with overlays as no-ops, causal t+21 forward returns, a centered/turnover-neutral tilt that reads only a backward feature). The result is HONESTLY reported (a small ~+0.05 Sharpe edge, explicitly flagged as near-threshold + noisy + large-cap-muted + survivorship/decay-caveated, not oversold). There is NO live engine change (diff = replay + test only; no flag flip). The recommendation correctly ESCALATES only k=0.5 to a SEPARATE operator-gated live step. k tested a priori {0.5,1.0}, both reported (no sweep, no cherry-pick). 5 non-tautological mechanism tests. All 5 harness-compliance items and all 3 deterministic checks pass.

[Independent replay re-run launched to reproduce the dSharpe sign/magnitude; result appended below. The PASS does not hinge on an exact number -- live yfinance drifts run-to-run, which Main disclosed -- it hinges on the A/B method being fair + causal + the production composite reused, all confirmed by code inspection.]
