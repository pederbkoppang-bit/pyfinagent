# phase-52.4 EVALUATE -- 2026-06-01

**Q/A verdict: PASS** (merged Layer-3 qa: deterministic-first + LLM judgment + code-review heuristics).
A rigorous, faithfully-implemented, fairly-compared, honestly-derived REJECT of residual
momentum IS a PASS of the step. The signal is a faithful Blitz/Gutierrez-Prinsky single-factor
12-1 residual momentum; the series are aligned; there is no look-ahead; the comparison is fair;
the REJECT is decisive (resid_mom is measurably WORSE, not under-powered); NO live change.

## 1. Harness-compliance audit (5-item, FIRST)

| Item | Status | Evidence |
|------|--------|----------|
| researcher BEFORE contract | PASS | `research_brief.md` header `# research_brief -- phase-52.4`; `STATUS: COMPLETE -- gate_passed: true` (line 22); 6 sources read in full incl. Blitz-Huij-Martens 2011 founding (RePEc #6 abstract verbatim) + Hanauer-Windmuller eq-9 via pdfplumber (65pp/121,537 chars, #3) + Chaves single-factor footnote #7. Recency scan present (lines 122-133). contract.md "Research-gate summary (PASSED)" cites researcher `afaa06ced01cfac95` + the 6 sources. |
| contract BEFORE generate; criteria verbatim | PASS | contract.md "Success criteria (IMMUTABLE -- verbatim from masterplan step 52.4)" lines 16-19 match masterplan 52.4 success_criteria 1-4 word-for-word (verified via json walk). |
| experiment_results + live_check present | PASS | both exist; `test -f live_check_52.4.md` -> "live_check present". |
| log-last | PASS | NO `phase=52.4` / `phase-52.4` entry in `handoff/harness_log.md`; masterplan 52.4 status still `pending`. Correct order (log + flip come AFTER this verdict). |
| first verdict; no prior CONDITIONALs | PASS | no 52.4 in harness_log -> 0 prior CONDITIONALs; this is the first 52.4 verdict. retry_count=0, max_retries=3 -> certified_fallback=false. |

## 2. Deterministic checks (reproduced)

- **pytest** `backend/tests/test_phase_52_4_residual_momentum.py -q` -> `5 passed in 1.42s` (exit 0). Matches live_check/experiment_results.
- **`test -f handoff/current/live_check_52.4.md`** -> present.
- **Gate reproduced from PINNED JSON** (independent re-run of `sharpe_diff_test` on `_residmom_paired_returns.json`, seed=42, n_boot=5000, ppy=12, ci=0.90):
  `delta=-0.249  p=0.7724  ci=[-0.883, 0.330]  sr_resid=1.082  sr_base=1.332  n=58`
  JSON-stored: `delta=-0.249 p=0.7724 ci=[-0.883,0.330] verdict=REJECT n_rebalances=59`.
  -> EXACT match to the live_check verbatim block (Sharpe 1.082 vs 1.332, delta -0.249, p 0.7724, CI [-0.883,+0.330], REJECT). Deterministic + reproducible.
- **git diff scope:** new replay + new test + pinned JSON + handoff docs + masterplan (52.4 step, still pending). `git diff --stat backend/tools/screener.py` = EMPTY (no screener change in the working tree). No autonomous_loop/flag change.

(Did NOT re-run the slow live-yfinance replay -- it drifts day-to-day; the pinned JSON + logic
audit is the deterministic path the contract pre-registered for reproducibility.)

## 3. The 4 IMMUTABLE criteria

| # | Criterion (verbatim) | Verdict | Evidence |
|---|----------------------|---------|----------|
| 1 | resid momentum (Blitz-Huij-Martens, price-only, regress on market, rank by trailing residual mom) measured ON-vs-OFF vs baseline momentum on S&P 500 via $0 replay, reporting Sharpe/return/turnover | **PASS** | replay table: baseline Sharpe 1.332 / 3.651%mo / 0.564 turnover; resid_mom 1.082 / 1.970% / 0.679. 58 paired rebalances, 2019-start data, W=504d single-factor OLS. Sharpe + return + turnover all reported. |
| 2 | improvement subjected to SAME Ledoit-Wolf SR-difference gate as 52.3 (paired stationary-bootstrap, a-priori R1 p<0.05 AND R2 delta>=+0.05 & CI_low>0); honest 'not robust' REJECT is valid | **PASS** | `sharpe_diff_test` reused verbatim from analytics.py (52.3): joint-resampled stationary bootstrap, seeded, one-sided H0 SR_a<=SR_b. R1 False (p=0.77), R2 False (delta -0.249, CI straddles 0) -> REJECT, honestly reported. |
| 3 | NO live engine change; US momentum core untouched | **PASS** | diff = new replay + new test + pinned JSON + docs. screener.py NOT modified (diff --stat empty); no autonomous_loop / momentum_52wh flag flip. Offline $0. |
| 4 | live_check records ON-vs-OFF + SR-difference stats + cited basis + promote/reject rec | **PASS** | live_check_52.4.md: verbatim table + LW gate stats + Blitz/Chaves cited basis + "Do NOT promote residual momentum (it's worse)" recommendation. |

## 4. Adversarial judgment -- is the REJECT rigorous, or a buggy/under-powered false-fail?

**4a. Is `resid_mom_signal` a FAITHFUL residual momentum (Blitz/Gutierrez-Prinsky)? -- YES.**
- Single-factor OLS (residual_momentum_replay.py:49-55): `var_m=mean((m-m̄)^2)`; `beta=mean((s-s̄)(m-m̄))/var_m` = cov/var; `alpha=s̄-beta*m̄`; `eps=s-alpha-beta*m`. Textbook OLS, correct.
- 12-1 formation (line 56): `form_eps = eps[-(form+skip):-skip]` with form=252, skip=21 -> residuals from t-273d to t-21d, SKIPPING the recent ~21d. This is the canonical 12-1 skip (Gutierrez-Prinsky/Blitz formation t-12..t-2 months).
- iMOM (line 60): `form_eps.sum() / form_eps.std(ddof=0)` -> Hanauer-Windmuller eq-9 (sum of formation residuals / std of same). Std-normalized. Correct.
- Direction pinned by tests (5/5 pass): +formation idio -> iMOM>0; -formation -> <0; recent-only spike -> NOT positive (skip + OLS-alpha absorption); too-short -> None; deterministic.
- NO spurious-underperformance bug: sign correct (rank desc by iMOM = high residual momentum first), slice correct (formation excludes skip), no future data in formation.

**4b. Are stock + market series ALIGNED? -- YES (the key false-REJECT trap, avoided).**
residual_momentum_replay.py:113 `df = pd.concat([s_w, m_w], axis=1).dropna()` then `df.iloc[:,0].to_numpy()` / `df.iloc[:,1].to_numpy()` (line 116). The stock return and equal-weight market return are joined on the date index and dropna'd TOGETHER, so the OLS regresses matched (s_i, m_i) pairs -- NOT misaligned series. The min-length guard `len(df) < int(W*0.8)` (line 114) ensures enough overlap. This is exactly the bug the prompt flagged; it is not present.

**4c. Is the comparison FAIR? -- YES.**
- Same universe (`closes.columns`), same rebalance dates (`rebal`), same loop for both configs.
- Both scored by the SAME `basket_fwd_return(basket, closes, t)` (line 124, shared `for name, basket` loop): equal-weight, horizon=21, strictly forward (`s.iloc[t+horizon]`, t+21 > t -> no look-ahead, no overlap with the [t-W,t] signal window).
- baseline = production `rank_candidates(rows, strategy="momentum")` -- the genuine composite (mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25, RSI/vol penalties; screener.py:295-308). resid_mom = rank by iMOM. Same pipeline, different signal. Same sample.
- Min-50-name guards applied symmetrically (baseline line 104, resid_mom line 119).
- NOTE (not a flaw): baseline uses a 260d feature window, resid_mom uses W=504d. This is INTRINSIC -- the baseline composite needs only ~126d, residual momentum structurally needs 504d beta + 252d formation. Each signal on its natural causal lookback, both scored identically forward. Pre-registered in the contract. Not rigging.

**4d. Is the REJECT honest, not under-powered? -- YES, decisively.**
- n=58 paired rebalances > 52.3's 47 -> MORE power, not less. Not under-powered.
- delta=-0.249: resid_mom Sharpe 1.082 < baseline 1.332. resid_mom is measurably WORSE, not merely "couldn't prove better."
- p_one_sided=0.7724 (H0 SR_resid<=SR_base nowhere near rejected -- correct, since resid_mom IS worse). CI [-0.883,+0.330] straddles 0. R1 False, R2 False -> REJECT.
- a-priori gate reused verbatim from 52.3 (same `sharpe_diff_test`, same R1/R2 rule, seed=42). Not rigged: I reproduced delta/p/CI independently from the pinned JSON; they match the live_check exactly. The economic story (modern-regime decay + long-only no-short-leg + large-cap low-idiosyncratic + single-factor partly recapturing already-rejected factor momentum + higher turnover) is consistent with the WORSE result and with the researcher's pre-registered adversarial prior.

**4e. Scope (criterion #3) -- diff = replay + test + pinned JSON ONLY? -- YES.**
`git diff --stat` = masterplan (52.4 step, pending), audit logs, contract, cycle_block_summary, experiment_results, research_brief. Untracked: residual_momentum_replay.py, test_phase_52_4_residual_momentum.py, _residmom_paired_returns.json, live_check_52.4.md, researcher memory. `git diff --stat backend/tools/screener.py` = EMPTY. NO screener.py / autonomous_loop / live-flag change.

## 5. Code-review heuristics (5 dimensions; offline ablation diff)
- Security: no secret-in-diff; no prompt-injection / command-injection (yf.download with literal args; no subprocess/eval/exec); $0 (no LLM call in the replay or test). No findings.
- Trading-domain: NO execution-path change -> kill-switch / stop-loss / perf-metrics / max-position invariants UNTOUCHED (this is offline measurement, not the live engine). No findings.
- Code quality: replay is a `scripts/` ablation -> `print()`/`log()` are allowed (negation list). `except Exception: continue` at residual_momentum_replay.py:79 is in the per-ticker download-extraction loop (NOT a risk-guard / execution path) -> NOTE-level at most, acceptable for a download-resilience loop in an ablation script. No type-hint requirement on a script. No findings rise above NOTE.
- Anti-rubber-stamp: financial-logic change (resid_mom_signal) HAS a behavioral test (5 tests pinning direction + 12-1 skip + determinism). No tautological assertions (tests check `>0`, `<0`, `<=0`, `<`, `is None`, equality of two real calls -- all behavioral). Not over-mocked (no mocks; synthetic data). PASS.
- LLM-evaluator anti-patterns: first verdict (no prior to flip); evidence cited file:line throughout; not 3rd-CONDITIONAL. No findings.

## Verdict

**PASS.** A rigorous REJECT. resid_mom_signal is a faithful single-factor Blitz/Gutierrez-Prinsky
12-1 residual momentum (correct OLS cov/var beta, alpha, residuals; 12-1 formation slice skipping
the recent month; std-normalized iMOM); the stock+market returns are aligned via concat+dropna (the
false-REJECT trap is avoided); there is no look-ahead (causal [t-W,t] signal, strictly-forward [t,t+21]
scoring); the comparison is fair (same universe/dates/forward-scoring; production baseline vs iMOM
ranking; symmetric guards); and the REJECT is honest and DECISIVE (n=58 > 52.3's 47 -> not under-powered;
delta=-0.249 resid_mom WORSE; p=0.77; CI [-0.883,+0.330] straddles 0; R1+R2 both False). The a-priori
Ledoit-Wolf gate is the 52.3 `sharpe_diff_test` reused verbatim, seeded/deterministic, and reproduced
independently from the pinned JSON. NO live engine change (diff = new replay + new test + pinned JSON
+ docs; screener.py/autonomous_loop/flag UNTOUCHED). All 4 immutable criteria met. The cited
price-based alpha-signal search is honestly exhausted; the +20% momentum engine stands.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Rigorous REJECT of residual momentum = PASS of the step. resid_mom_signal is a faithful single-factor Blitz/Gutierrez-Prinsky 12-1 residual momentum (correct cov/var OLS beta+alpha+residuals at residual_momentum_replay.py:49-55, 12-1 formation slice eps[-(form+skip):-skip] at :56, std-normalized iMOM at :60; direction+skip pinned by 5/5 passing tests). Stock+market series ALIGNED via pd.concat(...).dropna() at :113 (the false-REJECT trap avoided). NO look-ahead: causal [t-W,t] signal, strictly-forward [t,t+21] basket_fwd_return scoring. Comparison FAIR: same universe/rebalance-dates/forward-scoring, production rank_candidates(strategy='momentum') baseline vs iMOM ranking, symmetric min-50 guards. REJECT is HONEST+DECISIVE: n=58 (>52.3's 47, not under-powered), delta=-0.249 (resid_mom WORSE: Sharpe 1.082 vs 1.332), p_one_sided=0.7724, 90% CI [-0.883,+0.330] straddles 0 -> R1 False AND R2 False -> REJECT. Gate = 52.3 sharpe_diff_test reused verbatim (seeded, joint-resampled stationary bootstrap), reproduced independently from pinned JSON (delta/p/CI EXACT match to live_check). All 4 immutable criteria met. NO live change (diff=new replay+new test+pinned JSON+docs; screener.py diff --stat EMPTY; no autonomous_loop/flag flip). Harness 5-item audit clean (researcher gate_passed=true 6 sources before contract; criteria verbatim; results+live_check present; log-last respected; first verdict 0 prior CONDITIONALs).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "pytest_verification_command", "gate_reproduction_from_pinned_json", "git_diff_scope", "signal_faithfulness_audit", "series_alignment_check", "look_ahead_check", "fair_comparison_check", "power_and_honesty_check", "code_review_heuristics"]
}
```
