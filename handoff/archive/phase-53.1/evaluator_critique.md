# Evaluator Critique — phase-53.1 (Algorithm/quant elevation: no-trade rebalance band)

**Q/A agent (merged qa-evaluator + harness-verifier). FRESH single spawn.**
Main produced this; I did NOT self-evaluate. Deterministic-first, adversarial,
anti-rubber-stamp AND anti-false-fail. **Date:** 2026-06-01. **Mode:** in-place
working-tree read. **Verdict: PASS. ok: true.**

> This OVERWRITES the STALE phase-43.0 critique left in this rolling file. The
> verdict below is for **phase-53.1** only.

## CRITICAL FRAMING (why a REJECT outcome is a PASS for this step)

phase-53.1 criterion 3 explicitly states **"a 'not robust' REJECT is a VALID,
honestly-reported outcome."** The step is a MEASURE-FIRST quant-elevation
experiment: it PASSES iff (a) the research gate cleared and the lever is
literature-justified, (b) the lever was measured ON-vs-OFF on the production
universe reporting Sharpe/return/turnover/maxDD, (c) any improvement was put
through the SAME 52.3/52.4 SR-difference gate with an a-priori rule fixed BEFORE
the run, and (d) the change is config-gated default-OFF with NO live flip and a
live_check recording the keep/reject call. The lever LOSING the gate is NOT a
failure. My job is to confirm the REJECT is HONEST (not a dodge), the gate was
reused verbatim, the a-priori rule was pre-registered (no p-hacking), and
DO-NO-HARM holds. I do NOT demand the lever win. All four are satisfied.

---

## 0. 3rd-CONDITIONAL auto-FAIL rule — NOT triggered (verified)

`grep -nE "^##.*phase=53\.1" handoff/harness_log.md` returns EXIT 1 (zero hits).
There is NO `phase=53.1` cycle header in the log at all — this is the FIRST Q/A
for step-id 53.1. Zero prior CONDITIONALs. The auto-FAIL rule (3+ consecutive
CONDITIONALs) does not apply. (The only `53.1` string in the log is a forward
pointer in Cycle 38's phase-43.0 NEXT note.)

---

## 1. Harness-compliance audit (ran FIRST, per `feedback_qa_harness_compliance_first`) — 5/5 PASS

| # | Check | Result |
|---|-------|--------|
| 1 | researcher FIRST + gate passed | **PASS** — `research_brief.md` IS the 53.1 brief (complex tier). Envelope `{"tier":"complex","external_sources_read_in_full":7,"snippet_only_sources":11,"urls_collected":21,"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}`. 7 sources read in full (Ledoit-Wolf digest, Kitces tolerance-band, arXiv:2412.11575, arXiv:2411.07949, NBER Garleanu-Pedersen w15205, AQR TSMOM, DeMiguel-Garlappi-Uppal RFS-2009 `[ADVERSARIAL]`) — exceeds the ≥5 floor. Recency scan present (5 findings, 2024-2026). Lever is justified from the literature with a 5-candidate survey + 4 explicit rejections (vol-targeting already-rejected in 52.x, min-variance contraindicated by the DeMiguel 1/N adversarial finding, PBO/DSR not a construction lever, TSMOM a multi-asset-futures result) — NOT assumed. |
| 2 | `contract.md` BEFORE generate, N* delta + 4 criteria VERBATIM + **a-priori rule + dual legs PRE-REGISTERED** | **PASS** — N* delta present (`contract.md:6-11`). The 4 criteria are copied VERBATIM (`:39-51`) and match the masterplan byte-for-byte (I diffed them — see §1a). **Anti-p-hack confirmed:** the a-priori rule `p<0.05 AND delta>=+0.05 AND CI_low>0` is fixed in the contract at `:46-47` (criterion 3) AND restated at `:60-63` as plan-step 2, AND the dual legs (GROSS do-no-harm `ci_low>-0.05` + NET promote) are pre-registered at `:61-63` BEFORE any run. The guardrails (`:80`) explicitly say "the a-priori rule + dual legs are fixed BEFORE the run." This is the SAME gate 52.3/52.4 used. |
| 3 | `experiment_results.md` + `live_check_53.1.md` present w/ verbatim output | **PASS** — `experiment_results.md` has a files-changed table, a VERBATIM verification block (`:32-44`: 8 passed + the replay table + both SR-diff legs), and a verbatim criteria-mapping table (`:48-53`). `live_check_53.1.md` (52 lines) records the ON-vs-OFF table (Sharpe/return/turnover/maxDD gross+net), both pre-registered SR-diff legs with delta/p/CI, the cited basis, and the REJECT recommendation. |
| 4 | log-last / flip-last | **PASS** — `grep phase=53.1 harness_log.md` = EXIT 1 (no entry yet); masterplan `id:53.1 status=pending retry=0 max=3`. Both intact: the log + flip have NOT preceded this Q/A. |
| 5 | First Q/A spawn | **PASS** — no prior 53.1 critique or log entry exists (the file held the stale 43.0 critique). Not verdict-shopping. |

### 1a. Criteria-verbatim diff (contract vs masterplan)

I dumped the masterplan's 4 `success_criteria` and compared to `contract.md:41-51`.
They are identical (research-gate-passed+justified / measured-ON-vs-OFF-with-4-metrics
/ SAME-SR-diff-gate-with-a-priori-rule-REJECT-is-valid / config-gated-no-regression-no-flip-live_check).
The masterplan `live_check` field is `REQUIRED` and the command is `RESEARCH-FIRST
then define at GENERATE` — both honored. No criteria erosion.

---

## 2. Deterministic re-verification (ran every command myself) — all reproduce EXACTLY

### 2a. Tests, syntax, defaults

| Check | My independent run | Result |
|-------|--------------------|--------|
| unit tests | `pytest backend/tests/test_phase_53_1_rebalance_band.py -q` → **8 passed in 0.01s** | **PASS** |
| syntax | `ast.parse` on `rebalance_band.py` + `no_trade_band_replay.py` + `settings.py` → **AST OK all 3** | **PASS** |
| default OFF | `grep rebalance_band settings.py` → `rebalance_band_enabled: bool = Field(False, ...)` (`:99`) + `rebalance_band_pct: float = Field(0.2, ge=0.0, le=1.0, ...)` (`:100`) | **PASS** |
| OFF byte-identity (code) | `rebalance_band.py:41-43`: `base = list(ranked_tickers[:top_n]); if not enabled or band_pct <= 0 or not prev_holdings: return base` — OFF returns `ranked[:top_n]` exactly | **PASS** |

### 2b. THE DECISIVE GATE RE-RUN (seeded, no network) — reproduces byte-identically

I re-ran the SR-diff gate on the dumped arrays myself
(`handoff/current/_53_1_band_paired_returns.json`, 48 paired obs per arm):

```
GROSS 0.011 0.414 -0.071
NET   0.015 0.376 -0.066
promote? False
```

This matches the live_check (`:22-27`) and experiment_results (`:41-42`) EXACTLY:
GROSS dSharpe=+0.011/p=0.414/CI_low=-0.071; NET dSharpe=+0.015/p=0.376/CI_low=-0.066.
Seeded (seed=42, n_boot=5000) → it MUST reproduce, and it does. The dump's own
`verdict`/`promote`/`do_no_harm_ok` fields are `"REJECT ... honest negative
result"` / `False` / `False` — internally consistent.

### 2c. Turnover claim independently recomputed

The dump stores per-month turnover arrays. I computed the means myself:
- mean turnover baseline = **0.555** (live_check says 0.555 ✓)
- mean turnover band = **0.489** (live_check says 0.489 ✓)
- reduction = **11.9%** (live_check says "~12%" ✓ — honest, not rounded-up)

(2 of 47 months the band churned marginally MORE than full reconstitution — a
benign artifact of the slot-fill step when a previously-dropped name re-enters
top_n; it does not undermine the net 11.9% mean reduction or the REJECT.)

---

## 3. Honesty / anti-dodge / anti-p-hack judgment — the core of this gate

### 3a. The REJECT is HONEST (point estimates reported, gate correctly fails)

The band's directional wins ARE reported, not buried: turnover ↓ 11.9%, gross
Sharpe +0.011, net Sharpe +0.015, maxDD unchanged (live_check `:14-16`,
experiment_results `:57`). AND the gate correctly FAILS on the pre-registered
rule: net delta 0.015 < the 0.05 threshold, p=0.376 (>>0.05), CI90=[-0.066,+0.092]
straddles 0. The recommendation is a plain **REJECT** (live_check `:29`,
experiment_results `:3-5`/`:43`) — NOT a spun PASS, NOT a quiet promote. The
disposition is exactly right: ship as a config-gated default-OFF tested helper,
do NOT promote to live. This is the honest negative outcome criterion 3
explicitly sanctions.

### 3b. NO p-hacking (a-priori rule pre-registered; gate reused verbatim)

- The a-priori rule + dual gross/net legs were fixed in `contract.md` BEFORE the
  run (`:46-47`, `:60-63`, `:80`) — confirmed in §1 check 2. The verdict was NOT
  reverse-engineered to fit the data.
- The gate is the SAME `analytics.sharpe_diff_test` (Ledoit-Wolf 2008 SR-diff via
  Politis-Romano stationary bootstrap) that 52.3/52.4 used, with the SAME
  n_boot=5000 + seed=42. `no_trade_band_replay.py:21` imports it
  (`from backend.backtest.analytics import sharpe_diff_test`) and calls it at
  `:133-134`; `grep -c "def sharpe_diff_test|bootstrap|np.random"` on the replay
  = **0** — zero local reimplementation, zero weakening. The stat was reused
  verbatim, not forked.
- The net-of-cost axis is NOT cherry-picked to manufacture a PASS — BOTH legs are
  reported and BOTH fail, and the dominant decision axis (net) was justified in
  the research brief a-priori (`research_brief.md:197-209`).

### 3c. Tests exercise real behavior (anti-tautology, anti-over-mock)

The 8 tests assert concrete outcomes against the pure function — NOT tautologies
(`assert x==x`/`is not None`) and NOT over-mocking (no `@patch` of the module
under test):
- OFF / band_pct=0 / cold-start → `== RANKED[:TOP_N]` (byte-identity, 3 tests).
- held name at rank 11 inside the 12.0 exit band → RETAINED; the displaced top_n
  name NOT force-added (`test_held_name_inside_exit_band_is_retained`).
- held name at rank 13 beyond the band → DROPPED, slot filled from top_n
  (`test_held_name_beyond_exit_band_is_dropped`).
- band churn ≤ full-reconstitution churn (`test_band_reduces_churn...`).
- never exceeds top_n, no dups; maxDD math pinned (+10%/-20%/+5% → -0.20).

These pin the DO-NO-HARM contract (OFF = byte-identical) and the hysteresis logic
honestly.

---

## 4. DO-NO-HARM — confirmed

- **Default OFF byte-identical:** `rebalance_band_enabled=Field(False)`; the helper
  short-circuits to `ranked[:top_n]` when disabled (code §2a) and 3 tests pin it.
- **NOT wired into live `decide_trades`:** `grep -rn "apply_no_trade_band|rebalance_band_enabled|rebalance_band_pct" backend/services backend/agents backend/markets`
  → **EXIT 1 (zero hits)**. The only references in the repo are
  `settings.py`, `test_phase_53_1_rebalance_band.py`, `rebalance_band.py` (the
  helper), and `no_trade_band_replay.py` (the $0 replay). No money-path wiring.
- **No money-path regression:** `git diff --stat` shows only `settings.py` (+8)
  and handoff docs/audit-logs changed among TRACKED files; `rebalance_band.py`,
  `no_trade_band_replay.py`, the test, the live_check, and the paired-JSON are
  untracked-NEW. ZERO edits to `paper_trader.py` / `kill_switch.py` /
  `risk_engine.py` / `perf_metrics.py` / `backtest_engine.py` / `.env`. The +20%
  US momentum core is byte-identical.
- **$0 / measure-first:** no LLM call, no BQ write, no live cycle, no live flag
  flip. settings.py change is a dormant default-OFF gate.

---

## 5. Code-review heuristic sweep (SKILL: code-review-trading-domain) — worst severity NOTE

Diff does NOT touch `frontend/**` (no ESLint/tsc leg required). New Python is a
pure-function helper + a $0 offline replay script + a unit test.

- **Dimension 1 (security):** no secret-in-diff (grepped — no credential literals);
  no `subprocess`/`eval`/`exec`/`os.system`; no LLM-output→sink; no dep-pin removal
  (settings/helper add no deps). N/A or clean.
- **Dimension 2 (trading-domain):** no kill-switch path change; no stop-loss path;
  **perf-metrics-bypass [BLOCK] NOT triggered** — Sharpe is computed via the reused
  `analytics.sharpe_diff_test` / `compute_sharpe` machinery (the replay is an
  offline ablation, the canonical analytics module, not an inline re-impl); no
  `paper_trader`/`execute_buy`/`execute_sell` edit; no crypto re-enable. Clean.
- **Dimension 3 (code quality):** the helper has full type hints + docstrings;
  `print()` lives only in the replay SCRIPT (negation-listed — scripts are exempt);
  no broad-except in the helper. Clean.
- **Dimension 4 (anti-rubber-stamp):** `financial-logic-without-behavioral-test
  [BLOCK]` NOT triggered — the band logic + maxDD ship WITH 8 behavioral tests
  exercising real retain/drop/churn/byte-identity outcomes. No tautological
  asserts, no over-mock, no rename-as-refactor. The cost constant (round-trip
  0.002) is cited to `backtest_engine.py:668` convention (no formula-drift). Clean.
- **Dimension 5 (LLM-evaluator anti-patterns):** this is the FIRST 53.1 Q/A on
  fresh evidence — not sycophancy-under-rebuttal, not second-opinion-shopping
  (no prior verdict to flip; mtime of `experiment_results.md` is this cycle's).
  This critique cites file:line + verbatim command output throughout (no
  missing-chain-of-thought). Worst severity: **NOTE**. No BLOCK, no WARN.

---

## Verdict

**PASS. ok: true.** All four immutable criteria are met, and the REJECT is the
HONEST, criterion-3-sanctioned outcome — not a dodge.

- **Criterion 1 (research-gate + lever justified):** PASS — `gate_passed:true`,
  7 sources read in full, recency scan (5 findings), 5-lever survey with 4
  literature-grounded rejections; the band is justified from Garleanu-Pedersen +
  arXiv:2412.11575 + Kitces, not assumed.
- **Criterion 2 (measured ON-vs-OFF, 4 metrics):** PASS — 48 S&P-500 monthly
  rebalances 2022-2025; Sharpe/return/turnover/maxDD reported gross AND net; I
  independently recomputed the turnover means (0.555→0.489, 11.9%) and they match.
- **Criterion 3 (SAME SR-diff gate, a-priori rule, REJECT valid):** PASS —
  `analytics.sharpe_diff_test` reused VERBATIM (imported, 0 reimplementation),
  dual legs + a-priori rule pre-registered in the contract BEFORE the run, gate
  correctly REJECTs (net delta 0.015<0.05, p=0.376, CI straddles 0). I re-ran the
  seeded gate on the dumped arrays and it reproduces byte-identically
  (GROSS +0.011/0.414/-0.071, NET +0.015/0.376/-0.066, promote?False). Honest
  negative = a valid pass per the criterion's own text.
- **Criterion 4 (config-gated, no regression, no live flip, live_check):** PASS —
  `rebalance_band_enabled=False` default; OFF byte-identity pinned by 3 tests +
  the code short-circuit; ZERO live wiring (grep EXIT 1 in services/agents/markets);
  no money-path file touched; `live_check_53.1.md` records the full comparison +
  both SR-diff legs + cited basis + REJECT recommendation; NO live flag flip.

Harness 5/5 (researcher-first gate_passed:true; contract precedes generate with
N* delta + 4 criteria VERBATIM + a-priori rule & dual legs PRE-REGISTERED;
experiment_results + live_check present with verbatim output; harness_log has NO
53.1 entry + masterplan 53.1 pending retry=0 — log-last/flip-last intact; first
Q/A spawn). Anti-p-hack confirmed (rule fixed before run; gate reused verbatim).
DO-NO-HARM confirmed (default OFF byte-identical, no live wiring, no money-path
regression, $0). 3rd-CONDITIONAL auto-FAIL N/A (zero prior 53.1 verdicts). Code
review worst severity NOTE.

**Next:** append `harness_log.md` Cycle N `phase=53.1 result=PASS`, THEN flip
masterplan 53.1 to `done`, THEN auto-commit. The lever is REJECTED for live
promotion (correctly) and ships as a dormant default-OFF tested helper.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "phase-53.1 is a MEASURE-FIRST quant-elevation experiment whose criterion 3 explicitly states a 'not robust' REJECT is a VALID outcome; all 4 immutable criteria are met and the REJECT is HONEST, not a dodge. Harness 5/5: (1) researcher FIRST, gate_passed:true (7 sources read in full vs >=5 floor, recency scan with 5 findings, 21 URLs, 9 internal files; 5-lever survey with 4 literature-grounded rejections -- vol-targeting already-rejected in 52.x, min-variance contraindicated by DeMiguel-Garlappi-Uppal 1/N adversarial finding, PBO/DSR not a construction lever, TSMOM a multi-asset-futures result); (2) contract precedes generate with N* delta + 4 criteria copied VERBATIM (diffed vs masterplan, identical) + the a-priori rule p<0.05 AND delta>=0.05 AND CI_low>0 AND the dual gross/net legs PRE-REGISTERED at contract.md:46-47,60-63,80 BEFORE the run (anti-p-hack); (3) experiment_results + live_check_53.1.md present with verbatim output (8 passed + the ON-vs-OFF table + both SR-diff legs + REJECT rec); (4) harness_log has NO phase=53.1 entry (grep EXIT 1) + masterplan 53.1 status=pending retry=0 max=3 (log-last/flip-last intact); (5) first Q/A spawn. DETERMINISTIC RE-VERIFICATION (ran every command myself, all reproduce EXACTLY): pytest test_phase_53_1_rebalance_band.py = 8 passed in 0.01s; ast.parse OK on all 3 files; THE DECISIVE seeded gate re-run on the dumped paired arrays (handoff/current/_53_1_band_paired_returns.json, 48 obs/arm) reproduces byte-identically -> GROSS dSharpe=+0.011 p=0.414 CI_low=-0.071, NET dSharpe=+0.015 p=0.376 CI_low=-0.066, promote?=False (matches live_check + experiment_results exactly); I independently recomputed the turnover means from the per-month arrays = baseline 0.555, band 0.489, reduction 11.9% (matches live_check's 0.555->0.489 ~12% claim -- honest). HONESTY/ANTI-DODGE: the band's directional wins (turnover -11.9%, gross Sharpe +0.011, net +0.015, maxDD unchanged) ARE reported AND the gate correctly FAILS (net delta 0.015 < 0.05 threshold, p=0.376>>0.05, CI90=[-0.066,+0.092] straddles 0); recommendation is a plain REJECT (live_check:29, experiment_results:3-5/43), NOT a spun PASS, NOT a quiet promote -- the honest negative criterion 3 sanctions. ANTI-P-HACK: a-priori rule + dual legs fixed in contract BEFORE run; sharpe_diff_test reused VERBATIM (no_trade_band_replay.py:21 imports it from backend.backtest.analytics, called :133-134; grep -c 'def sharpe_diff_test|bootstrap|np.random' on the replay = 0 -- zero reimplementation/weakening; SAME n_boot=5000 + seed=42 as 52.3/52.4). DO-NO-HARM: rebalance_band_enabled=Field(False) default OFF; helper short-circuits to ranked[:top_n] when disabled (rebalance_band.py:41-43) pinned by 3 tests; grep 'apply_no_trade_band|rebalance_band_enabled|rebalance_band_pct' in backend/services backend/agents backend/markets = EXIT 1 (ZERO live wiring -- refs only in settings/tests/replay/helper); git diff --stat = only settings.py (+8) + handoff docs among tracked, new helper/replay/test/live_check/JSON untracked, ZERO edits to paper_trader/kill_switch/risk_engine/perf_metrics/backtest_engine/.env -- the +20% US momentum core byte-identical; $0 (no LLM/BQ/live cycle/flag flip). Tests exercise real behavior (concrete retain/drop/churn/byte-identity/maxDD asserts; no tautology, no over-mock). 3rd-CONDITIONAL auto-FAIL N/A (zero prior 53.1 verdicts; first spawn). Code-review heuristics: no frontend diff (no ESLint/tsc leg); pure-function helper + offline $0 replay + unit test; perf-metrics-bypass NOT triggered (Sharpe via reused analytics module); financial-logic-without-behavioral-test NOT triggered (8 behavioral tests); not sycophancy/verdict-shopping (first spawn, fresh evidence, cites file:line + verbatim output throughout); worst severity NOTE. The lever LOSING the gate is the correct, honest outcome -- PASS.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit_5of5", "research_brief_53_1_gate_envelope_7_sources", "contract_criteria_verbatim_diff_vs_masterplan", "apriori_rule_dual_legs_preregistered_in_contract", "experiment_results_completeness", "live_check_53_1_present_verbatim", "log_last_no_53_1_entry", "masterplan_status_pending_retry0", "first_qa_spawn", "third_conditional_rule_check", "pytest_8_passed", "ast_parse_3_files", "settings_default_off_grep", "off_byte_identity_code_inspection", "DECISIVE_seeded_srdiff_rerun_on_dumped_arrays", "turnover_means_recomputed_0555_to_0489", "verdict_promote_fields_in_dump_consistent", "sharpe_diff_test_reused_verbatim_zero_reimpl", "no_live_wiring_grep_services_agents_markets", "git_diff_stat_no_money_path_regression", "tests_exercise_real_behavior_anti_tautology", "code_review_heuristics"]
}
```
