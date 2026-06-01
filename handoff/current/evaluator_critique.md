# phase-51.2 EVALUATE -- 2026-06-01

**Step:** 51.2 -- sector diversification (measure-first; NEGATIVE result -> flag stays OFF)
**Evaluator:** Q/A (Layer-3, merged qa-evaluator + harness-verifier), single fresh instance
**Verdict: PASS**
**ok:** true | **certified_fallback:** false

This OVERWRITES the stale phase-51.1 critique (now superseded; the 51.1 PASS is
recorded in the harness log). A prior 51.2 Q/A instance truncated before writing
its verdict; this is the FIRST real 51.2 verdict on UNCHANGED, complete evidence
(NOT a re-spawn to overturn anything). 0 prior 51.2 verdicts, 0 prior 51.2
CONDITIONALs in `handoff/harness_log.md` -> 3rd-CONDITIONAL auto-FAIL rule N/A.

KEY FRAMING (the crux of this step): criterion #2 requires a COMPARISON + a
tradeoff REPORT, NOT that sector-neutral WINS. A rigorous, honestly-reported
NEGATIVE result (HARD sector-neutral hurts long-only Sharpe -0.166 -> flag stays
OFF) SATISFIES criterion #2 -- PROVIDED (a) the wiring (criterion #1) is genuinely
built and not a measurement dodge, and (b) the A/B is methodologically sound. Both
verified below from source. This is a textbook "measure before fixing" outcome.

---

## 1. Harness-compliance audit (5-item, FIRST)

| Item | State | Evidence |
|------|-------|----------|
| researcher BEFORE contract (TWO briefs) | PASS | `research_rotation_element2_verdict.md` (`gate_passed:true`, 8 sources -> REDIRECT away from winner-take-all rotation toward breadth-inside-the-engine) + `research_51_2_sector_div.md` (`gate_passed:true`, 9 sources read in full, floor 5, recency scan performed). Both cited in `contract.md` lines 7-12. The 51.2 brief is decisive: it located the exact no-op (`autonomous_loop.py:369` called `screen_universe` without `sector_lookup`) AND surfaced the Harvey et al. long-only caveat that the replay then confirmed. |
| contract BEFORE generate; 4 criteria VERBATIM | PASS | `contract.md` lines 18-21 are word-for-word identical to masterplan `phases[74].steps[1].verification.success_criteria` (4 criteria; diffed programmatically -- exact match, none amended). |
| experiment_results + live_check present/complete | PASS | `experiment_results.md` lists the file changes + verbatim verification output + artifact shape. `live_check_51.2.md` present (`test -f` exit 0) with the full verbatim replay table + a criterion-by-criterion table + sector distribution (baseline 4.73 vs sector-neutral 10.0 distinct GICS). |
| log-last | PASS | `handoff/harness_log.md` has ZERO 51.2 entries; masterplan 51.2 `status=pending`. Main logs + flips AFTER this PASS, in the correct order. |
| no verdict-shopping | PASS | No prior 51.2 verdict exists (earlier instance never wrote one -- evidence is UNCHANGED and complete, not a different opinion on a CONDITIONAL). First verdict. 0 prior consecutive CONDITIONALs. |

## 2. Deterministic checks (reproduced verbatim)

```
$ python -m pytest backend/tests/test_phase_51_2_sector_div.py -q
....                                                                     [100%]
4 passed in 0.22s

$ python -c "import ast; ast.parse(screener.py); ast.parse(autonomous_loop.py)"
AST OK

$ test -f handoff/current/live_check_51.2.md
live_check present
```

Independent byte-identity check (criterion #3 -- I constructed this myself, did
NOT trust Main's test): ranked a 6-row single-sector screen with vs without the
`sector` field, `sector_neutral=False`:
```
OFF byte-identical: True
```
=> the OFF (default-live) ranking is unaffected by the presence of the sector
field. The working US momentum core is untouched when the flag is OFF.

## 3. The 4 IMMUTABLE success criteria

**Criterion 1 -- candidates carry a sector AT rank time; no-op fixed: PASS.**
The no-op was GENUINE: `rank_candidates` regroups on `(s.get("sector") or "").strip()
or "_UNKNOWN_"` (`screener.py:452`); with no `sector_lookup`, every candidate hit
`_UNKNOWN_`, fell into one `global_pool` (`:458-459`), got a monotone percentile
transform -> identical sort to OFF. The fix:
- NEW `build_sector_map(tickers)` (`screener.py:64-88`) -> `{ticker: GICS sector}`
  from the Wikipedia S&P 500 table; intl/.DE/.KS -> `""` (global-pool fallback).
- `screen_universe` now attaches `row["sector"]` at screen time (`screener.py:233-239`,
  handling both dict and str lookup forms -- `build_sector_map` returns str values,
  caught at `:238-239`) BEFORE `rank_candidates` runs.
- `test_flag_on_spreads_across_sectors` proves the ON basket now spreads across >=2
  sectors and surfaces the best Health name via within-sector percentile;
  `test_flag_on_requires_sectors_to_work` documents the prior bug. The lever is
  genuinely functional, not a stub. NOT a dodge.

**Criterion 2 -- backtest compares ON vs OFF + reports the tradeoff: PASS.**
`scripts/ablation/sector_neutral_replay.py` is a SOUND A/B:
- Calls the PRODUCTION `rank_candidates` (imported `:23`). Both configs share the
  IDENTICAL `rows` (`screen_data`); the ONLY delta is `sector_neutral=sn` (`:170`).
- Forward returns are CAUSAL: `build_screen_row` uses prices up to AND INCLUDING
  the rebalance date (`:67-68, :164` -> `iloc[win_lo:t_idx+1]`); `basket_fwd_return`
  realizes at `t_idx + horizon` (t+21, `:104-106`). No lookahead.
- 48 monthly rebalances 2022-2025, 503 S&P 500 tickers, top_n=10, equal-weight.
- Result (verbatim, reproduced in `live_check_51.2.md`):
  ```
  baseline        ann_Sharpe 1.388  fwd 4.054%  sectors 4.73  turnover 0.555
  sector_neutral  ann_Sharpe 1.223  fwd 2.666%  sectors 10.0  turnover 0.638
  vol_scaled      ann_Sharpe 1.403  fwd 2.045%  sectors 4.73  turnover 0.555
  sector_neutral vs baseline: dSharpe=-0.166, dSectors=+5.27 -> KEEP? False
  ```
  HARD sector-neutral doubles breadth (+5.27 GICS) but COSTS -0.166 Sharpe + ~1.4%/mo
  return + more turnover. The tradeoff is REPORTED, evidence-based, not assumed. The
  Harvey et al. long-only caveat (~22% of long-only trials favor neutralizing) is now
  confirmed on pyfinagent's OWN universe. **A sound NEGATIVE result satisfies #2.**

**Criterion 3 -- config-gated; does NOT regress the US momentum core: PASS.**
The wiring at `autonomous_loop.py:377-388` sets `_sector_lookup = None` and ONLY builds
the map when `sector_neutral_momentum_enabled` OR `multidim_momentum_enabled` is True.
Flag OFF (default) -> `sector_lookup=None` -> `screen_universe` is called EXACTLY as
before AND there is no Wikipedia fetch on the live path (zero added live latency). Both
the project's `test_flag_off_is_byte_identical_with_or_without_sector` AND my independent
6-row check confirm the OFF ranking is byte-identical. No `paper_trader` / `kill_switch`
/ `risk_engine` / `decide_trades` / `perf_metrics` / `backtest_engine` touched (diff =
`screener.py` + `autonomous_loop.py` + new test + new ablation script). NO live flag flip.

**Criterion 4 -- live_check records the ON-vs-OFF comparison + sector distribution: PASS.**
`live_check_51.2.md` records the full verbatim replay table (the ON-vs-OFF Sharpe/return/
turnover comparison), the resulting sector distribution (baseline 4.73 vs sector-neutral
10.0 distinct GICS), a criterion-by-criterion table, and the OFF-stays decision. No live
flag flip (correctly deferred to a future evidence-gated operator-confirmed step).

## 4. Adversarial LLM judgment

- **Is the wiring a dodge to dress up a no-result? NO.** I traced the full chain from
  source: `build_sector_map` -> `screen_universe` attaches `row["sector"]` (`:233-239`)
  -> `rank_candidates` within-sector regroup (`:448-472`). The ON path genuinely groups
  within-sector now; the no-op is fixed (proven by `test_flag_on_spreads_across_sectors`).
  The measurement is real, not a placeholder.
- **Is the A/B confounded? NO.** Identical `screen_data` for both arms; sole delta is
  `sector_neutral=`. Causal forward returns (t+21/t). Real production ranker.
- **Survivorship caveat -- disclosed and non-fatal.** The replay uses today's S&P 500
  membership (survivorship bias). BUT the bias hits BOTH arms equally, and criterion #2
  is about the sector-neutral-vs-baseline DELTA, not the absolute Sharpe. The -0.166
  delta is robust to a bias that affects both arms identically. Contract line ack'd the
  ML engine can't measure this and a new replay was justified. Honest scope.
- **Over-claim check -- NONE.** experiment_results + live_check both lead with the
  NEGATIVE result and say the flag stays OFF. No "sector-neutral wins" spin. The
  vol_scaled +0.015 is correctly called "not compelling," not oversold. Honest reporting.
- **Scope honesty -- CONFIRMED.** Diff is minimal and surgical. No risk-guard, sizing,
  kill-switch, or perf-metrics path altered. No live trading change.

## 5. Code-review heuristics (5 dimensions evaluated)

No BLOCK/WARN fired.
- **Dimension 1 (security):** secret-in-diff clean (URLs + UA strings only, no
  credentials). No command/SQL/path injection sinks. `build_sector_map` fetches a
  fixed Wikipedia URL (`SP500_URL`), not an LLM-generated URL -> no SSRF.
- **Dimension 2 (trading-domain):** N/A by scope -- no execution-path file touched.
  kill-switch / stop-loss / max-position / paper-trader-broad-except all N/A.
  perf-metrics-bypass: the replay computes a RESEARCH-only basket Sharpe in a $0
  ablation script (`scripts/ablation/`, mirroring `run_ablation.py`), NOT in the live
  `services/perf_metrics.py` P&L path -> the single-metric-source rule (which governs
  live Sharpe/drawdown/alpha) is not violated; NOTE-level at most.
- **Dimension 3 (code quality):** `autonomous_loop.py:382` `except Exception as e:`
  LOGS (`logger.warning`) and degrades to the global pool -- it is in the sector-map
  BUILD path, NOT a risk/execution path, so broad-except-silences-risk-guard does NOT
  apply; the catch is intentional graceful degradation. Logger strings are ASCII
  (`->`, `--`). NOTE at most, not a degrade.
- **Dimension 4 (anti-rubber-stamp):** financial-logic-without-behavioral-test does NOT
  fire -- the ranking change ships WITH a new behavioral test file exercising both OFF
  byte-identity and ON spread. No tautological assertions; no over-mocking (tests assert
  real ranked orders on the production function).
- **Dimension 5 (LLM-evaluator anti-patterns):** first verdict, evidence unchanged ->
  no sycophancy / no second-opinion-shopping / no criteria-erosion. Every criterion
  above is backed by a file:line citation or reproduced command output.

## Verdict

**PASS.** This is the correct, rigorous outcome of a measure-first step. The wiring
(criterion #1) is GENUINELY built -- `build_sector_map` + the gated `sector_lookup=`
attach at screen time fixes a real silent no-op (proven by the ON test, traced from
source). The measurement (criterion #2) is a SOUND A/B on the production ranker with
causal forward returns over 48 rebalances; it honestly reports that HARD sector-neutral
HURTS long-only Sharpe (-0.166), confirming the Harvey et al. caveat on our own universe
-- and a sound negative result SATISFIES criterion #2 (the criterion asks for a comparison
+ tradeoff report, not a win). The change is config-gated and byte-identical when OFF
(criterion #3, proven by both the project test and my independent construction); the
working US momentum core is untouched and there is NO live flag flip. live_check records
the comparison + sector distribution (criterion #4). All 5 harness-compliance items pass;
log-last order intact; no code-review heuristic fired. "Measure before fixing" prevented
a Sharpe-reducing regression -- exactly the discipline this step exists to enforce.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met. (1) Wiring is genuinely built, not a dodge: build_sector_map (screener.py:64) + screen_universe attaches row['sector'] at screen time (screener.py:233-239) BEFORE rank_candidates -> the silent no-op (every candidate _UNKNOWN_ -> one global pool -> monotone -> OFF-identical) is fixed; test_flag_on_spreads_across_sectors proves the ON basket spreads >=2 sectors. (2) Sound A/B: scripts/ablation/sector_neutral_replay.py calls the production rank_candidates with IDENTICAL screen_data for both arms (sole delta sector_neutral=), causal fwd returns t+21/t, 48 rebalances x 503 tickers -> HARD sector-neutral HURTS Sharpe (1.388->1.223, -0.166) while doubling breadth (4.73->10.0 GICS); negative-but-rigorous result SATISFIES the comparison+tradeoff criterion. (3) Config-gated: flag OFF (default) -> _sector_lookup=None -> screen_universe byte-identical + no Wikipedia fetch on live path; project test + my independent 6-row check both confirm OFF ranking unchanged; no paper_trader/kill_switch/risk_engine/perf_metrics/decide_trades/backtest_engine touched; no live flag flip. (4) live_check_51.2.md records the ON-vs-OFF table + sector distribution (4.73 vs 10.0 GICS). Deterministic: pytest 4 passed, AST OK, live_check present, INDEPENDENT byte-identity True. Survivorship caveat disclosed and non-fatal (hits both arms equally; the delta is what matters). Two research briefs gate_passed; contract criteria VERBATIM; log-last intact; first 51.2 verdict (0 prior CONDITIONALs); no code-review heuristic fired.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "independent_byte_identity_proof", "contract_verbatim_match", "wiring_source_trace", "replay_ab_soundness_review", "experiment_results", "live_check", "scope_diff", "code_review_heuristics"]
}
```
