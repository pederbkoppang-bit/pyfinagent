# Evaluator Critique — Step 56.1 (FX/value/fee data-correctness fix)

**Verdict: PASS**
**Q/A:** single merged agent (deterministic-first + LLM judgment). **First spawn for 56.1.**
**Date:** 2026-06-10. **Isolation:** in-place.

---

## 0. Harness-compliance audit (5 items — all PASS)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher gate | PASS | `handoff/current/research_brief.md` is the 56.1 brief; envelope `{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":7,"urls_collected":13,"recency_scan_performed":true,"internal_files_inspected":11,"gate_passed":true}`. 6 sources read in full (Fowler Money, Wikipedia/Feathers, MDN NumberFormat, MDN format, Stripe currencies, CFA/GIPS) + recency scan present (3 complementary 2025-2026 findings, none contradicting). |
| 2 | Contract pre-commit | PASS | `contract.md` is for 56.1; its 4 immutable criteria match `.claude/masterplan.json` step 56.1 **verbatim** (programmatic json-extract compare — C1/C2/C3/C4 byte-match; verification command + live_check match). The contract documents the 3-line backend fix and the file:line plan BEFORE the diff. |
| 3 | Results artifact | PASS | `experiment_results.md` for 56.1 with verbatim verification output (`26 passed, 723 deselected`), the 8-file table, regression-proof, live-UI evidence, and an honest-limitations section disclosing the not-yet-restated rows + not-yet-restarted backend process. |
| 4 | Log-last | PASS | `handoff/harness_log.md` has NO `## Cycle … phase=56.1` entry (last cycle headers are 44=phase-55.2, 45=phase-55.3; the 3 incidental "56.1" string matches are line-numbers/axe-text in old cycle bodies). Masterplan 56.1 status still `pending`. |
| 5 | No verdict-shopping | PASS | First Q/A spawn for 56.1; no prior 56.1 critique to overturn. (The overwritten on-disk file was the archived 55.3 verdict, a different step.) |

---

## 1. Deterministic checks

**Immutable verification command** (verbatim):
```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && \
  python -m pytest backend/tests -k 'fx or paper_trader or krw' -q
26 passed, 723 deselected, 1 warning in 2.37s
exit=0
$ test -f handoff/current/live_check_56.1.md  ->  livecheck-ok
```
Isolated KRW class run: `test_phase_50_2_multicurrency.py` → `10 passed` incl.
`test_krw_buy_row_persists_usd_total_value`, `test_krw_sell_row_persists_usd_total_value_and_fee`,
`test_us_buy_row_byte_identical` all PASSED.

**Frontend gate (diff touches `frontend/**` — REQUIRED):**
- `cd frontend && npx eslint .` → **exit 0** (0 errors, 55 warnings; `react-hooks/rules-of-hooks` = 0 errors; all warnings pre-existing, none in the 5 changed files). Warnings do not fail the gate.
- `cd frontend && npx tsc --noEmit` → **exit 0**, zero output lines.

**Syntax / build:** `paper_trader.py` imports clean (pytest collected it); `npm run build` recorded green in live_check §E.

**Backfill NOT executed (live BQ query):**
```
1cc6ed96 005930.KS SELL total_value=1056195.94 fee=1056.2     (still corrupted KRW; migration pins expect_tv=1056195.94/expect_fee=1056.20)
a72a164e 066570.KS BUY  total_value=364175.06  fee=0.24        (still 364175.06, NOT restated to 238.40)
```
Confirms C3: the migration is present but not applied to live BQ.

**Do-no-harm diff scope:** `git diff --name-only HEAD` (code files) = exactly the 8 listed
(`paper_trader.py`, `test_phase_50_2_multicurrency.py`, `format.ts`, `useLiveNav.ts`,
`cockpit-helpers.tsx`, `trades-columns.tsx`, `layout.tsx`, `backfill_56_1_kr_trade_values.py`).
`git diff --stat HEAD` on `portfolio_manager.py / screener.py / backtest/ / kill_switch.py /
perf_metrics.py / risk_engine.py` = **empty** (decision/risk/backtest core untouched).
No emojis in the 5 changed frontend files (Unicode-symbol grep clean).

**Captures:** `handoff/current/captures_56.1/56_1_positions_cockpit_fixed.png` (146KB) and
`56_1_cockpit_KR_holdings_label.png` (152KB) exist.

---

## 2. Code-review heuristics (5 dimensions — no BLOCK, no WARN)

- **D1 Security:** no secret literal; no injection sink; no LLM-output-to-execution; no dep-pin removal. The migration builds SQL strings from **literal numeric values + hard-coded trade_id constants only** (no external/LLM input concatenated), so the SQL-injection heuristic does not fire. Clean.
- **D2 Trading-domain:** kill-switch reachability, stop-loss-always-set, max-position guard, crypto-ban all untouched (not in the diff). `perf-metrics-bypass` does NOT fire — the fix is at the paper_trader **writer** row-build; `perf_metrics.py:407,470` is the *consumer*, unchanged (the fix actually CORRECTS a ~1500x KR turnover-inflation bug there). No `except Exception` added to an execution path. The BQ UPDATE migration is operator-gated dry-run-default and is an UPDATE (not a NOT-NULL add or column drop), so `bq-schema-migration-safety` does not fire. Clean.
- **D3 Code quality:** `positionMarketValueUsd` is fully type-annotated; no print/unicode-in-logger; the +116-line test addition is a test file (negation-list exempt). Clean.
- **D4 Anti-rubber-stamp:** the financial-logic change (FX in trade rows) **has** behavioral tests; assertions are real magnitude+value bounds, NOT tautological; the tests mock the BQ/Router/FX **seams** and exercise the real `execute_buy`/`execute_sell` (not over-mocked); not rename-as-refactor. Clean.
- **D5 Evaluator anti-patterns:** first spawn; this critique cites file:line throughout; not a 3rd-CONDITIONAL. Clean.

`checks_run` includes `code_review_heuristics`.

---

## 3. LLM judgment vs the 4 immutable criteria

### C1 — USD persistence both paths + KRW fail-pre/pass-post + four-FX-point consistency — **PASS**
- **BUY** `paper_trader.py:266`: `round(quantity * exec_price * _local_to_usd, 2)`. `_local_to_usd` is validated non-None at `:209` (function returns `None`→skip-buy before the row-build), so no None-multiplication is reachable. CORRECT.
- **SELL** `paper_trader.py:416-417`: `total_value`/`transaction_cost` × `_l2u` at row-build ONLY. `_l2u` fail-soft to 1.0 at `:371-375`. Upstream `sell_value`/`tx_cost`/`net_proceeds` LEFT LOCAL — the cash credit at `:488` (`net_proceeds * _l2u`) and round-trip `realized_pnl_usd` at `:443` (`(price-entry)*qty*_l2u`) operate on the LOCAL value and are UNTOUCHED → **no double-conversion**. Exactly the contract's row-build-only design.
- **US byte-identity:** `_local_to_usd == _l2u == 1.0` for US → `round(x*1.0,2)==round(x,2)`. Proven by `test_us_buy_row_byte_identical` (`total_value == 1000.0` exactly, `transaction_cost == 1.0`).
- **Regression-proof:** live_check §B records the verbatim PRE-FIX failure (`2 failed, 8 passed`, `AssertionError: total_value=364175.1 looks like LOCAL currency (KRW), not USD`) → POST-FIX `10 passed`. Credible: the KR vs USD magnitude gap (~1500x) makes the `< 1000.0` guard robust to rounding. The SELL test asserts BOTH `total_value` (`<1000` and `≈238.53`) AND `transaction_cost` (`<1.0` and `≈0.24`) AND the cash-credit non-double-conversion (`captured_cash["cash"] ≈ 1000 + (sell_value_local−fee_local)*_KRW_USD`) — the exact triple-guard required.
- **Four-FX-point statement** present in live_check §B: (1) trade recording NOW USD (this fix, tested); (2) mark-to-market unchanged-correct (positions stored USD, 55.1-measured); (3) cash ledger asserted not-double-converted by the new test; (4) fees — BUY already USD, SELL now USD.
- **Consumer audit (anti-rubber-stamp sweep):** every reader of `paper_trades.total_value`/`transaction_cost` expects USD — `perf_metrics.py:407,470` (turnover/NAV-proxy), `slack_bot/formatters.py:213,712,735` (`${total_value:,.2f}`), `trades-columns.tsx:113,123` (`<Dollar>` / `$`-prefix). NO LOCAL-expecting reader exists, so writer-side conversion is unambiguously correct. `signals_server.py`/`orchestrator.py` `total_value` are DIFFERENT fields (portfolio equity / sector denominator). `realized_pnl_usd` round-trip field is untouched → unaffected.

### C2 — NAV root cause fixed (finding ID cited) + sane live UI + trades-columns/VS-KOSPI per 55.1 verdict — **PASS**
- **Root cause F-1** fixed at `useLiveNav.ts:34-43`: the old `lp * pos.quantity` (summed live KRW/EUR ticks as USD) → `positionMarketValueUsd(pos, livePrices[pos.ticker]?.price)`. Finding ID **F-1** cited in the code comment and the live_check map.
- **US do-no-harm preserved EXACTLY:** `positionMarketValueUsd` US branch returns `(livePrice ?? current_price ?? avg_entry_price) * quantity` — byte-identical to the old `useLiveNav` formula for a US-only book. The helper is the extracted `mvUsd` pattern (DRY, prevents NAV-card/donut drift).
- **RiskMonitorCard** `cockpit-helpers.tsx:309-311`: was `qty * current_price / navDenom` (KRW-as-USD → "1527.8%") → `positionMarketValueUsd(p) / navDenom`. For US (no livePrice arg) this is `(current_price ?? avg_entry_price)*qty`, equivalent to prior US behavior. CORRECT.
- **Live capture** (live_check §C): NAV card **23,856.94 USD** (was 345,950.68), Total P&L +19.28% (was +1,629.75%), Max position 3.0% (was 1,527.8%), donut $23,857, currency exposure USD 98.9%/KRW 1.0%. NAV 23,856.94 vs old 345,950.68 is the headline corruption-resolved evidence.
- **trades-columns.tsx:11 comment:** states the post-fix invariant (USD) AND adds the explicit caveat that the 7 KR rows written 2026-06-01..06-09 hold LOCAL until the backfill is approved. Honest resolution (make data match the comment, flag the legacy rows) — superior to rewriting it to "KRW" which would be wrong post-fix.
- **VS-KOSPI (F-12):** per the 55.1 verdict ("keeping/strengthening the already-disclosed tooltip limitation"), the non-US per-market card label is renamed `vs {LABEL}` → `{MKT} holdings` (capture `["KR HOLDINGS"]`), tooltip retained; ALL/US keep the true `vs SPY` excess. Backend fetches no ^KS11 (grep-confirmed), so true index excess is correctly deferred to phase-57. Satisfies the disjunctive criterion.

### C3 — backfill operator-gated migration only; GIPS disclosure + materiality; declined-path flagged — **PASS**
- `scripts/migrations/backfill_56_1_kr_trade_values.py`: **dry-run by default**, `--execute` gated (docstring: execute only after operator approval). **Idempotent:** each UPDATE pins `trade_id = '…' AND ABS(total_value - <old>) < 0.02` (and the fee for SELL rows), so a second run matches 0 rows and cannot double-convert. Explicit per-row USD literals (no live-FX dependence at migration time). GCP import deferred.
- **NOT executed** — confirmed by the live BQ query (rows still at 364175.06 / 1056195.94).
- **GIPS disclosure + materiality** in the script docstring: WHAT/WHEN/WHY + materiality classification (IMMATERIAL to composite returns; MATERIAL to ledger/TCA consumers → tier-3/4 correct-with-disclosure). Decline-path documented: rows stay flagged-not-fixed; caveat lives in trades-columns header comment + live_check §D; do NOT delete the script.
- **Declined-path flagging is live NOW** (operator has not approved yet): trades-columns caveat + live_check §D flag the 7 rows — not silently kept.

### C4 — every change cites a 55.x finding ID; mapping in live_check; US core untouched — **PASS**
- live_check §A maps every changed file to a finding ID: F-2 (paper_trader, tests, migration, trades-columns caveat), F-1 (format.ts helper, useLiveNav, RiskMonitor), F-12 (benchLabel relabel), F-13 (layout subtitle). Each code edit carries an inline `phase-56.1 (55.1 F-x)` comment. No change lacks a finding ID.
- **US momentum core untouched** — empty diff-stat on screener/optimizer/backtest/portfolio_manager/kill_switch/perf_metrics/risk_engine, plus the US byte-identity test (backend) and the US-branch preservation (frontend helper).

---

## 4. Scope-honesty assessment

The experiment_results "Honest limitations" section is candid and accurate:
(a) the 7 historical rows still display local magnitudes until the operator approves the backfill (disclosed in 3 places); (b) the live :8000 backend still runs the pre-fix code **in memory** — the F-2 row fix takes effect for trades written after the next backend restart, correctly deferred to an operator/deploy window (phase-58) rather than an unattended restart of a live trading process. Disclosed, not hidden. The frontend fixes hot-reload via `next dev` and are live.

---

## 5. Verdict

**PASS.** All 4 immutable criteria met with regression-proof tests; deterministic checks green (verification cmd exit 0 / eslint 0 / tsc 0); the backend fix is the minimal row-build-only conversion with no double-conversion and byte-identical US; the frontend NAV root cause is fixed with US behavior preserved exactly; the backfill is a correctly-gated non-executed migration with GIPS disclosure; every change cites a 55.x finding ID with the US momentum core untouched. No code-review heuristic (5 dimensions) fired at BLOCK or WARN.

**checks_run:** syntax, verification_command, frontend_eslint, frontend_tsc, bq_not_restated_query, do_no_harm_diff_stat, consumer_audit_total_value, mutation_resistance_test_read, migration_safety_review, captures_exist, emoji_scan, code_review_heuristics, research_gate, contract_precommit, results_artifact, log_last, no_verdict_shopping
