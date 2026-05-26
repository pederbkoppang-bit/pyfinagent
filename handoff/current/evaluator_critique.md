# Evaluator Critique -- Cycle 1: position-swap framework (zero-buy triage fix)

**Date:** 2026-05-26
**Cycle:** 1 (production-readiness mode, testing-phase mandate, north-star: maximize profit)
**Trigger:** trading-policy change introducing sell-to-buy-better path. Triage cycle -- no masterplan flip.
**Q/A spawn:** ONE (fresh, single-pass; no second-opinion shopping; prior cycle-77 verdict overwritten in this file because the cycle-1 trigger is new evidence, not a re-judgment on unchanged work).

---

## 1. Harness-compliance audit (5 items)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Researcher spawn | PASS | `handoff/current/research_brief_phase_zero_buy_triage.md` exists (21362 bytes, mtime 22:14). Contract cites `Researcher adc62c28569bf64cc, tier=deep, 7 sources read in full, 8 snippet-only, 15 URLs, recency scan performed, internal_files_inspected=9, gate_passed=true`. JSON envelope at brief L325-336 confirms `external_sources_read_in_full=7`, `recency_scan_performed=true`, `ai_in_trading_sources_cited=4`, `academic_method_sources_cited=3`, `gate_passed=true`. |
| 2 | Contract pre-commit | PASS | contract.md mtime `22:19` PRECEDES every modified source: settings.py `22:20`, test_portfolio_swap.py `22:23`, portfolio_manager.py `22:26`, test_dod4 `22:27`. The "Note on file collision" preamble at contract.md L7 correctly documents the autonomous-loop parameter-optimization overwrite at 19:56:50 UTC and supersedes it. |
| 3 | experiment_results.md content | PASS | exists (6617 bytes, mtime 22:28). Lists 1 NEW test (`test_portfolio_swap.py`) + 3 MODIFIED files (`settings.py`, `portfolio_manager.py`, `test_dod4_tier1_coverage_investment.py`). Verbatim pytest output included (4 passed in 0.06s; 37 passed in 2.37s). AST parse output included. Grep gates 3-10 with verbatim counts included. |
| 4 | harness_log absence | PASS | `grep -c "Cycle 1 -- 2026-05-26 -- position-swap" handoff/harness_log.md` = 0. Log-LAST discipline preserved. |
| 5 | No verdict-shopping | PASS | `grep -c "Cycle 1 -- 2026-05-26 -- position-swap" handoff/current/evaluator_critique.md` = 0 before this overwrite. Prior file content was a cycle-77 critique (different step-id); this Q/A is the first verdict on the cycle-1 trigger, not a re-judgment. No simultaneous-presentation rule violation. |

**Harness audit: 5/5 PASS.**

---

## 2. Deterministic checks (each command + verbatim tail)

### 2.1 Swap framework tests

```
$ source .venv/bin/activate && pytest backend/tests/test_portfolio_swap.py -v
backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap PASSED [ 25%]
backend/tests/test_portfolio_swap.py::test_swap_disabled_reproduces_zero_buy PASSED [ 50%]
backend/tests/test_portfolio_swap.py::test_swap_skips_below_threshold PASSED [ 75%]
backend/tests/test_portfolio_swap.py::test_swap_respects_max_per_cycle PASSED [100%]
============================== 4 passed in 0.05s ===============================
```
PASS (4/4 as required).

### 2.2 Regression suite

```
$ pytest backend/tests/ -k "portfolio_manager or paper_trader or test_portfolio_swap"
================ 37 passed, 582 deselected, 1 warning in 2.59s =================
```
PASS (37/37 as required).

### 2.3 AST parse

```
$ python -c "import ast; ast.parse(open('backend/services/portfolio_manager.py').read())"  -> OK portfolio_manager
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read())"             -> OK settings
```
PASS (both exit 0).

### 2.4 Grep gates

```
$ grep -c "paper_swap_enabled" backend/config/settings.py                    -> 1   (expected 1)   PASS
$ grep -c "paper_swap_min_delta_pct" backend/config/settings.py              -> 1   (expected 1)   PASS
$ grep -c "paper_swap_max_per_cycle" backend/config/settings.py              -> 1   (expected 1)   PASS
$ grep -c "_compute_swap_candidates" backend/services/portfolio_manager.py   -> 2   (def+call)     PASS
$ grep -c "swap_for_higher_conviction" backend/services/portfolio_manager.py -> 1                  PASS
$ grep -c "swap_buy" backend/services/portfolio_manager.py                   -> 1                  PASS
```

### 2.5 Frontend untouched

```
$ git diff --stat HEAD -- frontend/         -> (empty)   PASS
$ git diff HEAD -- frontend/package.json    -> (empty)   PASS
```

### 2.6 Live-check artifact

```
$ ls handoff/current/live_check_cycle_1_position_swap.md  -> 4130 bytes (exists)  PASS
```

**Deterministic checks: 6/6 PASS.**

---

## 3. LLM judgment (A-M)

### A. Citation gate (LOAD-BEARING per goal mandate)

**PASS.** Contract.md L13-18 cites by NAME with identifiers:
- AI-in-trading (4, threshold >=2): FinRL `arXiv:2011.09607`; TradingAgents `arXiv:2412.20138`; FinMem `arXiv:2311.13743`; KDD 2026 LLM Long-Run ADVERSARIAL `arXiv:2505.07078v5`.
- Academic-method (3, threshold >=2): Grinold-Kahn Fundamental Law of Active Management (CFI canonical); Kelly-Optimal Rebalancing `arXiv:1807.05265`; Resonanz Capital "upgrade-vs-exit" framework.

Research brief JSON envelope (L325-336) confirms `ai_in_trading_sources_cited: 4`, `academic_method_sources_cited: 3`. Both exceed the >=2 floor by 2x and 1.5x respectively. The adversarial KDD 2026 source is correctly load-bearing for the conservative 25% / 2-per-cycle defaults.

### B. Risk-gate preservation

**PASS.** Swap path at `portfolio_manager.py:503-521` re-checks the sector NAV-pct cap with the documented edge-case at L511-515 ("Block only if projected exceeds cap AND projected exceeds the pre-swap exposure"). Reductive swaps (existing_pct > cap, projected_pct <= existing_pct) allowed -- strict improvement. Position cap (`paper_max_positions=10`) preserved by virtue of each swap being +1 BUY / -1 SELL = net 0. Min-cash-reserve approximately preserved (each swap pair is cash-neutral within rounding).

### C. Sell-first-then-buy invariant

**PASS.** `portfolio_manager.py:382`: `orders.sort(key=lambda o: 0 if o.action == "SELL" else 1)`. Python's `list.sort` is stable, so within-group ordering (signal-SELL, stop-loss-SELL, swap-SELL) is preserved. Test `test_swap_framework_fills_zero_buy_gap` asserts `last_sell_idx < first_buy_idx` at L158-164 -- assertion passes.

### D. No regression in existing tests

**PASS.** `test_dod4_tier1_coverage_investment.py::_settings()` fixture at L795-798 sets `s.paper_swap_enabled = False` with a documented rationale: "Disable swap here so each cap test still characterizes its specific gate; the swap behavior is exercised by backend/tests/test_portfolio_swap.py." 37-test regression suite passes including the previously-flagged `test_portfolio_manager_decide_trades_sector_count_cap_blocks_third`.

### E. Conservative defaults per adversarial citation

**PASS.** `settings.py:208-219` sets `paper_swap_min_delta_pct=25.0` and `paper_swap_max_per_cycle=2`. Descriptions explicitly cite "Resonanz Capital upgrade-vs-exit + KDD 2026 adversarial overtrade evidence" (L212) and "Per KDD 2026 adversarial: LLMs overtrade; keep tight until backtest evidence supports loosening" (L218). Both values match contract spec.

### F. ASCII-only log messages

**PASS.** `grep -P "logger\.[a-z]+\([^)]*[^\x00-\x7F]" backend/services/portfolio_manager.py` returns empty (exit=1). Em-dashes appear only in the module docstring at line 2 and source-comment lines in settings.py -- NOT in any logger call. The 6 new logger lines in `_compute_swap_candidates` (`portfolio_manager.py:248,265,453,485,516,555`) and the existing 266 are all ASCII-only.

### G. Sector-blocked candidates captured

**PASS.** `portfolio_manager.py:239` initializes `sector_blocked: list[dict] = []`. The sector-COUNT-cap branch at L264-271 reads:
```python
if current_in_sector >= max_per_sector:
    logger.info("Skipping BUY %s: sector %s at cap (%d/%d) -- queued for swap check", ...)
    sector_blocked.append(cand)
    continue
```
Candidates are no longer silently dropped; they are append-then-continue with a diagnostic log line ending "queued for swap check".

### H. Epsilon-guarded denominator

**PASS.** `portfolio_manager.py:481`: `denom = max(abs(holding_score), 0.01)`. Comment at L476-480 explicitly justifies why `0.01` (not `1.0`): "do NOT clamp the denominator to 1.0 -- final_score lives in [0,1] so a 1.0 clamp would over-normalize every swap into ~score-delta-as-percentage rather than relative improvement."

### I. `_compute_swap_candidates` signature

**PASS.** `portfolio_manager.py:389-398` declares the function with exactly the contract-required parameters in correct order: `sector_blocked, current_positions, holding_lookup, sector_counts, sector_market_values, selling_tickers, settings, nav`. Return type `list[TradeOrder]`. Call site at L362-371 passes all arguments by name (kwargs) so name-drift would surface at call time.

### J. ZERO frontend changes

**PASS.** `git diff --stat HEAD -- frontend/` empty.

### K. ZERO new npm deps

**PASS.** `git diff HEAD -- frontend/package.json` empty.

### L. ZERO emojis introduced

**PASS.** No glyphs from the emoji block (`U+1F000`-`U+1FFFF`) in any modified file. The only non-ASCII characters detected are em-dashes in docstrings/comments (legitimate project convention; not in logger output, not in user-facing UI).

### M. North-star framing

**PASS.** contract.md L9-11 explicitly frames the policy as "above the testing-phase trade-count mandate" and aligned via "EXPECTED PROFIT UPLIFT, not raw signal-score delta". L36-42 translates the 25% threshold to economics: "~$2.50 expected gain per $1000 of position (NAV-relative; far exceeds the $1 round-trip cost). North-star positive in expectation." The framing is exactly the contract's required north-star integration.

**LLM judgment: 13/13 PASS.**

---

## 4. Code-review heuristics (trading-domain skill, 5 dimensions)

| Dimension | Heuristics checked | Findings |
|-----------|-------------------|----------|
| 1 Security | secret-in-diff, prompt-injection, command-injection, supply-chain, system-prompt-leakage, rag-poisoning, unbounded-llm-loop | None. The diff is pure trading-policy logic; no LLM call paths added, no new imports, no credential literals, no new endpoints. |
| 2 Trading-domain correctness | kill-switch-reachability, stop-loss-always-set, perf-metrics-bypass, position-sizing-div-zero, max-position-check-bypass, paper-trader-broad-except, crypto-asset-class | None. Swap path runs inside `decide_trades` which executes UPSTREAM of `paper_trader.execute_buy/execute_sell` -- kill-switch + stop-loss guards remain in the execution layer untouched. The new BUY orders carry `stop_loss_price=cand.get("stop_loss_price")` at L539 (forwarded from candidate analysis), preserving the existing stop-loss invariant. Position cap and crypto ban untouched. The `delta_pct` denominator IS epsilon-guarded (0.01) so `position-sizing-div-zero` is NOT triggered. |
| 3 Code quality | broad-except, no-type-hints, print-statement, unicode-in-logger, magic-number | None. `_compute_swap_candidates` has full type hints on all 8 parameters + return. No bare `except`. No `print()`. No Unicode in logger calls. The 25.0 threshold IS a named constant via `paper_swap_min_delta_pct` (not a magic number); 0.01 epsilon is justified by inline comment at L476-480. |
| 4 Anti-rubber-stamp | financial-logic-without-behavioral-test, tautological-assertion, over-mocked-test, rename-as-refactor, pass-on-all-criteria-no-evidence | None. The diff adds ~170 lines of NEW trading logic and is paired with 4 NEW behavioral tests at `test_portfolio_swap.py`. Tests are real (positive case, disabled-case baseline, below-threshold case, per-cycle-cap case) -- no tautological asserts, no over-mocked construction. Each test asserts specific tickers / counts / ordering. |
| 5 LLM-evaluator anti-patterns | sycophancy-under-rebuttal, second-opinion-shopping, missing-chain-of-thought, position-bias, criteria-erosion | None. Prior file contained a cycle-77 critique (DIFFERENT step-id); current cycle-1 is a NEW trigger, NOT a re-judgment on unchanged evidence. Every PASS criterion in this critique cites a specific file:line or command output (chain-of-thought present). No criteria erosion vs. contract immutable list. |

**Code-review heuristics: 0 BLOCK, 0 WARN, 0 NOTE.**

---

## 5. Verdict

**PASS.**

All harness-compliance items (5/5) PASS. All deterministic checks (6/6) PASS. All LLM-judgment items A-M (13/13) PASS. All code-review heuristics across the 5-dimensional framework produce 0 findings at any severity. The citation gate (item A, load-bearing per the goal mandate) is comfortably cleared with 4 AI-in-trading + 3 academic-method sources cited by name with arXiv/URL identifiers, exceeding the >=2 floor on both axes. The adversarial KDD 2026 source is correctly used to justify the conservative `paper_swap_min_delta_pct=25.0` and `paper_swap_max_per_cycle=2` defaults. The swap path preserves every risk gate (sector COUNT by net-zero construction, sector NAV-pct via re-check with documented reductive-swap edge case, position cap by net-zero, min-cash-reserve by approximate cash-neutrality, factor-correlation by staying in-sector). Sell-first-then-buy invariant is enforced by stable sort at L382. The existing tier-1 coverage tests pass because the `_settings()` fixture explicitly sets `paper_swap_enabled=False` with a documented rationale. No frontend changes, no new npm deps, no emojis, no Unicode in logger calls.

## 6. Violated criteria

None.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5/5 harness audit + 6/6 deterministic + 13/13 LLM judgment + 0 code-review findings. Citation gate cleared (4 AI-in-trading + 3 academic). Swap path preserves all risk gates. Sell-first-then-buy invariant enforced by stable sort at portfolio_manager.py:382.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "harness_compliance_audit", "code_review_heuristics", "live_check_artifact", "regression_suite", "frontend_diff_check"]
}
```
