# Q/A Evaluator Critique -- phase-50.4: Market-calendar gating

**Verdict: PASS** | Fresh Q/A (first for 50.4; no verdict-shopping) | 2026-05-30
**Reviewer:** Q/A subagent (merged qa-evaluator + harness-verifier), effort=max

---

## 5-item harness-compliance audit (run FIRST)

| # | Gate | Result | Evidence |
|---|------|--------|----------|
| 1 | researcher gate | PASS | `handoff/current/research_brief.md` envelope `gate_passed:true`, `external_sources_read_in_full:6`, `recency_scan_performed:true`, `urls_collected:17`, `internal_files_inspected:7`. Genuine 50.4 calendar brief (XETR/XKRX/XNYS, `is_session` API, latent `cal.days` bug). Cited by `contract.md` lines 6-13, 42. |
| 2 | contract-before-generate | PASS | `git log`: `4194088a phase-50.4: PLAN` precedes `6334a491 phase-50.4: GENERATE`. The 3 `success_criteria` in `contract.md` lines 19-21 are verbatim from masterplan 50.4 `verification.success_criteria` (programmatic substring match: all 3 PRESENT). |
| 3 | experiment_results present | PASS | `experiment_results.md` present: files list (markets.py:137, autonomous_loop.py intl block, test file), verbatim verification (`7 passed`), live evidence ref to `live_check_50.4.md`. |
| 4 | log-last | PASS | NO `phase=50.4` cycle header in `handoff/harness_log.md` (grep `phase=50\.4` / `## Cycle.*50\.4` = empty). masterplan 50.4 `status=in_progress`. Correct ordering preserved. |
| 5 | no verdict-shopping | PASS | First Q/A for 50.4 (no prior 50.4 entry in harness_log; on-disk critique before this write was the 50.3 PASS). Not a cycle-2 spawn; no simultaneous-presentation / 3rd-CONDITIONAL concern. |

---

## Deterministic checks (run by Q/A, not trusting the agent)

| Check | Result |
|-------|--------|
| `ast.parse` markets.py | OK |
| `ast.parse` autonomous_loop.py | OK |
| `import backend.services.autonomous_loop` | imports OK |
| `pytest backend/tests/test_phase_50_4_calendar.py -q` | **7 passed** in 1.50s |
| is_trading_day US/EU/KR matrix (assert block) | `calendar correct across US/EU/KR` |
| regression: not-always-true (`is_trading_day('2026-06-13','US') is False`) | `not always-true: confirmed fixed` |
| masterplan cmd (`'2026-01-01','EU' False` + `'2026-06-15','US' True`) | `masterplan assertions OK` |
| `test -f live_check_50.4.md` | `live_check_present` |
| **independent date verification vs exchange_calendars library** | all 8 dates match (XETR 05-01=False, XNYS 05-01=True, XKRX 02-17=False, XNYS 02-17=True, XKRX 09-25=False, XETR 01-01=False, XNYS 06-15=True, XNYS 06-13=False) |

Independent verification confirms the test's holiday assertions against my OWN ground-truth library query (not just the project's tests): Labour Day May 1 = Xetra closed; Seollal Feb 17 + Chuseok Sep 25 = KRX closed; all market-LOCAL dates correct. KR lunar holidays (which a UTC/US-date approach would miss) are correctly closed.

---

## Adversarial LLM judgment

### BYTE-IDENTITY (critical) -- PASS
The entry gate lives ENTIRELY inside `if _intl_markets:` (`autonomous_loop.py:331`). `_intl_markets = [m for m in _paper_markets if m != "US"]` (:330); with the default `paper_markets=["US"]`, `_intl_markets == []` -> the block is NEVER entered -> `universe` is byte-identical (`None` or russell/SP500). **The +20% engine is untouched on the default path.** Within a multi-market universe, `_open_today(sym)` returns `True` immediately for `market_for_symbol(sym) == "US"` (:347-348) -> US tickers are NEVER dropped from the universe. The Step-7 buy callsite (:1029-1060) applies NO calendar check (it threads `market=getattr(order,"market","US")` only, pre-existing 50.3) -- the gate is universe-filter-only (the documented primary gate (A); the optional buy-side guard (B) from the brief was not shipped, which is fine: a closed-market ticker is dropped from `universe` before `screen_universe`, so it never reaches `orders`). The gate cannot affect the US path. Confirmed by inspection at :345-358.

### Exits not gated -- PASS
The Step-7 sell loop (`:1007-1017`) calls `execute_sell(...)` with ZERO calendar check. `grep is_trading_day|is_session|_open_today|trading_day` in `paper_trader.py` + `portfolio_manager.py` = NO hits (execute_sell untouched). The Step-5.6 stop-loss block (`:868-918`) runs `check_stop_losses()` (:899) then fires `execute_sell(reason="stop_loss_trigger")` (:902-907) for each triggered stop -- NO calendar gate anywhere in that path. A breached KR stop fires even when KR is closed. This is the safe asymmetry the research mandated (gating an exit would strand a breached position -> unbounded loss).

### The latent-bug fix -- PASS
`is_trading_day` (`markets.py:137-158`) now calls `cal.is_session(ts.normalize())` (:155) instead of the broken `date in cal.days` (`.days` removed in exchange_calendars 4.0 -> bare-except swallow -> always True). Genuinely fixed: returns False for weekends (06-13) and holidays (independently verified). tz-aware input handled (`:153-154` `ts.tz_localize(None)` before is_session, which rejects tz-aware labels). Fail-open True if `cal is None` (:148-149) or on any error (:156-158) -- never blocks a trade because the lib is missing. `test_not_always_true_regression_guard` pins the fix.

### Date correctness -- PASS
EU/KR assertions use the market-LOCAL date and are correct vs my independent library query AND my own knowledge: Labour Day (May 1) = Xetra closed (German holiday, not a US closure); Seollal (Feb 16-18 2026, test uses Feb 17 Tue) and Chuseok (Sep 24-25 2026, test uses Sep 25 Fri) = KRX closed (lunar -- a weekday rule would miss these). The live-loop helper computes `datetime.now(utc).astimezone(ZoneInfo(market_tz)).date()` (:350-351) -- market-local, addressing the brief's ET->KST skew concern; `get_market_config` carries the IANA tz per market.

### Scope honesty (criterion #1 interpretation) -- ACCEPTABLE DISCLOSED CHOICE (not a violation)
Criterion #1 says "a US market holiday skips US." The shipped LIVE loop does NOT gate US (to preserve byte-identity -- the loop never gated US before, and the research proved adding it would CHANGE behaviour: the loop currently trades through weekday US holidays on stale yfinance data, and a US gate would start skipping them). This is disclosed THREE times: `contract.md:26` (explicit NOTE), `experiment_results.md:22`, `live_check_50.4.md:32`. Critically, the criterion's "skips US" CAPABILITY is satisfied by the function itself: `is_trading_day('<us holiday>','US') == False` is true and tested. A caller CAN gate US; the live loop deliberately doesn't. Given `is_trading_day` is correct for ALL markets (US included) and the non-US-only live gating is a deliberate, well-reasoned, triple-disclosed design choice that protects the working money engine, this satisfies criterion #1 under the disclosed interpretation rather than violating it. The honest disclosure (vs silently claiming "US is gated") is exactly the scope-honesty the protocol rewards. Criterion #1's second clause ("a German holiday skips EU/.DE independently") IS fully met by the live gate AND the function.

---

## Code-review heuristics (5 dimensions evaluated -- no BLOCK, no WARN)

Diff scanned: markets.py (is_trading_day rewrite), autonomous_loop.py (intl-block gate), test_phase_50_4_calendar.py (new).
- **financial-logic-without-behavioral-test**: NOT triggered -- diff touches `markets.py` + `autonomous_loop.py` (NOT perf_metrics/risk_engine/backtest_engine; no Sharpe/drawdown/sizing math changed) AND adds `test_phase_50_4_calendar.py` (7 behavioral tests exercising the new path incl. the regression guard).
- **broad-except-silences-risk-guard**: the `except Exception: return True` in `is_trading_day` (:156) and `_open_today` (:353-355) are NOT risk-guard suppressions -- they are deliberate fail-OPEN behaviour (never block/drop a trade on a calendar error, preserving today's "always trade" default). Correctly safe-by-default; both LOG via `logger.warning` (not silent `pass`).
- **kill-switch / stop-loss / max-position**: untouched. execute_sell, check_stop_losses, backfill_missing_stops, paper_max_positions all unchanged.
- **tautological-assertion / over-mocked-test**: none -- the 7 tests assert real per-exchange `is_session` results + `market_for_symbol` derivation; no `assert x==x`, no module-under-test patched.
- **mutation-resistance**: `test_not_always_true_regression_guard` IS the planted-violation guard -- asserts the function is NOT the old always-True no-op (would fail if the fix regressed). Independently re-run by Q/A: confirmed False for weekend + holiday.
- **secret-in-diff / command-injection / perf-metrics-bypass / crypto / supply-chain-pin**: N/A (no such code; exchange_calendars already imported, no new dep, no pin removal). Backend-only diff (no `frontend/**`) -> ESLint/tsc gate N/A.

---

## Quality criteria scoring (>=6 to pass each)
Infrastructure/correctness step, not a strategy change -- DSR/Sharpe criteria are N/A (no return-generating logic changed; the +20% engine is byte-identical on the default path). Robustness (fail-open on every error, exits ungated, per-market independence cross-validated vs published calendars) and Simplicity (minimal intervention inside an existing guarded block; one function rewrite) both strong. No criterion below 6.

---

## checks_run
syntax, verification_command, pytest, independent_date_verification, regression_mutation_test, evaluator_critique, experiment_results, harness_log_inspection, git_log_ordering, contract_alignment_verbatim, code_review_heuristics, byte_identity_trace, exits_ungated_trace, scope_honesty, research_gate_compliance

## Conclusion
All 3 immutable criteria met (criterion #1 fully met for EU/KR independence + satisfied for US under the disclosed non-US-only-live-gating interpretation, with `is_trading_day` correct for all markets incl. US). Byte-identity for `paper_markets=["US"]` proven by construction (gate inside `if _intl_markets:`, US ungated even in a multi-market universe, no buy-side calendar check). Exits never gated (execute_sell + stop-loss paths grep-clean and trace-clean). Latent always-True bug genuinely fixed with a dedicated regression test, independently re-verified. No US-path regression. **PASS.**
