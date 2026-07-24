# Research Brief — Step 64.3 (Backend gap tests)

Tier: moderate. Research gate for masterplan step 64.3. gate_passed: TRUE.

Objective: pure backend gap tests for (1) kill-switch state machine, (2)
currency/money add-on averaging + fx fallback, (3) per-market screener units,
(4) learnings reader error-vs-empty. All tests MUST be pure (no live
network/BQ) so the `requires_live` quarantine does NOT grow (criterion 1).

Immutable verification command (must be GREEN):
`python -m pytest backend/tests -k '64_3 or kill_switch_machine or currency_path or screener_market or learnings_reader' -q`

---

## Internal code inventory — the 4 pure testable seams

### 1. Kill-switch state machine — backend/services/kill_switch.py
| Seam | file:line | Purity |
|------|-----------|--------|
| `evaluate_breach(nav, dll_pct, tdd_pct)` | kill_switch.py:281 | PURE (reads `_state` snapshot; no net) |
| `check_auto_resume(nav, dll, tdd, enabled=False)` | kill_switch.py:340 | PURE (net only inside `raise_cron_alert_sync`, fail-open try/except; avoid by keeping breach clear + enabled path off) |
| `KillSwitchState.pause/resume/is_paused` | :157/:189/:127 | PURE with `trigger="test"` (in `_MANUAL_TRIGGERS` :171 -> skips Slack alert) |

- **Rail 5 (criterion 2)** = `docs/runbooks/away-ops-rules.md:17-18` VERBATIM:
  "Kill-switch stays paused after any breach; only `KILL SWITCH: RESUME`
  token resumes; auto-resume hysteresis stays OFF." The STAYS-PAUSED policy
  is enforced by `check_auto_resume(..., enabled=False)` :362 -> returns
  `{action:"no_op", reason:"auto_resume_disabled"}` and NEVER mutates state.
  Even `enabled=True` with an active breach -> `reason:"breach_still_active"`
  :379 (still no resume). Auto-resume fires ONLY at T+2h with no breach :388.
- `evaluate_breach` fail-safe: `nav<=0` -> `nav_invalid=True, any_breached=False`
  (:297, phase-69.1 — a BQ-timeout `or 0.0` must NOT phantom-flatten the book).
- **Isolation hazard**: module-level singleton `_state=KillSwitchState()` :271
  reads AND writes `_AUDIT_PATH` (handoff/kill_switch_audit.jsonl). The proven
  isolation helper is `_fresh_state(monkeypatch, tmp_path)` in
  `test_phase_38_1_kill_switch_auto_resume.py:26-32`:
  monkeypatch `kill_switch._AUDIT_PATH` -> tmp file, then monkeypatch
  `kill_switch._state` -> fresh `KillSwitchState()`. COPY THIS. Do NOT touch
  the real audit file.

### 2. Currency/money add-on avg_entry — backend/services/paper_trader.py
- Formula is INLINE in `execute_buy` :319-345 (not standalone). Flag
  `paper_avg_entry_fx_fix_enabled` (settings.py:455, default **False**):
  - FIX ON  :332 `new_avg = (old_qty*old_avg_LOCAL + qty*price_LOCAL)/new_qty`
    -> avg_entry stays LOCAL scale (KRW for .KS, EUR for .DE).
  - LEGACY  :334 `new_avg = new_cost(USD)/new_qty(LOCAL)` -> mixes USD cost
    with local shares -> corrupts non-US (the 61.3/70.3 bug). `round(new_avg,4)`.
  - cost_basis stays USD both paths; US byte-identical (fx=1).
- **fx fallback** — `execute_buy` :228-233: `_fx_usd_to_local`/`_fx_local_to_usd`
  (paper_trader.py:32/:56 -> `fx_rates.get_fx_rate`). If either is None ->
  logger.warning + `return None` (SKIP the buy; never silent USD). The deeper
  fallback chain is in `fx_rates.get_fx_rate` :228 -> `_usd_value_live` :78:
  yfinance -> FRED -> `_last_known_usd_value` (BQ last-known-good, phase-69.1
  :111) -> None. `from==to -> 1.0` (:236). Direction pitfall doc'd :9-14
  (KRW=X is KRW-per-USD -> invert; never KRWUSD=X).
- **Phase note**: task says "61.3 follow-through" but the CODE fix is phase-70.3.
  61.3 = currency-DISPLAY brief; 70.3 = the avg_entry fix. The 64.3
  `currency_path` test asserts 70.3-fix behavior in the SHAPE of 61.3's
  live_check (KR KRW-scale, EU EUR-scale).
- **Proven seam already exists**: `test_phase_70_3_atomic_swap.py:192-230`
  (`_kr_trader(fix_on)` + `test_avg_entry_fx_fix_local_consistent_for_kr`) is
  the exact PaperTrader mock harness and PASSES. 64.3 mirrors it and ADDS the
  EU (.DE / EUR) case (70.3 only covers KR). US byte-identical case is
  `test_dod4_tier1_coverage_investment.py:594` (avg (500+600)/10==110).

### 3. Per-market screener units
- Cleanest PURE seam: `validate_ohlcv(df, market="US", ticker="")` ->
  `(clean_df, report{dropped,flagged,reasons})` at
  `backend/tools/price_quality.py:48`. Pure pandas, NO IO. Wired into the
  screener at `screener.py:161-164` via `market=market_for_symbol(ticker)`.
  - US fast-path :55-56 `if df is None or market=="US": return df, report`
    -> BYTE-IDENTICAL (dropped==0, flagged==0).
  - R1 impossible OHLC (neg price / high<low) -> DROP; R2 identical-OHLC +
    zero-vol -> DROP, identical-OHLC + vol>0 -> FLAG (not drop); R3 |ret|>0.50
    -> DROP, z>3 -> FLAG; R4 >=4 identical closes -> FLAG.
- Companion PURE unit: `market_for_symbol(symbol)` `backend/backtest/markets.py:142`
  -> `.KS/.KQ`->KR, `.DE/.PA/.AS/.F`->EU, `.OL`->NO, bare->US. And
  `get_market_config` :116 / `market_currency` (fx_rates.py:56).
- NOTE: `screen_universe` :91 thresholds (`min_price=5.0`,
  `min_avg_volume=100_000`) are GLOBAL, not per-market; the market-specific
  behavior lives in `validate_ohlcv`. Test `validate_ohlcv`, not the yfinance
  download path in `screen_universe` (that is network).

### 4. Learnings reader error-vs-empty
- Two layers:
  - Reader `BigQueryClient.get_paper_trades_in_window(window_days)`
    `backend/db/bigquery_client.py:948` — has NO try/except: a BQ error from
    `self.client.query(...)` PROPAGATES (surfaces); zero rows -> `[]`. This is
    the clean error!=empty seam.
  - Aggregator `_compute_learnings(bq, window_days)`
    `backend/api/paper_trading.py:783` — wraps the reader in try/except and
    SWALLOWS to `[]` (:822-823 divergences, :852-853 kill-switch audit). So at
    THIS layer a genuine BQ-400 is indistinguishable from empty (the 61.4
    swallowed-error class).
- **PASS-able pure test** (target the reader, not the swallowing aggregator):
  build `bq = BigQueryClient.__new__(BigQueryClient)`; set `bq.client =
  MagicMock()` + `bq._pt_table = lambda t: "p.d.t"` (skips __init__/ADC).
  - ERROR: `bq.client.query.side_effect = RuntimeError("BQ 400")` ->
    `pytest.raises(RuntimeError): bq.get_paper_trades_in_window(30)` (surfaces).
  - EMPTY: `bq.client.query.return_value.result.return_value = []` ->
    `bq.get_paper_trades_in_window(30) == []`.
  - Pure pairing empty-safety: `pair_round_trips([]) == []`
    (`backend.services.paper_round_trips`).
  This PINS error!=empty and passes today. (Closing the aggregator swallow is
  a behavior change -> out of scope for a test-only step; flag it, don't fix.)

---

## backend/tests conventions + requires_live
- pytest.ini :7-9 registers ONLY the `requires_live` marker (skip unless
  `PYFINAGENT_LIVE_TESTS=1`). Quarantine list = 6 tests total (grep
  `@pytest.mark.requires_live`): test_phase_23_2_9/62_4/23_2_5/23_2_12/
  23_2_11/... The new 64.3 tests are PURE -> add NO `requires_live` mark ->
  list stays at 6 (criterion 1 satisfied).
- conftest.py sets `os.environ.setdefault("PYFINAGENT_TEST_NO_BQ","1")` at
  IMPORT time (before collection) — extra guard against buffered BQ writers
  leaking fixture rows (the 2026-07-08 prod-pollution incident).
- Established mock idioms (all in-repo): `MagicMock()` for the bq client;
  `patch.object(fx_rates,"get_fx_rate", side_effect=_fake_fx)` for FX;
  `patch("...paper_trader.ExecutionRouter")` for fills;
  `trader._maybe_notify_trade = lambda t: None` to silence Slack;
  `save_paper_position.side_effect = lambda row: captured.update(row=row)` to
  capture the persisted row; `get_settings().model_copy(update={flag: val})`
  to toggle a settings flag; `monkeypatch.setattr` for module singletons.

---

## External research

### Read in full (>=5; counts toward gate)
| # | URL | Accessed | Kind | Finding |
|---|-----|----------|------|---------|
| 1 | docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-07-17 | doc | `setattr/setitem/setenv/delattr` auto-teardown after test; autouse `no_requests` fixture `monkeypatch.delattr("requests.sessions.Session.request")` blocks ALL net; "patch the reference your code uses, not the stdlib original". |
| 2 | docs.pytest.org/en/stable/how-to/parametrize.html | 2026-07-17 | doc | `@pytest.mark.parametrize("in,expected",[(...),...])`; stack decorators for cartesian; `pytest.param(...,marks=xfail)`; `ids=` for readable names. |
| 3 | moderntreasury.com/journal/floats-dont-work-for-storing-cents | 2026-07-17 | industry | Never float for money; store integer minor units; ISO-4217 scale metadata; multi-currency has per-ccy scale (KRW/JPY 0dp, USD/EUR 2dp). |
| 4 | evanjones.ca/floating-point-money.html | 2026-07-17 | blog [ADVERSARIAL] | Counterpoint: floats ARE safe for money IF you round after every op; a 64-bit double holds 15 digits (<10 trillion) at 2dp. Validates pyfinagent's `round(new_avg,4)` discipline. |
| 5 | augmentcode.com/guides/unit-testing-best-practices... | 2026-07-17 | industry (2025) | Mock DB/net/fs behind interfaces + DI; "hermit mode" tests run anywhere; eliminate timing deps (no sleep); AAA pattern; stubs/mocks/fakes taxonomy. |

### Snippet-only (context; not counted)
pytest-with-eric monkeypatch guide; Real Python code-testing ref; DEV.to
BigDecimal practices; DZone never-float; Modern Treasury integers; Inspired
Python advanced fixtures; Medium test-doubles (Bruce Ho); Zencoder unit-test
practices; StartEarly 5 practices. (>=12 unique URLs collected total.)

### Recency scan (last 2 years, 2024-2026)
Searched "fast pure unit tests avoid DB/network mocking test doubles python
2025 2026" and "pytest state machine parametrize 2026". Result: NO new
findings that supersede canonical practice. The 2025 AugmentCode guide (#5)
restates the long-standing mock-boundaries + DI + determinism pattern; pytest
monkeypatch/parametrize APIs are stable. Money: the never-float vs
round-discipline debate is unchanged; pyfinagent already rounds. No API/tooling
change requires a plan revision.

### Money adversarial note (consensus vs debate)
Consensus (#3): don't store money as raw floats. Debate (#4): floats are OK
with disciplined rounding under ~10T. pyfinagent uses Python `float` +
`round(new_avg,4)` / `round(...,2)` throughout paper_trader — consistent with
the evanjones position; the 64.3 tests should assert with a tolerance
(`abs(avg-expected)<eps`), NOT bit-exact equality, matching the existing
70.3 test style (`abs(avg-70000.0)<500.0`).

---

## Proposed test files + key assertions
1. `backend/tests/test_64_3_kill_switch_machine.py` — copy `_fresh_state`
   (monkeypatch `_AUDIT_PATH`+`_state`). Assert: pause("test")->is_paused;
   resume("test")->not paused; `evaluate_breach` daily+trailing breach math
   (sod/peak set) and `nav<=0 -> nav_invalid, any_breached False`; **rail-5
   stays-paused**: after pause, `check_auto_resume(healthy_nav,4,10,enabled=
   False)["action"]=="no_op"` + `"auto_resume_disabled" in reason` + STILL
   `is_paused()`; `enabled=True` + active breach -> `"breach_still_active"` +
   still paused.
2. `backend/tests/test_64_3_currency_path.py` — mirror `_kr_trader`; add
   `_eu_trader` (.DE, market="EU", EUR/USD~1.16). Assert KR add-on avg_entry
   stays KRW-scale (~70000, ON) vs legacy tiny (OFF); EU add-on avg_entry
   stays EUR-scale (== EUR-weighted avg) vs legacy USD-inflated (OFF); US
   byte-identical (fx=1 -> avg unchanged by flag); fx-unavailable
   (`get_fx_rate`->None) -> `execute_buy` returns None (skips buy).
3. `backend/tests/test_64_3_screener_market.py` — pure `validate_ohlcv` +
   `market_for_symbol`. Assert US -> df returned UNCHANGED (dropped==0);
   EU/KR impossible bar (R1) dropped; R2 identical-OHLC zero-vol dropped vs
   vol>0 flagged-not-dropped; R3 >50% move dropped; `market_for_symbol`
   .KS->KR/.DE->EU/bare->US.
4. `backend/tests/test_64_3_learnings_reader.py` — `get_paper_trades_in_window`
   via `BigQueryClient.__new__` + MagicMock client. Assert BQ error PROPAGATES
   (`pytest.raises`); zero rows -> `[]`; `pair_round_trips([])==[]`. Pins
   error != empty.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5)
- [x] 10+ unique URLs total (>=12)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] >=1 [ADVERSARIAL] source (evanjones — floats-for-money counterpoint)

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 9,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "All 4 gap areas map to PURE testable seams. Kill-switch: evaluate_breach (kill_switch.py:281) + check_auto_resume(enabled=False) (:340) is the rail-5 stays-paused policy (away-ops-rules.md:17-18); isolate via _fresh_state monkeypatch of _AUDIT_PATH+_state (test_phase_38_1:26). Currency: add-on avg_entry inline in execute_buy (paper_trader.py:332 ON vs :334 legacy) gated by paper_avg_entry_fx_fix_enabled (default False, phase-70.3 not 61.3); mirror the proven _kr_trader harness (test_phase_70_3:192) and ADD an EU/.DE case; fx fallback returns None->skip buy. Screener: validate_ohlcv(df,market) (price_quality.py:48) US-noop/intl R1-R4 drop-flag + market_for_symbol (markets.py:142). Learnings: get_paper_trades_in_window (bigquery_client.py:948) RAISES on error / []-on-empty is the clean seam; _compute_learnings (:822) swallows (flag, don't fix). Mock bq via MagicMock/BigQueryClient.__new__ + patch fx_rates.get_fx_rate + ExecutionRouter; conftest sets PYFINAGENT_TEST_NO_BQ=1. All pure -> requires_live stays at 6.",
  "brief_path": "handoff/current/research_brief_64.3.md",
  "gate_passed": true
}
```
