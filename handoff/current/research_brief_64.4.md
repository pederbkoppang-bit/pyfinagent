# Research Brief — Step 64.4: Multi-Market E2E Fixture-Replay Test

**Tier:** complex | **Started:** 2026-07-17 | **Researcher gate**

## Objective (verbatim)
Multi-market e2e -- fixture-replayed US/KR/EU cycle (recorded yfinance fixtures,
no network) asserting per-market funnel counts >0 and currency invariants; plus
one requires_live-marked smoke excluded from default runs.

## Status: IN PROGRESS (write-first; appended as sources are read)

---

## Critical feasibility questions (to be answered)
1. Does the 65.2 per-market threshold flag/code ALREADY EXIST (built DARK)?
2. Smallest testable cycle seam producing per-market funnel counts offline?
3. Fixtures: record-once real yfinance vs synthetic OHLCV?
4. Currency invariants: reuse 64.3 fx pattern?
5. requires_live marker + exclusion pattern?

---

## Internal code inventory

### Q1 — Does the 65.2 per-market threshold flag/code exist? **NO.**
- grep for `per_market_screener` / `screener_threshold` / `market_thresholds` /
  `per_market_threshold` returned ZERO hits in backend/.
- masterplan: step **65.2 status = `pending`** (P0, depends_on 65.1). Its own
  verification greps for tests `-k 'screener_per_market or 65_2'` which do not exist.
- 65.1 status = `merged` (subsumed by 66.2 funnel criterion).
- **64.4 `depends_on_step` = "66.2"** (DONE 2026-07-09), NOT 65.2. So 64.4 is
  NOT formally blocked on 65.2 in the DAG — the "65.2 thresholds via test flag"
  language must be satisfied by a TEST-ONLY threshold override (monkeypatch),
  not the (non-existent) production 65.2 flag. See feasibility verdict below.
- Operator-token string for 65.2 exists in test fixtures only:
  `test_phase_62_2_operator_tokens.py:26 -> "65.2 EU SCREENER: ON"`.

### pytest marker registration (Q5)
- `pytest.ini:7-9` registers `requires_live` marker (phase-56.2): "skipped unless
  PYFINAGENT_LIVE_TESTS=1". Verification cmd uses `-m 'not requires_live'` to exclude.
- `conftest.py` sets `PYFINAGENT_TEST_NO_BQ=1` at IMPORT time (before collection) —
  BQ writes are guarded suite-wide. This is the offline-isolation seam.
- requires_live already used: test_phase_23_2_9, 23_2_5, 62_4, 23_2_12, 23_2_11.

### 64.3 currency/fx pattern to reuse (Q4)
- Flag: `settings.paper_avg_entry_fx_fix_enabled` (settings.py:455, default False).
- Reuse pattern: `test_64_3_currency_path.py:39` does
  `s = get_settings().model_copy(update={"paper_avg_entry_fx_fix_enabled": fix_on})`.
- Existing assertions: `test_64_3_currency_path_kr_avg_entry_stays_krw` (:73),
  `..._eu_avg_entry_stays_eur` (:88), `..._us_byte_identical` (:106),
  `..._fx_unavailable_skips_buy` (:117). These are the currency-invariant templates.
- paper_trader.py:331 is the live consumer of the flag.

### Q2 — Smallest testable cycle seam (per-market funnel, offline)
The live cycle screens ALL markets in ONE combined call (autonomous_loop.py:526
`universe = base + intl`; :576 single `screen_universe(tickers=universe,...)`),
and the `summary` funnel keys (universe_size/screened/candidates, :507/:965-966)
are AGGREGATE, not per-market. So the test must drive the PURE seam per market:

| Stage | Function | File:line | Offline? |
|-------|----------|-----------|----------|
| universe | market ticker list | universe_lists.py INTL_UNIVERSE (DAX-40 :17-26, KOSPI :31-41) | yes (static) |
| screening | `screen_universe(tickers, min_avg_volume, min_price, period, ...)` | tools/screener.py:91 | yes IF `yf.download` (:137, the ONLY net call) is monkeypatched |
| ranking | `rank_candidates(screen_data, top_n, strategy, ...)` | tools/screener.py:249 | yes — PURE (no I/O) |
| order-intent | `decide_trades(current_positions, candidate_analyses, holding_analyses, portfolio_state, settings, ...) -> list[TradeOrder]` | services/portfolio_manager.py:66 | yes — PURE (dicts injected; TradeOrder has `.market` :34) |

The ONE seam to mock = `backend.tools.screener.yf.download` (module-level `yf`).
Everything after :137 is pure pandas. rank_candidates + decide_trades take injected
Python objects. Per-market funnel = `{universe: len(tickers), screened:
len(screen_data), ranked: len(candidates), order_intent: len(orders)}`, computed
by calling the seam once per market. Bridge glue: rank_candidates output ->
minimal `candidate_analyses` dicts (recommendation="BUY" + risk_assessment) to
feed decide_trades (the live LLM analysis step is bypassed; hand-craft the BUY
recos — legitimate for an order-intent assertion).

### Q3 — Fixture approach: SYNTHETIC (recommend), template already exists
- `test_phase_50_5_dataquality.py:15` `_clean_series(n, start)` builds a valid
  OHLCV DataFrame (gentle drift, vol 1M+, no bad bars) that SURVIVES
  `validate_ohlcv` (price_quality.py:48). This is the fixture generator.
- yf.download shape: multi-ticker -> MultiIndex columns `(ticker, field)`; the
  mock builds `pd.concat({tkr: _clean_series() for tkr in tickers}, axis=1)`.
  Single-ticker -> flat frame (screener handles `len(tickers)==1` at :149).
- Fixtures must survive validate_ohlcv gates (price_quality.py): R1 impossible
  OHLC, R2 identical-OHLC+zero-vol DROP (:80), R3 |ret|>0.50 DROP (:34/:97), R4
  stale-run FLAG. `_clean_series` already avoids all four.
- Constraints: >= 20 bars (screener.py:172 `if len(close) < 20: continue`);
  price > min_price, avg_vol(20d) > min_avg_volume to pass :179.
- SYNTHETIC beats record-once: fully deterministic, no network EVER (record-once
  needs author-time net + risks capturing the real bad bars 50.5 drops), tiny,
  and lets us craft EU bars that FAIL US defaults but PASS lowered thresholds.
- Shape recommendation: 2-3 tickers/market x ~60 bars. US: AAPL/MSFT-like
  (price ~150, vol ~1M). EU: 2 DAX names (SAP.DE/SIE.DE) crafted to need lowered
  thresholds. KR: 2 KOSPI names (005930.KS/000660.KS), KRW-scale prices.

### Q1 refinement — "EU under the 65.2 thresholds via test flag"
`screen_universe` ALREADY accepts `min_avg_volume`/`min_price` as kwargs
(screener.py:93-94). The "test flag" = passing LOWERED kwargs for the EU call
(or a test-local settings override), NOT the (non-existent) production 65.2 flag.
Craft EU fixture to fail US defaults (100k vol / $5) but pass EU-lowered kwargs
-> demonstrates the 65.2 rationale WITHOUT depending on 65.2 being built.

### Q4 — Currency invariants (reuse 50.2 + 64.3)
- KR KRW-scale: assert screened EU/KR rows keep LOCAL-scale prices (KR ~
  1e4-1e6 KRW, EU ~ 1e1-1e3 EUR) at screen time; and/or drive `decide_trades ->
  TradeOrder` then `PaperTrader.execute_buy(market="KR"/"EU")` with mocked
  `fx_rates.get_fx_rate` (50.2 `_fake_fx_krw`, KRW->USD 0.000655) + assert
  avg_entry stays LOCAL while stored USD total is small (50.2:137, 64.3:73/:88).
- Flag: `paper_avg_entry_fx_fix_enabled` (settings.py:455) via
  `get_settings().model_copy(update={...})` (64.3 pattern :39).
- Helper `_mk_trader` (50.2:107) = PaperTrader + MagicMock BQ (no network).

### Q5 — requires_live variant (confirmed)
- Marker REGISTERED: `pytest.ini:8-9` `requires_live: ... skipped unless
  PYFINAGENT_LIVE_TESTS=1`. Exclusion `-m 'not requires_live'` (verification cmd).
- Precedent: `@pytest.mark.requires_live` at test_phase_23_2_9:83, 23_2_5:240,
  62_4:98, 23_2_12:138, 23_2_11:85. Add one `@pytest.mark.requires_live` smoke
  that hits real `yf.download` (no monkeypatch) for one US ticker.
- `conftest.py` sets `PYFINAGENT_TEST_NO_BQ=1` at import (suite-wide BQ guard).

## External research

### Read in full (7; >=5 required — gate cleared)
| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://docs.pytest.org/en/stable/example/markers.html | 2026-07-17 | official doc | Register markers in `[pytest] markers=`; select `-m NAME`, deselect `-m "not NAME"`; `@pytest.mark.NAME`; `pytest_collection_modifyitems`/`pytest_runtest_setup` for conditional skip. |
| 2 | https://github.com/kiwicom/pytest-recording | 2026-07-17 | official repo | VCR.py record/replay to YAML cassettes; DEFAULT record mode `none` (no unintentional net); `@pytest.mark.vcr`; `@pytest.mark.block_network` -> RuntimeError on net access; `--record-mode once/none/all/new_episodes`. |
| 3 | https://vcrpy.readthedocs.io/en/latest/usage.html | 2026-07-17 | official doc | 4 record modes; `none` = replay-only, error on any new request ("guarantees no new HTTP requests"); `once` = record if no cassette then error on new; cassettes = YAML, replay is fast+deterministic+offline. |
| 4 | https://til.simonwillison.net/pytest/pytest-recording-vcr | 2026-07-17 | authoritative blog | Record once (`--record-mode=once`), then replay with WiFi OFF (verified offline); `vcr_config` `filter_headers=["authorization"]` to redact secrets before commit; regenerate cassettes when API behavior changes. |
| 5 | https://medium.com/@nidhipandya1606/golden-tests-...0926b6384e9f | 2026-07-17 | blog | [ADVERSARIAL to synthetic] "You don't need hundreds of tests. You need a small set of queries whose answers must always be correct." Argues REAL recorded inputs > synthetic ("not synthetic test cases ... real queries"); data-driven systems fail silently (HTTP 200 + wrong result). |
| 6 | https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-8/ | 2026-07-17 | industry blog | Deterministic replay: capture events, "short-circuit external dependencies" with stubs, "clock virtualization / time warping" to override RNG+clock, hermetic env so outputs derive only from the trace. |
| 7 | https://qaskills.sh/blog/pytest-markers-custom-skip-xfail-guide-2026 | 2026-07-17 | blog (2026) | `--strict-markers` in `addopts` aborts on typo'd marks; quarantine pattern = fast `pytest -m "not slow"` on push, full suite nightly. Direct map to `requires_live`. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://pypi.org/project/pytest-recording/ | pkg page | dup of #2 |
| https://www.pluralsight.com/resources/blog/guides/explore-python-libraries-speed-up-http-tests-with-vcrpy | blog | dup of vcrpy |
| https://www.krython.com/tutorial/python/testing-external-services-vcr-py | tutorial | dup of vcrpy |
| https://docs.pytest.org/en/stable/how-to/mark.html | official doc | dup of #1 |
| https://oneuptime.com/blog/post/2026-02-02-pytest-markers-guide/view | blog (2026) | dup of #7 |
| https://debugg.ai/resources/deterministic-replay-or-bust-... | blog | dup of #6 |
| https://tianpan.co/blog/2026-04-12-deterministic-replay-... | blog (2026) | dup of #6 |
| https://www.mql5.com/en/blogs/post/769442 | trading blog | multi-currency reproducibility (single-threaded, real-tick chronological order) — JS-heavy, snippet sufficient |
| https://www.quantstart.com/articles/Forex-Trading-Diary-5-... | industry | multi-currency pairs context |
| https://vcrpy.readthedocs.io/en/latest/ (index) | official doc | covered by #3 |

### Search-query variants (three-variant discipline)
- Current-year frontier (2026): "pytest record replay ... 2026"
- Last-2-year window (2025): "multi-currency trading system e2e test fixtures reproducible 2025"
- Year-less canonical: "pytest custom markers -m deselect skip integration tests"; "golden file testing data pipeline deterministic replay no network"

## Recency scan (last 2 years)
Searched 2025-2026 literature on offline-replay/fixture testing + pytest markers.
Result: FOUND current guidance that COMPLEMENTS (does not supersede) the canonical
tooling. (a) pytest markers + `-m "not X"` deselection is stable/canonical (pytest
docs unchanged); the 2026 qaskills guide adds `--strict-markers` in `addopts` and
the fast-on-push / live-nightly quarantine pattern — directly applicable to our
`requires_live` marker. (b) VCR.py/pytest-recording (vcrpy 8.0.0) remain the
canonical HTTP record-replay tool; DEFAULT `none` mode + `block_network` are the
current no-network safety primitives. (c) 2026 "deterministic replay" writing
(Sakura Sky, tianpan, debugg.ai) generalizes the pattern to agents/pipelines:
short-circuit side-effecting components + virtualize clock/RNG — validates driving
the PURE screener seam (no clock/RNG) over the full loop (which has a
`datetime.now()` calendar gate). No finding overturns the monkeypatch-the-seam
approach; if anything the 2026 material reinforces "stub at the dependency boundary".

## Key findings
1. Marker exclusion is exactly our need — register in `[pytest] markers=`, mark
   with `@pytest.mark.requires_live`, exclude with `-m "not requires_live"`
   (Source: pytest docs #1; already implemented pytest.ini:8-9). qaskills 2026 (#7)
   adds `--strict-markers` hardening (enhancement, not required by 64.4).
2. Canonical record-replay = VCR cassettes, but VCR intercepts HTTP. yfinance
   (`yf.download`) returns a parsed DataFrame via curl_cffi/caching — an HTTP
   cassette is brittle for it. The robust seam is monkeypatching `yf.download`
   to return a synthetic DataFrame (the "short-circuit external dependency at the
   boundary" pattern, Source: Sakura Sky #6), which ALSO matches the existing
   phase-50 idiom (test_phase_50_5 constructs DataFrames directly).
3. `none`/`block_network` = the no-network guarantee (Source: vcrpy #3,
   pytest-recording #2). Our equivalent: monkeypatch (yf.download never dials out)
   + conftest `PYFINAGENT_TEST_NO_BQ=1`. Optional belt-and-suspenders: `pytest-socket`
   `disable_socket()` — NOT required by the criteria.
4. Small-curated-set / golden tests: "a small set of queries whose answers must
   always be correct" (Source: Pandya #5) — validates 2-3 tickers/market, not
   hundreds.
5. [ADVERSARIAL] Pandya (#5) argues REAL recorded inputs beat SYNTHETIC ("not
   synthetic test cases ... real queries") for catching silent data regressions.
   Counter-analysis for OUR use case below.

## Consensus vs debate (external)
- Consensus: stub side-effecting deps at the boundary; keep tests offline +
  deterministic; quarantine live/slow tests behind a marker excluded by default.
- Debate (synthetic vs recorded): golden-file advocates (#5) prefer REAL recorded
  inputs (higher fidelity, catches silent wrong-answer bugs). VCR record-once (#2/#4)
  is the middle path (real data captured once, replayed offline). RESOLUTION for
  64.4: the assertion is "per-market funnel counts >0 + currency SCALE invariants",
  a STRUCTURAL/deterministic gate, NOT a data-correctness oracle. Synthetic wins
  here because (i) real DAX/KOSPI bars carry the exact bad-bars price_quality.py
  DROPS (R1-R3) -> flaky counts; (ii) we must deterministically craft EU bars that
  fail US defaults but pass lowered thresholds; (iii) determinism > fidelity for a
  count>0 gate. Record-once is the fallback if a reviewer wants real-shape fidelity.

## Pitfalls (from literature + code)
- CLOCK nondeterminism: the FULL loop calls `datetime.now()` in the phase-50.4
  calendar gate (autonomous_loop.py:536-543) -> on a weekend it can drop ALL intl
  tickers (funnel=0, flaky). MITIGATION: drive the PURE screen_universe/rank/decide
  seam (calendar gate lives in the loop, NOT in screen_universe), so the clock is
  never touched. (Source: deterministic-replay clock-virtualization principle #6.)
- validate_ohlcv DROPS (price_quality.py R1-R3): synthetic bars must avoid
  impossible OHLC, identical-OHLC+zero-vol, and |1-day ret|>0.50. `_clean_series`
  (test_phase_50_5:15) already satisfies this — reuse it.
- Cassette/secret leakage (#4): N/A for synthetic fixtures (no secrets, no cassette).
- `--strict-markers`: if added later, ensure `requires_live` stays registered
  (it is) or collection aborts.

## Application to pyfinagent (file:line mapping)
- Screening seam: `screen_universe` (tools/screener.py:91), mock `yf.download`
  (screener.py:137). Thresholds via kwargs `min_avg_volume`/`min_price` (:93-94) =
  the "65.2 test flag".
- Ranking seam: `rank_candidates` (tools/screener.py:249) — pure.
- Order-intent seam: `decide_trades` (services/portfolio_manager.py:66) ->
  `list[TradeOrder]` (`.market` field :34) — pure; inject `portfolio_state` dict +
  hand-crafted BUY `candidate_analyses`. [CONFIRMED: decide_trades body :66-240 has
  ZERO bq/network/await/http refs -- pure, offline-drivable. `yf` is module-level
  (screener.py:11) so `patch.object(screener.yf,"download",fake)` mocks it cleanly.]
- Currency invariants: `_mk_trader` + `patch.object(fx_rates,"get_fx_rate",
  side_effect=_fake_fx_krw)` (test_phase_50_2:107/129) + `paper_avg_entry_fx_fix_enabled`
  (settings.py:455) via `model_copy(update=...)` (test_64_3:39).
- Fixture generator: reuse/extend `_clean_series` (test_phase_50_5:15) -> build
  per-ticker frames, `pd.concat({tkr: _clean_series() for tkr in tickers}, axis=1)`
  for the MultiIndex shape screen_universe expects when len(tickers)>1.
- Universe: `INTL_UNIVERSE` (universe_lists.py:17-41) — DAX .DE + KOSPI .KS seeds.
- requires_live smoke: `@pytest.mark.requires_live` (pytest.ini:8-9) hitting real
  `yf.download` for 1 US ticker; excluded by `-m 'not requires_live'`.

## Research Gate Checklist

Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (7 read + 10 snippet = 17)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (screener, autonomous_loop,
      portfolio_manager, markets, price_quality, universe_lists, conftest, pytest.ini,
      50.2/50.5/64.3 tests, settings)
- [x] Contradictions/consensus noted (synthetic-vs-recorded debate)
- [x] Claims cited per-claim

## FEASIBILITY VERDICT
64.4 is a SINGLE-CYCLE GENERATE. It is NOT blocked on 65.2 being built.
- Q1: the 65.2 production per-market-threshold flag/code does NOT exist (grep-empty;
  65.2 status=pending). But 64.4's DAG dep is 66.2 (DONE), and "65.2 thresholds via
  test flag" is satisfied by passing LOWERED `min_avg_volume`/`min_price` KWARGS to
  screen_universe (a test-only override) — screener already accepts them. No 65.2
  code needed. (CAVEAT: if the operator insists criterion 1 means flipping the real
  65.2 flag, THEN 64.4 blocks on 65.2 — flag this interpretation to Main.)
- Q2: smallest seam = screen_universe -> rank_candidates -> decide_trades, driven
  per market with `yf.download` monkeypatched. Pure after screener.py:137.
- Q3: SYNTHETIC deterministic fixtures via `_clean_series` (no network EVER).
- Q4: reuse 50.2/64.3 fx-mock + avg_entry-scale invariants.
- Q5: requires_live marker already registered + excluded.
Effort: ~1 focused GENERATE (one new test file `test_..._multi_market_e2e.py`
+ a small fixture helper). No multi-session build; no upstream dependency to build first.

## JSON envelope
```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "64.4 is a single-cycle GENERATE, not blocked on 65.2. The 65.2 production per-market-threshold flag does NOT exist (grep-empty; 65.2=pending), but 64.4's DAG dep is 66.2 (done) and 'EU under the 65.2 thresholds via test flag' is met by passing lowered min_avg_volume/min_price KWARGS to screen_universe (screener.py:93-94) -- no 65.2 code needed. Smallest offline seam: screen_universe (tools/screener.py:91, mock yf.download :137) -> rank_candidates (:249, pure) -> decide_trades (portfolio_manager.py:66, pure -> TradeOrder.market). Per-market funnel counts computed by driving the seam once per market (live loop only produces AGGREGATE funnel at autonomous_loop.py:1698). Fixtures: SYNTHETIC via _clean_series (test_phase_50_5:15) surviving validate_ohlcv -- no network ever. Currency invariants reuse 50.2 _mk_trader + mocked fx_rates.get_fx_rate + paper_avg_entry_fx_fix_enabled (64.3). requires_live marker already registered (pytest.ini:8-9), excluded by -m 'not requires_live'. Key pitfall: drive the PURE seam (not full loop) to avoid the datetime.now() calendar gate (autonomous_loop.py:536) that flakes on weekends.",
  "brief_path": "handoff/current/research_brief_64.4.md",
  "gate_passed": true
}
```
