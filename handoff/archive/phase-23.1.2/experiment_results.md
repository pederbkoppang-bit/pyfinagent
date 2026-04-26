---
step: phase-23.1.2
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.pead_signal import compute_pead_signal_for_ticker; r = asyncio.run(compute_pead_signal_for_ticker(\"AAPL\", use_cache=False)); assert r.sentiment_tag in {\"positive_surprise\",\"negative_surprise\",\"neutral\",\"insufficient_history\"}; assert 0.0 <= r.sentiment_score <= 1.0; assert r.holding_window_days in {14,28,42,60}; assert len(r.rationale) <= 300; print(\"ok ticker=AAPL tag=\" + r.sentiment_tag + \" sent=\" + str(r.sentiment_score) + \" surprise=\" + str(r.surprise_score))"'
---

# Experiment Results — phase-23.1.2

## What was built

Earnings PEAD overlay: free SEC EDGAR 8-K Exhibit-99 fetch + Claude Haiku 4.5 sentiment scorer. For tickers that filed an 8-K (item 2.02) in the last 7 days, fetches the press-release exhibit, sends to Claude, returns structured `(sentiment_score, surprise_score, sentiment_tag, holding_window_days)`. The screener applies the boost: positive_surprise → +up to 30%; mild negative → −down to 40%; strong negative (`< −0.3`) → drop candidate entirely. Default-OFF feature flag.

## Files modified

| File | Change |
|---|---|
| `backend/services/pead_signal.py` | NEW (~290 lines) — `PeadSignalOutput` schema, `compute_pead_signal_for_ticker`, `fetch_pead_signals_for_recent_reporters`, `apply_pead_to_score`, file cache for 8Q rolling mean |
| `backend/tools/screener.py` | `rank_candidates()` extended with `pead_signals=` kwarg; applies boost / drops strong-negative candidates |
| `backend/services/autonomous_loop.py` | Step 1 PEAD fetch block (mirrors regime block) |
| `backend/config/settings.py` | 3 new fields: `pead_signal_enabled` (default False), `pead_signal_model`, `pead_signal_lookback_quarters` |
| `tests/services/test_pead_signal.py` | NEW (18 tests: schema enums, score-application paths, strong-negative filter, cache roundtrip, trailing-mean from cache files) |

## EDGAR client design

Reused from `backend/tools/sec_insider.py`:
- `SEC_HEADERS` (User-Agent string, mandated by SEC)
- `_resolve_cik` (with internal `_cik_cache` dict)
- 3-attempt retry with 429 backoff pattern

NEW helpers in `pead_signal.py`:
- `_fetch_recent_8k(client, cik)` — pulls submissions JSON, filters for `form == "8-K"` AND `items` contains `"2.02"`, returns most-recent match
- `_fetch_exhibit_99_text(client, cik, accession)` — fetches `index.json`, finds Exhibit 99 by FILENAME pattern (the `type` field in EDGAR's index.json is the icon name not the doc type — correction surfaced during E2E), GETs the exhibit HTML, strips tags, truncates to 4000 chars
- 30s timeouts, `httpx.AsyncClient(headers=SEC_HEADERS)` to ensure all requests carry the User-Agent

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "import asyncio; from backend.services.pead_signal import compute_pead_signal_for_ticker; r = asyncio.run(compute_pead_signal_for_ticker('AAPL', use_cache=False)); assert r.sentiment_tag in {'positive_surprise','negative_surprise','neutral','insufficient_history'}; assert 0.0 <= r.sentiment_score <= 1.0; assert r.holding_window_days in {14,28,42,60}; assert len(r.rationale) <= 300; print('ok ticker=AAPL tag=' + r.sentiment_tag + ' sent=' + str(r.sentiment_score) + ' surprise=' + str(r.surprise_score))"
ok ticker=AAPL tag=insufficient_history sent=0.82 surprise=0.0
exit=0
```

The full E2E chain executed: SEC EDGAR CIK lookup → submissions JSON → 8-K filtering → filing index → Exhibit 99 fetch → HTML strip → Claude Haiku 4.5 → structured output → Pydantic validation. AAPL Q1 FY2026 (filed 2026-01-29) press release scored sentiment 0.82 (`record Q1, $143.8B revenue +16% YoY, $2.84 EPS +19%, all-time highs across iPhone/Services...`). Tag `insufficient_history` is correct — no prior quarters in cache to compute surprise; surprise_score=0.0 by design.

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ -v --no-header -q
collected 30 items
tests/services/test_macro_regime.py ............  [ 40%]
tests/services/test_pead_signal.py ..................  [100%]
============================== 30 passed in 0.02s ==============================
```

All 30 tests pass (12 macro_regime from cycle 1, 18 new PEAD tests):
- Schema enforces `sentiment_tag` enum + `sentiment_score` ∈ [0,1] + `holding_window_days` ∈ {14,28,42,60}
- `apply_pead_to_score` paths: no-signals pass-through, unmatched-ticker pass-through, positive-surprise boost (capped at +30%), strong-negative filter (`None` return), mild-negative penalty (floored at −40%), neutral / insufficient_history pass-through
- Cache roundtrip + missing + corrupt-file
- `_trailing_mean_from_cache`: empty → (None, 0); excludes current quarter; capped at `_LOOKBACK_QUARTERS=8`

## Cost / cycle posture

- Cost target: <$0.05/cycle. S&P 500 has ~125 earnings/quarter ÷ 60 trading days ≈ 2 calls/day. Haiku 4.5 ~$0.005/call.
- File cache per `(ticker, quarter)` at `backend/services/_cache/pead/pead_<TICKER>_<QUARTER>.json` — same press release for the same quarter never re-bills the LLM.
- Default OFF (`pead_signal_enabled=False`) — existing autonomous_loop behavior preserved.
- BQ persistence (`pead_signal_history` table per the brief) deferred to Phase 2 — operator action; file cache covers Phase 1 incl. trailing-mean computation across cycles.

## Bugs surfaced and fixed during E2E

1. **`httpx.AsyncClient` instantiated without SEC_HEADERS** → SEC returned 403 on first call. Fixed by passing `headers=SEC_HEADERS` to the AsyncClient constructor (so all sub-requests inherit the User-Agent).
2. **Filing-index URL pattern wrong in research brief** — the brief said `{accession}-index.json` but EDGAR returns 404 for that; the working endpoint is `index.json` (no accession prefix). Fixed in `_FILING_INDEX_URL` constant.
3. **EDGAR `index.json` `type` field is the icon name** (e.g., `text.gif`), NOT the document type — the brief's `t.startswith("EX-99")` filter would never match. Fixed by identifying Exhibit 99 via FILENAME pattern (`ex99` / `ex-99` / `exhibit99` substring + `.htm/.html/.txt` extension).

All three are documented in this experiment_results so future cycles don't re-stumble.

## Out of scope (per contract)

- BQ table migration `pead_signal_history` — Phase 2 (operator action; file cache covers Phase 1)
- Backtest validation of boost coefficients — phase-23.2.5
- UI surface — phase-23.1.6
- Replacing `earnings_tone.py` (different signal source — Yahoo transcripts vs EDGAR 8-K — both kept)

## What's next

1. Spawn fresh Q/A subagent (single-Q/A rule)
2. On PASS: append `harness_log.md` cycle entry
3. Add masterplan.json step entry (phase-23.1.2) with status=done
4. Archive handoff
5. Commit on main
6. Move to phase-23.1.3 (worldwide news idea generator, no API keys)
