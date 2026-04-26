---
step: phase-23.1.2
title: Earnings PEAD overlay (free SEC EDGAR 8-K + Claude sentiment-surprise vs trailing 8Q mean)
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.pead_signal import compute_pead_signal_for_ticker; r = asyncio.run(compute_pead_signal_for_ticker(\"AAPL\", use_cache=False)); assert r.sentiment_tag in {\"positive_surprise\",\"negative_surprise\",\"neutral\",\"insufficient_history\"}; assert 0.0 <= r.sentiment_score <= 1.0; assert r.holding_window_days in {14,28,42,60}; assert len(r.rationale) <= 300; print(\"ok ticker=AAPL tag=\" + r.sentiment_tag + \" sent=\" + str(r.sentiment_score) + \" surprise=\" + str(r.surprise_score))"'
research_brief: handoff/current/phase-23.1.2-research-brief.md
---

# Contract — phase-23.1.2

## Hypothesis

A daily-batch SEC EDGAR 8-K Exhibit-99 fetch + Claude Haiku 4.5 sentiment scorer can detect earnings sentiment-surprise (current sentiment − trailing-8Q mean) for tickers reporting in the last week. Positive surprise boosts the candidate's `composite_score` in `screener.rank_candidates`; strong negative surprise filters out the candidate. This implements the QuantPedia 5.89% CAR / Sharpe 0.76 PEAD-NLP strategy at zero data-vendor cost.

## Plan

1. **NEW `backend/services/pead_signal.py`** mirroring `macro_regime.py` design:
   - `PeadSignalOutput` Pydantic model with `ConfigDict(extra="forbid")` — fields: rationale, sentiment_score, surprise_score, sentiment_tag, holding_window_days, skip_reason
   - `compute_pead_signal_for_ticker(ticker, use_cache=True)` async — EDGAR fetch → Claude → cache
   - `fetch_pead_signals_for_recent_reporters()` async — pull tickers from `pyfinagent_data.calendar_events` (last 7 days) and run them in parallel under `asyncio.Semaphore(3)`
   - File cache per `(ticker, quarter)` at `backend/services/_cache/pead_signal_<ticker>_<quarter>.json`
   - 8Q rolling mean computed from local cache files (BQ table is Phase-2 deferred)
2. **EDGAR client** — reuse `SEC_HEADERS`, `_resolve_cik`, `_cik_cache` from `backend/tools/sec_insider.py`. Add helpers: `_fetch_recent_8k(cik)`, `_fetch_filing_index(cik, accession)`, `_fetch_exhibit_99(cik, accession, doc_name)`. 3-attempt retry with 429 backoff; 30s timeout.
3. **Schema-strip + clamp logic** — copy `_strip_unsupported_schema_keys` import from `macro_regime.py`. Clamp sentiment_score to [0.0, 1.0] post-parse.
4. **Extend `backend/tools/screener.py:151` `rank_candidates`** — add `pead_signals: dict[str, PeadSignalOutput] | None = None` kwarg. Apply: positive_surprise → `score *= 1 + min(surprise_score*0.5, 0.3)`; negative_surprise with `surprise_score < -0.3` → drop candidate; mild negative → `score *= max(1 + surprise_score*0.5, 0.6)`.
5. **Wire into `backend/services/autonomous_loop.py` Step 1** — after the regime block, fetch pead_signals when `pead_signal_enabled=True`. Pass to `rank_candidates`. Default-OFF.
6. **NEW settings fields** in `backend/config/settings.py`: `pead_signal_enabled: bool = False`, `pead_signal_model: str = "claude-haiku-4-5"`, `pead_signal_lookback_quarters: int = 8`.
7. **Tests** at `tests/services/test_pead_signal.py`:
   - Schema validation (enum, float ranges, holding_window_days enum)
   - `apply_pead_to_score` — positive_surprise boost, strong-negative filter, mild-negative penalty, no-signal pass-through
   - Cache roundtrip + expiry
   - Integration with mocked EDGAR + mocked Claude → returns PeadSignalOutput
   - 8Q rolling mean from cache files (insufficient history → tag="insufficient_history")
8. **Verification** — immutable command in front-matter calls `compute_pead_signal_for_ticker("AAPL")` end-to-end against real EDGAR + real Claude.

## Out of scope

- BQ table migration `pead_signal_history` (Phase 2 — operator-action; file cache covers Phase 1)
- Backtest validation of the boost coefficients (phase-23.2.5)
- Replacing `earnings_tone.py` (different signal — Yahoo transcripts vs EDGAR 8-K — both kept)
- UI surface (phase-23.1.6)

## Files modified

- `backend/services/pead_signal.py` — NEW (~250 LOC)
- `backend/tools/screener.py` — extend rank_candidates with pead_signals kwarg + apply
- `backend/services/autonomous_loop.py` — Step 1 PEAD fetch block
- `backend/config/settings.py` — 3 new fields
- `tests/services/test_pead_signal.py` — NEW
- `.claude/masterplan.json` — phase-23.1.2 step entry
- `handoff/current/{contract,experiment_results,evaluator_critique}.md` — rolling

## Verification (immutable)

The front-matter command runs against real SEC EDGAR + real Claude — no mocks. It verifies:
- EDGAR fetch chain works (CIK → submissions → filing index → Exhibit 99)
- Claude returns a parseable PeadSignalOutput
- All field constraints (sentiment_score ∈ [0,1], holding_window in {14,28,42,60}, rationale ≤ 300 chars)
- The function is reachable from a clean `python -c` invocation

## References

- `handoff/current/phase-23.1.2-research-brief.md` — full research brief (581 lines, 8 sources read in full, gate_passed: true)
- `backend/tools/sec_insider.py:17-30` — SEC_HEADERS + _resolve_cik to reuse
- `backend/services/macro_regime.py` — design template (just shipped phase-23.1.1)
- `backend/tools/screener.py:151` — rank_candidates extension surface
- `backend/services/autonomous_loop.py:113-128` — regime block (PEAD slots in after)
- `backend/agents/llm_client.py:690-840` — ClaudeClient structured-output gotchas (already documented)
