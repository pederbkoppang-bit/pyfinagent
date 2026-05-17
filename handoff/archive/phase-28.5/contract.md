# Contract — phase-28.5 — Short-interest exclusion filter

**Step ID:** phase-28.5
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.5-research-brief.md` (Researcher subagent; `gate_passed: true`)
- 6 external sources read in full: yfinance Ticker.info doc, FINRA short-interest files page, FINRA OTC short-interest API page, Quantpedia short-interest-effect long/short version, Oxford Academic 32-country short-interest study (RAPS 2022), practitioner short-percentage tutorial.
- 13 URLs collected, 7 snippet-only (SSRN/Wiley paywalls, binary PDFs, conference paper TLS error).
- Recency scan (2024-2026): anomaly remains valid; partial decay documented for the long-short version, NOT for exclusion-only filtering.
- Internal audit: `screener.py:127` is the filter-chain hook (immediately after the basic price/volume filter); `settings.py:182-197` shows the established feature-flag pattern (`macro_regime_filter_enabled`, `pead_signal_enabled`, etc., all default False).

## Hypothesis

The short-interest anomaly (Boehmer-Jones-Zhang 2008: high-short stocks underperform 1.16%/month; Oxford RAPS 2022: 32-country international confirmation) supports adding a **single-direction exclusion filter** (not a long-short trade) to the screener. By skipping tickers where `shortPercentOfFloat > 0.10` (approximate top-decile for S&P 500 large-caps), we expect to:

1. Improve hit rate by removing known-loser stocks identified by sophisticated short-sellers.
2. Avoid the "partial alpha decay" that has hit the LONG leg of the strategy (Quantpedia OOS note) since exclusion is a softer claim than long-short trading.
3. Cost zero LLM dollars and zero per-cycle HTTP calls if implemented via the FINRA bimonthly CSV (preferred); fallback to yfinance per-ticker is feasible but slower.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json::phase-28.steps[5].verification.success_criteria`)

1. `short_interest_field_collected_in_screen_universe`
2. `exclusion_filter_added_with_documented_threshold`
3. `feature_flag_short_exclusion_enabled_default_false`
4. `live_check_lists_excluded_tickers_for_one_cycle`

Immutable verification command:

```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE 'short.{0,30}(ratio|interest|exclusion)' backend/tools/screener.py
```

Immutable `live_check` shape:

> live_check_28.5.md: cycle log showing N excluded tickers + their shortRatio values

(Note: the brief recommends `shortPercentOfFloat` over the more ambiguous `shortRatio`. The live_check spec uses "shortRatio" as a generic label; we will report `shortPercentOfFloat` values and explain why in the live_check artifact.)

---

## Plan steps

1. **Research gate** — DONE. Brief recommends FINRA bimonthly CSV as primary, yfinance `Ticker.info["shortPercentOfFloat"]` as fallback. Threshold: 0.10 (10% of float). Feature-flagged default OFF.

2. **Add settings flags** (`backend/config/settings.py` after line 198):
   - `short_interest_filter_enabled: bool = Field(False, ...)`
   - `short_interest_threshold: float = Field(0.10, ...)`

3. **Add new service** (`backend/services/short_interest.py`):
   - `fetch_short_interest_lookup() -> dict[str, float]`
   - FINRA bimonthly CSV primary path with 14-day local cache
   - yfinance per-ticker fallback for tickers missing in FINRA
   - Returns empty dict on any unrecoverable error (default-OFF safety pattern, mirrors `news_screen` / `pead_signal`)

4. **Edit `backend/tools/screener.py`**:
   - Add `short_interest_lookup: Optional[dict[str, float]] = None` and `short_interest_threshold: float = 0.10` kwargs to `screen_universe`
   - Insert exclusion block immediately after the existing basic filter at line 127:
     ```python
     # phase-28.5: short-interest exclusion (high-short underperforms 1.16%/mo per Boehmer-Jones-Zhang 2008)
     if short_interest_lookup:
         short_pct = short_interest_lookup.get(ticker)
         if short_pct is not None and short_pct > short_interest_threshold:
             logger.debug(f"Excluding {ticker}: shortPercentOfFloat={short_pct:.3f} > {short_interest_threshold}")
             continue
     ```
   - When lookup is None or empty dict, no exclusion happens (back-compat with all current callers).

5. **Edit `backend/services/autonomous_loop.py`**:
   - In Step 1 (screening universe), when `settings.short_interest_filter_enabled` is True, pre-fetch the lookup via `fetch_short_interest_lookup()` and pass to `screen_universe(short_interest_lookup=..., short_interest_threshold=...)`.
   - Mirror the existing graceful-degradation pattern (try/except, log warning, continue).

6. **Run immutable verification command** — must EXIT 0.

7. **Smoke test** — `screen_universe(tickers=['AAPL','MSFT','TSLA'], short_interest_lookup={'TSLA': 0.15}, short_interest_threshold=0.10)` must return 2 results (TSLA excluded).

8. **Write `experiment_results.md`** with verbatim outputs.

9. **Write `live_check_28.5.md`** showing N excluded tickers + their values.

10. **Spawn Q/A** — fresh `qa` subagent.

11. **On PASS** — append harness_log Cycle entry, then flip status.

## References

- `handoff/current/phase-28.5-research-brief.md`
- `.claude/masterplan.json::phase-28.steps[5]`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #5)

## Risk / blast radius

- **Default OFF** — feature flag stays at False until validated. No change to current screener behavior.
- **Back-compat** — `short_interest_lookup` defaults to None; all current callers (autonomous_loop, backtest, tests) work unchanged.
- **Graceful degradation** — if FINRA download fails, the lookup is empty, exclusion never fires, cycle continues with the existing filter chain. Mirrors the `news_screen` / `pead_signal` failure pattern.
- **No LLM cost** — pure data filter; zero $/cycle.
- **Modest network cost when ON** — one FINRA CSV download every 14 days (a few MB), cached locally. If FINRA path fails, yfinance fallback adds 0.5s × N tickers (≤ 4 minutes for full S&P 500).
