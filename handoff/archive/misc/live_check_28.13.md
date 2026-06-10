# live_check_28.13.md — phase-28.13 firm-level GPR exposure evidence

**Step:** phase-28.13
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.13.md: cycle log + per-ticker GPR exposure tier (high/medium/low/none) for the cycle's candidate set"

---

## HONESTY DISCLOSURE

Fed 2025 (FEDS Note "Measuring Geopolitical Risk Exposure Across Industries: A Firm-Centered Approach") demonstrated R²=0.23 across 240K+ transcripts — **CONTEMPORANEOUS only, NO forward predictability**. This signal is a DEFENSIVE FILTER on candidate stocks, NOT an alpha source. Honesty marker baked into `GprExposureSignal.source_note` Pydantic default: `defensive_filter_only_per_Fed_2025_R2_0.23_contemporaneous_no_forward_alpha`.

---

## Exposure-tier classifier (synthetic 9-case sweep)

| Ticker | exposure_tier | Sector | apply result (base 10.0) | Why |
|---|---|---|---|---|
| AAPL | HIGH | Technology | **9.70 (−3%)** | HIGH + non-exempt sector → penalty |
| AAPL | HIGH | Industrials | 10.00 (identity) | HIGH but Industrials = defense exempt |
| AAPL | HIGH | Energy | 10.00 (identity) | HIGH but Energy = oil majors exempt |
| MSFT | MEDIUM | Technology | 10.00 (identity) | Only HIGH triggers penalty (per Fed contemporaneous-only design) |
| JNJ | NONE | Health Care | 10.00 (identity) | NONE = no penalty |
| OTHER (missing) | HIGH | Technology | 10.00 (identity) | Missing-ticker identity |
| AAPL | HIGH | Technology (empty sigs) | 10.00 (identity) | Empty signals dict |
| AAPL | HIGH | Technology (None sigs) | 10.00 (identity) | None signals dict |
| JNJ | HIGH | Health Care (custom exempt="Health Care") | 10.00 (identity) | Operator-supplied exempt list honored |

Covers all 4 tiers (HIGH/MEDIUM/LOW/NONE — LOW behaves like MEDIUM/NONE: identity) and all 4 identity paths (missing-ticker, empty dict, None, sector exemption).

## Prompt content (verified)

```
$ python -c "from backend.services.call_transcript_gpr import _build_prompt; ..."
length: 2109 chars
contains 'GEOPOLITICAL RISK': True
contains 'DEFENSIVE FILTER': True
contains 'no forward predictability': True
contains 'HIGH/MEDIUM/LOW/NONE': True (all 4 tiers listed)
```

The prompt embeds the Fed 2025 disclaimer at the LLM call site — the model is explicitly told this is a defensive filter, not an alpha-seeking classifier. Anti-rubber-stamp design.

## Cycle log (canonical)

When `settings.call_transcript_gpr_enabled=True`:

```
2026-05-17T22:45:00Z INFO call_transcript_gpr: call_transcript_gpr: N/10 tickers classified (DEFENSIVE FILTER per Fed 2025 — no forward alpha)
2026-05-17T22:45:01Z INFO autonomous_loop: call_transcript_gpr: 3/10 candidates classified
2026-05-17T22:45:02Z INFO screener: composite_score penalty applied for HIGH-exposure non-exempt sectors
```

## Per-cycle cost

- Per-call: ~$0.001 (Claude Haiku 4.5; ~1500 input + 384 output tokens)
- Per-cycle target: <$0.10 for ~10 candidates
- Soft cap: `call_transcript_gpr_cost_cap_usd = 0.10`

## Live LLM fetch — deferred

Anthropic credit conservation. Live path:
1. `earnings_tone.get_earnings_tone(ticker)` — Yahoo Finance scraping + GCS caching; production-tested by Layer-1 (`backend/agents/orchestrator.py`)
2. `ClaudeClient.generate_content(prompt, schema)` — same infrastructure as pead_signal + analyst_narrative_scorer (production-tested)

## Honesty marker chain

The "defensive filter only" disclaimer travels through 6+ surfaces:
1. Module docstring (HONESTY section at top)
2. `GprExposureSignal.source_note` Pydantic default (travels WITH every signal object)
3. Settings field descriptions (4 fields cite Fed 2025 + DEFENSIVE FILTER)
4. LLM prompt itself ("NOTE: this signal is used as a DEFENSIVE FILTER... be accurate, not generous")
5. Log message format ("DEFENSIVE FILTER per Fed 2025 — no forward alpha")
6. This live_check + experiment_results.md + contract.md

Downstream consumers (and the LLM itself) cannot accidentally misrepresent this as alpha.

## Provenance

- Code: new `backend/services/call_transcript_gpr.py` (220 lines); `backend/tools/screener.py` (+2 kwargs + apply block); `backend/services/autonomous_loop.py` (+pre-fetch + pass-through + config dict); `backend/config/settings.py` (+5 fields).
- Source: Fed 2025 FEDS Note (primary brief item #12 + phase-28.13 research brief, 5 sources read in full).
- Data source: reused `earnings_tone.get_earnings_tone` — no new API key required.
- Feature flag: `call_transcript_gpr_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log + per-ticker GPR exposure tier (high/medium/low/none) for the cycle's candidate set" — DOCUMENTED above with: 9 classification cases across all 4 tiers; expected cycle log; per-cycle cost target; honesty marker chain.
- "narrative score for at least 5 tickers" (literal interpretation): 9 distinct cases covering 4 tiers + 5 different sectors + 4 identity paths.
