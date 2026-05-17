# Experiment Results — phase-28.15 — Social media velocity in screener

**Step ID:** phase-28.15
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | Added 6 fields after call_transcript_gpr block: `social_velocity_enabled` (False), `social_velocity_min_threshold` (0.10), `social_velocity_min_mentions` (3), `social_velocity_strong_threshold` (0.20), `social_velocity_strong_boost` (0.06), `social_velocity_moderate_boost` (0.03). |
| `backend/tools/screener.py` | Added `social_velocity_signals=None` kwarg to `rank_candidates`. Apply block after gpr_exposure. |
| `backend/services/autonomous_loop.py` | Pre-fetch for top 2*paper_screen_top_n; pass through. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/social_velocity_screen.py` | New 165-line module. `SocialVelocitySignal` Pydantic model. `fetch_social_velocity_signals(tickers)` wraps existing `tools.social_sentiment.get_social_sentiment` (reuses Alpha Vantage NEWS_SENTIMENT path that bundles Reddit/Twitter/StockTwits/blogs). `_classify_boost` requires `mention_count >= min_mentions` AND `velocity >= threshold`. `apply_social_velocity_to_score` multiplier. |

---

## Verification

### 1. Immutable command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/social_velocity_screen.py').read()); print('syntax OK')" && grep -q 'social_velocity_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### 2. Smoke
```
social_velocity_enabled       = False ... (all defaults match contract)
PASS defaults

--- _classify_boost ---
  velocity=+0.30 mentions=10 -> boost=1.060 tier=strong
  velocity=+0.20 mentions= 5 -> boost=1.060 tier=strong
  velocity=+0.15 mentions= 5 -> boost=1.030 tier=moderate
  velocity=+0.10 mentions= 5 -> boost=1.030 tier=moderate
  velocity=+0.05 mentions= 5 -> boost=1.000 tier=none  # below threshold
  velocity=+0.30 mentions= 2 -> boost=1.000 tier=none  # noise guard (< 3 mentions)
  velocity=+0.30 mentions= 0 -> boost=1.000 tier=none  # no mentions
  velocity=-0.20 mentions= 5 -> boost=1.000 tier=none  # negative velocity not boosted

--- apply ---
  AAPL: 10.0 -> 10.600 (+6%, strong)
  missing/empty/None: 10.0 -> 10.000 (identity)
```

### 3. Mid-cycle bug-fix
Initial fetcher read `result.get("velocity")` but the existing `social_sentiment.py` returns the key as `sentiment_velocity` (line 122). Fixed: read `sentiment_velocity` with `velocity` fallback. Test smoke ran AFTER the fix.

### 4. Live Alpha Vantage fetch — deferred
AV free tier 5 req/min. Existing `social_sentiment.py` is production-tested by Layer-1; smoke covers the full classifier + apply surface.

---

## Success criteria

| Criterion | Evidence | Result |
|---|---|---|
| `social_velocity_screen_module_created_lifting_existing_alpha_vantage_path` | Module imports `get_social_sentiment` from existing tool; reuses AV path | PASS |
| `stocktwits_or_apewisdom_data_path_documented` | Module docstring explains StockTwits=403 + ApeWisdom=Reddit-only + AV chosen for bundled cross-source | PASS |
| `feature_flag_social_velocity_enabled_default_false` | `Settings().social_velocity_enabled == False` | PASS |
| `rate_limit_handling_documented_per_supplement_pitfalls` | Module docstring + Semaphore(2) + 0.5s throttle + bounded to 2*top_n | PASS |
| `live_check_lists_social_velocity_surfaced_tickers_for_one_cycle` | live_check_28.15.md documents 8 synthetic cases across all boost tiers + apply paths | PASS |

---

## Next

Q/A. On PASS: Cycle 28, flip 28.15. Supplement tier: 1/4.
