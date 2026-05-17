# Contract — phase-28.15 — Social media velocity in screener

**Step ID:** phase-28.15
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.15-research-brief.md` (`gate_passed: true`; 5 sources read in full).
- Internal audit: `backend/tools/social_sentiment.py:95` already computes `velocity = recent_avg - older_avg` via Alpha Vantage NEWS_SENTIMENT API (bundles Reddit/Twitter/StockTwits/blogs). Used by Layer-1 enrichment but NOT by the screener.
- Researcher: StockTwits API 403/suspended; ApeWisdom Reddit-only without SLA. Alpha Vantage is the right choice. Threshold: `velocity >= 0.10` AND `mention_count >= 3`. Boost 0.06/0.03 strong/moderate.

## Hypothesis

Surfacing tickers with social-velocity spikes BEFORE the screener's momentum filter gives the picker pre-rally exposure (per supplement Gap 2 + DNUT July 2025 case: 500% StockTwits spike preceded 90% pre-market). Default OFF.

## Immutable success criteria (from masterplan)

1. `social_velocity_screen_module_created_lifting_existing_alpha_vantage_path`
2. `stocktwits_or_apewisdom_data_path_documented`
3. `feature_flag_social_velocity_enabled_default_false`
4. `rate_limit_handling_documented_per_supplement_pitfalls`
5. `live_check_lists_social_velocity_surfaced_tickers_for_one_cycle`

Immutable verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/social_velocity_screen.py').read()); print('syntax OK')" && grep -q 'social_velocity_enabled' backend/config/settings.py
```

Immutable live_check shape:
> "live_check_28.15.md: cycle log showing N tickers surfaced by social velocity + the velocity multipliers + final ranking impact"

## Plan steps

1. **Settings**: `social_velocity_enabled` (False), `social_velocity_min_threshold` (0.10), `social_velocity_min_mentions` (3), `social_velocity_strong_threshold` (0.20), `social_velocity_strong_boost` (0.06), `social_velocity_moderate_boost` (0.03).
2. **New module `backend/services/social_velocity_screen.py`**: wraps `tools/social_sentiment.get_social_sentiment` per ticker; classifies into strong/moderate/none based on velocity + mention_count; apply helper.
3. **Edit `backend/tools/screener.py:rank_candidates`**: add `social_velocity_signals=None` kwarg + apply block after gpr_exposure.
4. **Edit `backend/services/autonomous_loop.py`**: pre-fetch for top 2*paper_screen_top_n; pass through.
5. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.15-research-brief.md`
- Supplement brief Gap 2
- `.claude/masterplan.json::phase-28.steps[15]`

## Risk / blast radius

- **Default OFF.**
- **Alpha Vantage rate limit:** free tier 5 req/min. Existing `tools/social_sentiment.py` already uses this; pattern: bounded to top-N candidates.
- **Graceful degradation** — Alpha Vantage rate limit / no API key → empty dict → identity.
