# Contract — phase-28.4 — Sector-neutral momentum scoring

**Step ID:** phase-28.4
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.4-research-brief.md` (`gate_passed: true`; 5 sources read in full incl. CFA Institute 2025, Quantpedia momentum-fix, RegimeFolio arXiv, Mamais 2025 Wiley, sector-momentum rotational).
- Internal audit: `screener.py:rank_candidates` already attaches `sector` to candidates (via the phase-23.1.13 `sector_lookup` propagation). `meta_scorer.py` consumes `composite_score` transparently — no changes needed downstream. Existing scoring loop builds `composite_score` per-stock; sector-neutral pass should happen AFTER all overlays are applied so the percentile is over the FINAL composite, not the raw momentum.
- Recommendation: two-pass approach inside `rank_candidates`, feature-flagged (`sector_neutral=False` default). Pass 1 = existing per-stock scoring + overlays. Pass 2 = group by sector, `pandas.Series.rank(method='average', pct=True)` within each group of N≥3; smaller groups (and missing-sector stocks) route to a global cross-sector pool ranked identically.

## Hypothesis

Per CFA Institute Dec 2025: sector-neutral momentum produces superior Sharpe with less regime sensitivity vs absolute (raw) momentum because it removes sector-concentration bias (the current picker often returns 6/10 tech). Within-sector percentile rank turns "best in absolute terms" into "best vs same-sector peers" — naturally diversifies the candidate set AND mitigates momentum crashes when one sector drags down its constituents en masse.

Feature-flagged default OFF preserves the validated Sharpe (1.1705); operator flips when ready to A/B-test.

## Immutable success criteria (from `.claude/masterplan.json::phase-28.steps[4].verification.success_criteria`)

1. `sector_neutral_branch_added_under_a_feature_flag`
2. `minimum_per_sector_threshold_documented`
3. `absolute_momentum_remains_default_until_validated`
4. `live_check_compares_top10_under_both_modes_for_one_cycle`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import rank_candidates; print('importable')" && grep -qE 'sector.{0,40}rank|percentile' backend/tools/screener.py
```

Immutable live_check shape:
> "live_check_28.4.md: cycle log + top-10 under absolute vs sector-neutral side-by-side"

## Plan steps

1. **Settings additions** (after crude_momentum block):
   - `sector_neutral_momentum_enabled: bool = Field(False, ...)`
   - `sector_neutral_min_group_size: int = Field(3, ...)`

2. **Edit `backend/tools/screener.py:rank_candidates`**:
   - Add `sector_neutral: bool = False` and `sector_neutral_min_group_size: int = 3` kwargs
   - After the existing scoring loop (which produces `scored` list with `composite_score`), when `sector_neutral=True`:
     - Group `scored` by sector
     - For groups with `len >= min_group_size`: compute within-sector percentile rank of `composite_score` using `pandas.Series.rank(method='average', pct=True)`; this becomes the NEW composite_score (preserved on a `composite_score_raw` field for transparency)
     - For groups with `len < min_group_size` + missing-sector stocks: aggregate into a "global pool"; rank them by raw composite_score percentile too
     - Final result: each stock has a percentile in [0, 1] under sector-neutral mode; ranking is by percentile (descending)
   - `news_only` synthetic baseline (score=5.5) gets clamped to a default percentile (e.g., 0.5) under sector-neutral mode

3. **Edit `backend/services/autonomous_loop.py`**:
   - Pass `sector_neutral=settings.sector_neutral_momentum_enabled` and `sector_neutral_min_group_size=settings.sector_neutral_min_group_size` to `rank_candidates`.

4. **Run masterplan verification** — must EXIT 0.

5. **Smoke test**:
   - Build synthetic `screen_data` covering 4 sectors (3+ tickers each)
   - Call `rank_candidates(screen_data, top_n=10)` (absolute mode)
   - Call `rank_candidates(screen_data, top_n=10, sector_neutral=True)` (sector-neutral mode)
   - Show top-10 of each + diff in ranking + sector distribution (should be more diverse under sector-neutral)

6. **Write `experiment_results.md`** + `live_check_28.4.md`.

7. **Spawn Q/A**.

8. **On PASS** — append harness_log Cycle 20, flip status.

## References

- `handoff/current/phase-28.4-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #4)
- `.claude/masterplan.json::phase-28.steps[4]`

## Risk / blast radius

- **Default OFF** — `sector_neutral_momentum_enabled = False`. Current Sharpe 1.1705 preserved.
- **Back-compat** — when False, the loop is unchanged. All current callers (autonomous_loop, backtest, tests) work unchanged.
- **Meta-scorer compatibility** — `meta_scorer.py` reads `composite_score`; percentile values [0,1] are valid floats so the LLM prompt continues to work (it sees smaller numbers, but the relative ordering is what it cares about).
- **`composite_score_raw` field** — sector-neutral mode adds this field for transparency; consumers ignoring extra fields are unaffected.
- **No cost change.** Zero LLM, zero network.
