# phase-28.4 — Design: Sector-neutral momentum scoring

**Step:** phase-28.4 (Candidate Picker Expansion — LAST pre-go-live item)
**Date:** 2026-05-17
**Effort:** S (2 new kwargs + ~30-line two-pass block + 2 settings fields + autonomous_loop wiring)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/tools/screener.py:rank_candidates` signature (post-edit):

```python
def rank_candidates(
    screen_data: list[dict],
    top_n: int = 10,
    strategy: str = "momentum",
    regime=None,
    pead_signals=None,
    news_signals=None,
    sector_events=None,
    revision_signals=None,
    sector_neutral: bool = False,           # NEW
    sector_neutral_min_group_size: int = 3, # NEW
) -> list[dict]: ...
```

Two-pass logic (after news_only injection, before final sort):

```python
if sector_neutral and scored:
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        key = (s.get("sector") or "").strip() or "_UNKNOWN_"
        groups[key].append(s)
    global_pool: list[dict] = []
    for key, members in list(groups.items()):
        if key == "_UNKNOWN_" or len(members) < sector_neutral_min_group_size:
            global_pool.extend(members)
            del groups[key]
    def _apply_pct_rank(members):
        raws = pd.Series([m.get("composite_score") or 0 for m in members])
        pcts = raws.rank(method="average", pct=True).tolist()
        for m, p in zip(members, pcts):
            m["composite_score_raw"] = m.get("composite_score")
            m["composite_score"] = round(float(p), 4)
    for members in groups.values():
        _apply_pct_rank(members)
    if global_pool:
        _apply_pct_rank(global_pool)
```

## Output shape change

When `sector_neutral=True`:
- `composite_score` is replaced by within-sector percentile in [0, 1]
- `composite_score_raw` is added with the original value for transparency
- `meta_scorer` reads `composite_score` transparently (treats it as a ranking signal)

When `sector_neutral=False` (default): unchanged.

## Edge cases

- Empty `scored` list → no-op.
- Stocks without a sector field → routed to `_UNKNOWN_` global pool.
- Sectors with fewer than `min_group_size` (default 3) members → merged into global pool.
- News-only synthetic candidates (no sector) → routed to global pool, ranked there.

## Test plan

1. Immutable verification (syntax + grep for sector-rank/percentile).
2. 3-file syntax check.
3. Settings defaults: False, 3.
4. `rank_candidates` signature has new kwargs.
5. Back-compat: `rank_candidates(...)` without new kwargs works unchanged.
6. Smoke: 15-candidate 4-sector dataset → sector distribution shifts from 5 Tech / 1 Health / 3 Energy / 1 Financials (absolute) to 3 / 2 / 3 / 2 (sector-neutral).
7. Edge cases: small group + missing sector → global pool.
8. Mutation test: enabling the flag DOES change top-10 ordering.
9. Q/A pass.

All nine passed; see `docs/audits/phase-28.4-smoke-test-2026-05-17.md`.

## Source rationale

- **CFA Institute Dec 2025** — primary brief item #4 + this step's brief — "sector-neutral momentum produces superior Sharpe with less regime sensitivity vs absolute momentum."
- **Quantpedia 3-methods-to-fix-momentum-crashes** — sector neutralization is documented as one of the three canonical fixes.
- **Mamais 2025 (Wiley)** — sector-momentum performance shifts across time/sectors; sector-neutral reduces the regime risk.

## Operator note

`sector_neutral_momentum_enabled = False` by default. Operator should A/B-test before flipping. The synthetic smoke shows clear sector-diversification benefit, but the validated Sharpe (1.1705) is on absolute mode — operator should backtest the sector-neutral mode against the existing baseline before production switch.

## References

- `handoff/current/phase-28.4-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.4.md`
- `docs/audits/phase-28.4-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[4]`
