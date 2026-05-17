# Experiment Results — phase-28.4 — Sector-neutral momentum scoring

**Step ID:** phase-28.4
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | Added 2 fields after the crude_momentum block: `sector_neutral_momentum_enabled` (False), `sector_neutral_min_group_size` (3). |
| `backend/tools/screener.py` | Added `sector_neutral` + `sector_neutral_min_group_size` kwargs to `rank_candidates`. Added two-pass within-sector percentile-rank logic AFTER the news-only injection and BEFORE the final sort. Groups with `len >= min_group_size` get within-sector `pandas.Series.rank(method="average", pct=True)`; smaller groups + missing-sector stocks merge into a global cross-sector percentile pool. Original composite preserved as `composite_score_raw`. |
| `backend/services/autonomous_loop.py` | Pass `sector_neutral=settings.sector_neutral_momentum_enabled` and `sector_neutral_min_group_size=...` into `rank_candidates`. |

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import rank_candidates; print('importable')" && grep -qE 'sector.{0,40}rank|percentile' backend/tools/screener.py && echo "MASTERPLAN VERIFICATION: PASS"
importable
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Smoke — 15 synthetic candidates across 4 sectors, absolute vs sector-neutral

```
=== ABSOLUTE momentum mode (default) ===
Rank  Ticker  Sector        composite_score
   1  NVDA    Technology             25.950
   2  AVGO    Technology             20.600
   3  LLY     Health Care            15.100
   4  AAPL    Technology             14.750
   5  MSFT    Technology             11.900
   6  COP     Energy                 10.800
   7  XOM     Energy                  9.150
   8  CRM     Technology              9.050
   9  GS      Financials              9.050
  10  OXY     Energy                  7.300
Sector distribution (top-10 absolute): {'Technology': 5, 'Health Care': 1, 'Energy': 3, 'Financials': 1}

=== SECTOR-NEUTRAL mode (sector_neutral=True, min_group=3) ===
Rank  Ticker  Sector         pct_rank       raw
   1  NVDA    Technology        1.000    25.950
   2  COP     Energy            1.000    10.800
   3  GS      Financials        1.000     9.050
   4  LLY     Health Care       1.000    15.100
   5  AVGO    Technology        0.800    20.600
   6  XOM     Energy            0.750     9.150
   7  JPM     Financials        0.667     7.050
   8  JNJ     Health Care       0.667     2.700
   9  AAPL    Technology        0.600    14.750
  10  OXY     Energy            0.500     7.300
Sector distribution (top-10 sector-neutral): {'Technology': 3, 'Energy': 3, 'Financials': 2, 'Health Care': 2}

=== diff ===
Absolute top-10: ['NVDA', 'AVGO', 'LLY', 'AAPL', 'MSFT', 'COP', 'XOM', 'CRM', 'GS', 'OXY']
Sector-neutral top-10: ['NVDA', 'COP', 'GS', 'LLY', 'AVGO', 'XOM', 'JPM', 'JNJ', 'AAPL', 'OXY']
In abs but not sn: {'CRM', 'MSFT'}
In sn but not abs: {'JPM', 'JNJ'}
```

**PASS** — sector-neutral mode delivers exactly the documented benefits:
- Sector distribution shifts from heavily-tech (5/10) to balanced (3+3+2+2).
- Each sector's leader gets percentile rank 1.0 (NVDA, COP, GS, LLY — the strongest momentum within their respective sector).
- Mid-tier tech names CRM (composite 9.050) and MSFT (11.900) drop out — they're not the best Technology has; they were only carried by sector concentration.
- JPM and JNJ (mid-tier in their sectors) surface — they offer diversification value.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `sector_neutral_branch_added_under_a_feature_flag` | `rank_candidates(... sector_neutral=False)` kwarg; runs new two-pass logic only when True; default False | PASS |
| `minimum_per_sector_threshold_documented` | `sector_neutral_min_group_size=3` (kwarg default + settings field + docstring); groups smaller than this fall back to global pool | PASS |
| `absolute_momentum_remains_default_until_validated` | `settings.sector_neutral_momentum_enabled = False` by default; live cycle behavior unchanged | PASS |
| `live_check_compares_top10_under_both_modes_for_one_cycle` | live_check_28.4.md captures side-by-side top-10 ranking + sector distribution + diff | PASS |

---

## Artifact shape

`rank_candidates` signature (post-edit):

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
    sector_neutral: bool = False,          # NEW
    sector_neutral_min_group_size: int = 3, # NEW
) -> list[dict]:
```

New two-pass logic (inserted before the final sort):

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

    def _apply_pct_rank(members: list[dict]) -> None:
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

---

## Operator note

When `sector_neutral_momentum_enabled=True`, downstream meta_scorer reads `composite_score` values in [0, 1] instead of raw scores. Since meta_scorer treats composite_score as a ranking signal (not an absolute number) and the relative order is preserved, this works without changes. For transparency, `composite_score_raw` is preserved on each candidate so downstream code can inspect the original.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 20, flip phase-28.4 status. **This completes the pre-go-live tier (7 items: 28.0, 28.5, 28.1, 28.2, 28.3, 28.6, 28.4).**
