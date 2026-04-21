# Experiment Results — phase-7 / 7.8 (Satellite/geospatial deferred)

**Step:** 7.8 **Date:** 2026-04-19 **Cycle:** 1 (closure).

## What was done

No code, no doc write — this is a closure cycle. Criterion was already satisfied by phase-7.0's compliance doc (qa_70_v1 even validated the same grep as a bonus check).

## Verification output

```
$ grep -n 'Phase 8' docs/compliance/alt-data.md
30:**Phase 8**), private messaging platforms, LinkedIn profile data (post-hiQ
161:| 7.8 | Satellite / geospatial | **DEFERRED to Phase 8** (see Section 8) | -- | -- | deferred |
251:## 8. Open Items / Deferred (including Phase 8)
253:- **7.8 Satellite/geospatial proxies** -- DEFERRED to **Phase 8**. Scope:

$ grep -q 'Phase 8' docs/compliance/alt-data.md && echo "GREP EXIT=0"
GREP EXIT=0

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Contract criterion

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `grep -q 'Phase 8' docs/compliance/alt-data.md` | PASS | 4 literal "Phase 8" references in the compliance doc across Section 1, the per-source policy row, and Section 8. |

## Caveats

1. **Criterion satisfaction is structural.** "Phase 8" does not yet exist as a masterplan phase. The deferral is a forward-pointer with budget rationale (Planet Labs/Maxar/Spire enterprise licenses exceed current LLM-API budget).
2. **No satellite vendor integration happens in this cycle or in any of phase-7.** Signal extraction from parking-lot / oil-tank / cargo imagery is a Phase 8 agenda item.
