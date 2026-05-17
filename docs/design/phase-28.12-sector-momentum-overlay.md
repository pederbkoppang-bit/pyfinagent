# phase-28.12 — Design: Sector-ETF momentum overlay

**Step:** phase-28.12 (Candidate Picker Expansion — first post-launch item)
**Date:** 2026-05-17
**Effort:** S (new 175-line module + screener kwarg + autonomous_loop wiring + 6 settings fields)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/services/sector_momentum.py`:

```python
class RankedSector(BaseModel):
    sector: str
    etf: str
    momentum: float
    rank: int                  # 1 = highest momentum, 11 = lowest
    boost_multiplier: float    # 1.15 if rank==1, 1.10 if rank<=top_n, else 1.0

async def fetch_sector_momentum_ranks(...) -> dict[str, RankedSector]: ...
def apply_sector_momentum_to_score(base, sector, ranks) -> float: ...
```

`screener.py:rank_candidates` extended with `sector_momentum_ranks=None` kwarg.

## Mechanism

1. Single yfinance batch download for 11 SPDR ETFs (XLK, XLV, XLF, XLY, XLC, XLI, XLP, XLE, XLU, XLRE, XLB).
2. Compute trailing N-month total return per ETF (default 12m = Quantpedia canonical).
3. Sort descending; rank 1-11.
4. Assign boost: 1.15× to #1, 1.10× to ranks 2-top_n, 1.0× otherwise.
5. In `rank_candidates`, multiply composite_score by `boost_multiplier` for the stock's sector.

## Feature flag

`sector_momentum_enabled = False` default. Production behavior unchanged.

## Cache

24h JSON file cache at `backend/services/_cache/sector_momentum/ranks.json`. Matches monthly rebalance cadence with safety margin.

## Source rationale

- **Quantpedia sector momentum rotational system** — top-3 monthly rotation → 13.94%/yr, Sharpe 0.54, +4%/yr vs passive.
- **Faber sector rotation** — canonical 12-month sector momentum robust over decades.
- **Alvarez Quant Trading** — persistence confirmed post-2020.

Boost magnitudes (1.10 / 1.15) are practitioner heuristics — small enough to preserve other signals but visible. Operator should A/B-test before flipping.

## Sector naming

Uses `screener.py::SECTOR_ETFS` canonical keys. `sector_analysis.py:13-25` has a duplicate map with slightly different keys; that's a separate Layer-1 analysis path and was NOT touched.

## Test plan

All 8 tests passed (immutable verification, 4-file syntax, settings defaults, module imports, signature kwarg, live fetch, apply unit, Q/A pass). See `docs/audits/phase-28.12-smoke-test-2026-05-17.md`.

## References

- `handoff/current/phase-28.12-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.12.md`
- `docs/audits/phase-28.12-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[12]`
