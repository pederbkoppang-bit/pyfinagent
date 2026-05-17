# phase-28.10 — Design: Opportunistic insider buying signal (CMP classifier)

**Step:** phase-28.10 (Candidate Picker Expansion — post-launch)
**Date:** 2026-05-17
**Effort:** M (new 195-line module + screener kwarg + autonomous_loop wiring + 7 settings fields)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/services/insider_signal_screen.py`:

```python
class InsiderSignal(BaseModel):
    ticker: str
    n_opportunistic_buys: int
    aggregate_usd: float
    n_unique_insiders: int
    boost_multiplier: float
    boost_tier: str  # strong | moderate | none

async def fetch_insider_signals(tickers, lookback_months=48, ...) -> dict[str, InsiderSignal]: ...
def apply_insider_signal_to_score(base, ticker, signals) -> float: ...
```

`screener.py:rank_candidates` extended with `insider_signals=None` kwarg.

## CMP classifier (Cohen-Malloy-Pomorski 2012)

```
ROUTINE       = insider traded in same calendar month in EACH of 3 prior consecutive years
OPPORTUNISTIC = all other trades by insiders with >= 3 years of history
UNKNOWN       = insiders with < 3 years of history (cold-start guard)
```

82bps/month value-weighted alpha for OPPORTUNISTIC; ~0 for ROUTINE.

## Aggregation

Per ticker:
- Sum dollar value of OPPORTUNISTIC + BUY trades in last 30 days
- ≥$500K → moderate boost 1.04
- ≥$2M → strong boost 1.07
- <$500K → no signal (filtered out)

## Cost guard

Per-ticker SEC EDGAR fetch (48-month lookback). Bounded to top 2×paper_screen_top_n (~20 candidates per cycle). `Semaphore(3)` + 0.5s throttle (more conservative than options_flow — SEC rate-limit safety).

## Feature flag

`insider_signal_screen_enabled = False` default.

## Source rationale

- **Cohen-Malloy-Pomorski 2012** — primary brief item #9 — 82bps/mo
- **Duong-Pi-Sapp 2025** — 13D pre-filing extension (supplement Gap 3)
- **MDPI 2025 herding insiders** — confirms opportunistic-class persistence

## Test plan

All 5 immutable criteria evidenced. 21-assertion unit-test suite by Q/A. Live SEC fetch deferred (rate-limited; synthetic covers full surface; underlying get_insider_trades is production-tested by Layer-1).

## References

- `handoff/current/phase-28.10-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.10.md`
- `docs/audits/phase-28.10-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[10]`
