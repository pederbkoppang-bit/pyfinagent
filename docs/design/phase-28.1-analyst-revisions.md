# phase-28.1 — Design: Analyst EPS revision-breadth plug-in

**Step:** phase-28.1 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** S (4-file change, one new 165-line module)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

New module `backend/services/analyst_revisions.py`:

```python
class RevisionSignal(BaseModel):
    ticker: str
    breadth: float           # (n_up - n_down) / (n_up + n_down), [-1.0, 1.0]
    n_up: int
    n_down: int
    n_total: int
    lookback_days: int

async def fetch_revision_signals(
    tickers: list[str],
    lookback_days: int = 100,
    min_analysts: int = 3,
) -> dict[str, RevisionSignal]: ...

def apply_revisions_to_score(
    base_score: float,
    ticker: Optional[str],
    revision_signals: Optional[dict[str, RevisionSignal]],
    threshold: float = 0.10,
    weight: float = 0.15,
) -> float: ...
```

`backend/tools/screener.py:rank_candidates` signature extended with `revision_signals=None`.

## Formula

`breadth = (n_up - n_down) / (n_up + n_down)` over the `analyst_revisions_lookback_days` window (default 100, the Mill Street canonical).

Only `Action` values `up` and `down` count. `main` (reiteration, ~80% of rows), `init`, and `reit` are filtered out (signal-free per yfinance contract).

Multiplier applied via `score *= (1 + breadth * weight)` ONLY when `|breadth| > threshold` (deadband). Default `weight=0.15` means a full `breadth=+1.0` yields a +15% boost; `breadth=-1.0` yields a -15% penalty.

## Data source

`yf.Ticker(t).upgrades_downgrades` — per-ticker HTTP to Yahoo Finance. Returns `GradeDate`-indexed DataFrame (tz-naive `datetime64[s]`) with columns `[Firm, ToGrade, FromGrade, Action, priceTargetAction, currentPriceTarget, priorPriceTarget]`.

Cost: $0 LLM, per-ticker HTTP throttled at 0.3s/call with `asyncio.Semaphore(4)` concurrency cap. For top-N (10-30 tickers), total fetch is ~3-10s.

## Integration points

- `backend/services/autonomous_loop.py` — when `settings.analyst_revisions_enabled=True`, fetches revisions for `2 * paper_screen_top_n` candidates AFTER first-pass `screen_universe` (bounds cost to candidate set, not full S&P 500), passes to `rank_candidates(revision_signals=...)`. Mirrors existing graceful-degradation pattern (try/except, log warning, continue).
- `backend/tools/screener.py:rank_candidates` — applies `apply_revisions_to_score` in the per-stock loop AFTER `sector_events` block. Back-compat: when `revision_signals=None`, no overlay fires.

## Feature flag

`settings.analyst_revisions_enabled = False` by default — production behavior unchanged until explicitly enabled.

Supporting fields (all customizable):
- `analyst_revisions_lookback_days = 100` (Mill Street canonical; shorter = faster signal, lower IC)
- `analyst_revisions_min_analysts = 3` (statistical-noise guard)
- `analyst_revisions_threshold = 0.10` (deadband edge — |breadth| ≤ 0.10 = no multiplier)
- `analyst_revisions_weight = 0.15` (multiplier intensity)

## Test plan

1. Immutable verification command (masterplan).
2. 4-file syntax check.
3. Settings field defaults (False, 100, 3, 0.10, 0.15).
4. `rank_candidates` signature includes `revision_signals`.
5. Live `fetch_revision_signals` returns real data; AMD (4 up, 3 down in 100d) produces production-grade signal.
6. `apply_revisions_to_score` deadband works (breadth=0.0 → no change; breadth=+0.143 → +2.1% boost; breadth=+1.0 → +15% boost).
7. Back-compat: `rank_candidates(...)` without `revision_signals` works.
8. Synthetic ranking demo shows top-3 conviction shifts (AAPL +1.155, TSLA +0.915, AMD +0.148).
9. Q/A pass with 5-item audit + 8 deterministic checks.

All nine passed; see `docs/audits/phase-28.1-smoke-test-2026-05-17.md`.

## Source rationale

- **Mill Street Research 19-yr backtest** (primary brief item #1, plus this step's research brief): revision-breadth top vs bottom decile spread = 7.6% annualized, t=2.93, p=0.003; combined with price momentum Sharpe ~1.60.
- **arXiv 2502.20489 (sell-side analyst reports)**: LLM extraction of analyst Strategic Outlook generates 68bps/month alpha — confirms text-driven analyst signals are alpha-generative.
- **arXiv 2410.20597 (analyst networks)**: analyst-network attention extracts alpha — cross-validates the underlying mechanism.

## Mid-cycle bug-fix (single-evidence revision, fully disclosed)

Initial smoke: 0/5 tickers produced signals. Root cause: `_compute_breadth` used a tz-AWARE cutoff while yfinance returns a tz-NAIVE `datetime64[s]` index → `TypeError`, silently swallowed by outer `try/except`. Fixed by switching cutoff to tz-naive (`datetime.now()`) and adding a `tz_convert(None)` fallback for tz-aware indexes.

Q/A noted the outer broad-except remains and could swallow other errors — same pattern as pead/news/sector. Future cycle's polish, not a defect.

## References

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/live_check_28.1.md`
- `handoff/current/phase-28.1-research-brief.md`
- `docs/audits/phase-28.1-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[1]`
