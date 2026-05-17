# Contract — phase-28.1 — Analyst EPS revision-breadth plug-in

**Step ID:** phase-28.1
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.1-research-brief.md` (Researcher subagent; `gate_passed: true`)
- 5 external sources read in full: yfinance API ref, Mill Street Research methodology, arXiv 2502.20489 (sell-side analyst reports), arXiv 2410.20597 (analyst networks alpha), MDPI 2025 (momentum-based normalization).
- 15 URLs collected, 10 snippet-only.
- Recency scan 2024-2026: arXiv preprints confirm the analyst-revision-breadth signal persists; the 2025 work on momentum-based normalization shows the signal can be enhanced.
- Data path: yfinance `Ticker.upgrades_downgrades` returns a `GradeDate`-indexed DataFrame with `Action` in `{up, down, main, init, reit}`. Only `up` and `down` count toward revision breadth (`main` is ~80% of rows = noise and must be filtered out).

## Hypothesis

The analyst-revision-breadth signal (Mill Street Research 19-year backtest: top vs bottom decile spread = 7.6% annualized, t=2.93, p=0.003; Sharpe ~1.60 when combined with price momentum) supports a small-magnitude multiplier overlay on the screener's composite score. The signal fires 1-2 weeks before price momentum becomes visible, directly addressing the early-detection gap for reference cases like Sandisk/memory (analysts were aggressively raising MU/WDC EPS mid-2024) and oil majors (analysts revise energy EPS within 1-2 weeks of crude price moves).

## Immutable success criteria (copied verbatim from `.claude/masterplan.json::phase-28.steps[1].verification.success_criteria`)

1. `analyst_revisions_module_created_and_syntax_OK`
2. `feature_flag_analyst_revisions_enabled_default_false`
3. `wired_into_rank_candidates_or_meta_scorer`
4. `smoke_run_with_flag_on_produces_non_empty_signal_for_recent_reporters`
5. `cycle_cost_delta_under_0_05_USD`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/analyst_revisions.py').read()); from backend.services.analyst_revisions import fetch_revision_signals; print('module importable')" && grep -q 'analyst_revisions_enabled' backend/config/settings.py && grep -q 'analyst_revisions' backend/services/autonomous_loop.py
```

Immutable live_check shape:
> "live_check_28.1.md: cycle log + screener output diff showing N revisions-scored tickers, top-3 conviction shifts vs baseline"

## Plan steps

1. **Settings**: add 5 fields after the short-interest block:
   - `analyst_revisions_enabled` (bool, False)
   - `analyst_revisions_lookback_days` (int, 100 — Mill Street canonical window)
   - `analyst_revisions_min_analysts` (int, 3 — guard against statistical noise)
   - `analyst_revisions_threshold` (float, 0.10 — boost/penalty zone)
   - `analyst_revisions_weight` (float, 0.15 — multiplier intensity)

2. **New module `backend/services/analyst_revisions.py`**:
   - `RevisionSignal` Pydantic model: `breadth, n_up, n_down, n_total, lookback_days, ticker`
   - `fetch_revision_signals(tickers: list[str], lookback_days: int = 100, min_analysts: int = 3) -> dict[str, RevisionSignal]`
     - per-ticker `yf.Ticker(t).upgrades_downgrades`, filtered to `Action in (up, down)` within the lookback
     - concurrency-capped async (Semaphore 4, throttled 0.3s per call) to bound runtime + 429 risk
     - returns empty dict on any unrecoverable error (default-OFF safety)
   - `apply_revisions_to_score(base_score, ticker, revision_signals, threshold=0.10, weight=0.15) -> float`
     - identity when no signal
     - applies `score *= (1 + breadth * weight)` only when `|breadth| > threshold`

3. **Edit `backend/tools/screener.py:rank_candidates`**:
   - Add `revision_signals=None` to signature (mirror `pead_signals`, `news_signals` pattern)
   - Insert apply block after `sector_events` block in per-stock loop

4. **Edit `backend/services/autonomous_loop.py`**:
   - When `settings.analyst_revisions_enabled` is True, fetch revisions for top-N candidates AFTER first-pass `screen_universe` (to bound cost to candidate set, not the full universe), pass to `rank_candidates(revision_signals=...)`.
   - Mirror the existing graceful-degradation pattern (try/except, log warning, continue).

5. **Run masterplan verification command** — must EXIT 0.

6. **Smoke test**:
   - `fetch_revision_signals(['AAPL','MSFT','NVDA','TSLA','GME'])` returns dict with at least one non-empty signal
   - `apply_revisions_to_score(1.0, 'AAPL', {...})` returns adjusted score
   - back-compat: `rank_candidates(screen_data, ...)` without `revision_signals=` works unchanged

7. **Write `experiment_results.md`** with verbatim outputs.

8. **Write `live_check_28.1.md`** showing N revisions-scored tickers + top-3 conviction shifts vs baseline.

9. **Spawn Q/A**.

10. **On PASS** — append harness_log Cycle entry, flip status.

## References

- `handoff/current/phase-28.1-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #1)
- `.claude/masterplan.json::phase-28.steps[1]`

## Risk / blast radius

- **Default OFF** — production behavior unchanged.
- **Back-compat** — `revision_signals=None` default in screener; all current callers unchanged.
- **Cost** — $0 LLM cost; yfinance only. Per-ticker network cost bounded by top-N (typically 10-30 tickers), not full S&P 500.
- **Graceful degradation** — yfinance failures yield empty signal dict; multiplier becomes identity; cycle continues.
