# Contract — phase-28.12 — Sector-ETF momentum overlay

**Step ID:** phase-28.12
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.12-research-brief.md` (`gate_passed: true`; 5 sources read in full: Quantpedia sector momentum rotational system, Quantpedia how-to-improve-etf-sector-momentum, Faber sector rotation ChartSchool, Alvarez Quant Trading ETF sector rotation, LuxAlgo sector momentum rotation).
- Internal audit: `screener.py:20-25` has the 11-SPDR `SECTOR_ETFS` dict. `sector_analysis.py:13-25` has a duplicate map with slightly different key names ("Healthcare" vs "Health Care", "Financial" vs "Financials") — must unify on screener.py names. No existing module computes 12m sector momentum.

## Hypothesis

Quantpedia's sector momentum rotational system (top-3 SPDR sector ETFs by 12-month return, monthly rebalance) generated 13.94% annual / Sharpe 0.54 — +4%/yr over passive. Adding a small score multiplier (1.10× for top-3, 1.15× for #1) to stocks in winning sectors gives the picker exposure to sector-rotation alpha without requiring a sector-rotation backbone.

## Immutable success criteria (from `.claude/masterplan.json::phase-28.steps[12].verification.success_criteria`)

1. `sector_momentum_module_created`
2. `top_3_sector_logic_documented`
3. `feature_flag_sector_momentum_enabled_default_false`
4. `live_check_lists_winning_sectors_and_boost_recipients`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/sector_momentum.py').read()); print('syntax OK')" && grep -q 'sector_momentum_enabled' backend/config/settings.py
```

Immutable live_check shape:
> "live_check_28.12.md: cycle log showing 11 sector ETF 12-month momentum ranks + which 3 won + N tickers boosted"

## Plan steps

1. **Settings additions** (after sector_neutral block):
   - `sector_momentum_enabled: bool = Field(False, ...)`
   - `sector_momentum_lookback_months: int = Field(12, ...)`
   - `sector_momentum_top_n: int = Field(3, ...)`
   - `sector_momentum_boost_top: float = Field(1.10, ...)` (multiplier for top-3)
   - `sector_momentum_boost_leader: float = Field(1.15, ...)` (multiplier for #1)
   - `sector_momentum_cache_hours: int = Field(24, ...)`

2. **New module `backend/services/sector_momentum.py`**:
   - `RankedSector` Pydantic model: sector, etf, momentum_12m, rank, boost_multiplier
   - `async fetch_sector_momentum_ranks() -> dict[str, RankedSector]` mapping GICS sector → ranked entry
   - yfinance batch `yf.download(list_of_11_ETFs, period="1y", interval="1d", group_by="ticker")` (single API call)
   - Compute trailing 12-month total return for each, rank, assign boost multiplier (1.0 for non-top, top_boost for top-N, leader_boost for #1)
   - Cache to JSON for 24h
   - `apply_sector_momentum_to_score(score, sector, ranks)` returning boosted score (identity when sector missing or rank not in top-N)

3. **Edit `backend/tools/screener.py:rank_candidates`**:
   - Add `sector_momentum_ranks=None` kwarg
   - In per-stock loop, AFTER analyst_revisions block (line ~283), apply `apply_sector_momentum_to_score` when ranks present

4. **Edit `backend/services/autonomous_loop.py`**:
   - Pre-fetch ranks when flag is on; pass to rank_candidates. Add log line + summary field.

5. **Run masterplan verification** — must EXIT 0.

6. **Smoke test**:
   - Live fetch — verify 11 ETFs downloaded, ranks ordered, top-3 identified
   - Apply test: stocks in top-3 sectors get boost; non-top sectors unchanged
   - Back-compat: rank_candidates without new kwarg works

7. **Write experiment_results.md + live_check_28.12.md** with verbatim outputs.

8. **Spawn Q/A**.

9. **On PASS** — Cycle 21, flip status.

## References

- `handoff/current/phase-28.12-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #13)
- `.claude/masterplan.json::phase-28.steps[12]`

## Risk / blast radius

- **Default OFF** — `sector_momentum_enabled = False`.
- **Back-compat** — `sector_momentum_ranks=None` default in rank_candidates; existing callers unchanged.
- **Cost** — single yfinance batch (11 ETFs in one call). Zero LLM.
- **Graceful degradation** — yfinance failure → empty dict → no boosts. Cycle continues.
- **Sector name unification** — code uses screener.py keys; sector_analysis.py duplicate is NOT touched (it's a separate Layer-1 analysis path).
