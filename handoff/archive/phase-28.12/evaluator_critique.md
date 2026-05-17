# Evaluator Critique -- phase-28.12 -- Sector-ETF momentum overlay

**Step ID:** phase-28.12
**Date:** 2026-05-17
**Cycle:** 1
**Evaluator:** Q/A subagent (Opus 4.7 xhigh, single-spawn)

---

## Verdict: PASS

All four immutable success criteria evidenced, deterministic checks pass,
live yfinance batch reproduces the reported 11-sector ranking verbatim,
6 of 6 settings defaults match, integration wiring is correct in screener
and autonomous_loop, graceful-degradation paths are real (try/except + `or
None` pass-through), sector_analysis.py duplicate left intentionally
untouched per the brief, default-OFF discipline holds. One NOTE on a
field-name mismatch between the runtime model (`momentum`) and the
documented artifact shape (`momentum_12m`) -- non-blocking; recorded for
the next cycle to fix doc-side.

---

## STEP 1: harness-compliance audit (5 items)

1. **Researcher gate before contract** -- PASS. `handoff/current/phase-28.12-research-brief.md` exists with `gate_passed: true`, 5 sources read in full (Quantpedia sector momentum rotational system, Quantpedia how-to-improve-etf-sector-momentum, Faber sector rotation ChartSchool, Alvarez Quant Trading, LuxAlgo), 13 URLs (5 read + 8 snippet), recency scan 2024-2026 documented, three-variant query discipline visible (current-year + last-2-year + year-less canonical).
2. **Contract before generate** -- PASS. `handoff/current/contract.md` describes the phase-28.12 sector-momentum overlay and carries the four immutable criteria verbatim.
3. **Results verbatim** -- PASS. `experiment_results.md` contains literal verification command output ("syntax OK / MASTERPLAN VERIFICATION: PASS") and the literal 11-row ranking table.
4. **Log-last not violated** -- PASS. `grep -E 'phase=28\.12' handoff/harness_log.md` returns no PASS line yet; Cycle 21 will be appended AFTER this verdict.
5. **No verdict-shopping** -- PASS. First Q/A spawn for phase-28.12 (no prior `phase=28.12` entries in harness_log.md).

---

## STEP 2: deterministic checks (verbatim output)

### Immutable masterplan verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/sector_momentum.py').read()); print('syntax OK')" && grep -q 'sector_momentum_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 4-file syntax check

```
OK: backend/services/sector_momentum.py
OK: backend/tools/screener.py
OK: backend/services/autonomous_loop.py
OK: backend/config/settings.py
```

### Settings defaults (6 fields)

```
$ python -c "from backend.config.settings import Settings; s=Settings(); print(s.sector_momentum_enabled, s.sector_momentum_lookback_months, s.sector_momentum_top_n, s.sector_momentum_boost_top, s.sector_momentum_boost_leader, s.sector_momentum_cache_hours)"
False 12 3 1.1 1.15 24
```

Matches spec verbatim: `sector_momentum_enabled=False` (default-OFF), lookback=12, top_n=3, boost_top=1.10, boost_leader=1.15, cache_hours=24.

### Module imports

```
$ python -c "from backend.services.sector_momentum import fetch_sector_momentum_ranks, apply_sector_momentum_to_score, RankedSector; print('OK')"
OK
```

### rank_candidates signature

```
$ python -c "from backend.tools.screener import rank_candidates; import inspect; print('sector_momentum_ranks' in inspect.signature(rank_candidates).parameters)"
True
```

### autonomous_loop wiring (fetch + pass-through)

```
$ grep -n 'sector_momentum' backend/services/autonomous_loop.py
248: sector_momentum_ranks = {}
249: if getattr(settings, "sector_momentum_enabled", False):
251: from backend.services.sector_momentum import fetch_sector_momentum_ranks
252: sector_momentum_ranks = await fetch_sector_momentum_ranks(...)
253-257: 5 kwarg pass-throughs (cache_hours, lookback_months, top_n, boost_top, boost_leader)
259: logger.info("sector_momentum ranks loaded: %d sectors", ...)
260-262: summary["sector_momentum_top"] surfaces top-3 sector names
264: logger.warning("sector_momentum fetch failed (non-fatal): %s", e)
320: sector_momentum_ranks=sector_momentum_ranks or None
```

Confirmed: pre-fetch is gated by `sector_momentum_enabled`, wrapped in `try/except` (logs warning, does not raise), and the call-site uses `or None` so an empty dict is passed as `None` -- which is the identity path in `apply_sector_momentum_to_score`. Graceful degradation is REAL: yfinance failure -> empty dict -> None pass-through -> no boost -> cycle continues.

### screener.py overlay block (lines 288-293)

```
288: # phase-28.12: sector-ETF momentum overlay (Quantpedia top-3 rotation).
289: # Boost candidates in top-N momentum sectors. Identity when ranks dict is None
290: # or sector missing/non-top.
291: if sector_momentum_ranks:
292:     from backend.services.sector_momentum import apply_sector_momentum_to_score
293:     score = apply_sector_momentum_to_score(score, stock.get("sector"), sector_momentum_ranks)
```

Placement is correct (AFTER analyst_revisions block at line 284, BEFORE `scored.append` at line 295). Order matches the contract.

### Live `fetch_sector_momentum_ranks()` -- real yfinance batch

```
Returned 11 sectors
  1 Technology               XLK   +51.43%  boost=1.15 <- TOP
  2 Energy                   XLE   +43.93%  boost=1.10 <- TOP
  3 Industrials              XLI   +23.55%  boost=1.10 <- TOP
  4 Materials                XLB   +20.25%  boost=1.00
  5 Communication Services   XLC   +16.71%  boost=1.00
  6 Health Care              XLV   +14.69%  boost=1.00
  7 Utilities                XLU   +13.79%  boost=1.00
  8 Real Estate              XLRE   +9.80%  boost=1.00
  9 Consumer Staples         XLP    +9.40%  boost=1.00
 10 Consumer Discretionary   XLY    +8.70%  boost=1.00
 11 Financials               XLF    +1.86%  boost=1.00

top-3 with boost > 1.0: 3/3
non-top at identity:    8/8
leader boost = 1.15
2nd-place boost = 1.10
3rd-place boost = 1.10
LIVE FETCH SMOKE: PASS
```

Q/A independently reproduced experiment_results.md's exact numbers down to two decimal places (Technology +51.43%, Energy +43.93%, Industrials +23.55%, Financials +1.86%). Top-3 vs leader differentiation (1.10 vs 1.15) is REAL and observable. Single-batch download confirmed (one `yf.download(...)` call for all 11 ETFs).

### apply_sector_momentum_to_score unit tests (9/9 PASS, non-tautological)

```
Tech (rank 1):        10 -> 11.5 OK (1.15x leader)
Health Care (rank 6): 10 -> 10.0 OK (identity)
Missing sector:       10 -> 10.0 OK (identity)
None sector:          10 -> 10.0 OK (identity)
Empty ranks:          10 -> 10.0 OK (identity)
None ranks:           10 -> 10.0 OK (identity)
Energy (rank 2):      10 -> 11.0 OK (1.10x top-3)
Industrials (rank 3): 10 -> 11.0 OK (1.10x top-3)
Mutation (boost=1.0): 10 -> 10.0 OK (proves boost field is read, not hard-coded)
```

Mutation test passes: synthetic ranks dict with `boost_multiplier=1.0` for Technology yields identity, proving the function actually consumes the field rather than hard-coding a multiplier.

---

## STEP 3: LLM judgment

### Contract alignment

All 4 immutable success criteria are evidenced:

| Criterion | Evidence | Result |
|---|---|---|
| `sector_momentum_module_created` | `backend/services/sector_momentum.py` exists (200 lines), syntactically valid, importable, `RankedSector` + `fetch_sector_momentum_ranks` + `apply_sector_momentum_to_score` all present | PASS |
| `top_3_sector_logic_documented` | Module docstring cites Quantpedia 13.94%/yr & Sharpe 0.54; `top_n=3` is parameterized default; settings field descriptions cite the same; experiment_results documents top-3 vs leader (1.10 vs 1.15) | PASS |
| `feature_flag_sector_momentum_enabled_default_false` | `Settings().sector_momentum_enabled == False` confirmed by Q/A independently | PASS |
| `live_check_lists_winning_sectors_and_boost_recipients` | `live_check_28.12.md` lists all 11 sector ranks, names the 3 winners (Tech / Energy / Industrials), and gives approximate ticker counts boosted per sector (~70 / ~22 / ~75 ~= 167 of ~500 S&P 500) | PASS |

### Default-OFF discipline

- `sector_momentum_enabled = False` is the default in `settings.py:225`.
- `autonomous_loop.py:249` gates the pre-fetch on the flag, so production behavior is unchanged until the operator flips it.
- `screener.py:291` overlay block only fires when `sector_momentum_ranks` is truthy.
- Existing callers of `rank_candidates` that don't pass the new kwarg get `None` default -> overlay is no-op. Back-compat preserved.

### Graceful degradation

Three layers:

1. yfinance import failure -> `return {}` (line 108).
2. yfinance batch download failure -> `return {}` (line 121).
3. Empty DataFrame -> `return {}` (line 125).

At the call site in autonomous_loop: `try` around the whole fetch with `except Exception` -> warning log + continue. Then `sector_momentum_ranks or None` ensures empty dict -> None at rank_candidates -> identity. The cycle survives yfinance flakiness, transient network errors, and import issues. This matches the brief's "Pitfalls" entry that NaN scores must be avoided.

### Sector naming consistency

`backend/services/sector_momentum.py:39-51` mirrors `backend/tools/screener.py::SECTOR_ETFS` exactly (canonical GICS labels: "Health Care", "Financials", "Communication Services"). The duplicate dict in `backend/tools/sector_analysis.py:13-25` (with "Healthcare", "Financial", etc.) was NOT touched -- `git log` confirms sector_analysis.py has no commit in the current cycle, and the module docstring explicitly calls out the discrepancy and points readers to the brief. This matches the contract's "Risk / blast radius" section.

### Single-batch yfinance call

`sector_momentum.py:113-118` -- one `yf.download(list_of_11_tickers, period=..., interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)` call inside `asyncio.to_thread`. Confirmed live: Q/A's smoke ran in under 4 seconds end-to-end, consistent with one HTTP round-trip for all 11 ETFs (not 11 separate calls).

### Top-3 vs leader boost differentiation

Live output shows Technology = 1.15 (leader) and Energy/Industrials = 1.10 (top-3-not-leader). The logic at lines 156-161 is the textbook two-tier branch:

```python
if rank == 1:
    mult = boost_leader  # 1.15
elif rank <= top_n:
    mult = boost_top     # 1.10
else:
    mult = 1.0
```

Differentiation is REAL and observable.

### Cache TTL (24h)

`sector_momentum_cache_hours=24` matches monthly-rebalance cadence per the brief's key finding #4 -- "Daily recalculation is overkill; weekly would drift. 24h file cache (same as `macro_regime.py` convention) is appropriate." Cache hit path at lines 95-102 short-circuits the yfinance call when fresh, with a logged "cache hit" line.

### Research-gate compliance in contract

Contract references `phase-28.12-research-brief.md` by name with the gate_passed flag, lists all 5 read-in-full sources, calls out the internal audit (screener.py:20-25 vs sector_analysis.py:13-25 mismatch), and grounds the 1.10/1.15 multipliers in the brief's key-finding #3. Contract-research linkage is clean.

### Anti-rubber-stamp / mutation resistance

Q/A's mutation test (synthetic ranks dict with `boost_multiplier=1.0` for Technology) confirms the function reads the field rather than hard-coding the multiplier -- if the field were ignored, the test would have returned 11.5 (the leader multiplier baked in) instead of 10.0.

### Scope honesty in experiment_results

The "N tickers boosted" line in live_check_28.12.md is honestly framed as an "approximate distribution" using "S&P 500 universe" plus a note that "Specific tickers depend on which pass the existing basic filters" -- not overclaimed as enumerated tickers. Top-3 sectors are named with real % returns from a real fetch. Cycle log block is labeled "canonical" (i.e. what it WOULD print when the flag is on), not falsely presented as captured live output. Honest disclosure throughout.

---

## NOTES (non-blocking, recorded for next cycle)

**N1 -- Field name doc/code drift.** The Pydantic model field is `momentum` (`sector_momentum.py:59`) but experiment_results.md "Artifact shape" example uses `momentum_12m` in two RankedSector constructions. The runtime code is the source of truth and is consistent across module + screener + autonomous_loop (none of those reference `momentum_12m`). Suggest a future doc-only cycle to align the artifact-shape example with the actual field name OR rename the field to `momentum_12m` (more descriptive). Not blocking PASS because: (1) it's documentation-only, (2) no code path depends on the wrong name, (3) the Q/A reproduction with the correct field name passes 9/9 tests including mutation.

**N2 -- xls cache header.** Module docstring is dense; consider a one-line example invocation in the next refactor pass. Pure style; no behavioral impact.

---

## STEP 4: Final verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_before_contract": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": {
    "immutable_verification_exit": 0,
    "syntax_4_files": "PASS",
    "settings_defaults_match_spec": "PASS (False 12 3 1.1 1.15 24)",
    "module_imports": "PASS",
    "rank_candidates_kwarg": "PASS",
    "autonomous_loop_fetch_plus_passthrough": "PASS",
    "live_fetch_11_sectors_reproduced": "PASS (Tech +51.43%, Energy +43.93%, Industrials +23.55%, Financials +1.86%)",
    "apply_unit_tests_9_of_9": "PASS",
    "mutation_test": "PASS (boost=1.0 in dict yields identity)",
    "sector_analysis_py_NOT_touched": "PASS (git log + git diff confirm)"
  },
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "syntax",
    "verification_command",
    "settings_defaults",
    "module_imports",
    "signature_check",
    "live_fetch_smoke",
    "apply_unit_tests_with_mutation",
    "screener_integration_inspection",
    "autonomous_loop_integration_inspection",
    "graceful_degradation_inspection",
    "code_review_heuristics",
    "evaluator_critique_overwrite"
  ]
}
```
