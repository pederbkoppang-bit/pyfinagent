# Evaluator Critique — phase-28.6 — Crude-oil (CL=F) cross-asset trend signal

**Step ID:** phase-28.6
**Cycle:** 1
**Date:** 2026-05-17
**Evaluator:** Q/A subagent (Opus 4.7 xhigh, merged qa-evaluator + harness-verifier)
**Verdict:** **PASS**

---

## STEP 1 — 5-item harness-compliance audit

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Researcher gate before contract | PASS | `handoff/current/phase-28.6-research-brief.md` exists; ends with envelope `gate_passed: true`, `external_sources_read_in_full: 6`, `urls_collected: 13`, `recency_scan_performed: true`. Mtime is 21:55, BEFORE contract (21:56). |
| 2 | Contract before generate | PASS | `contract.md` mtime 21:56; `experiment_results.md` mtime 21:58. Contract written before results. Step ID, immutable criteria copied verbatim from masterplan, plan steps, references all present. |
| 3 | Results verbatim | PASS | `experiment_results.md` contains literal Bash output blocks: masterplan command exit-0 transcript, settings defaults dump, live `_fetch_crude_momentum()` output (+6.69% momentum, +0.137 z-score, above=False), `_apply_gpr_tilt` reuse trace. |
| 4 | Log-last not violated | PASS | `grep -nE "phase=28.6\|phase-28.6" handoff/harness_log.md` returned ZERO matches — no `phase=28.6 result=PASS` row exists yet. Append is correctly deferred until after this Q/A verdict. |
| 5 | No verdict-shopping | PASS | First Q/A spawn for phase-28.6. No prior `evaluator_critique.md` entries for 28.6 (file currently holds phase-28.3 content, being replaced). |

All 5 harness-compliance items PASS. No protocol breach.

---

## STEP 2 — Deterministic checks (verbatim output)

### 2a. Immutable verification command (from masterplan.json::phase-28.steps[6].verification.command)

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'CL=F|crude|brent|oil_trend' backend/services/macro_regime.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### 2b. Syntax check `backend/config/settings.py`

```
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read()); print('settings.py syntax OK')"
settings.py syntax OK
```
**PASS.**

### 2c. Settings new fields & defaults

```
$ python -c "from backend.config.settings import Settings; s=Settings(); print(s.crude_momentum_enabled, s.crude_momentum_window_days, s.crude_momentum_lookback_days, s.crude_momentum_zscore_threshold, s.crude_momentum_cache_hours, s.crude_momentum_sector_etfs)"
False 21 252 1.0 24 XLE
```
Expected: `False 21 252 1.0 24 XLE`. **Exact match. PASS.**

Direct read of settings.py:214-219 confirms 6 fields with `phase-28.6:` markers, descriptions citing 1.0 z-score = ~84th percentile under normal assumption, and 78% crude implied vol calibration rationale.

### 2d. Importability of new helper + reused helper

```
$ python -c "from backend.services.macro_regime import _fetch_crude_momentum, _apply_gpr_tilt; print('OK')"
OK
```
**PASS.**

### 2e. phase-28.6 markers in macro_regime.py

```
$ grep -n 'phase-28.6' backend/services/macro_regime.py
63:# phase-28.6: WTI crude (CL=F) 1-month momentum secondary trigger. Orthogonal to
201:    """phase-28.6: Fetch WTI crude (CL=F) 1-month momentum z-score.
291:    """phase-28.3 (reused by phase-28.6): When `gpr_info["above_threshold"]` is True,
493:    # phase-28.6: Optional WTI crude (CL=F) 1m-momentum post-process. Orthogonal to GPR.
```
Markers present at module header, helper docstring, reused-helper docstring update, and post-LLM hook block. **PASS.**

### 2f. CL=F ticker references

```
$ grep -n 'CL=F' backend/services/macro_regime.py
63:# phase-28.6: WTI crude (CL=F) 1-month momentum secondary trigger. Orthogonal to
201:    """phase-28.6: Fetch WTI crude (CL=F) 1-month momentum z-score.
204:        current_momentum: float -- trailing `window_days` percent change of CL=F close
236:            lambda: yf.download("CL=F", period=period, interval="1d",
240:        logger.warning("yfinance CL=F download failed: %s", e)
244:        logger.warning("yfinance CL=F returned empty")
252:            logger.warning("CL=F insufficient history: %d closes", len(close))
493:    # phase-28.6: Optional WTI crude (CL=F) 1m-momentum post-process. Orthogonal to GPR.
```
`CL=F` literal used in the yfinance call at line 236, plus surfaces in 3 logger.warning paths (download failure / empty result / insufficient history) — graceful degradation is real, not just claimed. **PASS.**

### 2g. Live `_fetch_crude_momentum()` against real WTI futures data

```
$ python -c "import asyncio; from backend.services.macro_regime import _fetch_crude_momentum
async def main():
    info = await _fetch_crude_momentum(cache_hours=0, window_days=21, lookback_days=252, zscore_threshold=1.0)
    print(f'current_momentum: {info[\"current_momentum\"]:+.4f}')
    print(f'zscore: {info[\"zscore\"]:+.3f}')
    print(f'above_threshold: {info[\"above_threshold\"]}')
    print(f'keys: {sorted(info.keys())}')
asyncio.run(main())"

current_momentum: +0.0669
zscore: +0.137
above_threshold: False
keys: ['above_threshold', 'current_momentum', 'last_date', 'lookback_days', 'mean', 'n_observations', 'std', 'threshold', 'window_days', 'zscore']
```

Numeric output **identical** to experiment_results.md (`+6.69%`, z-score `+0.137`, `above_threshold: False`). Dict shape covers the contract's required keys (`current_momentum`, `zscore`, `above_threshold`) plus diagnostic keys for the log line. **PASS.**

### 2h. Unit test on `_apply_gpr_tilt` reuse with synthetic crude_info dict

```
$ python -c "from backend.services.macro_regime import _apply_gpr_tilt, MacroRegimeOutput, SectorWeights
def make():
    return MacroRegimeOutput(
        rationale='test', regime='mixed', conviction=0.5, conviction_multiplier=1.0,
        sector_hints=SectorWeights(overweight=['XLK'], underweight=[]),
        series_used=['VIXCLS'], computed_at='2026-05-17T00:00:00Z',
    )
crude_above = {'above_threshold': True, 'zscore': 1.5, 'current_momentum': 0.10}
print('above_threshold=True overweight:', list(_apply_gpr_tilt(make(), crude_above, 'XLE').sector_hints.overweight))
crude_below = {'above_threshold': False, 'zscore': 0.13, 'current_momentum': 0.06}
print('above_threshold=False overweight:', list(_apply_gpr_tilt(make(), crude_below, 'XLE').sector_hints.overweight))"

above_threshold=True overweight: ['XLK', 'XLE']
above_threshold=False overweight: ['XLK']
```
- `above_threshold=True` → XLE appended after XLK, deduped, preserved order. Inject path works.
- `above_threshold=False` → identity. Below-threshold path works.

Confirms `_apply_gpr_tilt` is correctly generic over the trigger info dict. The reuse is legitimate, not a name collision — only the `above_threshold` key is consumed. **PASS.**

### 2i. Post-LLM hook ordering (GPR → crude → _save_cache)

Read of `backend/services/macro_regime.py` lines 470-516:
- Line 476: `if getattr(settings, "gpr_signal_enabled", False):` — GPR hook (phase-28.3)
- Line 496: `if getattr(settings, "crude_momentum_enabled", False):` — crude hook (phase-28.6)
- Line 516: `_save_cache(parsed)` — cache persist

**Order is correct: GPR fires first, crude fires second, both modify `parsed` before `_save_cache`.** Both hooks wrapped in `try/except Exception` with `logger.warning("... (non-fatal): %s", e)` — graceful degradation per the contract. **PASS.**

---

## STEP 3 — LLM judgment

### Contract alignment

| Immutable criterion (from masterplan.json) | Evidence | Verdict |
|---|---|---|
| `crude_oil_trend_signal_added_to_macro_regime` | New `_fetch_crude_momentum` helper at line 195; second post-LLM hook at line 496 in `compute_macro_regime` (gated on `settings.crude_momentum_enabled`); CL=F yfinance call at line 236. | PASS |
| `threshold_documented` | settings.py:218 description: "Z-score threshold above which the trigger fires. 1.0 = ~84th percentile under a normal assumption (calibrated for ~78% crude implied vol)." Research brief documents the 1.0 z-score choice over fixed +5% percent-change due to high implied vol regime. | PASS |
| `fallback_when_yfinance_unavailable_does_not_break_cycle` | `_fetch_crude_momentum` returns `None` on yfinance ImportError / empty DataFrame / insufficient history (logger.warning at lines 240, 244, 252). Post-LLM hook at line 504 (`if crude_info:`) skips tilt when None. Outer `try/except` at lines 497-514 catches any unhandled exception with `logger.warning("Crude momentum tilt application failed (non-fatal): %s", e)`. Identical degradation pattern to phase-28.3. | PASS |
| `live_check_shows_oil_trend_value_and_resulting_sector_action` | `live_check_28.6.md` (3424 bytes) shows real CL=F data: 1m momentum +6.69%, z-score +0.137, threshold 1.0, above_threshold=False, overweight ['XLK'] → ['XLK'] (identity). Plus a synthetic above_threshold=True trace showing the inject path. Plus a contrast table vs phase-28.3 showing concurrent state. | PASS |

All 4 immutable criteria evidenced. **Contract alignment: PASS.**

### Default-OFF discipline

`crude_momentum_enabled` defaults to `False` (verified via `Settings()` instantiation). Production behavior unchanged unless explicitly enabled. The post-LLM hook block at line 496 is fully guarded by the feature flag. **PASS.**

### Graceful degradation when yfinance fails

Three layered defenses verified by code-read:
1. Helper-internal: `_fetch_crude_momentum` wraps yfinance import + download + history check in three separate `try/except` / length-guard branches (lines 240, 244, 252), each returning `None` with a warning log.
2. Hook-level: `if crude_info:` (line 504) skips `_apply_gpr_tilt` when `None`.
3. Outer try/except: `except Exception as e: logger.warning(... non-fatal: %s, e)` at line 513-514 catches anything escaping the helper.

Identical 3-layer pattern to phase-28.3 (consistent with the cross-trigger pattern the contract claims to reuse). **PASS.**

### Honest scope disclosure (below-threshold)

The experiment_results explicitly notes (lines 53-65):
> "WTI crude (CL=F) is up +6.69% over the trailing 21 trading days. The trailing 252-day distribution has mean +4.77% and std 13.94% (very volatile), giving z-score +0.137 — **just slightly above the mean, far below the 1.0 trigger threshold**. The picker would NOT inject XLE via this trigger today. This is the OPPOSITE outcome of phase-28.3 (where GPR-Acts is well above its threshold), showing the two triggers are appropriately calibrated and behave independently."

This is exactly the honest disclosure pattern the Q/A spec demanded. The author:
- Quoted the real numeric value (+0.137).
- Stated the threshold (1.0) and the comparison result (False).
- Explicitly contrasted with phase-28.3's TRUE outcome.
- Did NOT overclaim that the trigger "is working" because it fires today — it correctly notes the trigger is working because it CORRECTLY does NOT fire today and the inject path is exercised via the synthetic test in section 4.

Author did the harder thing (validate both inject and identity paths separately) rather than the easier thing (claim the live below-threshold result alone proves both paths). **PASS.**

### Reuse of `_apply_gpr_tilt` is correct

Read of `_apply_gpr_tilt` body (lines 290-308): consumes only `gpr_info.get("above_threshold")`. Crude info dict supplies this key (verified at line 504 in `_fetch_crude_momentum` return value). Function is genuinely generic. Docstring update (line 291) explicitly calls out the dual-purpose use: "phase-28.3 (reused by phase-28.6)". No new injection code path was needed. **DRY discipline observed.** **PASS.**

### Post-LLM hook ordering

Confirmed in §2i above. GPR (line 476) → crude (line 496) → _save_cache (line 516). When both flags are enabled, GPR adds XLE first; crude tries to add XLE again, but `_apply_gpr_tilt`'s order-preserving dedupe at line 304-306 (`if e not in existing: existing.append(e)`) prevents double-add. The cache persists the combined tilt. **PASS.**

### Research-gate compliance

`phase-28.6-research-brief.md` envelope: `gate_passed: true`, 6 sources read in full (yfinance CL=F doc, RepEc 2024 paper, EIA STEO May 2026, yfinance BZ=F doc, stockanalysis.com XLE, Medium ETF rotation blog), 13 URLs collected, recency scan performed (2024-2026, 3 new findings reported). Three-variant search discipline followed: current-year frontier query (2025), last-2-year window (2022-2024), year-less canonical. Snippet-only table present (7 entries). All hard-blocker checklist items satisfied. Contract's "Research gate summary" section cites the brief by path. **PASS.**

---

## Code-review heuristics (5 dimensions)

| Dimension | Findings |
|---|---|
| **Security** | None. No new secrets, no LLM prompt path, no subprocess/eval, yfinance is a pinned dep. |
| **Trading-domain correctness** | None. No new kill-switch path, no stop-loss change, no perf-metrics formula. The macro_regime tilt is sector-overweight metadata, not order execution. |
| **Code quality** | NOTE only: helper has one broad `except Exception:` block at line 513-514, but this is the documented graceful-degradation pattern explicitly required by the contract criterion `fallback_when_yfinance_unavailable_does_not_break_cycle`. Inside a risk-guard path this would be a BLOCK; in a non-fatal post-process tilt it is the correct pattern. Mirrors phase-28.3's identical exception swallow. No findings. |
| **Anti-rubber-stamp on financial logic** | None. The added "logic" is a sector-tilt suggestion, not a return/Sharpe/sizing formula. Helper has been exercised with real CL=F data (live) AND a synthetic above_threshold=True path (unit test). Both paths verified. |
| **LLM-evaluator anti-patterns** | None. First Q/A spawn, single critique, no prior 28.6 verdict to flip. |

**Code-review heuristics: clean.**

---

## Mutation-resistance evidence

Author exercised both paths of `_apply_gpr_tilt` with synthetic crude_info dicts:
- `{above_threshold: True}` → XLE injected (mutation: inject path)
- `{above_threshold: False}` → identity (mutation: skip path)

This is real mutation testing of the new code path, not just confirmation of the happy path. Live yfinance call exercises the helper end-to-end (HTTP → DataFrame → percent_change → z-score → dict). **PASS.**

---

## STEP 4 — Verdict

**PASS** — all 4 immutable criteria evidenced with verbatim verification output. Below-threshold behavior faithfully captured (z=+0.137, threshold=1.0, no XLE injection today). Author honestly disclosed the trigger does NOT fire today, contrasted with phase-28.3 which DOES fire, and validated the inject path via synthetic test rather than overclaim. `_apply_gpr_tilt` reuse is correct — function is genuinely generic over `above_threshold`. Post-LLM hook ordering is correct (GPR → crude → save). Graceful-degradation pattern matches phase-28.3 (3-layer try/except + length guards). Research gate cleared (6 sources read in full, recency scan, three-variant search). Default OFF.

No violated criteria. No certified-fallback signal.

---

## JSON envelope (canonical Q/A return)

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
  "deterministic_checks": [
    "masterplan_immutable_command: exit 0 (syntax OK, grep matched)",
    "settings_py_syntax: PASS",
    "settings_defaults: False 21 252 1.0 24 XLE (exact match)",
    "helper_importable: _fetch_crude_momentum + _apply_gpr_tilt importable",
    "phase-28.6_markers: 4 lines (header / docstring / reused-helper docstring / hook)",
    "cl_f_ticker_references: 8 occurrences incl yfinance call + 3 logger.warning paths",
    "live_fetch_crude_momentum: current=+0.0669 zscore=+0.137 above=False (matches experiment_results verbatim)",
    "apply_gpr_tilt_reuse_unit_test: above=True->['XLK','XLE'], above=False->['XLK']",
    "post_llm_hook_ordering: line 476 (GPR) -> line 496 (crude) -> line 516 (_save_cache), both wrapped in try/except"
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 9
}
```
