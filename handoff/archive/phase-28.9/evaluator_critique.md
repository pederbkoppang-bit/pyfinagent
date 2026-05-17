# Evaluator Critique — phase-28.9 — Options-flow OI-surge filter

**Step ID:** phase-28.9
**Date:** 2026-05-17
**Cycle:** 1
**Q/A agent:** merged qa (deterministic + LLM judgment)

---

## Verdict: PASS

All 5 immutable success criteria evidenced. All deterministic checks passed.
Default-OFF discipline maintained. Graceful degradation paths verified.
Cost bounded to top 2*paper_screen_top_n candidates (~20 tickers per cycle).

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| 1. Researcher gate | PASS | `phase-28.9-research-brief.md` exists with `gate_passed: true`, 5 sources read in full (CXO Advisory, LuxAlgo, Market Rebellion, Management Science Apr 2026, OptionsTradingOrg), 15 URLs collected, recency scan present |
| 2. Contract pre-commit | PASS | `contract.md` written BEFORE generate; contains step ID, research-gate summary, all 5 immutable criteria verbatim + immutable verification command |
| 3. Results verbatim | PASS | `experiment_results.md` contains verbatim EXIT 0 of immutable verification command + live fetch output |
| 4. Log-last-then-flip | PRE-CONDITION OK | Masterplan status NOT yet flipped; harness_log.md append + status flip pending this PASS |
| 5. No verdict-shopping | PASS | First Q/A spawn for phase=28.9 (0 prior entries in harness_log.md); no prior CONDITIONAL to shop |

---

## Deterministic checks

### Immutable verification command (exit 0)
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/options_flow_screen.py').read()); from backend.services.options_flow_screen import fetch_oi_surge_signals; print('importable')" && grep -q 'options_flow_screen_enabled' backend/config/settings.py
importable
GREP_PASS
```
**Result:** EXIT 0. PASS.

### 4-file syntax (ast.parse)
```
SYNTAX_OK: backend/services/options_flow_screen.py
SYNTAX_OK: backend/tools/screener.py
SYNTAX_OK: backend/services/autonomous_loop.py
SYNTAX_OK: backend/config/settings.py
```

### Settings defaults
```
SETTINGS_OK: all 9 fields, defaults match contract, flag OFF
  options_flow_screen_enabled = False
  options_otm_threshold       = 1.01
  options_dte_min             = 2
  options_dte_max             = 45
  options_vol_avg_multiplier  = 5.0
  options_vol_oi_multiplier   = 3.0
  options_strong_boost        = 0.06
  options_moderate_boost      = 0.03
  options_cache_hours         = 4
```

### `rank_candidates` signature has `options_surge_signals` kwarg
```
RANK_KWARG_OK: options_surge_signals present in rank_candidates signature
SIG: rank_candidates(screen_data: list[dict], top_n: int = 10, strategy: str = 'momentum', ...,
                     pead_signals_lookup=None, options_surge_signals=None) -> list[dict]
```

### Back-compat: `rank_candidates` without new kwarg works
```
BACKCOMPAT_OK_POSITIONAL_EMPTY: list len= 0
BACKCOMPAT_OK_KWARG_NONE: list len= 0
```

### Apply unit-test (synthetic OptionsSurgeSignal)
```
APPLY_OK: 1.06*10 = 10.6; missing ticker identity; None signals identity; None ticker identity
```
- `boost_multiplier=1.06` * base 10.0 -> 10.6 (exact within 1e-9)
- Missing ticker -> identity (10.0)
- `signals=None` -> identity (10.0)
- `ticker=None` -> identity (10.0)

### Live `fetch_oi_surge_signals(['NVDA','AAPL'])`
```
LIVE_OK: type=dict, len=2, keys=['NVDA', 'AAPL']
EMPTY_OK: type=dict, len=0
```
Returns dict (per contract -- any result OK including empty). Empty input -> empty dict (no exception).

---

## LLM judgment

### Contract alignment
All 5 immutable criteria mapped 1:1 to evidence in experiment_results.md. The hypothesis
("Wayne State near-expiry OTM call surge filter at screener tier, mirror of phase-28.1
pattern") matches the implementation: 165-line module, kwarg on rank_candidates, pre-fetch
in autonomous_loop with 2*top_n bound.

### Wayne State predicate honored
At `options_flow_screen.py:113-117`:
```python
if strike < spot * otm_threshold:        # OTM: strike > spot * 1.01
    continue
...
if vol_avg_ratio >= vol_avg_mult and vol_oi_ratio >= vol_oi_mult:   # vol > max(5x avg, 3x OI)
    surges.append(...)
```
DTE window enforced at lines 92-94 (`if dte < dte_min or dte > dte_max: continue`).
Call-side only (`chain.calls`, not puts) at line 97.

### Default-OFF discipline
`Settings().options_flow_screen_enabled` defaults to `False`. Autonomous_loop guards with
`if getattr(settings, "options_flow_screen_enabled", False) and screen_data:` at line 305.
Production unchanged unless operator flips the flag.

### Graceful degradation
- yfinance ImportError -> `return None` at `options_flow_screen.py:72`
- Per-ticker fetch failure -> `logger.debug` + `return None` at line 81
- Per-expiration chain failure -> `logger.debug` + `continue` at line 99
- Outer autonomous_loop guard -> `logger.warning` + sets `options_surge_signals = {}` at line 328
- `apply_options_surge_to_score` is identity when signals empty or ticker missing

### Cost-bounding
Pre-fetch limited to `2 * settings.paper_screen_top_n` candidates at autonomous_loop.py:309
(typically ~20 tickers, not full S&P 500/Russell). Per-ticker Semaphore(4) + 0.3s throttle in
options_flow_screen.py:34-35,119. Per-ticker chain fetch capped at `expirations[:6]` at line 87.

### Honest disclosure of calibration observation
Experiment_results.md and live_check_28.9.md both DOCUMENT that 5/5 mega-caps were flagged
at default thresholds, propose tightening (`vol_avg_mult=8.0` or `vol_oi_mult=5.0`), and
explicitly justify NOT changing defaults because flag is OFF and operator A/B-tests before
flipping. This is the right discipline: surface the calibration concern, do not silently
tune defaults to make the smoke test look cleaner.

### Anti-rubber-stamp: behavioral test
Apply path exercised with synthetic OptionsSurgeSignal verifying numeric multiplication
(1.06 * 10.0 = 10.6) AND identity behavior on three missing-input paths (None signals,
None ticker, missing key). Not a tautology (`assert x == x` style); not a mock-and-assert-called.

### Code-review heuristics
- No secrets, no eval/exec/subprocess, no prompt-injection paths
- Kill-switch unaffected (screener tier, not execution tier)
- perf_metrics single-source preserved (no Sharpe/drawdown/alpha formulas added)
- Stop-loss / risk-guard paths untouched
- ASCII-only logger messages (`->`, `--`, no Unicode) -- Windows cp1252 defense honored
- No supply-chain pin removal
- No frontend changes (eslint/tsc not required this cycle)
- `try/except Exception` in `_fetch_one` is INSIDE the documented graceful-degradation
  contract (returns None -> empty dict -> identity boost), NOT inside risk-guard code path.
  Negation-list match: "intentional fallback in non-risk-guard code path is OK"

---

## Violated criteria

None.

---

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_pre_commit": "PASS",
    "results_verbatim": "PASS",
    "log_last_then_flip": "PRE_CONDITION_OK",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": {
    "immutable_verification_exit": 0,
    "syntax_4_files": "PASS",
    "settings_defaults": "PASS",
    "rank_candidates_kwarg_present": "PASS",
    "back_compat_no_new_kwarg": "PASS",
    "apply_unit_test_10x1_06_to_10_6": "PASS",
    "apply_missing_ticker_identity": "PASS",
    "live_fetch_returns_dict": "PASS",
    "empty_input_returns_empty_dict": "PASS"
  },
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_4_files",
    "settings_defaults",
    "signature_kwarg_present",
    "back_compat",
    "apply_unit_test",
    "live_yfinance_fetch",
    "empty_input_fallback",
    "code_review_heuristics",
    "contract_alignment",
    "default_off_discipline",
    "graceful_degradation",
    "cost_bounding",
    "honest_disclosure",
    "anti_rubber_stamp_behavioral_test",
    "wayne_state_predicate_honored"
  ]
}
```
