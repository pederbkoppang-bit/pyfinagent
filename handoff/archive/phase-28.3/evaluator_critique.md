# Evaluator Critique — phase-28.3 — GPR-triggered energy-sector tilt in macro_regime.py

**Step ID:** phase-28.3
**Cycle:** 1
**Date:** 2026-05-17
**Evaluator:** Q/A subagent (merged qa-evaluator + harness-verifier), Opus 4.7 xhigh
**Verdict:** PASS

---

## STEP 1 — 5-item harness-compliance audit

| # | Check | Evidence | Result |
|---|---|---|---|
| 1 | Researcher gate before contract | `handoff/current/phase-28.3-research-brief.md` exists; envelope `{tier:simple, external_sources_read_in_full:7, snippet_only_sources:10, urls_collected:17, recency_scan_performed:true, internal_files_inspected:1, gate_passed:true}`. Tier-1/2 hierarchy strong (peer-reviewed ERL ×2, PMC ×1, ECB, official policyuncertainty.com + Caldara author page, KPMG industry). Recency scan section present with 2024-2026 findings (Brent surge, CFA 2026 framework). | PASS |
| 2 | Contract before generate | `handoff/current/contract.md` written before code edits; contains immutable success criteria copied verbatim from `.claude/masterplan.json::phase-28.steps[3].verification.success_criteria` (4 criteria: gpr_index_fetcher_implemented_with_caching, sector_tilt_branch_added_to_macro_regime, threshold_documented_in_audit_basis, live_check_shows_XLE_overweight_when_gpr_above_threshold). | PASS |
| 3 | Results verbatim | `experiment_results.md` contains literal `$ source .venv/bin/activate && python ...` output incl. `syntax OK`, `MASTERPLAN VERIFICATION: PASS`, settings defaults dump, multi-ETF inject test, AND the xlrd install failure → fix → success path is honestly disclosed in Section 4 ("First live fetch failed... Fix: `pip install xlrd>=2.0.1`... After install, the live fetch succeeded"). | PASS |
| 4 | Log-last not violated | `grep "phase=28.3" handoff/harness_log.md` returns no hits — last entry is `Cycle 17 phase=28.2 result=PASS`. Log will be appended AFTER this PASS, BEFORE status flip. | PASS |
| 5 | No verdict-shopping | This is the FIRST Q/A spawn for phase-28.3. No prior critique to overturn. | PASS |

All 5 protocol items PASS. No anti-pattern.

---

## STEP 2 — Deterministic checks (verbatim)

### 2a. Immutable verification command (from contract.md and `.claude/masterplan.json`)

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'gpr|geopolitical' backend/services/macro_regime.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

Exit 0. PASS.

### 2b. settings.py syntax + requirements.txt sanity

```
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read()); print('settings.py syntax OK')" && wc -l backend/requirements.txt
settings.py syntax OK
      55 backend/requirements.txt
```

PASS — requirements.txt is 55 lines (xlrd at line 20 as documented).

### 2c. Settings defaults (must print `False 0.9 24 XLE`)

```
$ python -c "from backend.config.settings import Settings; s=Settings(); print(s.gpr_signal_enabled, s.gpr_signal_quantile, s.gpr_signal_cache_hours, s.gpr_signal_sector_etfs)"
False 0.9 24 XLE
```

Exact match. PASS.

### 2d. Helper importability

```
$ python -c "from backend.services.macro_regime import _fetch_gpr_acts, _apply_gpr_tilt; print('OK')"
OK
```

PASS.

### 2e. phase-28.3 markers in macro_regime.py

```
$ grep -n 'phase-28.3' backend/services/macro_regime.py
46:# phase-28.3: Caldara-Iacoviello GPR-Acts (geopolitical events) energy sector tilt
101:    """phase-28.3: Fetch latest GPR-Acts value from matteoiacoviello.com.
128:                    resp = await client.get(url, headers={"User-Agent": "PyFinAgent/2.0 (phase-28.3)"})
185:    """phase-28.3: When GPR-Acts is above the quantile threshold, inject configured
365:    # phase-28.3: Optional GPR-Acts post-process. When enabled AND latest GPRA exceeds
```

5 anchored comments. PASS — threshold_documented_in_audit_basis criterion met.

### 2f. Unit test of `_apply_gpr_tilt`

```
$ python -c "from backend.services.macro_regime import _apply_gpr_tilt, MacroRegimeOutput, SectorWeights; ..."
Test 1 (above_threshold=True, XLE): ['XLK', 'XLE'] -- PASS
Test 2 (above_threshold=False, identity): ['XLK'] -- PASS
Test 3 (multi-ETF): ['XLK', 'XLE', 'XOM', 'CVX'] -- PASS
Test 4 (dedup): ['XLK', 'XLE', 'XOM'] -- PASS
Test 5 (empty CSV identity): ['XLK'] -- PASS
All 5 unit tests PASSED
```

All five behavioral tests PASS:
- Above-threshold + single ETF: `['XLK']` → `['XLK','XLE']` ✓
- Below-threshold: identity → `['XLK']` unchanged ✓
- Multi-ETF inject: `['XLK','XLE','XOM','CVX']` ✓
- Dedup (XLE already present + add XOM): `['XLK','XLE','XOM']` not `['XLK','XLE','XLE','XOM']` ✓
- Empty CSV: identity → `['XLK']` ✓

Mutation-resistance: the dedup + identity tests would FAIL if `_apply_gpr_tilt` blindly appended or short-circuited on `above_threshold=False`. These are real behavioral tests, not tautological.

### 2g. Optional live `_fetch_gpr_acts()` (network call)

```
$ python -c "import asyncio; from backend.services.macro_regime import _fetch_gpr_acts; ..."
current=285.35
threshold=184.93
above_threshold=True
rolling_n=60
last_date=2026-04-01 00:00:00
PASS: live fetch returned valid dict
```

Live data matches what `experiment_results.md` and `live_check_28.3.md` report (285.35 / 184.93 / 2026-04-01). The cache at `backend/services/_cache/gpr/data_gpr_export.xls` is 2.7MB and fresh — second call hit cache as expected. `xlrd` 2.0.2 importable. PASS.

---

## STEP 3 — LLM judgment

### Contract alignment

All 4 immutable success criteria satisfied:

| Criterion | Evidence |
|---|---|
| `gpr_index_fetcher_implemented_with_caching` | `_fetch_gpr_acts()` at macro_regime.py:100-181 reads cache if age < `cache_hours`; downloads only when stale; primary + fallback URL pattern; logs bytes downloaded. Cache verified at `backend/services/_cache/gpr/data_gpr_export.xls` (2.7MB). |
| `sector_tilt_branch_added_to_macro_regime` | `_apply_gpr_tilt()` at macro_regime.py:184-200 + post-LLM hook at macro_regime.py:365-383 (inside `compute_macro_regime`). |
| `threshold_documented_in_audit_basis` | Multi-line comment macro_regime.py:46-55 cites Caldara-Iacoviello AER 2022 + IMF GFSR 2025 + explicit "calibrated practitioner heuristic" disclosure on the 90th-percentile choice. Settings descriptions (settings.py:210-213) all carry "phase-28.3" tags. |
| `live_check_shows_XLE_overweight_when_gpr_above_threshold` | `live_check_28.3.md` records GPR-Acts=285.35 > threshold=184.93 from real 2026-04-01 data; before/after `sector_hints.overweight` documented (`['XLK']` → `['XLK','XLE']`); multi-ETF mode also recorded. Reproduced verbatim by this Q/A in deterministic check 2g. |

### Default-OFF discipline

`gpr_signal_enabled` defaults to `False` (settings.py:210). The post-LLM hook at macro_regime.py:368 reads `getattr(settings, "gpr_signal_enabled", False)` — even if the field were missing from settings (back-compat), it would default to OFF. Production behavior unchanged unless an operator explicitly flips the flag. PASS.

### Graceful degradation

The post-LLM hook is wrapped in `try / except Exception` (macro_regime.py:369-383); any exception logs `warning("GPR tilt application failed (non-fatal): %s", e)` and the regime returned is unchanged. The `_fetch_gpr_acts()` function returns `None` on any of: cache-stat error, both URLs fail AND no cached file, pandas parse error, missing GPRA column, empty DataFrame. The `_apply_gpr_tilt()` is identity when `above_threshold` is False OR when the ETF CSV is empty. **No code path can cause `compute_macro_regime()` to fail because of GPR** — the existing FRED-regime LLM call result is preserved untouched on any GPR failure. This is the correct discipline for an opt-in external-dependency feature. PASS.

### Honest disclosure of mid-cycle xlrd dependency add

`experiment_results.md` Section 4 explicitly names the failure mode (`Import xlrd failed`), the fix (`pip install xlrd>=2.0.1`), and the persistence (`backend/requirements.txt line 20`). `live_check_28.3.md` repeats the disclosure. The "Known follow-ups" section in `experiment_results.md` notes the operator action required for fresh-venv environments. The added requirement carries a `# phase-28.3: parse matteoiacoviello.com GPR .xls (legacy Excel format)` inline comment in `backend/requirements.txt`. No silent dependency. PASS.

### Threshold rationale (90th-pct quantile) source-cited

Research brief Section "Key findings" item 7 explicitly states: "No published paper directly validates a 90th-percentile GPRA cutoff for energy overweighting, but the literature consistently supports extreme-event nonlinearity. Historical GPRA: baseline ~50-80 (peacetime), spikes to 150-400+ (Gulf War, 9/11, Ukraine). A 90th-percentile cutoff on rolling 5-year history (~120-150) captures genuine Acts-phase events without false positives." This is honest disclosure that the threshold is a practitioner calibration, not a peer-reviewed cutoff. The macro_regime.py comment at line 53-55 mirrors this honesty ("calibrated practitioner heuristic — no peer-reviewed paper validates the exact cutoff"). PASS — no overclaim.

### Post-LLM hook ordering — correctness

The hook fires at macro_regime.py:365-383, INSIDE `compute_macro_regime()`, AFTER:
- `parsed = MacroRegimeOutput.model_validate(raw)` (line 357 — LLM output validated)
- `parsed = parsed.model_copy(update={"series_used": available})` (line 363 — series_used backfilled)

and BEFORE:
- `_save_cache(parsed)` (line 385 — cache write)
- final `logger.info(...)` (line 387 — regime computed log)

Ordering implications:
1. **Non-disruptive to FRED-regime logic** — the LLM call has fully resolved before any GPR work. The LLM's regime tag, conviction, and conviction_multiplier are NOT modified. Only `sector_hints.overweight` is augmented (appended-to). FRED-driven sector_hints are preserved, GPR adds to them. ✓
2. **GPR result is cached** — the post-tilt `parsed` is what `_save_cache` persists. Cached MacroRegimeOutput already contains the GPR tilt result so a subsequent within-24h request returns the tilted result (consistent with `gpr_signal_cache_hours=24`). ✓
3. **Cache-bypass when disabled** — if `gpr_signal_enabled=False`, the hook never runs and the cached MacroRegimeOutput is identical to pre-phase-28.3 behavior. ✓
4. **`apply_regime_to_score` (line 393-417) unchanged** — the existing 1.05× / 0.95× multiplier on overweight/underweight ETF matches automatically picks up XLE. No change needed. ✓

PASS — the post-LLM hook ordering is correct.

### Code-review heuristics (5-dimension sweep)

| Dim | Heuristic | Finding |
|---|---|---|
| Security — secret-in-diff | grep on diff | No literal API keys/tokens. PASS |
| Security — command-injection | subprocess/eval/exec scan | None. PASS |
| Security — prompt-injection | new LLM-bound user input? | None (GPR data is numeric, not free-text). PASS |
| Trading — kill-switch-reachability | new execution path bypass? | N/A — this is a macro regime sector tilt, not an execution path. PASS |
| Trading — perf-metrics-bypass | inline Sharpe/drawdown? | None. PASS |
| Trading — crypto-asset-class | re-enable crypto? | No — only XLE/XOM/CVX/COP/OXY mentioned. PASS |
| Quality — broad-except | `except Exception` audit | 9 hits at lines 119/134/145/171/212/239/343/358/382. ALL are: (a) outside risk-guard paths (this is a sector tilt feature, not kill_switch / stop_loss / paper_trader execution), (b) log the error, (c) degrade to identity / None. This is the correct discipline for an opt-in external-dependency feature. NOTE only — pre-existing pattern in this file (e.g. cache load at line 212 predates phase-28.3). |
| Quality — print-statement | print() in non-test code | None. PASS |
| Anti-rubber-stamp — financial-logic-without-behavioral-test | `_apply_gpr_tilt` covered? | 5 behavioral tests in experiment_results Section 2 + verified live by this Q/A (above-threshold inject, below-threshold identity, multi-ETF, dedup, empty-CSV identity). PASS |
| Anti-rubber-stamp — tautological-assertion | `assert x == x` patterns? | All tests assert specific expected lists (`['XLK','XLE']` etc.). PASS |
| LLM-evaluator — sycophancy/verdict-shop | unchanged-evidence flip? | First Q/A spawn; no prior verdict to flip. PASS |

No BLOCK, no WARN. One NOTE on the pre-existing `except Exception` density in macro_regime.py — out of scope for phase-28.3 to refactor.

### Scope honesty

Researcher brief honestly disclosed:
- Mid-cycle xlrd dependency need (item 5 of "Pitfalls").
- 90th-percentile threshold is a practitioner heuristic, not a peer-reviewed cutoff.
- WTI vs Brent asymmetry caveat (US oil majors trade Brent-correlated despite US listing).
- Regime-dependent (not linear) response — supports threshold approach.

Experiment results honestly disclosed:
- The xlrd install was a mid-cycle remediation (Section 4).
- Multi-ETF mode tested but XLE-only is the default ("Single inject by default; multi-ETF available via comma-separated config").
- Real live GPRA (285.35) — not a synthetic stub.

No overclaim. PASS.

---

## STEP 4 — Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate_before_contract": true,
    "contract_before_generate": true,
    "results_verbatim": true,
    "log_last_not_violated": true,
    "no_verdict_shopping": true
  },
  "deterministic_checks": {
    "immutable_verification_cmd_exit": 0,
    "settings_py_syntax": "OK",
    "requirements_txt_lines": 55,
    "settings_defaults_exact": "False 0.9 24 XLE",
    "helpers_importable": true,
    "phase_28_3_markers_count": 5,
    "apply_gpr_tilt_unit_tests": "5/5 PASS",
    "live_fetch_current": 285.35,
    "live_fetch_threshold": 184.93,
    "live_fetch_above_threshold": true,
    "live_fetch_last_date": "2026-04-01 00:00:00",
    "xlrd_version_installed": "2.0.2",
    "gpr_cache_size_bytes": 2705408
  },
  "immutable_success_criteria_evidenced": {
    "gpr_index_fetcher_implemented_with_caching": true,
    "sector_tilt_branch_added_to_macro_regime": true,
    "threshold_documented_in_audit_basis": true,
    "live_check_shows_XLE_overweight_when_gpr_above_threshold": true
  },
  "default_off_preserved": true,
  "graceful_degradation_verified": true,
  "mid_cycle_xlrd_disclosed": true,
  "post_llm_hook_ordering_correct": true,
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item",
    "syntax",
    "verification_command",
    "settings_defaults",
    "helpers_importable",
    "phase_marker_grep",
    "apply_gpr_tilt_5_unit_tests",
    "live_fetch_smoke",
    "xlrd_dep_check",
    "code_review_heuristics",
    "evaluator_critique",
    "experiment_results",
    "live_check_artifact",
    "research_brief"
  ]
}
```

---

## Recommendation to Main

PASS. All 4 immutable success criteria evidenced, default-OFF preserved, mid-cycle xlrd add honestly disclosed and persisted in `requirements.txt` with a phase-28.3 comment. Live fetch reproduces the documented values (285.35 / 184.93 / above_threshold=True). Graceful degradation verified — any GPR-side failure becomes a `warning` log and identity behavior, never breaks the regime call.

Proceed: append `## Cycle 18 -- 2026-05-17 ... -- phase=28.3 result=PASS` block to `handoff/harness_log.md`, THEN flip `.claude/masterplan.json::phase-28.steps[3].status` to `done`.
