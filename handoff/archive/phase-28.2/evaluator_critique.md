# Evaluator Critique — phase-28.2 — 12-quarter SUE stacking in pead_signal.py

**Step ID:** phase-28.2
**Cycle:** 1
**Date:** 2026-05-17
**Evaluator:** Q/A subagent (merged qa-evaluator + harness-verifier), Opus 4.7 xhigh
**Verdict:** PASS

---

## STEP 1 — 5-item harness-compliance audit

| # | Check | Evidence | Result |
|---|---|---|---|
| 1 | Researcher gate before contract | `handoff/current/phase-28.2-research-brief.md` exists; envelope `{tier:simple, external_sources_read_in_full:5, urls_collected:14, recency_scan_performed:true, gate_passed:true}`. Three-variant queries documented. Sources hierarchy: 1 reference (Wikipedia), 1 academic textbook (iangow Bernard & Thomas), 1 doc/strategy (Quantpedia), 2 practitioner blogs (QuantConnect, Quantpedia NLP). | PASS |
| 2 | Contract before generate | `handoff/current/contract.md` written before code edits; contains immutable success criteria copied verbatim from `.claude/masterplan.json::phase-28.steps[2].verification.success_criteria` (4 criteria: lookback_quarters_increased_to_12, weighting_scheme_added_or_documented, back-compat_with_existing_cache_files, syntax_OK_and_pead_signal_still_importable). | PASS |
| 3 | Results verbatim | `experiment_results.md` contains literal `$ source .venv/bin/activate && grep ...` output (`PASS` + `MASTERPLAN VERIFICATION: PASS`) and verbatim synthetic-smoke output. | PASS |
| 4 | Log-last not violated | `grep "phase=28.2 result=PASS" handoff/harness_log.md` returns no hits (verified via `grep "phase-28\|phase=28" ... | tail -10` — last log entry is Cycle 16 phase=28.1). Log will be appended AFTER this PASS, BEFORE status flip. | PASS |
| 5 | No verdict-shopping | This is the FIRST Q/A spawn for phase-28.2. No prior critique to overturn. | PASS |

All 5 audit items PASS — proceeding to deterministic checks.

---

## STEP 2 — Deterministic checks (re-run by Q/A in this session)

### Immutable verification command (from masterplan.json)

```
$ source .venv/bin/activate && grep -qE '_LOOKBACK_QUARTERS\s*=\s*12' backend/services/pead_signal.py && python -c "import ast; ast.parse(open('backend/services/pead_signal.py').read()); print('PASS')"
PASS
```
Exit code: 0. **PASS.**

### Supplementary checks

| Check | Command | Output | Result |
|---|---|---|---|
| Module constant = 12 | `python -c "from backend.services.pead_signal import _LOOKBACK_QUARTERS; print(_LOOKBACK_QUARTERS)"` | `12` | PASS |
| Settings default = 12 | `python -c "from backend.config.settings import Settings; print(Settings().pead_signal_lookback_quarters)"` | `12` | PASS |
| Description updated to 12Q | `grep -nE 'rolling.{0,3}12Q\|rolling-12Q' backend/services/pead_signal.py` | `62:        description="sentiment_score - rolling-12Q mean (phase-28.2; was 8Q)...."` | PASS |
| phase-28.2 comment present | `grep -n 'phase-28.2' backend/services/pead_signal.py` | `38:# phase-28.2 (2026-05-17): ...`, `62:        description="...phase-28.2..."` | PASS |
| Synthetic 12-cache smoke | wrote 12 synthetic `pead_QATEST_<DATE>.json` files in tmp dir, patched `_CACHE_DIR`, called `_trailing_mean_from_cache("QATEST","2026-05-17")` | `trailing_mean = 0.5475, n_quarters = 12` (matches equal-weight expected) | PASS |

The synthetic smoke independently reproduces the 0.5475 mean reported in `experiment_results.md`, confirming `_trailing_mean_from_cache` correctly slices `[:12]` and uses equal-weight arithmetic mean.

### Frontend lint/typecheck gate
Not applicable — diff does NOT touch `frontend/**`.

---

## STEP 3 — Code-review heuristics (5 dimensions)

### Dimension 1 — Security audit
- secret-in-diff: no secrets in the 14-line diff
- prompt-injection-path: no LLM call signature changes
- command-injection / yaml-unsafe-load / pickle / supply-chain-dep-pin-removal: none touched
- excessive-agency / owasp-headers-bypass: no new tool, no new router

**Verdict: clean.**

### Dimension 2 — Trading-domain correctness
- `pead_signal.py` is NOT in the canonical risk-guard list (`kill_switch` / `risk_engine` / `paper_trader` / `perf_metrics`). The single execution path (`_trailing_mean_from_cache`) is unmodified — only the constant slice depth changed.
- `perf-metrics-bypass`: no Sharpe/drawdown/alpha math added; not applicable
- `stop-loss-always-set` / `kill-switch-reachability`: no execution-path change in trader code
- `position-sizing-div-zero`: not applicable
- `bq-schema-migration-safety`: no schema change
- `crypto-asset-class`: not touched
- Cache back-compat: filenames `pead_<TICKER>_<YYYY-MM-DD>.json` (line 77) don't encode lookback depth → existing cache files are read identically. Synthetic smoke independently verified.
- Production exposure: signal is still GATED by `pead_signal_enabled = False` default (settings.py:185). Production candidate picker behavior is unchanged when the flag is OFF, exactly as contract risk section discloses.

**Verdict: clean.**

### Dimension 3 — Code quality
- No `except: pass` / `print()` / global mutable state introduced
- Em-dash `—` appears in the multi-line Python COMMENT at lines 41-42. ASCII-logger rule (security.md) scope is `logger.*()` calls explicitly; comments are not at risk for cp1252 crashes (parser is UTF-8). No violation.
- Type hints unchanged on `_trailing_mean_from_cache` (existing signature preserved)

**Verdict: clean (no NOTE.)**

### Dimension 4 — Anti-rubber-stamp on financial logic
- `financial-logic-without-behavioral-test`: diff touches `pead_signal.py` (a service/signal module, not in the explicit canonical list of `perf_metrics.py`/`risk_engine.py`/`backtest_engine.py`/`backtest_trader.py`). The contract+results+live_check provide a behavioral demo (8Q vs 12Q delta on synthetic data), and Q/A independently re-ran the smoke in this session and produced `n_quarters == 12` and the exact expected mean 0.5475. Live behavioral evidence present.
- `tautological-assertion`: smoke uses `assert n == 12` (non-trivial) and `assert abs(mean - expected_mean) < 1e-9` (non-trivial). Not tautological.
- `rename-as-refactor`: pure constant bump + docstring/description sync — no rename
- `pass-on-all-criteria-no-evidence`: this critique cites file:line for every criterion
- `formula-drift-without-citation`: ScienceDirect 2025 paper cited verbatim in the in-source comment at lines 38-45 + settings.py:187 description

**Verdict: clean.**

### Dimension 5 — LLM-evaluator anti-patterns
- `sycophancy-under-rebuttal`: N/A (first cycle for this step)
- `second-opinion-shopping`: N/A (no prior critique)
- `missing-chain-of-thought`: this critique cites file:line throughout
- `3rd-conditional-not-escalated`: N/A (0 prior CONDITIONALs)
- `criteria-erosion` / `sycophantic-all-criteria-pass`: every criterion mapped to explicit evidence

**Verdict: clean.**

---

## Success criteria mapping (immutable, from masterplan.json)

| Criterion | Evidence | Result |
|---|---|---|
| `lookback_quarters_increased_to_12` | `_LOOKBACK_QUARTERS = 12` at `backend/services/pead_signal.py:46`; module import returns `12`; `Settings().pead_signal_lookback_quarters == 12` (settings.py:187) | PASS |
| `weighting_scheme_added_or_documented` | Multi-line phase-28.2 comment at `pead_signal.py:38-45` explicitly documents equal-weight (arithmetic mean) choice with ScienceDirect 2025 rationale; settings.py:187 description also says "equal-weighted mean" | PASS |
| `back-compat_with_existing_cache_files` | Cache filename pattern `pead_<TICKER>_<YYYY-MM-DD>.json` at `pead_signal.py:77` does NOT encode lookback depth. Q/A independently wrote 12 cache files in this exact format in a tmp dir and confirmed `_trailing_mean_from_cache` reads all of them and returns `n=12` with the expected equal-weight mean. | PASS |
| `syntax_OK_and_pead_signal_still_importable` | `python -c "import ast; ast.parse(...)"` exit 0; `from backend.services.pead_signal import _LOOKBACK_QUARTERS, PeadSignalOutput` succeeded; `PeadSignalOutput.model_fields` still includes all 6 fields per live_check_28.2.md line 77 | PASS |

All 4 immutable criteria PASS with citations.

---

## Live check artifact

`handoff/current/live_check_28.2.md` exists and conforms to the immutable shape:
> "one ticker's PEAD before/after with 8Q vs 12Q stack, surprise_score diff and resulting holding_window_days"

Includes synthetic TESTQ ticker with 12-quarter sentiment table, before/after means (0.6025 vs 0.5475, Δ=−0.055), surprise_score delta (+0.0550), and holding_window_days delta (28 → 42, +14 days longer hold). Cycle log canonical line example provided. Cache back-compat note included.

---

## Operator-impact disclosure (honesty check)

The contract explicitly discloses (line 73): "Default behavior change. Unlike 28.1 / 28.5 (feature-flagged OFF), this step CHANGES the default lookback." The `experiment_results.md` operator-impact note (line 108-110) clarifies that the PEAD signal itself is still GATED by `pead_signal_enabled=False` (settings.py:185), so production picker behavior is unchanged when that flag is OFF, and modestly shifted (typical Δ≈+0.055 in surprise_score) when ON. This is honest scope disclosure, not overclaim.

---

## Verdict

**PASS** — all 4 immutable criteria evidenced with file:line citations; deterministic immutable verification command exits 0; supplementary checks all pass; synthetic 12-cache smoke independently reproduced by Q/A; code-review heuristics clean across all 5 dimensions; honest operator-impact disclosure; first cycle (no verdict shopping); researcher gate cleared with 5 sources read in full + three-variant queries + recency scan.

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable success criteria met with file:line evidence: (1) _LOOKBACK_QUARTERS = 12 at pead_signal.py:46 + settings.py:187 sync, (2) equal-weight documented at pead_signal.py:38-45 + settings.py:187, (3) cache back-compat verified by Q/A-side synthetic smoke (12 files written in legacy format, all read correctly, n_quarters=12, expected mean 0.5475 reproduced), (4) syntax OK + import succeeds. All 5 harness-compliance items pass (research brief gate_passed:true with 5 sources; contract written before generate; verbatim outputs; log-last preserved; first Q/A spawn). Code-review heuristics clean across all 5 dimensions.",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": {
    "immutable_verification_command_exit": 0,
    "module_constant_value": 12,
    "settings_default_value": 12,
    "rolling_12Q_in_description": true,
    "phase_28_2_comment_present": true,
    "synthetic_smoke_n_quarters": 12,
    "synthetic_smoke_mean_matches_equal_weight": true
  },
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "syntax",
    "verification_command",
    "module_constant_value",
    "settings_default_value",
    "description_update_grep",
    "phase_marker_grep",
    "synthetic_cache_smoke",
    "diff_review",
    "code_review_heuristics",
    "live_check_artifact_review"
  ]
}
```
