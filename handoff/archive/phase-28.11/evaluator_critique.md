# Evaluator Critique -- phase-28.11 -- LLM analyst-narrative signal (MVP proxy)

**Step ID:** phase-28.11
**Date:** 2026-05-17
**Cycle:** 1
**Q/A agent:** merged qa (deterministic + LLM judgment), Opus 4.7 xhigh

---

## Verdict: PASS

All 5 immutable success criteria evidenced. All deterministic checks passed.
Module name says "analyst" but the data source is explicitly disclosed as
"management 8-K Exhibit 99" in 13+ visible places (module docstring x3, Pydantic
default, prompt body, fetch_narrative_signals docstring, logger message, settings
field descriptions x3, autonomous_loop comment, contract, experiment_results,
live_check). The honesty-about-proxy is GENUINELY visible and not buried -- it
travels WITH the signal via the Pydantic `source_note` default. Default-OFF
discipline maintained. EDGAR + 8-K infrastructure REUSED from pead_signal (no
duplicate code paths). Cost-bounded to ~10 recent reporters (`2 * paper_screen_top_n`)
at ~$0.001/call, target <$0.10/cycle, soft cap in settings. Graceful degradation:
missing Anthropic key OR EDGAR fetch failure OR LLM call failure -> per-ticker
None -> outer try/except logs warning -> identity boost -> cycle continues.

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| 1. Researcher gate | PASS | `phase-28.11-research-brief.md` present with `gate_passed: true`, `external_sources_read_in_full: 5` (arXiv 2502.20489v1, IntuitionLabs LLM finance, XBRL.org sentiment, PMC Frontiers, arXiv 2502.16789v2 AlphaAgent), `urls_collected: 15`, `recency_scan_performed: true`. Three-variant query discipline visible (current-year, last-2-year, year-less canonical). |
| 2. Contract pre-commit | PASS | `contract.md` written BEFORE generate; cites brief filename, includes all 5 immutable criteria verbatim from masterplan, verbatim verification command, immutable live_check spec |
| 3. Results verbatim | PASS | `experiment_results.md` contains verbatim EXIT 0 output of immutable verification command (`syntax OK`, `MASTERPLAN VERIFICATION: PASS`) + synthetic 9-tier classifier sweep + identity-path table + honesty-marker assertion |
| 4. Log-last-then-flip | PRE_CONDITION_OK | Masterplan status NOT yet flipped; harness_log.md append + status flip pending this PASS |
| 5. No verdict-shopping | PASS | First Q/A spawn for phase=28.11 (`grep -c "phase=28.11" handoff/harness_log.md` returns 0); no prior verdict to shop, 3rd-CONDITIONAL counter reset on new step-id anyway |

---

## Deterministic checks

### Immutable verification command (exit 0)
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/analyst_narrative_scorer.py').read()); print('syntax OK')" && grep -q 'analyst_narrative_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
**Result:** EXIT 0. PASS.

### 4-file syntax (ast.parse)
```
SYNTAX_OK: backend/services/analyst_narrative_scorer.py
SYNTAX_OK: backend/tools/screener.py
SYNTAX_OK: backend/services/autonomous_loop.py
SYNTAX_OK: backend/config/settings.py
```

### Settings defaults (6 contracted + 1 derived field)
```
OK: analyst_narrative_enabled = False
OK: analyst_narrative_model = 'claude-haiku-4-5'
OK: analyst_narrative_cost_cap_usd = 0.1
OK: analyst_narrative_strong_threshold = 0.7
OK: analyst_narrative_strong_boost = 0.05
OK: analyst_narrative_moderate_boost = 0.025
OK: analyst_narrative_weak_threshold = 0.3  (used by _classify_boost + autonomous_loop)
SETTINGS_DEFAULTS: PASS  -- all match contract, flag defaults OFF
```

### Public surface importable
```
$ python -c "from backend.services.analyst_narrative_scorer import fetch_narrative_signals, apply_narrative_signal_to_score, AnalystNarrativeSignal; print('IMPORTS_OK')"
IMPORTS_OK
```

### `rank_candidates` signature has `narrative_signals` kwarg with `default=None`
```
RANK_KWARG: PASS -- narrative_signals present in rank_candidates
DEFAULT: None (back-compat preserved)
```

### Unit tests -- `_classify_boost` 5 tiers (9 score points)
```
OK: outlook_score=0.95 -> boost=1.050 tag=strongly_bullish
OK: outlook_score=0.75 -> boost=1.050 tag=strongly_bullish
OK: outlook_score=0.65 -> boost=1.025 tag=bullish
OK: outlook_score=0.55 -> boost=1.025 tag=bullish
OK: outlook_score=0.50 -> boost=1.000 tag=neutral
OK: outlook_score=0.45 -> boost=1.000 tag=neutral
OK: outlook_score=0.35 -> boost=0.975 tag=bearish
OK: outlook_score=0.25 -> boost=0.950 tag=strongly_bearish
OK: outlook_score=0.10 -> boost=0.950 tag=strongly_bearish
CLASSIFY_BOOST: PASS  -- symmetric 5-tier classifier (strongly_bullish/bullish/neutral/bearish/strongly_bearish)
```

### Unit tests -- `apply_narrative_signal_to_score` (6 identity / hit paths)
```
OK: AAPL apply (hit):    10.0 -> 10.50  (1.05 multiplier)
OK: aapl (case-insens):  10.0 -> 10.50  (ticker.upper() lookup works)
OK: MISSING ticker:      10.0 -> 10.00  (identity)
OK: empty dict:          10.0 -> 10.00  (identity)
OK: None signals:        10.0 -> 10.00  (identity)
OK: None ticker:         10.0 -> 10.00  (identity)
APPLY_IDENTITY: PASS
```

### Honesty-marker default in `AnalystNarrativeSignal` (built-in)
```
source_note default: 'management_8k_proxy: NOT canonical analyst_strategic_outlook'
HONESTY_DEFAULT_BUILT_IN: PASS -- "NOT canonical" + "proxy" both present
```

### `_build_prompt` PROXY honesty visible at the LLM-call site
```
prompt length: 918 chars
contains 'PROXY':      True
contains 'forward'/'guidance' language: True
PROMPT_HONESTY: PASS
```

### Harness log scan for prior phase-28.11 entries
```
$ grep -c "phase=28.11" handoff/harness_log.md
0
```
First Q/A spawn. No prior CONDITIONAL to escalate.

---

## LLM judgment

### Contract alignment (all 5 immutable criteria mapped 1:1)

| Criterion | Evidence | Result |
|---|---|---|
| `analyst_narrative_scorer_module_created` | `backend/services/analyst_narrative_scorer.py` (243 lines as read; `IMPORTS_OK` verified) | PASS |
| `data_source_decision_documented_paid_vs_EDGAR_vs_free` | Module docstring lines 1-24 explicitly state: paid Investext ($10K-$100K/yr) NOT viable -> 8-K Exhibit 99 free proxy via reused pead_signal infrastructure -> interface repointable when paid data available. Research brief item #1 confirms canonical signal requires Investext. Contract + experiment_results + live_check all echo. | PASS |
| `feature_flag_analyst_narrative_enabled_default_false` | `Settings().analyst_narrative_enabled == False` verified; autonomous_loop guards at line 305 with `getattr(settings, "analyst_narrative_enabled", False)`; both Pydantic default and getattr fallback are False | PASS |
| `cost_budget_per_cycle_documented` | `analyst_narrative_cost_cap_usd = 0.10` settings field + module docstring "~$0.001 per LLM call (Claude Haiku); per-cycle target <$0.10 (10 recent reporters analyzed)" + live_check Per-cycle LLM cost section | PASS |
| `live_check_includes_narrative_score_for_at_least_5_tickers` | `live_check_28.11.md` documents 9 distinct outlook_score values mapping to 5 distinct boost outcomes (covers >=5 narrative scores criterion) + apply identity paths + per-cycle cost + canonical cycle-log line shape | PASS |

### Honesty-about-canonical-vs-proxy (visible in 13+ places, not buried)

Counted occurrences of the proxy/NOT-canonical/management_8k disclosure:

1. **Module docstring** (`analyst_narrative_scorer.py:4-7`): "canonical 68bps/month signal ... requires Thomson Reuters Investext, a paid commercial feed ($10K-$100K/yr) -- not viable for this local-only deployment"
2. **Module docstring** (`analyst_narrative_scorer.py:9`): "This module is a **MVP PROXY**: it scores MANAGEMENT FORWARD-LOOKING TONE"
3. **Module docstring** (`analyst_narrative_scorer.py:19-20`): "When/if paid analyst-report data becomes available, this interface can be repointed without changing downstream callers"
4. **Pydantic default** (`analyst_narrative_scorer.py:58-61`): `source_note: str = Field(default="management_8k_proxy: NOT canonical analyst_strategic_outlook", description="Honesty marker -- data source is 8-K Exhibit 99 management text, not paid analyst reports.")`
5. **LLM prompt body** (`analyst_narrative_scorer.py:80-83`): "This is a PROXY for the canonical analyst Strategic Outlook signal (which needs paid data); focus on management's own forward language"
6. **fetch_narrative_signals docstring** (`analyst_narrative_scorer.py:204-205`): "HONESTY: this is a PROXY for the canonical analyst Strategic Outlook signal (which requires paid data)"
7. **Logger info line** (`analyst_narrative_scorer.py:225`): "MVP proxy via 8-K, not canonical analyst reports"
8. **Settings section comment** (`settings.py:258,261`): "phase-28.11: LLM analyst-narrative signal (MVP: management-outlook proxy from 8-K Exhibit 99)"
9. **Settings field description** (`settings.py:262`): "phase-28.11: LLM-scored management outlook tone from 8-K Exhibit 99 (MVP proxy for canonical analyst Strategic Outlook signal). Default OFF."
10. **autonomous_loop comment** (`autonomous_loop.py:301-303`): "management-outlook narrative overlay (MVP proxy for canonical analyst Strategic Outlook signal -- which needs paid data)"
11. **contract.md** line 15, 19, 21, 44, 60 -- 5 explicit disclosure points
12. **experiment_results.md** section "HONESTY UPFRONT" (lines 9-15) + criterion-mapping table
13. **live_check_28.11.md** section "HONESTY DISCLOSURE" (lines 10-12) + "Honesty marker baked into every signal" section (lines 61-74)

The disclosure travels WITH the signal via Pydantic default -- downstream consumers cannot construct an `AnalystNarrativeSignal` without the `source_note` being populated to a string containing "NOT canonical" + "proxy". This is the highest-leverage anti-misrepresentation guard. PASS.

### EDGAR infrastructure reuse (no duplicate code)

```python
# analyst_narrative_scorer.py imports
from backend.services.pead_signal import _fetch_exhibit_99_text, _fetch_recent_8k
from backend.tools.sec_insider import SEC_HEADERS, _resolve_cik
```

Three EDGAR helpers reused, zero re-implemented. Verified by `grep -n "_fetch_exhibit_99_text\|_fetch_recent_8k\|_resolve_cik\|SEC_HEADERS"` -- all four are imports + call-sites, no local redefinitions. PASS.

### Cost-bounding (documented + structurally enforced)

- Per-call cost: ~$0.001 (Claude Haiku 4.5, 256 max_output_tokens, ~600 input tokens after 8-K Exhibit 99 first 4000 chars + prompt scaffold)
- Per-cycle target: <$0.10 for ~10 recent reporters (those with a recent 8-K)
- Soft cap: `settings.analyst_narrative_cost_cap_usd = 0.10` (operator-monitored, not enforced as hard kill)
- Pre-fetch upper bound: `screen_data[: 2 * settings.paper_screen_top_n]` at `autonomous_loop.py:309` -- mirrors phase-28.10 insider-signal pattern exactly. Typical: 20 candidates, but many have no recent 8-K and return None early -> actual LLM call count much lower.
- Concurrency cap: `_CONCURRENCY = 3` semaphore (more conservative than yfinance-style services)

PASS -- cost is bounded structurally, not just documented.

### Graceful degradation (5 layers verified by code-read)

1. `_fetch_one_narrative:107-117` -- no Anthropic key -> `return None`
2. `_fetch_one_narrative:119-132` -- EDGAR `_resolve_cik` / `_fetch_recent_8k` / `_fetch_exhibit_99_text` failure -> outer try/except -> `return None`
3. `_fetch_one_narrative:134-143` -- `ClaudeClient` init failure -> `return None`
4. `_fetch_one_narrative:157-170` -- LLM call exception -> `return None`
5. `_fetch_one_narrative:172-179` -- JSON parse failure -> `return None`
6. `fetch_narrative_signals:220-223` -- per-ticker None is dropped from result dict
7. `autonomous_loop.py:325-326` -- outer try/except wraps the whole fetch_narrative_signals call; on Exception logs warning, narrative_signals stays `{}`, cycle continues uninterrupted
8. `apply_narrative_signal_to_score:237-242` -- 4 identity paths (None signals, empty dict, None ticker, missing ticker) -- all verified by tests above

PASS -- failure at any layer collapses cleanly to identity boost without killing the cycle.

### PEAD vs narrative signal-overlap honesty

Contract `risk` section: "Signal overlap with PEAD -- high correlation expected. Future cycle: add correlation analysis + decorrelation if needed."

Module docstring `analyst_narrative_scorer.py:13-17`:
```
- pead_signal:     sentiment_score vs trailing 12Q mean (surprise-vs-baseline)
- analyst_narrative: outlook_score from forward-looking language (guidance, strategy)
The two signals are likely correlated (same 8-K Exhibit 99 source) -- correlation
analysis required before joint deployment. Boost magnitude conservatively 50% of
PEAD scale pending live A/B validation.
```

PASS -- the correlation risk is acknowledged in 3 places (contract, module docstring, experiment_results) and conservative boost magnitudes (half PEAD's 0.10 strong / 0.05 moderate -> 0.05 strong / 0.025 moderate) pre-emptively de-risk pending A/B.

### Default-OFF discipline

- `Settings.analyst_narrative_enabled = False` (verified at runtime)
- `autonomous_loop.py:305`: `if getattr(settings, "analyst_narrative_enabled", False) and screen_data:` -- both Pydantic default and explicit getattr fallback are False
- `screener.py:325`: `if narrative_signals:` -- empty/None dict is a no-op
- All 7 settings fields default to safe values that produce zero signal change unless the flag is flipped

PASS.

### Anti-rubber-stamp: behavioral test

- `_classify_boost` test exercises EXACT tier-boundary values (0.75, 0.65, 0.55 immediately above; 0.50, 0.45 around neutral midpoint; 0.35 bearish; 0.25 strongly_bearish, 0.10 saturated) -- catches off-by-one in the inequality chain
- Identity-path test covers 4 distinct degenerate inputs (None signals, empty dict, None ticker, missing ticker) -- catches single-point-of-failure regressions
- Case-insensitive ticker lookup verified at `apply_narrative_signal_to_score:239` (`signals.get(ticker.upper())`)
- Honesty-marker test reads the actual Pydantic default and asserts the literal string content -- if a future commit shortened or weakened the disclosure, this assertion catches it

No `assert x == x`, no mock-and-assert-called, no over-mocking. The tests genuinely exercise the new financial-tier logic.

### Code-review heuristics (Dimensions 1-5)

- **secret-in-diff** [BLOCK]: NOT TRIGGERED -- no API keys/tokens/credentials in diff. Anthropic key sourced from `settings.anthropic_api_key` via `get_secret_value()` pattern at lines 107-117.
- **kill-switch-reachability** [BLOCK]: NOT TRIGGERED -- screener-tier change, kill_switch unaffected
- **stop-loss-always-set** [BLOCK]: NOT TRIGGERED -- no execution path changes
- **prompt-injection-path** [BLOCK]: NOT TRIGGERED -- LLM input is 8-K Exhibit 99 SEC-filed text, NOT user-supplied. SEC EDGAR is the authoritative source. The 4000-char truncation at the prompt boundary also limits any pathological inputs.
- **broad-except-silences-risk-guard** [BLOCK]: NOT TRIGGERED -- `try/except Exception` at lines 114, 130, 141, 168, 177 and `autonomous_loop:325` are INSIDE the documented graceful-degradation contract (returns None -> identity boost), NOT in risk-guard code path. Negation-list match.
- **financial-logic-without-behavioral-test** [BLOCK]: NOT TRIGGERED -- 9-point classifier sweep + 6-path apply test cover the new financial logic surface
- **tautological-assertion** [BLOCK]: NOT TRIGGERED -- tier-edge probing + honesty-string assertion are real behavioral checks
- **perf-metrics-bypass** [WARN]: NOT TRIGGERED -- module computes a score MULTIPLIER (`boost_multiplier`), not Sharpe/drawdown/alpha. `grep "perf_metrics\|sharpe\|drawdown\|alpha"` on the new module returns ZERO hits.
- **command-injection** [BLOCK]: NOT TRIGGERED -- no subprocess/eval/exec
- **excessive-agency-scope-creep** [WARN]: NOT TRIGGERED -- new LLM call, but it's read-only scoring (no writes, no actions). Cost-bounded ($0.10/cycle cap) and default-OFF.
- **position-sizing-div-zero** [WARN]: NOT TRIGGERED -- no vol divisor; multiplier-only
- **criteria-erosion** [WARN]: NOT TRIGGERED -- all 5 immutable criteria evaluated 1:1
- **sycophantic-all-criteria-pass** [WARN]: NOT TRIGGERED -- critique cites file:line for every claim (line counts >250)
- **supply-chain-dep-pin-removal** [WARN]: NOT TRIGGERED -- no dep manifest edits
- **unicode-in-logger** [NOTE]: NOT TRIGGERED -- all 6 logger calls (lines 109, 131, 142, 169, 178, 224) use ASCII-only format strings (`%s`, `%d/%d`, no Unicode)
- **frontend lint/typecheck**: NOT REQUIRED -- no `frontend/**` files in diff
- **sycophancy-under-rebuttal**: NOT TRIGGERED -- first Q/A spawn for this step-id, no prior verdict to flip
- **second-opinion-shopping**: NOT TRIGGERED -- first spawn (0 prior log entries)
- **3rd-conditional-not-escalated**: NOT TRIGGERED -- 0 prior CONDITIONALs

### Naming-tension observation (NOTE, not verdict-affecting)

The module is named `analyst_narrative_scorer.py` per the masterplan spec, but the data source is management 8-K Exhibit 99, not analyst reports. The team chose to honor the masterplan spec name and disclose the proxy nature in 13+ visible places rather than rename. This is the right tradeoff -- the immutable spec field is the masterplan's `criteria` list, and renaming the module would create churn for the masterplan filename, the contract, and the live_check spec without improving the user-visible honesty. The naming-tension is itself disclosed in contract.md line 21 ("Naming: the module is `analyst_narrative_scorer.py` per the masterplan spec name, but..."). NOTE-level only.

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
    "settings_defaults_7_fields": "PASS",
    "public_surface_importable": "PASS",
    "rank_candidates_kwarg_present": "PASS",
    "rank_candidates_default_None": "PASS",
    "classify_boost_5_tiers_9_points": "PASS",
    "apply_identity_6_paths": "PASS",
    "honesty_default_built_in_pydantic": "PASS",
    "prompt_honesty_PROXY_visible": "PASS",
    "harness_log_prior_phase_28_11_entries": 0
  },
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_4_files",
    "settings_defaults_7_fields",
    "public_surface_importable",
    "rank_candidates_signature",
    "rank_candidates_back_compat_None",
    "unit_tests_classify_boost_9_points_5_tiers",
    "unit_tests_apply_score_6_paths",
    "honesty_marker_pydantic_default",
    "honesty_marker_prompt_body",
    "honesty_marker_count_13_locations",
    "edgar_infrastructure_reuse_no_duplication",
    "cost_bounding_structural_2x_top_n",
    "graceful_degradation_5_layers_plus_outer",
    "default_off_discipline",
    "pead_overlap_honesty",
    "anti_rubber_stamp_behavioral_tier_edges",
    "code_review_heuristics_dimensions_1_through_5",
    "logger_ascii_only",
    "no_perf_metrics_bypass",
    "harness_log_prior_entries_scan"
  ]
}
```
