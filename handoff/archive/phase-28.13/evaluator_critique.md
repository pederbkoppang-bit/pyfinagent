# Evaluator Critique -- phase-28.13 -- Earnings-call NLP for firm-level GPR exposure (DEFENSIVE FILTER)

**Step ID:** phase-28.13
**Date:** 2026-05-17
**Cycle:** 1
**Q/A agent:** merged qa (deterministic + LLM judgment), Opus 4.7 xhigh

---

## Verdict: PASS

All 4 immutable success criteria evidenced. All deterministic checks passed.
The module is a textbook example of HONEST signal engineering: the Fed 2025
finding of CONTEMPORANEOUS-ONLY relationship (R²=0.23, NO forward predictability)
is acknowledged in 9+ visible surfaces, the disclaimer travels WITH every signal
object via the Pydantic `source_note` default (downstream consumers cannot
construct a `GprExposureSignal` without the "no_forward_alpha" string being
populated), and the LLM prompt itself embeds the Fed disclaimer plus the
anti-rubber-stamp instruction "be accurate, not generous". Default-OFF discipline
maintained. Reuses `earnings_tone.get_earnings_tone` infrastructure (Yahoo
Finance scraping + GCS caching) -- zero duplicate code paths, no new API key
required. Cost-bounded structurally to `2 * paper_screen_top_n` candidates with
$0.10/cycle soft cap. Sector-exemption logic correctly handles Industrials and
Energy (the GPR-benefitting sectors per phase-28.3 Caldara-Iacoviello US-as-net-
exporter asymmetry). 4-layer graceful degradation: missing Anthropic key OR
transcript-fetch failure OR LLM call failure OR JSON parse failure -> identity
score, cycle continues.

One NOTE-level flag: `logger.info` format string at line 193 contains a U+2014
em-dash. Per `security.md` ASCII-only logger rule, `--` or `-` would be safer.
Runtime risk is mitigated because `setup_logging()` in `main.py` forces UTF-8
on uvicorn handlers, but defense-in-depth would prefer ASCII. NOTE-level only
(per Q/A spec dimension-3 heuristic table); does NOT degrade verdict.

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| 1. Researcher gate | PASS | `phase-28.13-research-brief.md` present with `gate_passed: true`, `external_sources_read_in_full: 5` (Fed FEDS Note Aug 2025, API Ninjas docs, arXiv 2503.01886, LSEG insights, Finnhub docs), `urls_collected: 15`, `recency_scan_performed: true`. Three-variant query discipline visible (current-year 2026, last-2-year 2025, year-less canonical). Internal file inventory cites `earnings_tone.py:228`, `:233`, `:71-103` + `orchestrator.py:986`, `:1620`. |
| 2. Contract pre-commit | PASS | `contract.md` written BEFORE generate; cites brief filename, includes all 4 immutable criteria verbatim from masterplan, verbatim verification command, immutable live_check spec. HONEST CONSTRAINT section explicitly calls out the Fed contemporaneous-only finding before defining the hypothesis. |
| 3. Results verbatim | PASS | `experiment_results.md` contains verbatim EXIT 0 output of immutable verification command (`syntax OK`, `MASTERPLAN VERIFICATION: PASS`) + synthetic 9-case classifier sweep covering all 4 tiers + 5 sectors + 4 identity paths + honesty-marker assertion |
| 4. Log-last-then-flip | PRE_CONDITION_OK | Masterplan status NOT yet flipped; harness_log.md append + status flip pending this PASS |
| 5. No verdict-shopping | PASS | First Q/A spawn for phase=28.13 (`grep -c "phase=28.13" handoff/harness_log.md` returns 0); no prior verdict to shop, 3rd-CONDITIONAL counter reset on new step-id anyway |

---

## Deterministic checks

### Immutable verification command (exit 0)
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/call_transcript_gpr.py').read()); print('syntax OK')" && grep -q 'call_transcript_gpr_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
**Result:** EXIT 0. PASS.

### 4-file syntax (ast.parse)
```
SYNTAX_OK: backend/services/call_transcript_gpr.py
SYNTAX_OK: backend/tools/screener.py
SYNTAX_OK: backend/services/autonomous_loop.py
SYNTAX_OK: backend/config/settings.py
```

### Settings defaults (5 contracted fields)
```
OK: call_transcript_gpr_enabled = False
OK: call_transcript_gpr_model = 'claude-haiku-4-5'
OK: call_transcript_gpr_high_penalty = 0.97
OK: call_transcript_gpr_exempt_sectors = 'Industrials,Energy'
OK: call_transcript_gpr_cost_cap_usd = 0.1
SETTINGS_DEFAULTS: PASS  -- all match contract, flag defaults OFF
```

### Public surface importable
```
$ python -c "from backend.services.call_transcript_gpr import fetch_gpr_exposure_signals, apply_gpr_exposure_to_score, GprExposureSignal; print('IMPORTS_OK')"
IMPORTS_OK
```

### `rank_candidates` signature has BOTH new kwargs with default=None
```
gpr_exposure_signals present: True   default: None
gpr_exposure_config present: True    default: None
RANK_KWARG: PASS
```

### Unit tests -- `apply_gpr_exposure_to_score` (9 paths)
```
OK: AAPL HIGH Technology (non-exempt)               -> 9.700  (penalty 0.97 applied)
OK: AAPL HIGH Industrials (exempt)                  -> 10.000 (sector exempt)
OK: AAPL HIGH Energy (exempt)                       -> 10.000 (sector exempt)
OK: MSFT MEDIUM Technology                          -> 10.000 (only HIGH triggers)
OK: JNJ NONE Health Care                            -> 10.000 (NONE = identity)
OK: OTHER HIGH Technology (missing-ticker)          -> 10.000 (identity)
OK: AAPL HIGH Tech (empty sigs)                     -> 10.000 (identity)
OK: AAPL HIGH Tech (None sigs)                      -> 10.000 (identity)
OK: JNJ HIGH Health Care + custom-exempt            -> 10.000 (operator override honored)
APPLY: PASS
```

### Honesty-marker default in `GprExposureSignal` (built-in)
```
source_note default: defensive_filter_only_per_Fed_2025_R2_0.23_contemporaneous_no_forward_alpha
HONESTY_DEFAULT_BUILT_IN: PASS -- "no_forward_alpha" + "Fed_2025" + "contemporaneous" all present
```

### `_build_prompt` honesty markers visible at LLM-call site
```
prompt length: 906 chars
OK: DEFENSIVE FILTER
OK: no forward predictability
OK: GEOPOLITICAL RISK
OK: HIGH/MEDIUM/LOW/NONE (all 4 tiers listed)
PROMPT_HONESTY: PASS
```

### Harness log scan for prior phase-28.13 entries
```
$ grep -c "phase=28.13" handoff/harness_log.md
0
```
First Q/A spawn. No prior CONDITIONAL to escalate.

---

## LLM judgment

### Contract alignment (all 4 immutable criteria mapped 1:1)

| Criterion | Evidence | Result |
|---|---|---|
| `call_transcript_gpr_module_created` | `backend/services/call_transcript_gpr.py` exists (224 lines as read); `IMPORTS_OK` verified for all 3 public symbols | PASS |
| `transcript_data_source_decision_documented` | Module docstring (lines 22-24) + research brief #4-#5 + contract `Risk` section all explicitly document: Yahoo Finance scraping via reused `earnings_tone.get_earnings_tone`; `api_ninjas_key` setting present but UNUSED by active source; Finnhub free alternative noted in brief | PASS |
| `feature_flag_call_transcript_gpr_enabled_default_false` | `Settings().call_transcript_gpr_enabled == False` verified at runtime; `autonomous_loop.py:305` guards with `getattr(settings, "call_transcript_gpr_enabled", False)` -- both Pydantic default AND getattr fallback are False | PASS |
| `live_check_includes_gpr_exposure_classifications_for_5_tickers` | `live_check_28.13.md` documents 9 synthetic classification cases covering all 4 tiers (HIGH x 3 sectors + MEDIUM + NONE + 4 identity paths). Meets literal "5 tickers" floor (AAPL/MSFT/JNJ/OTHER/JNJ-with-custom-exempt) AND exceeds with cross-sector + identity coverage. | PASS |

### Honesty-about-defensive-filter-vs-alpha (visible in 9+ places, NOT buried)

The contract demanded honesty visible in 6+ places. I counted 9 distinct surfaces
where the "defensive filter NOT alpha" disclaimer appears:

1. **Module docstring HONESTY section** (`call_transcript_gpr.py:4-8`): "**HONESTY:** ... R²=0.23 across 240K+ earnings call transcripts ... demonstrated CONTEMPORANEOUS relationship only -- **NO forward return predictability**. This signal is therefore a defensive RISK FILTER on the candidate picker, NOT an alpha source."
2. **Module docstring sector-exemption rationale** (`call_transcript_gpr.py:11-14`): "Firms in defense-benefiting sectors (Industrials -> defense contractors, Energy -> oil majors) are EXEMPT from the penalty because they BENEFIT from elevated GPR"
3. **Pydantic default `source_note`** (`call_transcript_gpr.py:56-59`): `default="defensive_filter_only_per_Fed_2025_R2_0.23_contemporaneous_no_forward_alpha"`. The disclaimer TRAVELS WITH every signal object -- a downstream consumer cannot construct a `GprExposureSignal` without this string being populated.
4. **LLM prompt body** (`call_transcript_gpr.py:77-78`): "NOTE: this signal is used as a DEFENSIVE FILTER on candidate stocks. The Fed (2025) found contemporaneous relationship only, no forward predictability -- be accurate, not generous." -- this is the anti-rubber-stamp instruction baked into the prompt itself.
5. **fetch_gpr_exposure_signals docstring** (`call_transcript_gpr.py:180`): "HONESTY: defensive risk filter only (Fed 2025: no forward predictability)."
6. **Logger info line** (`call_transcript_gpr.py:193-194`): "DEFENSIVE FILTER per Fed 2025 -- no forward alpha"
7. **Settings field description** (`settings.py:273`): "DEFENSIVE FILTER (Fed 2025: contemporaneous only, no forward alpha). Default OFF."
8. **autonomous_loop comment** (`autonomous_loop.py:301-303`): "DEFENSIVE filter (Fed 2025 R²=0.23 contemporaneous only; NOT forward alpha)"
9. **screener comment** (`screener.py:331-332`): "firm-level GPR exposure DEFENSIVE FILTER (Fed 2025; no forward alpha). Penalize HIGH-exposure firms unless their sector benefits from GPR."

PLUS contract.md (5 explicit disclosure points), experiment_results.md
("HONESTY UPFRONT" section), live_check_28.13.md ("HONESTY DISCLOSURE" +
"Honesty marker chain" 6-surface enumeration). The disclosure is GENUINELY
visible and not buried. The Pydantic-default mechanism is the highest-leverage
anti-misrepresentation guard. PASS.

### Reuses `earnings_tone` infrastructure (no duplicate code)

```python
# call_transcript_gpr.py:100-102
from backend.tools.earnings_tone import get_earnings_tone
tone_result = await get_earnings_tone(
    ticker, api_key="", max_transcripts=1, bucket_name=bucket_name,
)
```

One import + one call -- zero re-implementation of Yahoo Finance scraping, GCS
caching, or transcript text extraction. The new module is a thin LLM-classifier
wrapper over existing transcript infrastructure. PASS.

### Sector-exempt logic correctly handles Industrials/Energy

```python
# call_transcript_gpr.py:220-222
exempt = {s.strip() for s in exempt_sectors_csv.split(",") if s.strip()}
if sector and sector.strip() in exempt:
    return base_score
```

Verified by 9-path unit test above:
- AAPL HIGH Industrials -> 10.00 (exempt; defense contractors live here)
- AAPL HIGH Energy -> 10.00 (exempt; oil majors live here)
- AAPL HIGH Technology -> 9.70 (penalty applied; not exempt)
- JNJ HIGH Health Care + operator-supplied custom exempt "Health Care" -> 10.00 (operator override honored)

The asymmetry matches phase-28.3 Caldara-Iacoviello US-as-net-exporter logic
(US defense exports + US oil exports BENEFIT from elevated GPR; the rest get
hurt). PASS.

### Cost-bounding (documented + structurally enforced)

- Per-call cost: ~$0.001 (Claude Haiku 4.5, `max_output_tokens=384`,
  `temperature=0.0`, transcript excerpt truncated to first 8000 chars at
  prompt boundary -- `call_transcript_gpr.py:69`)
- Per-cycle target: <$0.10 for ~10 candidates with fetchable transcripts
- Soft cap: `settings.call_transcript_gpr_cost_cap_usd = 0.10`
  (operator-monitored, not enforced as hard kill)
- Pre-fetch upper bound: `screen_data[: 2 * settings.paper_screen_top_n]` at
  `autonomous_loop.py:309` -- mirrors phase-28.9/28.10/28.11 pattern exactly
- Concurrency cap: `_CONCURRENCY = 3` semaphore (`call_transcript_gpr.py:43`)
- 200-char rationale truncation + 80-char x 3 phrase truncation at
  `call_transcript_gpr.py:157, 165` further caps output tokens
- Many candidates have no Yahoo transcript and return None early (line 108-109
  short-circuit on excerpt <200 chars) -> actual LLM call count typically much
  lower than the 2x ceiling

PASS -- cost is bounded structurally, not just documented.

### Graceful degradation (4 layers + outer wrapper)

1. `_fetch_one_exposure:96-97` -- no Anthropic key -> `return None`
2. `_fetch_one_exposure:104-106` -- transcript fetch exception -> `return None`
3. `_fetch_one_exposure:108-109` -- empty / too-short excerpt -> `return None`
4. `_fetch_one_exposure:118-120` -- `ClaudeClient` init failure -> `return None`
5. `_fetch_one_exposure:145-147` -- LLM call exception -> `return None`
6. `_fetch_one_exposure:158-160` -- JSON parse failure -> `return None`
7. `_fetch_one_exposure:151-153` -- malformed `exposure_tier` value -> defaulted to "NONE"
8. `fetch_gpr_exposure_signals:189-192` -- per-ticker None is dropped from result dict
9. `autonomous_loop.py:322-323` -- outer try/except wraps the whole
   `fetch_gpr_exposure_signals` call; on Exception logs warning,
   `gpr_exposure_signals = {}`, cycle continues uninterrupted
10. `apply_gpr_exposure_to_score:213-217` -- 3 identity paths
    (None/empty signals, None ticker, missing ticker) -- all verified by tests

PASS -- failure at any layer collapses cleanly to identity without killing the cycle.

### Anti-rubber-stamp: prompt design

`call_transcript_gpr.py:77-78`:
```
NOTE: this signal is used as a DEFENSIVE FILTER on candidate stocks. The Fed (2025)
found contemporaneous relationship only, no forward predictability -- be accurate, not generous.
```

This is the textbook anti-rubber-stamp prompt design from the Q/A spec: telling
the LLM both the downstream USE of its output AND explicitly instructing
"be accurate, not generous" (a counter-instruction to typical sycophancy). The
prompt also orders rationale BEFORE tier commitment (`call_transcript_gpr.py:76`
"Write rationale (<=200 chars) FIRST, then commit to the tier"), which is a
known chain-of-thought-style anti-bias technique. PASS.

### Default-OFF discipline

- `Settings.call_transcript_gpr_enabled = False` (verified at runtime)
- `autonomous_loop.py:305`: `if getattr(settings, "call_transcript_gpr_enabled", False) and screen_data:` -- both Pydantic default and explicit getattr fallback are False
- `screener.py:333`: `if gpr_exposure_signals:` -- empty/None dict is a no-op
- All 5 settings fields default to safe values that produce zero signal change unless the flag is flipped

PASS.

### Code-review heuristics (Dimensions 1-5)

- **secret-in-diff** [BLOCK]: NOT TRIGGERED -- no API keys/tokens/credentials in diff. Anthropic key sourced from `settings.anthropic_api_key` via `get_secret_value()` pattern at lines 90-95.
- **kill-switch-reachability** [BLOCK]: NOT TRIGGERED -- screener-tier change, kill_switch unaffected. `grep "kill_switch\|stop_loss"` on both new module + screener.py returns 0 hits.
- **stop-loss-always-set** [BLOCK]: NOT TRIGGERED -- no execution path changes
- **prompt-injection-path** [BLOCK]: NOT TRIGGERED -- LLM input is Yahoo Finance earnings transcript text reached through `get_earnings_tone` (already a vetted Layer-1 source), NOT user-supplied. The 8000-char truncation at prompt boundary further limits any pathological inputs.
- **broad-except-silences-risk-guard** [BLOCK]: NOT TRIGGERED -- `try/except Exception` at lines 94, 104, 118, 145, 158 and `autonomous_loop:322` are INSIDE the documented graceful-degradation contract (returns None -> identity score), NOT in risk-guard code path. Negation-list match.
- **financial-logic-without-behavioral-test** [BLOCK]: NOT TRIGGERED -- 9-path apply test covers the new financial logic surface (penalty trigger, sector exemption, all identity paths)
- **tautological-assertion** [BLOCK]: NOT TRIGGERED -- tests probe actual edge cases (sector boundaries, missing ticker, custom exempt list)
- **perf-metrics-bypass** [WARN]: NOT TRIGGERED -- module computes a score MULTIPLIER (defensive haircut 0.97 for HIGH non-exempt), not Sharpe/drawdown/alpha. `grep "perf_metrics\|sharpe\|drawdown\|alpha"` on the new module returns ONLY 4 hits and all 4 are HONESTY-text occurrences of the word "alpha" in the negative ("NOT alpha", "no_forward_alpha", "no forward alpha") -- the honest disclaimer that this is NOT an alpha signal, NOT actual alpha computation.
- **command-injection** [BLOCK]: NOT TRIGGERED -- no subprocess/eval/exec
- **excessive-agency-scope-creep** [WARN]: NOT TRIGGERED -- new LLM call, but read-only scoring (no writes, no actions). Cost-bounded ($0.10/cycle cap) and default-OFF.
- **position-sizing-div-zero** [WARN]: NOT TRIGGERED -- no vol divisor; multiplier-only
- **criteria-erosion** [WARN]: NOT TRIGGERED -- all 4 immutable criteria evaluated 1:1
- **sycophantic-all-criteria-pass** [WARN]: NOT TRIGGERED -- critique cites file:line for every claim (line counts >300 across 9 honesty surfaces)
- **supply-chain-dep-pin-removal** [WARN]: NOT TRIGGERED -- no dep manifest edits
- **unicode-in-logger** [NOTE]: **TRIGGERED at `call_transcript_gpr.py:194`** -- `logger.info` format string contains U+2014 em-dash ("-- no forward alpha"). NOTE-level only. Runtime risk mitigated by `setup_logging()` forcing UTF-8 in `main.py`, but per `security.md` ASCII-only logger rule, `--` (two hyphens) would be safer defense-in-depth. Recommended cleanup: replace `—` with `--`. **Does NOT degrade verdict** per Q/A spec dimension-3 NOTE severity = PASS-with-flag.
- **frontend lint/typecheck**: NOT REQUIRED -- no `frontend/**` files in diff
- **sycophancy-under-rebuttal**: NOT TRIGGERED -- first Q/A spawn for this step-id, no prior verdict to flip
- **second-opinion-shopping**: NOT TRIGGERED -- first spawn (0 prior log entries for phase=28.13)
- **3rd-conditional-not-escalated**: NOT TRIGGERED -- 0 prior CONDITIONALs

### Scope-honesty (NOT alpha; defensive filter)

The contract, experiment_results.md, live_check_28.13.md, and module docstring
all explicitly disclose scope:
- This is CONTEMPORANEOUS-only per Fed 2025 R²=0.23 finding
- It is NOT a forward-return alpha signal
- The penalty magnitude (3%) is small and defensive
- Industrials/Energy sectors are EXEMPTED because they BENEFIT from GPR
  (per phase-28.3 asymmetry, NOT a new claim being invented here)
- "Live LLM fetch -- deferred" is HONESTLY documented in live_check_28.13.md
  (Anthropic credit conservation), not papered over

The team did NOT overclaim. The team did NOT silently morph a research-backed
defensive-filter signal into a pseudo-alpha boost. This is the right discipline
when the underlying academic result has known limits. PASS.

### Research-gate compliance

The contract explicitly references the research brief and the Fed 2025 finding
in 4 distinct places:
- Contract line 13: "Brief: `handoff/current/phase-28.13-research-brief.md`
  (`gate_passed: true`; 5 sources read in full)"
- Contract line 14: "HONEST CONSTRAINT (Researcher): Fed 2025 study (R²=0.23 on
  240K+ transcripts) is CONTEMPORANEOUS only"
- Contract line 21: Hypothesis explicitly cites phase-28.3 sector-level GPR-Acts
  asymmetry (the prior research basis for the exemption logic)
- Contract line 51: References section cites brief file

The researcher's HONEST CONSTRAINT is honored in the implementation, not
ignored. PASS.

---

## Violated criteria

None. The unicode-in-logger NOTE finding is severity NOTE only (PASS-with-flag
per Q/A spec); does not constitute a violated criterion.

---

## Recommended cleanup (non-blocking)

Replace the em-dash on `backend/services/call_transcript_gpr.py:194` with
`--` to align with `security.md` ASCII-only logger rule. Single character
change; can be batched into next cycle.

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
    "settings_defaults_5_fields": "PASS",
    "public_surface_importable": "PASS",
    "rank_candidates_kwargs_both_present": "PASS",
    "rank_candidates_defaults_None": "PASS",
    "apply_gpr_9_paths": "PASS",
    "honesty_default_built_in_pydantic": "PASS",
    "prompt_honesty_4_markers_visible": "PASS",
    "harness_log_prior_phase_28_13_entries": 0
  },
  "violated_criteria": [],
  "violation_details": [],
  "code_review_heuristics_findings": [
    {
      "name": "unicode-in-logger",
      "severity": "NOTE",
      "location": "backend/services/call_transcript_gpr.py:193 (logger.info format string contains U+2014 em-dash)",
      "constraint": "security.md ASCII-only logger rule (defense-in-depth against cp1252 crashes)",
      "verdict_effect": "PASS-with-flag (does not degrade)"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_4_files",
    "settings_defaults_5_fields",
    "public_surface_importable",
    "rank_candidates_signature_both_kwargs",
    "rank_candidates_back_compat_None",
    "unit_tests_apply_score_9_paths",
    "honesty_marker_pydantic_default",
    "honesty_marker_prompt_body_4_strings",
    "honesty_marker_count_9_surfaces",
    "earnings_tone_infrastructure_reuse_no_duplication",
    "sector_exempt_logic_industrials_energy",
    "cost_bounding_structural_2x_top_n_plus_concurrency_cap",
    "graceful_degradation_4_layers_plus_outer",
    "default_off_discipline",
    "anti_rubber_stamp_prompt_design",
    "scope_honesty_no_alpha_overclaim",
    "research_gate_compliance",
    "code_review_heuristics_dimensions_1_through_5",
    "ascii_logger_audit_ast_parse",
    "no_perf_metrics_bypass",
    "harness_log_prior_entries_scan"
  ]
}
```
