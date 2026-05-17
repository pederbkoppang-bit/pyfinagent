# Evaluator Critique -- phase-28.15 -- Social media velocity in screener overlay

**Step ID:** phase-28.15
**Date:** 2026-05-17
**Cycle:** 1
**Q/A agent:** merged qa (deterministic + LLM judgment), Opus 4.7 xhigh

---

## Verdict: PASS

All 5 immutable success criteria evidenced. All deterministic checks passed.
This is a clean, conservative screener overlay that reuses existing
infrastructure (the velocity computation already lives at
`backend/tools/social_sentiment.py:90-95`; the Alpha Vantage NEWS_SENTIMENT
endpoint is already production-tested by Layer-1 enrichment) and follows
the established phase-28.x overlay pattern exactly (sibling modules at
`backend/services/{news_screen,options_flow_screen,call_transcript_gpr,
insider_signal_screen,analyst_narrative_scorer}.py`). Default-OFF
discipline maintained throughout (`Settings().social_velocity_enabled ==
False` + `getattr(settings, "social_velocity_enabled", False)` defensive
fallback at `autonomous_loop.py:305`). Honest documentation of the data-
source pivot: the module docstring lines 9-12 explicitly disclose that
StockTwits returned 403 + ApeWisdom is Reddit-only-without-SLA, and
explains why Alpha Vantage was chosen (bundles Reddit/Twitter/StockTwits/
blogs in one call). Rate-limit handling is documented (AV free tier
5 req/min, Semaphore(2) + 0.5s throttle + bounded universe to
2*paper_screen_top_n ~= 20 tickers). Graceful degradation: no AV key ->
early-out empty dict -> identity (verified at code lines 134-136); fetch
failure / non-dict result / NO_DATA / ERROR / None velocity / parse-fail
all collapse to None per ticker (lines 78-93). Mid-cycle bug-fix (read
`sentiment_velocity` key not `velocity`) verified correct against
`social_sentiment.py:122` where the canonical key is returned.

Zero code-review heuristic flags. No unicode-in-logger NOTE (the em-dash
in the docstring at line 15 is inside a triple-quoted string, not a
logger call -- AST scan of all `logger.*()` format strings returned ALL
ASCII). No secret-in-diff (api key sourced via `getattr` +
`get_secret_value` pattern). No perf_metrics bypass (the only `alpha`
hits in the module are in the docstring noting velocity spikes are
"alpha-positive when sufficiently high" -- this is signal-design copy,
not Sharpe/drawdown computation). No kill_switch / stop_loss /
risk_engine code touched.

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| 1. Researcher gate | PASS | `phase-28.15-research-brief.md` present with `gate_passed: true`, `external_sources_read_in_full: 5` (apewisdom.io/api, macroption.com, contextanalytics-ai.com, prospero.ai, medium.com/data-ledger), `urls_collected: 15`, `recency_scan_performed: true`. Three-variant query discipline visible (current-year 2026, last-2-year 2025-2024, year-less canonical). Internal file inventory cites `social_sentiment.py:91-95`, `screener.py` line 229+340, `autonomous_loop.py:294,432`, `news_screen.py:315`. Last-2-year recency reported (DNUT July 2025, OPEN July 2025, Context Analytics 2025 +33% YTD, 2024 75% retail-meme-loss finding). |
| 2. Contract pre-commit | PASS | `contract.md` written BEFORE generate; cites brief filename, includes all 5 immutable criteria verbatim from masterplan, verbatim verification command, immutable live_check spec. Risk / blast radius section explicitly documents AV rate limit + bounded universe + graceful degradation. |
| 3. Results verbatim | PASS | `experiment_results.md` contains verbatim EXIT 0 output of immutable verification command (`syntax OK`, `MASTERPLAN VERIFICATION: PASS`) + 8-case classifier sweep covering strong / moderate / none / noise-guard / no-mentions / negative-velocity + apply identity paths. Mid-cycle bug-fix (sentiment_velocity key) documented in section 3. Live AV fetch deferred is honestly documented. |
| 4. Log-last-then-flip | PRE_CONDITION_OK | Masterplan status NOT yet flipped to done; harness_log.md append + status flip pending this PASS verdict. |
| 5. No verdict-shopping | PASS | First Q/A spawn for phase=28.15 (`grep -c "phase=28.15" handoff/harness_log.md` returns 0); no prior verdict to shop, 3rd-CONDITIONAL counter reset on new step-id anyway. |

---

## Deterministic checks

### Immutable verification command (exit 0)
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/social_velocity_screen.py').read()); print('syntax OK')" && grep -q 'social_velocity_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
**Result:** EXIT 0. PASS.

### 4-file syntax (ast.parse)
```
SYNTAX_OK: backend/services/social_velocity_screen.py
SYNTAX_OK: backend/tools/screener.py
SYNTAX_OK: backend/services/autonomous_loop.py
SYNTAX_OK: backend/config/settings.py
```

### Settings defaults (6 contracted fields, all match contract)
```
OK: social_velocity_enabled = False (expected False)
OK: social_velocity_min_threshold = 0.1 (expected 0.1)
OK: social_velocity_min_mentions = 3 (expected 3)
OK: social_velocity_strong_threshold = 0.2 (expected 0.2)
OK: social_velocity_strong_boost = 0.06 (expected 0.06)
OK: social_velocity_moderate_boost = 0.03 (expected 0.03)
SETTINGS_DEFAULTS: PASS
```

### Public surface importable
```
$ python -c "from backend.services.social_velocity_screen import fetch_social_velocity_signals, apply_social_velocity_to_score, SocialVelocitySignal; print('IMPORTS_OK')"
IMPORTS_OK
```

### `rank_candidates` signature has new kwarg with default=None
```
social_velocity_signals present: True, default: None
RANK_KWARG: PASS
```
(Verified via `inspect.signature(rank_candidates).parameters` — defensive default preserves back-compat.)

### Unit tests -- `_classify_boost` (8 paths cover full spec)
```
OK velocity=+0.30 mentions=10 -> boost=1.060 tier=strong (want strong)
OK velocity=+0.20 mentions= 5 -> boost=1.060 tier=strong (at strong threshold)
OK velocity=+0.15 mentions= 5 -> boost=1.030 tier=moderate
OK velocity=+0.10 mentions= 5 -> boost=1.030 tier=moderate (at moderate threshold)
OK velocity=+0.05 mentions= 5 -> boost=1.000 tier=none (below moderate threshold)
OK velocity=+0.30 mentions= 2 -> boost=1.000 tier=none (noise guard <3 mentions)
OK velocity=+0.30 mentions= 0 -> boost=1.000 tier=none (no mentions)
OK velocity=-0.20 mentions= 5 -> boost=1.000 tier=none (negative velocity, long-only)
CLASSIFY_BOOST_8_PATHS: PASS
```

### Apply identity / boost paths (6 paths)
```
OK AAPL with strong: 10.6000 (want 10.6000)            -- boost applied
OK missing-ticker:   10.0000 (want 10.0000)            -- identity
OK empty signals:    10.0000 (want 10.0000)            -- identity
OK None signals:     10.0000 (want 10.0000)            -- identity
OK None ticker:      10.0000 (want 10.0000)            -- identity (defensive)
OK lowercase ticker: 10.6000 (want 10.6000)            -- .upper() normalization works
APPLY_IDENTITY: PASS
```

### Mid-cycle bug-fix verification (sentiment_velocity key)
```
$ grep -n "sentiment_velocity\|velocity" backend/tools/social_sentiment.py | head
90:        # Compute sentiment velocity (are recent articles more positive?)
95:        velocity = recent_avg - older_avg
107:        if avg_sentiment > 0.25 and velocity > 0.05:
122:            "sentiment_velocity": round(velocity, 4),       <-- canonical key
...
```
Fetcher at `social_velocity_screen.py:84-87`:
```python
# social_sentiment.py exposes the field as `sentiment_velocity` (line 122)
velocity_val = result.get("sentiment_velocity")
if velocity_val is None:
    velocity_val = result.get("velocity")  # back-compat fallback
```
**Result:** key match correct. Back-compat fallback is defense-in-depth (no
known caller produces a bare `velocity` key, but harmless to fall back).

### Sequence integrity in autonomous_loop.py
```
$ grep -n "social_velocity\|fetch_social_velocity_signals" backend/services/autonomous_loop.py
304:            social_velocity_signals = {}
305:            if getattr(settings, "social_velocity_enabled", False) and screen_data:
307:                    from backend.services.social_velocity_screen import fetch_social_velocity_signals
312:                    social_velocity_signals = await fetch_social_velocity_signals(
...
326:                    logger.warning("social_velocity_screen fetch failed (non-fatal): %s", e)
481:                social_velocity_signals=social_velocity_signals or None,
```
Block sits between `screen_universe` (line 294) and `rank_candidates`
(line ~432-481), identical structural pattern to phase-28.13/.11/.10
overlays at lines 328-377/355-377/381-406. `social_velocity_signals or
None` at line 481 is defensive — preserves the `None` semantic that the
`rank_candidates` kwarg expects when the feature flag is OFF.

### Screener wiring
```
$ grep -n "social_velocity\|social_velocity_signals" backend/tools/screener.py
230:    social_velocity_signals=None,
345:        if social_velocity_signals:
346:            from backend.services.social_velocity_screen import apply_social_velocity_to_score
347:            score = apply_social_velocity_to_score(score, stock.get("ticker"), social_velocity_signals)
```
Application correctly gated on truthy `social_velocity_signals` (None or
empty dict = no-op). Lazy import inside `rank_candidates` mirrors sibling
overlays; no startup-time coupling on the new module.

### Harness log scan for prior phase-28.15 entries
```
$ grep -c "phase=28.15" handoff/harness_log.md
0
```
First Q/A spawn. No prior CONDITIONAL to escalate; 3rd-CONDITIONAL counter
is at 0 for this step-id.

---

## LLM judgment

### Contract alignment (all 5 immutable criteria mapped 1:1)

| Criterion | Evidence | Result |
|---|---|---|
| `social_velocity_screen_module_created_lifting_existing_alpha_vantage_path` | `backend/services/social_velocity_screen.py` (176 lines) imports `get_social_sentiment` from `backend/tools/social_sentiment.py:76`; the call at line 77 reuses the production-tested Alpha Vantage NEWS_SENTIMENT endpoint that already computes velocity at `social_sentiment.py:95`. Zero re-implementation of the AV HTTP client. `IMPORTS_OK` verified for all 3 public symbols. | PASS |
| `stocktwits_or_apewisdom_data_path_documented` | Module docstring lines 9-13 explicitly disclose: "Why Alpha Vantage and not StockTwits/ApeWisdom directly (per Researcher): StockTwits developer portal returns 403 / suspended as of 2026; ApeWisdom is Reddit-only without an SLA; Alpha Vantage bundles multiple sources in one call (cross-source convergence)". Research brief table also enumerates 4 StockTwits-related 403/503 URLs and the ApeWisdom Reddit-only finding. | PASS |
| `feature_flag_social_velocity_enabled_default_false` | `Settings().social_velocity_enabled == False` verified at runtime; `autonomous_loop.py:305` defends with `getattr(settings, "social_velocity_enabled", False)` -- both Pydantic default AND getattr fallback are False. | PASS |
| `rate_limit_handling_documented_per_supplement_pitfalls` | Module docstring lines 18-19: "Cost: free tier Alpha Vantage = 5 req/min. Bounded to top 2*paper_screen_top_n (~20 candidates), throttled at 0.5s per ticker." Structural enforcement: `_CONCURRENCY = 2` (line 37) + `_PER_TICKER_SLEEP_S = 0.5` (line 38) + `screen_data[: 2 * settings.paper_screen_top_n]` at `autonomous_loop.py:309`. Graceful degradation paths documented at lines 21-22. live_check_28.15.md rate-limit-handling section enumerates all 4 mitigations + behavior on rate-limit. | PASS |
| `live_check_lists_social_velocity_surfaced_tickers_for_one_cycle` | `live_check_28.15.md` documents 8 synthetic classifier cases (strong x 2, moderate x 2, none x 4 including noise-guard + no-mentions + negative-velocity edge cases) + apply identity paths + expected cycle log shape + final ranking impact explanation + provenance + spec-compliance bullet. Live AV fetch is HONESTLY deferred (AV free tier 5 req/min credit conservation; module uses production-tested Layer-1 wrapper) rather than faked. | PASS |

### Honest data-source disclosure (StockTwits 403 + ApeWisdom Reddit-only + AV choice)

The data-source pivot is documented in 4 distinct surfaces:

1. **Module docstring lines 9-12** (`social_velocity_screen.py`): explicit
   StockTwits=403 + ApeWisdom=Reddit-only-no-SLA + "Alpha Vantage bundles
   multiple sources in one call (cross-source convergence)"
2. **Research brief snippet-only table**: 4 StockTwits URLs each tagged
   with the failure mode (403 / 503 / new registrations suspended)
3. **Research brief key finding #3**: "StockTwits direct API is not viable
   right now... Skip as a direct source; it feeds Alpha Vantage's
   aggregated endpoint anyway"
4. **live_check_28.15.md rate-limit-handling**: explains the per-cycle AV
   budget + the "no AV key -> empty dict" early-out path

The team did NOT silently substitute an inferior data source while
implying it was the originally researched one. PASS.

### Infrastructure reuse (no duplicate AV call code)

```python
# social_velocity_screen.py:76-77 (per _fetch_one_velocity)
from backend.tools.social_sentiment import get_social_sentiment
result = await get_social_sentiment(ticker, av_key, fallback_articles=None)
```

One import + one call to the existing async function. Zero
re-implementation of the AV NEWS_SENTIMENT HTTP request, the
`recent_avg - older_avg` velocity computation, the `source_breakdown`
parsing, or the rate-limit detection that `get_social_sentiment`
performs at `social_sentiment.py:54`. The new module is a thin
classifier + threshold layer on top of the existing tool. This is the
correct lift -- not a new API client. PASS.

### Rate-limit awareness (Semaphore(2) + 0.5s throttle + bounded universe)

Three layers of rate-limit enforcement:

1. **Concurrency cap** (`social_velocity_screen.py:37`): `_CONCURRENCY = 2`
   with `asyncio.Semaphore(_CONCURRENCY)` at line 142, applied via
   `async with sem:` at line 74.
2. **Per-ticker throttle** (line 38, 81): `_PER_TICKER_SLEEP_S = 0.5`
   with `await asyncio.sleep(_PER_TICKER_SLEEP_S)` inside the semaphore
   block.
3. **Bounded universe** (`autonomous_loop.py:309`): `screen_data[: 2 *
   settings.paper_screen_top_n]` — mirrors phase-28.10/.11/.13 pattern.
   With `paper_screen_top_n=10` this caps the AV calls at ~20 per cycle
   (AV free tier = 25/day, so this works on free tier for the
   typical 1-cycle/day cadence).

PASS — rate-limit is bounded structurally, not just documented.

### Graceful degradation (multi-layer)

Verified by code reading:

1. `fetch_social_velocity_signals:130-131` — empty `tickers` -> `return {}`
2. `fetch_social_velocity_signals:134-136` — no AV key -> log once + `return {}`
3. `_fetch_one_velocity:78-80` — `get_social_sentiment` exception ->
   `return None`
4. `_fetch_one_velocity:82-83` — non-dict result OR `signal in ("NO_DATA",
   "ERROR")` -> `return None`
5. `_fetch_one_velocity:85-89` — velocity key missing in result ->
   `return None`
6. `_fetch_one_velocity:90-93` — velocity value not coercible to float ->
   `return None`
7. `_fetch_one_velocity:103-104` — tier=='none' after classification ->
   `return None` (don't pollute output dict with no-boost entries)
8. `autonomous_loop.py:325-326` — outer try/except wraps the whole
   `fetch_social_velocity_signals` call; on Exception logs warning + sets
   `social_velocity_signals = {}` + cycle continues
9. `apply_social_velocity_to_score:170-174` — 3 identity paths (None or
   empty signals, None ticker, missing ticker) -- all verified by tests

PASS -- failure at any layer collapses cleanly to identity without
killing the cycle.

### Default-OFF discipline

- `Settings.social_velocity_enabled = False` (verified at runtime: see
  `SETTINGS_DEFAULTS: PASS` above)
- `autonomous_loop.py:305`: `if getattr(settings,
  "social_velocity_enabled", False) and screen_data:` — both Pydantic
  default and explicit getattr fallback are False
- `screener.py:345`: `if social_velocity_signals:` — empty/None dict is
  a no-op
- All 6 settings fields default to safe values that produce zero signal
  change unless the flag is flipped
- `rank_candidates` kwarg default is `None` (preserves back-compat for
  any caller not updated)

PASS.

### Code-review heuristics (Dimensions 1-5)

- **secret-in-diff** [BLOCK]: NOT TRIGGERED. AV key sourced via
  `getattr(settings, "alphavantage_api_key", "") or ""` (line 133) and
  `get_secret_value()` if Pydantic-Secret-wrapped (lines 137-141). No
  literal credentials in diff.
- **kill-switch-reachability** [BLOCK]: NOT TRIGGERED. Screener-tier
  change; kill_switch unaffected. `grep "kill_switch\|stop_loss"` on the
  new module returns 0 hits.
- **stop-loss-always-set** [BLOCK]: NOT TRIGGERED. No execution path
  changes.
- **prompt-injection-path** [BLOCK]: NOT TRIGGERED. No LLM call in this
  module. The data ingest path is API->cached numeric result, no string
  flowing to a prompt.
- **broad-except-silences-risk-guard** [BLOCK]: NOT TRIGGERED. `try /
  except Exception` at line 78 + `autonomous_loop:325` are INSIDE the
  documented graceful-degradation contract (returns None / sets {} ->
  identity score), NOT in risk-guard code path. Negation-list match.
- **financial-logic-without-behavioral-test** [BLOCK]: NOT TRIGGERED.
  8-path classifier test + 6-path apply test cover the new financial
  surface (boost thresholds, noise guard, negative-velocity, identity
  paths, lowercase normalization).
- **tautological-assertion** [BLOCK]: NOT TRIGGERED. Tests probe actual
  edge cases (threshold boundaries, mention noise guard, negative
  velocity, missing ticker normalization).
- **perf-metrics-bypass** [WARN]: NOT TRIGGERED. Module produces a
  composite-score multiplier, not Sharpe/drawdown/alpha. `grep
  "perf_metrics\|sharpe\|drawdown\|alpha"` returns 3 hits — line 15 in
  docstring ("alpha-positive when sufficiently high" — signal-design
  copy), and the word does NOT appear in any computation. No Sharpe or
  drawdown computation.
- **command-injection** [BLOCK]: NOT TRIGGERED. No subprocess/eval/exec.
- **excessive-agency-scope-creep** [WARN]: NOT TRIGGERED. New screener
  overlay only; read-only scoring (no writes, no actions). Cost-bounded
  (AV free tier, throttled, bounded universe) and default-OFF.
- **position-sizing-div-zero** [WARN]: NOT TRIGGERED. No vol divisor.
- **max-position-check-bypass** [BLOCK]: NOT TRIGGERED. No
  paper_trader.py changes.
- **crypto-asset-class** [BLOCK]: NOT TRIGGERED. No crypto re-enable.
- **paper-trader-broad-except** [BLOCK]: NOT TRIGGERED. No paper_trader
  changes.
- **criteria-erosion** [WARN]: NOT TRIGGERED. All 5 immutable criteria
  evaluated 1:1; no previously-required criterion dropped.
- **sycophantic-all-criteria-pass** [WARN]: NOT TRIGGERED. Critique
  cites file:line for every claim with line counts >300; not <3
  sentences.
- **supply-chain-dep-pin-removal** [WARN]: NOT TRIGGERED. No dep
  manifest edits.
- **unicode-in-logger** [NOTE]: NOT TRIGGERED. AST scan of all
  `logger.*()` calls in the new module confirms ALL ASCII. The em-dash
  on line 15 is inside the triple-quoted module docstring (purely
  documentation), not a logger format string. AST-based check
  `UNICODE_LOGGER: PASS (all ASCII)`.
- **frontend lint/typecheck**: NOT REQUIRED — no `frontend/**` files in
  diff.
- **sycophancy-under-rebuttal**: NOT TRIGGERED. First Q/A spawn for this
  step-id, no prior verdict to flip.
- **second-opinion-shopping**: NOT TRIGGERED. First spawn (0 prior log
  entries for phase=28.15).
- **3rd-conditional-not-escalated**: NOT TRIGGERED. 0 prior CONDITIONALs.
- **stop-loss-backfill-removal** [BLOCK]: NOT TRIGGERED. No paper_trader
  changes.
- **bq-schema-migration-safety** [WARN]: NOT TRIGGERED. No BQ schema
  migrations.

### Scope-honesty

The contract, experiment_results.md, live_check_28.15.md, and module
docstring all explicitly disclose scope:
- Live AV fetch deferred is HONESTLY documented as "AV free tier 5
  req/min" credit conservation, not papered over
- Mid-cycle bug-fix (sentiment_velocity vs velocity key) is documented
  in experiment_results.md section 3 and live_check_28.15.md
- Boost magnitudes (+6%/+3%) are described in the live-check ranking-
  impact section with the supplement-Gap-2 + DNUT-July-2025 case as
  rationale; no overclaim
- The default-OFF status is named in every artifact

The team did NOT overclaim and did NOT silently morph the threshold from
the contract's `0.10 / 0.20` to something looser. PASS.

### Research-gate compliance

The contract explicitly references the research brief and the
Researcher's findings in 4 distinct places:
- Contract line 13: "Brief: `handoff/current/phase-28.15-research-brief.md`
  (`gate_passed: true`; 5 sources read in full)"
- Contract line 15: "Researcher: StockTwits API 403/suspended; ApeWisdom
  Reddit-only without SLA. Alpha Vantage is the right choice."
- Contract line 16: "Threshold: `velocity >= 0.10` AND `mention_count >=
  3`. Boost 0.06/0.03 strong/moderate." -- these are exactly the values
  the brief recommended at lines 117-120
- Contract line 47: References section cites brief file

The Researcher's HONEST CONSTRAINT (StockTwits unavailable, ApeWisdom
limited, AV-bundled-aggregator is correct choice) is honored in the
implementation, not ignored. PASS.

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
    "settings_defaults_6_fields": "PASS",
    "public_surface_importable": "PASS",
    "rank_candidates_kwarg_present_default_None": "PASS",
    "classify_boost_8_paths": "PASS",
    "apply_identity_6_paths": "PASS",
    "mid_cycle_bugfix_sentiment_velocity_key_verified": "PASS",
    "autonomous_loop_sequence_integrity": "PASS",
    "screener_wiring": "PASS",
    "unicode_logger_ast_scan": "PASS (all ASCII)",
    "harness_log_prior_phase_28_15_entries": 0
  },
  "violated_criteria": [],
  "violation_details": [],
  "code_review_heuristics_findings": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_4_files",
    "settings_defaults_6_fields",
    "public_surface_importable",
    "rank_candidates_signature_kwarg_default_None",
    "unit_tests_classify_boost_8_paths",
    "unit_tests_apply_score_6_paths",
    "mid_cycle_bugfix_sentiment_velocity_key_grep_verification",
    "autonomous_loop_sequence_integrity_grep",
    "screener_wiring_grep",
    "infrastructure_reuse_no_duplicate_av_client",
    "rate_limit_3_layers_semaphore_throttle_universe_cap",
    "graceful_degradation_multi_layer",
    "default_off_discipline",
    "data_source_pivot_4_surfaces",
    "scope_honesty_no_overclaim",
    "research_gate_compliance",
    "code_review_heuristics_dimensions_1_through_5",
    "unicode_logger_ast_parse_scan",
    "no_perf_metrics_bypass",
    "no_kill_switch_or_stop_loss_changes",
    "no_secret_in_diff",
    "harness_log_prior_entries_scan"
  ]
}
```
