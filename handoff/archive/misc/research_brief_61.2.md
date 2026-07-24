# Research Brief — phase-61.2 (Analysis-integrity: degraded-row persistence, retry-on-empty, timeout, company_name, signal_downgrade, RiskJudge context)

Tier: complex. Written 2026-07-08 by Researcher (write-first skeleton; findings appended as gathered).

## Internal findings (file:line anchors)

### 1. Persistence paths + consumers of final_score/recommendation

**Fabrication chain (live-confirmed 5/5 on cycle 0725d2aa, live_check_66.2.md §5b):**
- `orchestrator.py:1526-1532` — synthesis loop fallback: `_parse_json_with_fallback(draft_text, "Synthesis-Final")` fails -> returns `{"error": "Failed to parse final report.", "synthesis_iterations": N}`. This dict becomes `report["final_synthesis"]` (orchestrator.py:2164).
- `autonomous_loop.py:1592-1612` — full-path result assembly: `rec = synthesis.get("recommendation", {})` is `{}` on error dict -> `"recommendation": rec.get("action", "HOLD")` = **synthetic HOLD** (:1599); `"final_score": synthesis.get("final_weighted_score", synthesis.get("final_score", 0))` = **synthetic 0** (:1607-1609). Neither checks `synthesis.get("error")` nor missing `scoring_matrix`.
- The dict then flows to (a) `_persist_analysis` (BQ `analysis_results` row), (b) same-cycle `trader.decide_trades` input — both poisoned.

**Consumers of final_score / recommendation (compat table candidates):**
- `autonomous_loop.py:1848-1875` `_degraded_scoring_check` (phase-56.2 F-5): degraded iff `float(a.get("final_score") or a.get("score") or 0) == 0.0` OR confidence==0 with uppercase rec. **NULL-score row: `None or 0` -> 0.0 -> still counted degraded. Compatible unchanged.** Exception path (TypeError/ValueError) also counts degraded.
- `portfolio_manager.py` decide_trades: admission `rec in _BUY_RECS` only (:50,:161); signal_downgrade at :114-127 (see §5). A degraded marker rec (non-BUY) is not tradeable — safe. BUT once signal_downgrade is fixed, a synthetic HOLD on a HELD ticker would trigger a downgrade SELL — degraded rows MUST be excluded from downgrade matching (interaction hazard, see §5).
- funnel/cycle-health/frontend/learnings/meta-evolution: (see below, being inventoried)

### 2. Retry surface (client vs orchestrator) + RailGuard interaction

- Swallow site: `claude_code_client.py:556-570` — `generate_content` catches `ClaudeCodeError`, calls `_rail_guard_record_failure(str(exc))`, returns **empty LLMResponse** with `thoughts=f"errored: {exc}"`. Rail-guard skip path (:538-544) returns empty LLMResponse with `thoughts=f"rail_guard_skipped: {reason}"` and **records nothing** (deliberate: no phantom rows).
- Orchestrator retry: `orchestrator.py:755-862` `_generate_with_retry` retries ONLY on exceptions (TimeoutError :841-845, GCP ServiceUnavailable/ResourceExhausted :846-852, name-matched transient :853-862). A successfully-returned empty response is passed through at :834 (`return response`).
- **Key affordance for retry-on-empty:** the empty response is distinguishable — `.text == ""` plus `.thoughts` prefix `"errored:"` (retryable) vs `"rail_guard_skipped:"` (NOT retryable — breaker open / probe-dead; retrying is retrying into an open breaker).
- **Breaker accounting:** each `generate_content` call independently checks `_rail_guard_blocked()` (:538) and records failure/success (:565/:585). So an orchestrator-level retry naturally (a) counts each attempt as a new rail call toward the consecutive-failure breaker (threshold 20, `claude_code_client.py:~100-107` `_rail_breaker_threshold`), and (b) stops retrying once the breaker opens (subsequent attempts return `rail_guard_skipped`, the non-retryable marker). Client-internal retry would need to re-check the breaker between attempts manually — orchestrator-level placement gets breaker interaction for free.

### 3. Timeout plumbing

- CLI subprocess default: `claude_code_invoke(timeout_s: int = 120)` at `claude_code_client.py:219`; `ClaudeCodeClient.__init__(..., timeout_s: int = 120)` at :479-481; passed at :552.
- The file's own note: `recommended_step_timeout = 150` class attribute at :477 (comment :474-476: 120s subprocess timeout fails first).
- `orchestrator.py:323-345` `_resolve_step_timeout` already lifts the ThreadPoolExecutor STEP budget to `recommended_step_timeout` (150) — but the SUBPROCESS still dies at 120s, so the step-budget lift is moot for the rail. Fix belongs at ClaudeCodeClient construction: new settings field consumed where the client is instantiated (site: see below).

### 1b. Persistence layer + downstream-consumer compatibility (NULL score + degraded marker)

- **Second fabrication site:** `_persist_analysis` (autonomous_loop.py:2429-2475) re-coerces `final_score=float(analysis.get("final_score") or 0.0)` (:2462) and `recommendation=analysis.get("recommendation") or "Hold"` (:2463). Fixing :1599/:1607 alone is insufficient — this layer would re-fabricate 0.0/"Hold".
- **BQ schema accepts NULLs** (verified live via ADC 2026-07-08): `financial_reports.analysis_results` — `final_score FLOAT NULLABLE`, `recommendation STRING NULLABLE`, `company_name STRING NULLABLE`, `full_report_json JSON NULLABLE`. Table lives in `financial_reports` (us-central1), NOT `pyfinagent_data` (`bigquery_client.py:36` reports_table from `bq_dataset_reports`).
- **Consumer compat table (NULL final_score / NULL recommendation on a degraded row):**

| Consumer | Anchor | Behavior on NULL | Change needed |
|---|---|---|---|
| `/api/reports/` list | `backend/api/models.py:96-97` (`final_score: float`, `recommendation: str` non-Optional) + `backend/api/reports.py:28-37` | Pydantic ValidationError -> HTTP 500 for the WHOLE reports list (rk=1 dedup means a degraded row CAN be a ticker's latest) | `Optional[float]` / `Optional[str]` + degraded passthrough field |
| Reports page score cell | `frontend/src/components/reports-columns.tsx:95` `row.original.final_score.toFixed(2)` | TypeError, page crash | null guard (em-dash + degraded badge); `types.ts:120` `final_score: number` -> `number \| null` |
| 30D sparkline map | `reports-columns.tsx:147` `tail.map(r => r.final_score)` | nulls in series | filter nulls |
| Slack digest | `backend/slack_bot/formatters.py:417,:532` `data.get("final_score", 0)` | `.get` default does NOT fire on present-but-None -> format TypeError | `or 0` guard / skip degraded rows |
| funnel_report | `scripts/diagnostics/funnel_report.py:102-110` `COALESCE(final_score,0)`, `(r.recommendation or "HOLD")` | already NULL-safe; NULL score counts into the existing degraded leg -> honest "degraded scoring" verdict | none (optionally count JSON `$._degraded` explicitly) |
| `_degraded_scoring_check` | autonomous_loop.py:1848-1875 | in-memory `None or 0` -> 0.0 -> counted degraded | none |
| decide_trades | `portfolio_manager.py:114` `analysis.get("recommendation", "HOLD").upper()` | **AttributeError crash** — `.get(k, default)` returns None when key present as None | degraded analyses must NEVER enter `candidate_analyses`/`holding_analyses` (see design) |
| signal_attribution | `backend/services/signal_attribution.py:185-199` | None-safe (`isinstance` guard; weight 0.0) | none (unreached if excluded from decide input) |

- **The existing lite-fallback routing already exists but is unreachable for synthesis errors:** `_run_single_analysis` full-path `except` (autonomous_loop.py:1630-1650) catches EXCEPTIONS from `run_full_analysis`, stamps `_fallback_reason` (feeds the 60.1 fallback-rate P1 at :960-971), and retries via `_select_lite_analyzer`. The synthesis-error dict does NOT raise, so it bypasses this whole machinery. Raising (or explicitly routing) on `synthesis.get("error") or missing scoring_matrix` after :1584 reuses ALL existing fallback + alarm plumbing for free.
- **Both-paths-fail today = returns None (:1648-1650) -> no BQ row at all, ticker silently vanishes from the cycle.** A NULL+marker degraded row on this branch adds observability without touching decide_trades (None results are already filtered from analyses lists at :910/:917).
- Analyses assembly: `_run_and_persist_one` (:860-902) -> `candidate_analyses` :910, `holding_analyses` :917, decide_trades call :1176-1184.

### 4. company_name fallback

- `_persist_analysis` reads `market_data = full_report.get("market_data") or {}` (:2452) -> `company_name=market_data.get("name") or None` (:2461). Only the LITE path puts `market_data` into full_report (:2413-2419 via `_integrity_market_data`); the full path's full_report is the orchestrator report (no `market_data` key) -> NULL company_name on every full-path autonomous row.
- Manual path works via the quant dict: `tasks/analysis.py:212` `company_name=quant.get("company_name", "N/A")`. Full-path full_report DOES carry `quant` (`{**report, ...}` at :1612; `report["quant"]`).
- Fix shape (pure bug fix, ungated): `market_data.get("name") or (full_report.get("quant") or {}).get("company_name") or None`.

### 5. signal_downgrade dead exit path

- `paper_trader.py:305` and `:329` write `"recommendation": reason` into the paper_positions row — the TRADE reason (`"new_position"`, `"swap_for_higher_conviction"`), not the analysis recommendation.
- `portfolio_manager.py:117` reads `old_rec = (pos.get("recommendation") or "").upper()` and the downgrade rule `:127` requires `old_rec in _BUY_RECS` (`{"BUY","STRONG_BUY"}`, :50-52). Trade reasons never match -> the signal_downgrade SELL exit (:127-134) has never fired. `_DOWNGRADE_RECS = {"HOLD","SELL","STRONG_SELL"}` (:48-49).
- Fix shape: thread the analysis recommendation into `execute_buy`'s pos_row (TradeOrder has no such field today — portfolio_manager.py:19-43; add `recommendation` field populated from the candidate analysis, or pass separately).
- **INTERACTION HAZARD (load-bearing):** reviving signal_downgrade makes synthetic HOLDs lethal — a rail failure on a held ticker's re-eval would emit HOLD -> `_DOWNGRADE_RECS` match -> SELL of a healthy position. The synthesis-integrity fix (degraded rows excluded from `holding_analyses`) MUST be active at-or-before the downgrade revival. Flag design must not allow downgrade-ON + integrity-OFF.
### 6. RiskJudge advisory portfolio context

- `autonomous_loop.py:791-800` (phase-57.1 F-8): `_rj_portfolio_ctx = ""` and only built `if getattr(settings, "paper_risk_judge_reject_binding", False)`. Criterion 6 = compute it regardless of the binding flag (advisory injection), binding stays OFF. `_build_portfolio_sector_context(positions)` already exists; positions already fetched at :787.

### 6b. Meta-scorer fallback (criterion 4)

- Clamp site: `meta_scorer.py:138-142` `_fallback_conviction` = `max(1, min(10, int(round(composite_score))))` — composites 78-163 all saturate to 10 -> constant "conviction 10.00".
- Fallback reason strings: `_fallback_all` (:249-254, `"fallback (LLM unavailable)"` — fires from the LLM-call except :203-205 AND the parse except :215-217), `"fallback (no API key)"` (:173-179), `"below batch cap (composite-score fallback)"` (tail, :234-238 — only head `_MAX_BATCH` is LLM-scored; tail ALWAYS uses fallback conviction, so rank-normalization must apply to the tail too or head/tail scales diverge).
- Existing single-cycle alarm: `_all_conviction_fallback` predicate (autonomous_loop.py:1902-1908) fires P1 `conviction_overlay_degraded` at :757-775 via `backend.services.observability.alerting.raise_cron_alert` (the 66.1-fixed import path). The 56.2 comment at :749-756 explicitly DEFERRED changing fallback VALUES as a live-behavior change -> rank-normalization belongs behind the 61.2 flag.
- "2 consecutive all-fallback cycles -> WARN": needs cross-cycle state. Precedent: start-of-cycle heartbeat file (autonomous_loop.py:195, `handoff/.cycle_heartbeat.json`) — a counter field there survives restarts; a module-level counter alone does not (backend is kickstart-restarted routinely).
- Root cause (criterion 4's diagnosis leg) is already evidence-backed: direct-API Anthropic credit exhaustion — `'credit balance is too low'`, live_check_66.2.md §5d, firing once per cycle since ~2026-07-03; meta_scorer constructs a direct `ClaudeClient` with `anthropic_api_key` (:185-190), NOT the cc_rail. The 06-03..06-10 window predates the July credit death — same fallback strings observed; experiment_results must distinguish (June window = the same credit-balance error per 66.2 §5d "since ~07-03" needs checking against June llm_call_log rows; do this at GENERATE with a bounded BQ query).

### 7. Flag design precedents

- Established idiom (settings.py): one bool Field per COHERENT BEHAVIOR THEME, default False, long audit-trail description — `paper_data_integrity_enabled` (:42, bundles 3 sub-behaviors), `paper_risk_judge_reject_binding` (:283), `paper_swap_churn_fix_enabled` (:317, bundles 3 sub-fixes). Scalar knobs are separate int/float Fields (`claude_rail_breaker_threshold` :176-180, `fallback_alarm_threshold` :46).
- Goal-doc rule: "behavior changes config-gated default OFF; pure bug fixes exempt but test-covered".
- Timeout plug point: `llm_client.py:1972` `ClaudeCodeClient(model_name=model_name)` inside `make_client` — settings in scope; add `timeout_s=settings.claude_code_timeout_s`.
- Flags absent from `settings_api.py _FIELD_TO_ENV` are manual-.env-only (61.1 audit note) — decide intentionally whether the new flag(s) join the Settings UI map.

## External sources

### Read in full (8; >=5 required — all via WebFetch, full pages not abstracts, 2026-07-08)
| URL | Kind | Key finding |
|---|---|---|
| https://learn.microsoft.com/en-us/azure/architecture/patterns/retry | official doc | "Implement retry logic only where the full context of a failing operation is understood... It might be better to configure the lower-level task to fail fast and report the reason... back to the task that invoked it." Retry must stop when the circuit breaker says the fault isn't transient; log early failures as informational, only the LAST retry failure as error; idempotent ops are inherently safe to retry. |
| https://martinfowler.com/bliki/CircuitBreaker.html | authoritative blog (canonical) | Open breaker: "all further calls... return with an error, without the protected call being made at all" — never retry into an open breaker; "any change in breaker state should be logged." |
| https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ | official vendor eng blog (Brooker, canonical) | Plain exponential backoff still clusters; FULL jitter (sleep = random(0, min(cap, base*2^attempt))) ~halves total calls vs no-jitter at N=100 contenders; full ≈ decorrelated > equal jitter. |
| https://sre.google/sre-book/handling-overload/ | official doc (Google SRE, canonical) | Per-request budget ~3 attempts ("If a request has already failed three times, we let the failure bubble up"); retry at EXACTLY ONE layer ("only... the layer immediately above it. If multiple layers retried, we'd have a combinatorial explosion"); explicit "'overloaded; don't retry' error response" as a non-retryable signal — maps 1:1 to our `rail_guard_skipped` marker. |
| https://platform.claude.com/docs/en/api/errors | official vendor doc | Anthropic SDKs "automatically retry transient failures... with exponential backoff, twice by default"; retryable = connection errors/429/5xx/529; non-retryable = 400/401/402/403 (the meta-scorer's `billing_error` credit exhaustion is NON-retryable — fallback, don't retry). |
| https://tenacity.readthedocs.io/en/latest/ | official lib doc | `retry_if_result` retries on a RESULT value (e.g. `is_none_p`), not just exceptions — the exact retry-on-empty primitive; `stop_after_attempt(n)` + `wait_random_exponential`; `before_sleep` logs only retried failures. (We implement in-house in `_generate_with_retry`; tenacity validates the semantics, no new dep needed.) |
| https://arxiv.org/html/2605.08563 | peer-reviewed preprint (2026) `[ADVERSARIAL]` | "Why Retrying Fails" — retry with contaminated context raises per-step error (SWE-bench ratio eps1/eps0 ≈ 7.1, super-critical); BUT "context-clearing before retry strictly reduces required attempts." Qualifies naive retry enthusiasm; our CLI rail call is a STATELESS fresh subprocess per attempt (no session resume), i.e. the always-fresh-context regime where bounded retry is effective. |
| https://minimalmodeling.substack.com/p/sentinel-free-schemas-a-thought-experiment | practitioner data-modeling essay | Sentinels ("UNKNOWN", 0, fake dates) "produce definite but potentially incorrect results" — precisely the 0.00/HOLD bug (a fabricated definite value destroyed two live BUY consensuses); alternatives: NULL + "explicit enums for meaningful absence" -> our NULL score + explicit `_degraded` marker. |

### Snippet-only (identified, not fetched in full; does not count toward gate)
~20 relevant of 35+ unique URLs surfaced: AWS prescriptive-guidance circuit-breaker; glich.co circuit-breaker-vs-retry; GeeksforGeeks retry strategies + cb-vs-retry; codereliant.io retries-backoff-jitter; dev.to akdevcraft "When Resilience Backfires" (retries trip tight breakers — widen thresholds when retries in play); refactorfirst resilience4j; dasroot.net 2026 LLM timeouts/retries; dev.to correctover "10,000 LLM calls" (5-15% first-attempt failure; 200-OK quality failures need result-based checks); llmgateway.io failover; buildmvpfast 2026 fallback strategies; apxml retry mechanisms for LLM calls; futureagi 2026 structured-output modes (provider constrained decoding 99.7-99.9% schema compliance); zenvanriel Instructor guide (validation-error-fed retry, "one retry resolves the large majority"); aisecurityinpractice output validation; keyhole/bytesizeddesign/jeffbailey/oneuptime retry storms; matrixtrak Polly backoff; Wikipedia + Grokipedia sentinel value; HN sentinel thread; layrs retry course. Reasons: redundant with read-in-full canon, or community-tier weight.
(The original AWS builders-library timeouts-retries page 301-redirects to builder.aws.com which serves a JS shell with no extractable text — substituted Brooker's canonical backoff post + SRE book, same authors/lineage.)

## Recency scan (2024-2026) — MANDATORY SECTION

Performed via dedicated 2025- and 2026-scoped searches ("LLM pipeline reliability retry degraded output handling 2026"; "retry storm outage backoff jitter incident lessons 2025"). Findings that complement the canon:
1. **Oct 19-20 2025 AWS outage**: synchronized client retries materially lengthened recovery — retry budgets/caps are now standard incident guidance (multiple 2025-26 postmortem writeups). Supports bounded (<=2 extra) attempts + full jitter.
2. **arXiv 2605.08563 (2026)**: fresh-context retries strictly dominate contaminated retries (~21% better resolution at equal budget) — directly validates retrying the stateless CLI subprocess, and warns against ever feeding the failed output back in.
3. **2026 practitioner consensus on LLM rails**: 5-15% first-attempt failure is normal; "200 OK but empty/wrong output" is a distinct failure class requiring RESULT-based (not exception-based) retry predicates — exactly criterion 1's gap.
4. **Provider-native constrained decoding (2025-26)** reaches 99.7-99.9% schema compliance — long-term, moving synthesis to schema-enforced decoding shrinks the parse-failure class; the cc_rail's `--json-schema` flag (claude_code_client.py:528-533) is already plumbed for dict schemas but the orchestrator passes Pydantic classes (schema-in-prompt convention). Follow-on lever, out of 61.2 scope.
No finding supersedes the canonical sources; all four complement them.

## Queries run (3-variant discipline)

Year-less canonical: "retry backoff jitter idempotency distributed systems best practices"; "circuit breaker retry pattern interaction don't retry into open breaker"; "LLM structured output parse failure retry validation Anthropic OpenAI best practice"; "sentinel values considered harmful data pipeline null vs magic number". Current-year frontier: "LLM pipeline reliability retry degraded output handling 2026". Last-2-year window: "retry storm outage backoff jitter incident lessons 2025".

## Recommended design

### Flags (recommendation: TWO gated bools + two ungated scalars; pure fixes ungated)

1. **`paper_synthesis_integrity_enabled: bool = Field(False)`** (settings.py, mirror the :42/:317 idiom) — ONE umbrella flag for the analysis-input theme, matching the 60.2/60.3 precedent of bundling coherent sub-behaviors under a single operator promotion decision:
   - (a) synthesis-error detection: after `run_full_analysis` returns (autonomous_loop.py:1584-1590), if `synthesis.get("error")` or no `scoring_matrix` -> `raise` a `SynthesisDegradedError` INSIDE the existing try so the EXISTING except (:1630-1650) routes to the lite fallback with `_fallback_reason="synthesis_error: ..."` — reuses the 60.1 fallback-rate P1 and produces a REAL scored row (criterion 1 first option).
   - (b) both-paths-fail: persist a degraded row (schema below) then return None (decide_trades never sees it — position keeps its slot; no crash at portfolio_manager.py:114; no synthetic-HOLD downgrade/displacement).
   - (c) `_persist_analysis` NULL-passthrough when the degraded marker is present (:2462-2463 coercions bypassed only for marker rows).
   - (d) retry-on-empty (below).
   - (e) RiskJudge advisory context: build `_rj_portfolio_ctx` at :791-800 when EITHER this flag OR the binding flag is on (criterion 6; binding flag stays OFF).
   - (f) meta-scorer fallback rank-normalization + 2-consecutive-cycle WARN (criterion 4).
2. **`paper_position_recommendation_fix_enabled: bool = Field(False)`** — signal_downgrade revival (criterion 5): thread the ANALYSIS recommendation into `execute_buy`'s pos_row (new TradeOrder field; paper_trader.py:305/:329 write it instead of the trade reason). SEPARATE flag because its blast radius is SELLs of healthy held positions — different risk class, operator may stage it after observing flag 1. **Code must WARN if flag2 ON while flag1 OFF** (synthetic HOLDs + revived downgrade = rail-failure sells; the interaction hazard in §5). Old rows carry trade reasons and never match — no backfill needed.
3. **`claude_code_timeout_s: int = Field(150)`** — ungated scalar (criterion 2 is immutable: ">= 150s and configurable"; reliability fix, not a decision change). Consumed at `llm_client.py:1972` (`ClaudeCodeClient(model_name=..., timeout_s=settings.claude_code_timeout_s)`). ALSO set instance `self.recommended_step_timeout = timeout_s + 30` in `__init__` (:479-481) so `_resolve_step_timeout` (orchestrator.py:342) keeps the step budget ABOVE the subprocess timeout — otherwise 150/150 recreates the race the :474-476 comment warns about.
4. **`claude_code_empty_retry_max: int = Field(2)`** — attempt knob for (d); effective only when flag 1 is ON.
5. **Ungated pure fixes** (goal-doc exemption, test-covered): company_name quant fallback in `_persist_analysis` (:2461 -> `market_data.get("name") or (full_report.get("quant") or {}).get("company_name") or None`); read-side NULL guards (models.py Optional, frontend guards, slack formatter guards — safe unconditionally since NULL rows cannot exist while flag 1 is OFF).
   - Decide explicitly whether the two bools join `settings_api.py _FIELD_TO_ENV` (61.1 lesson: absent = manual-.env-only).

### Retry policy (criterion 1 supporting leg)

- **Placement: orchestrator `_generate_with_retry` ONLY** (one layer — Google SRE; Azure "lower-level task fail fast"). NOT inside ClaudeCodeClient (would multiply with the orchestrator's exception retries; client lacks step context).
- **Predicate (result-based, Tenacity `retry_if_result` semantics):** after :834's successful return path, classify: `text==""` AND `thoughts` startswith `"errored:"` -> RETRYABLE empty; `thoughts` startswith `"rail_guard_skipped:"` -> NON-RETRYABLE (the SRE "overloaded; don't retry" signal — never retry into an open breaker, Fowler/Azure); non-empty text -> return.
- **Bounds:** max total attempts = 1 + `claude_code_empty_retry_max` (default 3 total = the SRE per-request budget; Anthropic's own SDK default is 2 retries). Backoff: full jitter, `sleep = random.uniform(0, min(15, 2 * 2**attempt))` (Brooker).
- **Breaker interaction (already free):** each attempt re-enters `generate_content` -> re-checks `_rail_guard_blocked()` (:538) and records failure via `_rail_guard_record_failure` (:565). So every retry COUNTS as a new rail call (llm_call_log ok=False row per attempt — honest metering), accelerates the consecutive-20 breaker during a real outage instead of hiding failures, and stops the moment the breaker opens (subsequent attempts return the non-retryable skip marker). No RailGuard changes needed.
- **Idempotency/cost:** the CLI call is a stateless fresh subprocess per attempt (no session resume) — the arXiv 2605.08563 fresh-context regime; no side effects beyond quota + log rows. Worst-case cost: 2 extra 150s calls per synthesis-class step; breaker caps outage-mode waste.
- Log retries at WARNING, final failure at ERROR (Azure logging model).

### Degraded-row persistence schema (criterion 1 second option, both-paths-fail only)

`final_score=NULL`, `recommendation=NULL` (schema verified NULLABLE), `summary="DEGRADED: <reason>"`, `full_report_json`: `$._degraded=true`, `$._degraded_reason` (<=500 chars), `$._path` preserved — the JSON-marker idiom already used for `$._path` (bigquery_client.py:268). Never a fabricated 0.0/HOLD (sentinel-free: Minimal Modeling). Downstream compat changes per the §1b table: `ReportSummary.final_score: Optional[float]`, `recommendation: Optional[str]`, add `degraded` passthrough; `types.ts` `number|null`; `reports-columns.tsx:95` null guard + degraded badge, `:147` filter nulls; `formatters.py:417/:532` `or 0`-guard/skip; funnel_report needs NO change (COALESCE + degraded leg already honest).

### Meta-scorer (criterion 4)

Percentile-rank the cycle's composites -> `conviction = 1 + round(9 * pct_rank)` inside a set-aware fallback (replace per-candidate clamp `:138-142` for `_fallback_all` AND the below-cap tail so head/tail share one scale). Keep reason strings. 2-consecutive-all-fallback WARN: counter persisted in the cycle-heartbeat file (autonomous_loop.py:195 precedent — module state dies on kickstart restarts), fire `raise_cron_alert(severity="WARN"/"P2", error_type="conviction_fallback_streak")` alongside the existing per-cycle P1 (:757-775). Root-cause leg: direct-API Anthropic `billing_error` credit exhaustion (live_check_66.2.md §5d; NON-retryable class per Anthropic errors doc — falling back is correct, the constant 10.00 VALUE was the bug); at GENERATE, run one bounded llm_call_log BQ query over 06-03..06-10 to confirm the June window shows the same signature, document in experiment_results.md.

### Test plan (verification `-k 'synthesis or persist or downgrade or meta_scorer or 61_2'`; name files `test_phase_61_2_*` — the 59.1 lesson: names must match the -k expression)

1. **Criterion-1 regression (immutable shape):** flag ON + mocked `run_full_analysis` returning `final_synthesis={"error": ...}` -> assert `save_report` never called with 0.0/HOLD; lite fallback attempted; with lite also failing -> degraded row (score=None, rec=None, `$._degraded`) AND `_run_single_analysis` returns None (same-cycle decide input not neutralized). Flag OFF -> byte-identical legacy behavior.
2. Retry-on-empty: errored-empty then success -> 2 attempts; `rail_guard_skipped` -> zero retries; flag OFF -> zero retries; per-attempt rail failure recording asserted (breaker counting).
3. Timeout: `claude_code_timeout_s` threads through `make_client` to `_timeout_s`; instance `recommended_step_timeout == timeout_s + 30`.
4. company_name: full-path dict (no market_data, quant.company_name present) -> persisted non-null.
5. signal_downgrade: flag2 ON -> pos_row carries analysis rec; pm:127 fires SELL(signal_downgrade) on fresh HOLD re-eval of a BUY-rec position; degraded/absent re-eval -> NO sell; flag2-without-flag1 warning.
6. Meta-scorer: composites 78-163 -> non-constant spread across 1-10; heartbeat-counter WARN at streak 2.
7. RJ advisory: `_rj_portfolio_ctx` non-empty with binding OFF + flag1 ON.
Live check per the immutable field: post-fix cycle BQ rows — non-null company_name on full-path, zero final_score=0.0 AND `$.final_synthesis.error` rows, non-constant convictions.

### Dependency notes

- Ships dark (both bools OFF): every decision-path change is byte-identical OFF; timeout + read-side guards + company_name are the only live deltas post-restart (all pure fixes).
- 60.2 interaction: the both-fail None-return leaves a holding out of `holding_lookup`; with `paper_swap_churn_fix_enabled` ON (61.1 recommendation) it is excluded from swap displacement — coherent. If 60.2 were OFF, the absent re-eval re-exposes the sentinel-displacement bait; disclose in the contract (out of 61.2 scope).

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 20,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "report_md": "handoff/current/research_brief_61.2.md",
  "gate_passed": true
}
```
