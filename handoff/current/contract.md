# Contract -- masterplan step 75.5

**Step id**: `75.5`
**Name**: Audit75 S5 -- live CC-rail schema enforcement, metered-bypass guards, model-retirement + telemetry/cost correctness
**Phase**: phase-75 | **Priority**: P0 | **Cycle**: 1 | **Date**: 2026-07-20
**BOUNDARY (from the step text)**: $0 metered strengthened; **no model-tier pins changed** -- literals become constants only.

---

## 1. Research gate

**PASSED.** Run `wf_0cea9f6a-482`, two legs (researcher + independent adversarial verifier).
Envelope: `tier=complex`, `external_sources_read_in_full=6`, `snippet_only=24`,
`urls_collected=30`, `recency_scan_performed=true`, `internal_files_inspected=17`,
`gate_passed=true`. Not audit-class. Brief: `handoff/current/research_brief_75.5.md`.

### 1a. Claim verdicts (adversarially re-derived, not read off the step text)

| Finding | Verdict | Note |
|---|---|---|
| llmeng-01 | **PARTIAL** | Defect exact at `claude_code_client.py:535-537`; but "the ONLY schema shape" is **REFUTED** |
| llmeng-03 | **CONFIRMED** | All three limbs; only a line-number slip |
| llmeng-04 | **PARTIAL** | Gap real; **"single owner" premise already false**; `:1684` is the wrong line |
| llmeng-06 | **PARTIAL** | All 5 line numbers exact; but "5 pins" **undercounts** (13 behavioural) |
| llmeng-10 | **CONFIRMED** | Money direction **correct, not backwards** |
| llmeng-11 | **CONFIRMED** | Line slip only (`:1140` def, not `:1214`) |
| arch-04 | **CONFIRMED** | All limbs; plus a discovered defect (§6) |

### 1b. Corrections that bind this plan

1. **llmeng-01 -- "the ONLY schema shape the pipeline passes" is REFUTED.** All 9 *pipeline*
   `response_schema` values are Pydantic classes, so `--json-schema` is dead code **on that
   path**. But **6 production services pass DICT schemas** (`meta_scorer.py:232`,
   `news_screen.py:288`, `pead_signal.py:292`, `macro_regime.py:525`,
   `analyst_narrative_scorer.py:161`, `call_transcript_gpr.py:139`), each pre-cleaned via
   `_strip_unsupported_schema_keys(...model_json_schema())`. The `isinstance(dict)` branch is
   **live for those six** -- the fix must ADD class handling, never replace the dict path.
2. **llmeng-01 `$defs`/`$ref` trap MEASURED and CLEARED.** Anthropic lists `$ref`/`$defs` as
   SUPPORTED (external `$ref` not). 4 of our 6 schemas emit `$defs` (CriticVerdict 5 refs,
   SynthesisReport 3, ModeratorConsensus 2, RiskJudgeVerdict 1); **zero** carry an
   unsupported keyword. No flattening needed. But `minimum`/`maxLength`/recursive schemas
   **400** -- so a future `Field(ge=…)` would silently start failing. Guard test required.
3. **llmeng-04 -- the "single owner" premise is ALREADY FALSE.** Two independent
   max_tokens doubling re-requests exist today. The contract names the owner **verbatim**:
   > **The sole max_tokens re-request owner on the `generate_content` path is
   > `ClaudeClient.generate_content`'s phase-4.14.4 MF-26/27 `stop_reason` dispatch,
   > `backend/agents/llm_client.py:1656-1681`, re-request at `:1672`, budget
   > `min(max_tokens*2, 8192)`.**

   The Layer-2 MAS loop (`multi_agent_orchestrator.py:1363-1394`, re-request `:1381`,
   budget `min(_max_tokens*2, 32768)`, `_mt_retried_turn` guard) is a **separate,
   pre-existing owner on a separate path** -- **out of scope, not to be unified here**.
   Also: owner #1 only retries on a **tool_use tail** (`:1659`); plain-text truncation just
   logs (`:1682-1685`) with no retry.
4. **llmeng-04 -- `LLMResponse` is at `llm_client.py:653-679`, NOT `:1684`** (that line is a
   log string). And **there is no shared JSON-parse helper**: there are **three duplicated**
   ones (`debate.py:122` and `risk_debate.py:118` are byte-identical; `orchestrator.py:309`),
   all taking a plain `text: str` -- they **cannot** expose `stop_reason` without a signature
   change. Criterion 3 says "the shared JSON-parse helper"; I will introduce a single shared
   helper and route the call sites to it, rather than pretend one exists.
5. **llmeng-04 -- enum case differs.** Claude `stop_reason` is lowercase
   (`end_turn`/`max_tokens`/`tool_use`/…); Gemini `finishReason` is UPPERCASE
   (`STOP`/`MAX_TOKENS`/…). **Normalization policy: store the provider-native string on
   `LLMResponse.stop_reason`, and compare case-insensitively via one helper.** The CC
   envelope already carries `stop_reason` (`claude_code_client.py:245,:364-368`).
6. **llmeng-06 -- "5 pins" undercounts.** 13 behavioural pins exist (unlisted:
   `directive_review.py:159`, `directive_rewriter.py:202`, `news/sentiment.py:81`,
   `harness_memory.py:322,:503`, `services/autonomous_loop.py:2648,:2663`,
   `api/agent_map.py:132`). **Criterion 4 is scoped to the 5 named files so it stays
   satisfiable as written** -- I fix those 5 and **queue the remaining 8** rather than
   silently widening scope. Also **`GEMINI_DEEP_THINK` does not exist** -- only
   `GEMINI_WORKHORSE` (`model_tiers.py:50`); deep-think lives inside a dict at `:97`. I must
   create the constant, not assume it.
7. **llmeng-10 -- direction CONFIRMED, and the over-count risk is CHECKED and CLEARED.**
   Anthropic verbatim: *"The `input_tokens` field represents only the tokens that come after
   the last cache breakpoint in your request - not all the input tokens you sent"* and
   *"total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens +
   input_tokens"*. Current `cost_tracker.py:176`
   `regular_input = max(0, input_tokens - cache_read - cache_creation)` double-subtracts.
   Worked example (Opus 4.8, input=1000, cache_read=5000): today prices **$0.002500**,
   correct is **$0.007500** -- a **66.7% under-report**; the `max(0,…)` clamp is what makes it
   silent. `anthropic==0.96.0` exact pin, no version skew.
   **Critical nuance**: `record()` is shape-polymorphic -- it reads Gemini's
   `prompt_token_count` but Anthropic's cache field names, and **Gemini semantics are the
   OPPOSITE** (`prompt_token_count` INCLUDES cached). This is safe **only** because
   GeminiClient never populates the cache fields (`llm_client.py:1072-1076` omits them;
   `:1091-1092` hardcodes 0). That safety is **incidental, not asserted** -> a guard test is
   mandatory, or a future Gemini caching change silently over-counts.
8. **Criterion 5 contains a kwarg that does not exist.** `UsageMeta(input=1000,
   cache_read=5000)` raises `TypeError`; the real fields (`llm_client.py:643-650`) are
   `prompt_token_count` / `candidates_token_count` / `total_token_count` /
   `cache_creation_input_tokens` / `cache_read_input_tokens`, and `record()` takes a
   **response object** (`cost_tracker.py:153` `getattr(response,'usage_metadata')`).
   **The criterion is IMMUTABLE and will NOT be amended.** I read `input=`/`cache_read=` as
   **descriptive shorthand** and satisfy its *intent* with the real API:
   `SimpleNamespace(usage_metadata=UsageMeta(prompt_token_count=1000,
   cache_read_input_tokens=5000))` must price exactly 1000 uncached input tokens. This
   reading is recorded here so Q/A does not mistake it for criteria-weakening.
9. **llmeng-03 SecretStr trap.** Use `unwrap_secret(getattr(settings,'anthropic_api_key',''))`
   per the existing idiom at `:1953` -- **never `or ""`**, because a non-empty SecretStr is
   truthy and `or ""` returns the **wrapper** (auto-memory `project_secretstr_dead_overlays`;
   this exact bug silently killed 4 alpha overlays for 3 weeks).
10. **Corrected line anchors**: `advisor_call` def `:2075` (not `:2110`, the `Anthropic()`
    construction); `OpenAIClient.generate_content` def `:1140` (not `:1214`, the return).
    OpenAIClient has **no latency timer** (add `_t0` around `:1204`) and **serves both direct
    OpenAI and GitHub Models** (routed `:2035-2039` via `base_url`) -> `provider` must be
    conditional, not hardcoded.

### 1c. Measured test breakage (not assumed)

- **(g)** `tests/slack_bot/test_scheduler_wiring_phase991.py:150` monkeypatches
  `cost_budget_watcher._default_fetch_spend` and **will break**. It lives under top-level
  `tests/`, **outside** the verification command's `backend/tests/` scope, so it would
  **regress silently**. Mitigation: keep a backwards-compatible alias AND run that file
  explicitly as evidence. *(The verifier flagged it did not independently re-confirm the
  monkeypatch line -- I will verify it myself before relying on the mitigation.)*
- **(e)** `test_cost_tracker_pricing.py` and `verify_phase_25_{C9,D9,E9,S_1}` all pass
  `cache=0` -> safe. `verify_phase_25_A9` claim-1 is a **source scan** and claim-2 a
  **tautology** (expected==actual, same literal) -> **neither can catch (e)**; do not treat
  their passing as evidence. `verify_phase_25_B9:147-151` does not break, but its **fixture
  encodes the WRONG inclusive semantics** (`input_tokens=5050` with `cache_read=5000`) and
  must be re-shaped to `input_tokens=50` in the same commit.

---

## 2. Hypothesis

Seven defects on the LLM/cost rail, all *silent*: a schema flag that never fires on the
pipeline path, two metered-spend guards that a live call site skips entirely, truncation
that is invisible to callers, model pins that will hard-stop on 2026-10-16, an input-cost
under-report of 66.7% on every cached call, a provider that writes no telemetry, and a
money guard reached into via a private symbol. None crash; all under-report or under-guard.

**(e) is the money-critical item** and the one most likely to be got backwards -- which is
exactly why the gate quoted the doc verbatim and worked the arithmetic before I touched it.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

> 1. New backend/tests/test_phase_75_llm_rail.py passes offline and asserts: a config carrying a Pydantic model CLASS yields a CC argv containing --json-schema whose JSON sets additionalProperties:false (no CLI spawned -- argv construction only)
> 2. Test asserts advisor_call invokes _check_cost_budget and raises the routing-breach ValueError when settings.paper_use_claude_code_route is True (monkeypatched), and llm_client no longer builds a raw Anthropic client from os.getenv in advisor_call
> 3. Test asserts LLMResponse exposes stop_reason populated by the Claude, Gemini, and CC client paths, and the shared JSON-parse helper sets a `degraded` flag on a max_tokens truncation WITHOUT issuing its own retry (no-double-retry guard verified); the sole max_tokens re-request owner remains the existing ClaudeClient retry layer, asserted by the test
> 4. Source scan in the test proves zero 'gemini-2.5' literals outside config/model_tiers.py across the 5 listed files, and the 2.5-family retirement warning fires under a frozen >=2026-09-15 date
> 5. Test asserts UsageMeta(input=1000, cache_read=5000) prices exactly 1000 uncached input tokens (no clamp to zero), and OpenAIClient.generate_content calls log_llm_call (mocked transport)
> 6. fetch_spend is importable from backend.services.observability; llm_client, api/cost_budget_api, and slack_bot/jobs/cost_budget_watcher all resolve it from there (import-path scan); guard degradation increments a counter/alert seam instead of only a WARNING line

**Command**: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_llm_rail.py -q`
**live_check**: `handoff/current/live_check_75.5.md` -- verbatim command output (exit 0) +
`git diff --stat`; ON-vs-OFF $0 diff for flag-gated live-loop behavior; Playwright/curl for
UI-touching parts. Findings: llmeng-01, -03, -04, -06, -10, -11, arch-04.

*No criterion amended. Criterion 5's `UsageMeta(input=…, cache_read=…)` is read as
descriptive shorthand per §1b.8; criterion 3's "the shared JSON-parse helper" is satisfied
by creating one per §1b.4; criterion 4 stays scoped to the 5 named files per §1b.6.*

---

## 4. Plan

- **(a)** In `claude_code_client.py:535-537`, keep the `isinstance(dict)` branch **intact**
  (6 live dict callers) and ADD: Pydantic class -> `model_json_schema()` ->
  `_ensure_additional_properties_false`. `--append-system-prompt` only if it costs nothing.
- **(b)** `advisor_call` (`:2075`): add `_check_cost_budget` + the routing-breach guard
  (copy `:2012-2021` verbatim), resolve the key via `unwrap_secret(...)`, drop the raw
  `os.getenv` client build.
- **(c)** Add `stop_reason` to `LLMResponse` (`:653-679`), populated on all three paths.
  Create **one** shared JSON-parse helper that sets `degraded` on truncation and **never
  retries**; route the three duplicates to it. Add the no-double-retry guard. The owning
  layer is named verbatim in §1b.3 and must be repeated in the code comment.
- **(d)** Route the **5 named** pins through `model_tiers.py` constants (creating
  `GEMINI_DEEP_THINK`, which does not exist). Startup warning when `date >= 2026-09-15` and
  any resolved gemini model is 2.5-family. **No tier pin values change** (boundary).
- **(e)** `cost_tracker.py:176` -> `regular_input = input_tokens`. Plus a **guard test**
  pinning the incidental Gemini safety (§1b.7).
- **(f)** `OpenAIClient.generate_content` (`:1140`): `log_llm_call` retrofit, fail-open,
  `_t0` latency timer, **conditional** provider (OpenAI vs GitHub Models).
- **(g)** Promote `fetch_spend()` into `backend/services/observability` (package exists;
  `alerting.py` is the natural home for the degradation counter). Repoint all consumers,
  keep fail-open, **keep a back-compat alias** for the out-of-scope test.

## 5. Mutation matrix (mandatory -- a guard that cannot fail does not count)

Per the durable rule from cycle 131: **a matrix licenses only "these N were killed", never
"no vacuous guards".** Mutate the fixture too. Minimum set, each must FAIL a test:
M1 revert the isinstance gate so a class is dropped; M2 emit a schema without
`additionalProperties:false`; M3 remove `_check_cost_budget` from advisor_call; M4 remove
the breach guard; M5 restore the raw `os.getenv` client; M6 drop `stop_reason` from each of
the 3 paths **separately**; M7 make the parse helper retry (double-retry); M8 restore a
`gemini-2.5` literal in **one** of the 5 files; M9 freeze the date to 2026-09-14 (warning
must NOT fire -- negative control); **M10 revert `regular_input` to the subtracting form**
(the money mutant); M11 populate Gemini cache fields (the incidental-safety guard must
fire); M12 remove `log_llm_call` from OpenAIClient; M13 point a consumer back at the private
`_default_fetch_spend`; **M14 HARNESS: replace the real `record()` with a stub** -- the
anti-stub clause must break.

## 6. Risks / discovered defects -> queued, not disclosed in prose

- **DISCOVERED (arch-04)**: `_default_fetch_spend` queries
  `INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at $6.25/TiB -- it measures
  **BigQuery** spend, but `settings.py:379` documents `cost_budget_daily_usd` as the
  *"Daily LLM-spend cap"* and CLAUDE.md calls the $25/day cap the **LLM** circuit breaker.
  **The guard does not measure what its name says.** Promoting it unchanged (as (g)
  specifies) correctly preserves behavior -- but this must be **queued as 75.5.1**, not
  silently carried.
- **75.5.2**: the 8 remaining gemini-2.5 behavioural pins outside criterion 4's 5 files.
- **75.5.3**: schema-keyword guard -- a future `Field(ge=/le=)` would silently 400 the CC
  rail (§1b.2).
- **75.5.4**: unify or document the two max_tokens retry owners (§1b.3) -- explicitly NOT
  done here.
- **R-live**: (b), (e), (g) touch the live money/cost path. (e) changes reported cost
  **upward** (it was under-reporting) -- this is a *correction*, but it will move dashboards
  and could trip the $25/day guard sooner. Must be called out in `experiment_results.md`.

## 7. References

`handoff/current/research_brief_75.5.md` (`wf_0cea9f6a-482`); Anthropic prompt-caching +
structured-outputs + stop_reason docs; `ai.google.dev/gemini-api/docs/deprecations`
(2.5 shutdown 2026-10-16); auto-memories `project_secretstr_dead_overlays`,
`feedback_mutation_test_guards_and_fixtures`, `feedback_measure_dont_assert_claims`;
`handoff/harness_log.md` Cycle 131.
