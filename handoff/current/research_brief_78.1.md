# Research Brief — phase-78.1: Rewire six signal-overlay services (C1-C6) off direct `ClaudeClient` onto the CC rail

**Tier:** moderate. **Audit-class:** false. **Started:** 2026-07-24.
**Status:** COMPLETE. `gate_passed: true` (6 sources read in full, 30 URLs,
recency scan performed). Two disclosed gaps in §13 (backend/.env denied; the
CLI output-ceiling figure is an in-repo assertion, unverified upstream).

---

## 0. Scope

REWIRE `meta_scorer`, `news_screen`, `macro_regime`, `pead_signal`,
`analyst_narrative_scorer`, `call_transcript_gpr` off direct `ClaudeClient(...)`
construction onto `make_client(...)` so `PAPER_USE_CLAUDE_CODE_ROUTE` applies.
No behavior change to the signals.

---

## 1. Internal code inventory — the six call sites

All six anchors RE-CONFIRMED by grep on 2026-07-25 (exact `ClaudeClient(` lines):

| # | Service | ctor line | ctor kwargs | schema arg | model setting |
|---|---------|-----------|-------------|-----------|---------------|
| C1 | `backend/services/meta_scorer.py` | :221 | `model_name=settings.meta_scorer_model or "claude-haiku-4-5"`, `api_key=anthropic_key`, `enable_prompt_caching=False` | `_strip_unsupported_schema_keys(MetaScorerBatch.model_json_schema())` → **dict** (:218) | `meta_scorer_model` |
| C2 | `backend/services/news_screen.py` | :267 | same 3 kwargs, model `news_screen_model` | `NewsSignalBatch.model_json_schema()` stripped → **dict** (:274) | `news_screen_model` |
| C3 | `backend/services/macro_regime.py` | :506 | same 3 kwargs, model `macro_regime_model` | `MacroRegimeOutput.model_json_schema()` stripped → **dict** (:519) | `macro_regime_model` |
| C4 | `backend/services/pead_signal.py` | :279 | same 3 kwargs, model `pead_signal_model` | `PeadSignalOutput.model_json_schema()` stripped → **dict** (:286) | `pead_signal_model` |
| C5 | `backend/services/analyst_narrative_scorer.py` | :135 | `model_name=model` (**function ARG**, not settings), `api_key`, `enable_prompt_caching=False` | hand-written literal dict, then stripped (:145-155) | caller-supplied `model=` |
| C6 | `backend/services/call_transcript_gpr.py` | :113 | `model_name=model` (**function ARG**), `api_key`, `enable_prompt_caching=False` | hand-written literal dict, then stripped (:123-133) | caller-supplied `model=` |

### 1.1 Per-service call shape (the differences that matter)

**Common to all six** — `await asyncio.to_thread(client.generate_content, prompt, {...})`
with a config dict of exactly four keys: `response_schema` (dict),
`response_mime_type: "application/json"`, `max_output_tokens`, `temperature: 0.0`.
**None of the six passes `_role`, `_ticker`, `system`, `system_instruction`,
`thinking`, or `effort`.** That single fact drives findings §5 (no house
instructions on the rail) and §6 (bare `cc_rail` agent tag).

| # | key acquisition | `max_output_tokens` | retry | parse-fail behavior | cache-on-disk |
|---|---|---|---|---|---|
| C1 meta_scorer | `unwrap_secret(settings.anthropic_api_key)` :203; empty-key → per-candidate fallback + **returns early** :204-212 | `min(8192, 250*len(shuffled))` :234 | none | `_fallback_all(head+tail)` :252 | no |
| C2 news_screen | `unwrap_secret` :261; empty-key → **`return {}`** :262-264 | `min(48000, max(8192, 250*len(deduped)))` :280 (phase-69.3) | **2 attempts**, `for _attempt in range(2)` :282 | `continue` → after 2 tries `return {}` :309-310 | **yes** `_save_cache` :324 (success only) |
| C3 macro_regime | `unwrap_secret` :501; empty-key → `_fallback_regime(..., "ANTHROPIC_API_KEY not configured")` :502-503 | **512** :527 | none | `_fallback_regime(indicators, "LLM output parse error")` :548 | **yes** `_save_cache` :596 (success only); 24h TTL |
| C4 pead_signal | `unwrap_secret` :249; empty-key → `_fallback("no_anthropic_key")` :273-274 | **512** :294 | none | `_fallback("parse_error")` :316 | **yes** `_save_pead_cache` :318 (success only) |
| C5 analyst_narrative | **manual** `getattr(...) or ""` + `hasattr(k,'get_secret_value')` unwrap :106-116 (does NOT use `unwrap_secret`) | **256** :163 | none | `return None` :178 | no |
| C6 call_transcript_gpr | **manual** unwrap :90-97 (same hand-rolled shape) | **384** :141 | none | `return None` :160 | no |

Additional per-service asymmetries:

- **C1** shuffles the batch with a **fixed seed** `random.Random(0xC0FFEE)` (:214-216)
  — determinism is already an explicit design goal here, which makes the
  temperature loss in §3 a direct regression against intent.
- **C1** post-clamps `conviction_score` to `[1,10]` and truncates
  `conviction_reason` to 200 chars BEFORE `model_validate` (:244-249) — so schema
  drift is partially absorbed.
- **C3** post-clamps `conviction` to `[0,1]` and `conviction_multiplier` to
  `[0.5,1.5]` (:541-544) with the comment "Anthropic structured outputs cannot
  enforce numerical … constraints" — same absorption.
- **C3** and **C4** both wrap the LLM call in `try/except Exception` and return a
  DIFFERENT fallback reason for call-failure vs parse-failure (`"LLM error:
  {type}"` :533 vs `"LLM output parse error"` :548; `f"llm_error:{type}"` :300 vs
  `"parse_error"` :316). Because the CC rail **returns an empty response instead
  of raising** (§3.2), a rail failure lands in the PARSE branch, so the persisted
  `rationale`/`reason` string changes. Observable, but signal-neutral.
- **C5/C6** are the only two that construct the client inside a
  `try/except → return None` (:133-142 / :111-120) and the only two whose model
  name is a function parameter — so their rewire must thread `settings` in, or
  call `get_settings()` (they already do at :105 / :89).
- **C5/C6** wrap the whole body in `async with sem:` (a caller-supplied
  `asyncio.Semaphore`) → these two are the concurrency-sensitive pair (§9.3).

## 2. `make_client` routing (`backend/agents/llm_client.py:2044-2175`)

Order of evaluation for a `claude-haiku-4-5` model name:

1. `:2076` gemini branch — not taken.
2. `:2099-2113` **CC-rail branch** — taken iff `model_name.startswith("claude-")`
   AND `settings.paper_use_claude_code_route` is truthy. Returns
   `ClaudeCodeClient(model_name=..., timeout_s=int(settings.claude_code_timeout_s or 150))`.
   Note it catches **only `ImportError`** (:2114) — any other exception from the
   `ClaudeCodeClient` construction propagates out of `make_client`.
3. `:2120-2139` direct-Anthropic branch. **Routing-breach guard at :2128-2137**:
   if `paper_use_claude_code_route` is True and control still reaches here (i.e.
   the CC import failed), it **raises `ValueError`**. For the six services this
   is a *desirable* loud failure — but it is an EXCEPTION, and C1/C3/C4 call
   `make_client` OUTSIDE their `try` today's `ClaudeClient(...)` sits in, so the
   rewire must place `make_client` inside a guarded block or the ValueError
   escapes to the caller (see §9 risk R2). When the flag is False it returns
   `ClaudeClient(model_name=..., api_key=...)` with
   **`enable_prompt_caching` left at its default `True`** (§3.1 diff D7).
4. `:2148` GitHub-Models branch — **cannot capture these calls**: it is guarded by
   `model_name in GITHUB_MODELS_CATALOG and github_token`, and it is only reached
   when branch 3 did not match, i.e. when `anthropic_key` is empty. All six
   services already `return`/fallback on an empty key BEFORE constructing a
   client, so as long as the rewire keeps the existing empty-key early-returns,
   branch 4 is unreachable. **If a rewire deletes the empty-key guard**, a
   catalog-listed model + `GITHUB_TOKEN` would silently route to
   `models.github.ai` — a real, metered, third-party rail. Verified catalog
   membership below (§2.1).

### 2.1 Is `claude-haiku-4-5` in the GitHub catalog?

**YES** — `llm_client.py:481-529` `GITHUB_MODELS_CATALOG` contains
`"claude-haiku-4-5"` (verified by printing the literal set). So branch 4 is a
LIVE hazard if the empty-key guard is dropped. Mitigation is simply: **do not
touch the empty-key early-return blocks.** (Whether `GITHUB_TOKEN` is actually
set could not be verified — `backend/.env` reads are denied to the researcher
sandbox; Main should confirm. Treat as set.)

## 3. `ClaudeClient` vs `ClaudeCodeClient` — every API difference

`ClaudeClient.generate_content` = `llm_client.py:1437-1924`.
`ClaudeCodeClient.generate_content` = `claude_code_client.py:517-646`
(class built lazily by `_make_claude_code_client_class()` at :453).

### 3.1 Config-key handling matrix

| Config key | `ClaudeClient` | `ClaudeCodeClient` | Impact on the six |
|---|---|---|---|
| `response_schema` (dict) | `:1682-1701` → `output_config.format = {"type":"json_schema","schema": _ensure_additional_properties_false(dict)}` (**mutates in place? no — operates on the caller's dict**, see D9) | `:557-560` → `--json-schema` after `_ensure_additional_properties_false(copy.deepcopy(schema))` | **compatible** (both emit the same sealed schema) |
| `response_mime_type` | `:1446` used only to decide the system-prompt JSON nudge | `:530` **gates the whole schema path** — no `application/json` ⇒ no `--json-schema` | all six set it ⇒ OK |
| `max_output_tokens` | `:1441` → `max_tokens` hard API cap | `:523` read, passed to `claude_code_invoke`, then `:280` `_ = max_tokens` **NO-OP** | **D1 — real difference** |
| `temperature` | `:1442`/`:1489` → API `temperature` | **never read** | **D2 — real difference** |
| `system` / `system_instruction` | **never read** (builds its own from `_HOUSE_INSTRUCTIONS`) | `:524` → `--append-system-prompt` | **D3 — real difference** |
| `_role` | `:1897` → `agent=` on the llm_call_log row; `:1616-1621` → effort hint | `:527` → `agent="cc_rail:<role>"` | **D4** — none of the six sets it |
| `_ticker` | `:1906` | `:528` | neither sets it |
| `thinking`, `effort`, `citations`, `skill_file_id`, `data_prompt` | supported | **silently ignored** | not used by the six |
| ctor `enable_prompt_caching` | `:1345`, `:1475-1484` cache_control block | **no such ctor param** | **D7** |

### 3.2 The nine differences, ranked

**D1 — `max_output_tokens` is a no-op on the rail** (`claude_code_client.py:273-280`).
The comment is explicit: "`--max-tokens` is the SDK option name, NOT the CLI flag …
Q/A cycle-5 caught that ~63% of calls were rejected with `error: unknown option
'--max-tokens'`. Drop the flag entirely." Confirmed against the official CLI
reference (§9.2): there is **no** output-token flag; the only budget control is
`--max-budget-usd`. Consequences per service:
  - C3/C4 (512) and C5 (256), C6 (384): the cap only ever *truncated* pathological
    output. Removing it **relaxes** a limit → cannot break the parse; at worst a
    longer response (all four post-truncate their strings anyway). Low risk.
  - C1 (`min(8192, 250*n)`): same, relaxing.
  - **C2 news_screen (`min(48000, max(8192, 250*n))`, :280)** — the caller's
    hypothesis is correct that something hides here, but the direction is the
    *opposite* of a regression: phase-69.3 RAISED this cap precisely because the
    old 8192 truncated big batches. On the rail the cap disappears and the model
    default applies. Per the code comment at `claude_code_client.py:275-278` the
    CLI "uses model-default ceilings (32K for Haiku, 64K for Opus, 4K for Sonnet
    via Max plan)". **If that 32K Haiku figure is right, C2's effective budget
    FALLS from 48 000 → 32 768 on the rail** — a real, silent 32% reduction in the
    worst case (batch of 192+ headlines), landing exactly on the busiest-news-day
    failure mode 69.3 was written to fix. That figure is an in-repo assertion, NOT
    something I could confirm in Anthropic's docs (§9.2: no documented per-plan
    output ceiling table). **Treat as unverified and measure it** (§11 M-C2).
  - Mitigation that preserves intent without new machinery: keep C2's existing
    2-attempt retry, and additionally lower `max_headlines` OR chunk the batch
    when on the rail. A cheaper, honest option: leave it, and add a
    `stop_reason == "max_tokens"` check — `LLMResponse.stop_reason` IS populated
    on the rail (`claude_code_client.py:636`), so truncation is now *detectable*
    where it previously was not.

**D2 — `temperature` is unreachable on the rail.** No CLI flag exists (§9.2).
All six pass `temperature: 0.0`; C1 additionally seeds its shuffle
(`meta_scorer.py:214`) to make the batch deterministic. On the rail the sampling
temperature is whatever the CLI/session default is. **This is a genuine,
unavoidable behavior change to the signals** and cannot be mitigated inside the
six modules. It is not fatal — every service post-clamps and falls back — but
run-to-run conviction scores will vary where they previously did not. Any live
A/B of "flag on vs off" must not treat a score delta as a bug.

**D3 — the house system prompt disappears.** `ClaudeClient` builds
`system = _HOUSE_INSTRUCTIONS` (`llm_client.py:69`, a ~200-line financial-analyst
charter: cite-or-discard, conviction calibration bands, FACT_LEDGER discipline,
no-hallucinated-news, JSON output rules) **+** (for a dict schema, which is what
all six pass) the string `"\n\nYou MUST respond with a valid JSON object only."`
(`:1454-1461`; the richer `model_json_schema()` branch at :1455 is NOT taken
because `hasattr(dict, "model_json_schema")` is False). On the rail,
`config["system"]` is absent for all six ⇒ `--append-system-prompt` is never
passed ⇒ the six run against **Claude Code's default agent system prompt**
(a coding-agent persona), per the CLI reference: `--append-system-prompt`
"Append custom text to the end of the default system prompt". **This is the
single largest silent prompt change in the step** and it is invisible in a diff
that only swaps a constructor.
  - Mitigation (recommended, minimal): add `"system": <text>` to each config
    dict. `ClaudeClient` **ignores** the key entirely, so the OFF path stays
    byte-identical, while the rail forwards it. To avoid copy-pasting the
    charter six times, export a helper from `llm_client` (e.g.
    `build_house_system_prompt(json_only: bool) -> str`) that returns exactly
    `_HOUSE_INSTRUCTIONS + "\n\nYou MUST respond with a valid JSON object only."`
    — that touches `llm_client.py`, which is outside the stated boundary
    ("the six service modules + tests"), so either widen the boundary by one
    additive function or import the private `_HOUSE_INSTRUCTIONS` directly.
  - Caveat: `--append-system-prompt` APPENDS; it does not replace. The coding-agent
    preamble remains. Only `--system-prompt` replaces it, and the rail does not
    expose that flag. Full parity with the direct API is therefore **impossible**
    without extending `claude_code_invoke`.

**D4 — agent tag will be the BARE `cc_rail` shape.** `claude_code_client.py:504`:
`agent=f"cc_rail:{agent}" if agent else "cc_rail"`. None of the six passes
`_role`/`_agent`, so **the rewired rows land in the bare bucket**, adding to the
2 241-row/4.1 M-token population step 75.5.12 measured. Spend-exclusion status:
`backend/tests/test_phase_75_5_1_spend_metric.py:93` shows production now uses
`agent != 'cc_rail'` (exact equality) **in addition to** `NOT LIKE 'cc_rail:%'`,
and `:197-220` asserts the bare shape contributes **zero** priced spend. So the
rows are correctly excluded **either way** — but only once 75.5.12 lands. Two
consequences: (a) if 78.1 ships BEFORE 75.5.12, the six overlays' flat-fee tokens
get priced at API rates and inflate the $25/day metric; (b) attribution is lost
(six services collapse into one bucket). **Recommendation: pass
`"_role": "<service_name>"` in each config.** It is safe on BOTH clients —
on `ClaudeClient` it only sets the log tag and an effort hint that is
subsequently dropped for Haiku (`model_supports_effort` false, `:1625-1630`) —
and it produces the `cc_rail:meta_scorer` shape which the existing
`NOT LIKE 'cc_rail:%'` exclusion already covers, removing the 75.5.12 ordering
dependency entirely. This is the highest-value single line in the rewire.

**D5 — errors are returned, not raised.** `ClaudeClient` re-raises
`RateLimitError`/`APIStatusError` (`:1733-1746`); `ClaudeCodeClient` catches
`ClaudeCodeError` and returns `LLMResponse(text="", thoughts="errored: …")`
(`:599-614`). Also the rail-guard short-circuit returns
`text=""`, `thoughts="rail_guard_skipped: …"` (`:582-588`) **without spawning a
subprocess**. Net effect on the six: a rail failure is routed into the *parse*
except-branch instead of the *call* except-branch. All six still fall back
safely, but the recorded reason string changes (§1.1). No signal impact; audit
strings change. **A follow-on hazard**: C3/C4/C1 will now log
`"parse failed … raw="` with an EMPTY raw string on every rail failure, which
reads like a model defect rather than a rail outage. Recommend an explicit
`if not response.text: <call-failure branch>` in each service so the logs stay
diagnosable.

**D6 — the cost-budget hard block is skipped.** `ClaudeClient.generate_content`
calls `_check_cost_budget()` on line 1439 (raises `BudgetBreachError` when the
daily/monthly cap is tripped, `llm_client.py:396-423`). `ClaudeCodeClient` has no
such call. Defensible (the rail is flat-fee) but it IS a removed guard; note it
so nobody later reads the absence as an oversight.

**D7 — `enable_prompt_caching=False` cannot be expressed.** `make_client` does
not accept the kwarg at all; its direct-Anthropic branch (`:2139`) constructs
`ClaudeClient(model_name=…, api_key=…)` with the **default `True`**. So the
rewire silently flips prompt caching ON for all six on the **flag-OFF path**.
Practical impact is small — the cache write only registers when the cached block
exceeds the per-model minimum (4096 tokens on Haiku 4.5, per the comment at
`:1470-1474`), and an over-minimum write costs 1.25x — but `_HOUSE_INSTRUCTIONS`
is a large static block, so a write may well register, and cache-write tokens
are billed. This is a **cost** change on the direct rail, not a signal change.
Options: (a) accept it (and note that a HIT is then likely across the six, since
they share the same prefix — probably net-cheaper); (b) keep the direct path on a
hand-built `ClaudeClient(..., enable_prompt_caching=False)` and only use
`make_client` when the flag is on (defeats the purpose); (c) extend `make_client`
with an optional kwarg (out of boundary). **Recommend (a) + measure**, and say so
explicitly in `experiment_results.md` rather than leaving it implicit.

**D8 — no `stop_reason` recovery loop on the rail.** `ClaudeClient` retries once
on `stop_reason=="max_tokens"` with a `tool_use` tail (`:1766-1791`) and returns
a `"[refused: …]"` sentinel on `stop_reason=="refusal"` (`:1796-1805`). The rail
surfaces `stop_reason` (`:636`) but acts on nothing. For the six: the tool_use
tail is unreachable (no tools), and a refusal on the rail arrives as ordinary
text that will fail `json.loads` → fallback. Acceptable.

**D9 — schema mutation.** `ClaudeCodeClient` `copy.deepcopy`s the schema before
sealing (`:558-559`); `ClaudeClient` passes the caller's dict into
`_ensure_additional_properties_false` without a copy (`:1697`). All six build a
fresh schema per call, so no cross-call contamination either way — but C5/C6
build a literal dict inline per call and C1-C4 call `model_json_schema()` per
call, so this is benign today. Do not "optimize" any of them into a module-level
constant without re-checking this.

## 4. `max_tokens` on the rail

See **D1**. Summary answer to the caller's question: the 48 000-token cap in
`news_screen.py:280` is discarded at `claude_code_client.py:280`; the flag was
removed in phase-cycle-5 after ~63% of calls were rejected with
`unknown option '--max-tokens'`. The official CLI reference (fetched 2026-07-25)
lists **no** output-token flag — only `--max-budget-usd` and `--max-turns` — so
this is not fixable at the call site. **Yes, a real behavior change hides there**,
and it is the one place where the change could plausibly make a signal WORSE
(C2's whole-screen `return {}` on truncation). Guard it with a `stop_reason`
check rather than pretending the cap survives.

## 5. Prompt caching

Two separate questions:
- **Rail side:** `claude_code_invoke` builds argv at `:263-272` and passes no
  caching control; the CLI has no caching flag. Claude Code manages caching
  internally per session, and each invocation is a **fresh session** (no
  `--continue`/`--resume`), so cross-call prompt caching is effectively absent —
  though on the Max rail the cost is flat-fee, so the impact is **latency**, not
  dollars (§9.4).
- **Call-site side:** `enable_prompt_caching=False` is simply unrepresentable via
  `make_client` → see **D7**.

## 6. Agent tagging / llm_call_log

See **D4**. Shape today: **bare `cc_rail`** (because no `_role` is passed).
Recommended: add `_role`, producing `cc_rail:meta_scorer`, `cc_rail:news_screen`,
`cc_rail:macro_regime`, `cc_rail:pead_signal`, `cc_rail:analyst_narrative`,
`cc_rail:call_transcript_gpr`. Both shapes are excluded from metered spend once
75.5.12 lands (`test_phase_75_5_1_spend_metric.py:180-220`); only the tagged
shape is excluded by the PRE-75.5.12 predicate. Note also the rail writes via
`backend/services/observability/api_call_log.py::log_llm_call`
(`claude_code_client.py:498`) whereas `ClaudeClient` writes via
`backend/services/observability::log_llm_call` (`llm_client.py:1887`) — different
import paths; do not assume one patch covers both in tests.

## 7. Existing tests

| File | LLM-path coverage | Breaks on rewire? |
|---|---|---|
| `tests/services/test_meta_scorer.py` | **3 tests** patch `backend.agents.llm_client.ClaudeClient` (:161, :186, :205) with `settings_mock = MagicMock()` | **YES — all three.** See §7.1 |
| `tests/services/test_news_screen.py` | none (schema, dedup, cache, apply-fn only) | no |
| `tests/services/test_macro_regime.py` | none (schema, apply-fn, cache only) | no |
| `tests/services/test_pead_signal.py` | none (schema, apply-fn, cache, trailing-mean only) | no |
| C5 `analyst_narrative_scorer` | **NO test file exists** | n/a (gap) |
| C6 `call_transcript_gpr` | **NO test file exists** | n/a (gap) |
| `backend/tests/test_phase_75_llm_rail.py:86-108` | `test_dict_schema_path_is_preserved_not_replaced` — already guards the dict branch these six depend on, and its docstring names all six | no (should stay green) |

### 7.1 Why the three meta_scorer tests break

They build `settings_mock = MagicMock()` and set only `anthropic_api_key` +
`meta_scorer_model`. Every other attribute is an **auto-created truthy
MagicMock**. So after the rewire, inside `make_client`:
`getattr(settings, "paper_use_claude_code_route", False)` → truthy MagicMock →
the CC branch at `llm_client.py:2099` is taken → `int(getattr(settings,
"claude_code_timeout_s", 150))` → `int(MagicMock())` raises **TypeError**, which
is NOT caught (`:2114` catches `ImportError` only) → the test errors out. And
even if it did not, `patch("backend.agents.llm_client.ClaudeClient")` would no
longer intercept anything. **This is a feature, not just breakage**: it means a
naive rewire fails loudly in CI rather than silently. Fix by setting the flag
explicitly on the mock in each test (`settings_mock.paper_use_claude_code_route
= False`), which also documents which rail each test exercises.

## 8. Consumers / output contract

| Service | Consumer | Contract |
|---|---|---|
| C1 meta_scorer | `autonomous_loop.py:926-927` → `meta_score_candidates(candidates, regime=regime)`; gated by `meta_scorer_enabled` | `list[dict]` with `conviction_score` + `conviction_reason` — unchanged |
| C2 news_screen | `autonomous_loop.py:449-450`, gate `news_screen_enabled`; applied via `screener.py` `apply_news_to_score` | `dict[str, NewsHeadlineSignal]` — unchanged |
| C3 macro_regime | `autonomous_loop.py:425-426`, gate `macro_regime_filter_enabled`; consumed at **`backend/tools/screener.py:320-324`** → `apply_regime_to_score(score, sector, SECTOR_ETFS, regime)` | `MacroRegimeOutput` — unchanged |
| C4 pead_signal | `autonomous_loop.py:439-440`, gate `pead_signal_enabled`; `screener.py:327-331` (`apply_pead_to_score`, which can return `None` → candidate dropped) | `dict[str, PeadSignalOutput]` — unchanged |
| C5 analyst_narrative | `autonomous_loop.py:760-779`, gate `analyst_narrative_enabled`; `screener.py:374-376` | `dict[str, AnalystNarrativeSignal]` — unchanged |
| C6 call_transcript_gpr | `autonomous_loop.py:734-752`, gate `call_transcript_gpr_enabled`; `screener.py:380-388` | `dict[str, GprExposureSignal]` — unchanged |

**Type contract is unchanged by a client swap** — every service converts the LLM
response into its own Pydantic type before returning, and all six loop call sites
are individually `try/except → non-fatal`. The *values* can change (D2/D3), which
is a signal-quality question, not a contract question. Which of the six
`*_enabled` gates are ON in production could not be read (`backend/.env` denied);
Main must confirm before claiming live impact.



## 4. max_tokens on the rail

_(pending)_

## 5. Prompt caching on the rail

_(pending)_

## 6. Agent tagging / llm_call_log shape

_(pending)_

## 7. Existing tests

_(pending)_

## 8. Consumers / output contract

_(pending)_

## 9. External research

### 9.0 Read in full (>= 5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| S1 | https://code.claude.com/docs/en/headless | 2026-07-25 | official doc | WebFetch | `--json-schema` takes **inline JSON** (docs' own example); invalid schema ⇒ `Error: --json-schema is not a valid JSON Schema`; `--bare` "will become the default for `-p` in a future release"; without `--bare`, `claude -p` "loads the same context an interactive session would, including anything configured in the working directory or `~/.claude`" |
| S2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-07-25 | official doc | WebFetch | API structured outputs "guarantee schema-compliant responses through **constrained decoding**"; GA "for Claude 4.5 and later"; unsupported: `minimum`/`maximum`/`minLength`/`maxLength`/recursive/external `$ref`; "If you use an unsupported feature, you'll receive a 400 error" |
| S3 | https://code.claude.com/docs/en/cli-reference | 2026-07-25 | official doc | WebFetch | `--json-schema`: "Get **validated** JSON output matching a JSON Schema **after the agent completes its workflow**"; `--append-system-prompt` = append to default, `--system-prompt` = replace; **no temperature and no max-output-token flag exists**; `--bare` "skip auto-discovery of hooks, skills, plugins, MCP servers, auto memory, and CLAUDE.md"; `--strict-mcp-config` without `--mcp-config` ⇒ no MCP servers load |
| S4 | https://code.claude.com/docs/en/agent-sdk/structured-outputs | 2026-07-25 | official doc | WebFetch | "the SDK validates the output against it, **re-prompting on mismatch**. If validation does not succeed within the retry limit, the result is an error"; failure surfaces as `subtype: error_max_structured_output_retries`; "A result can also end with subtype `success` but no `structured_output` value … Treat that case as a failure"; "The SDK validates schemas with **JSON Schema draft-07**, so schemas that declare a newer version are rejected" |
| S5 | https://blog.e-infra.cz/blog/claude-benchmark/ | 2026-07-25 | industry benchmark (Apr 2026, 600+ headless runs) | WebFetch | "Claude Code's system prompt (tool definitions included) weighed roughly **21K tokens**"; median headless latency **32–64 s** for native Claude; Track C failures at a 300 s timeout |
| S6 | https://www.claudecodecamp.com/p/how-prompt-caching-actually-works-in-claude-code | 2026-07-25 | practitioner analysis | WebFetch | cache "expires after roughly **5 minutes of inactivity**"; cached layers = system prompt + "Tool definitions and MCP tools" + "**CLAUDE.md file content**" + history; cache reads 90% cheaper, cache **writes +25%** |
| S7 | https://github.com/anthropics/claude-code/issues/9058 | 2026-07-25 | issue tracker | WebFetch (**PARTIAL** — replies failed to load; body only) | The original `--json-schema` request explicitly asked for "constrained decoding"/"schema compliance at the token generation level"; **NOT counted toward the gate** |

### 9.1 Are the API and the rail's schema guarantees the SAME? — **NO.**

This is the single most consequential external finding.

- **API (`ClaudeClient` today):** *"Structured outputs guarantee schema-compliant
  responses through **constrained decoding** … Always valid: No more
  `JSON.parse()` errors"* (S2). Grammar-constrained sampling; a violating token
  is never emitted.
- **CLI rail (`--json-schema`):** *"Get **validated** JSON output matching a JSON
  Schema **after the agent completes its workflow**"* (S3) and *"the SDK
  validates the output against it, **re-prompting on mismatch**. If validation
  does not succeed within the retry limit, the result is an error instead of
  structured data"* (S4).

So the rail is **post-hoc validate + internal re-prompt**, not constrained
decoding. Three concrete consequences for the six:

1. **Fail-safe, but only by luck of our error handling.** A validation
   exhaustion produces `subtype = "error_max_structured_output_retries"`.
   `claude_code_invoke` raises `ClaudeCodeError` on any non-`success` subtype
   (`claude_code_client.py:362-370`), which `ClaudeCodeClient` converts to an
   empty `LLMResponse` (`:599-614`), which every one of the six turns into its
   own fallback. **Verified by code reading, not by a live run.**
2. **The hidden `success`-without-`structured_output` case (S4).** Our
   `extract_result_text` (`:429-443`) then falls through to `envelope["result"]`
   — free-form prose — which fails `json.loads` and lands in the fallback. Also
   safe, also silent. Worth an explicit log line.
3. **Latency and quota are non-deterministic**: an internal re-prompt loop means
   one logical call can be several model turns. On a flat-fee rail that is
   latency + Max-quota, not dollars.

### 9.2 What the CLI can and cannot express vs the API

| Capability | API (`ClaudeClient`) | CLI rail | Source |
|---|---|---|---|
| JSON schema | `output_config.format`, constrained decoding | `--json-schema`, post-hoc validate + re-prompt | S2, S3, S4 |
| Schema dialect | Anthropic's own supported-keyword list | **draft-07**; newer `$schema` declarations rejected | S2, S4 |
| max output tokens | `max_tokens` | **no flag** (`--max-budget-usd`, `--max-turns` only) | S3 |
| temperature | `temperature` | **no flag** | S3 |
| system prompt | full control | `--append-system-prompt` (append) / `--system-prompt` (replace) — our wrapper only wires the append form (`claude_code_client.py:269-270`) | S1, S3 |
| tool use | request-level | agentic loop, tools removed by name via `--disallowedTools` | S3 |
| prompt caching | explicit `cache_control` + `ttl:"1h"` | implicit, ~5 min inactivity TTL, per session | S6 |

Verified against our schemas: none of the four Pydantic models emits a
`$schema` key, and `_strip_unsupported_schema_keys`
(`macro_regime.py:330-341`, keys = `exclusiveMaximum, exclusiveMinimum,
maxLength, maximum, minLength, minimum`) removes every constraint the API
rejects — measured live below, so the draft-07 rejection risk is **low**:

```
MetaScorerBatch    raw risky: conviction_score.minimum=1 / .maximum=10   -> after strip: none  ($defs: yes, 824 B)
NewsSignalBatch    raw risky: none                                        -> after strip: none  ($defs: yes, 1769 B)
MacroRegimeOutput  raw risky: rationale.maxLength=300, conviction 0..1,
                              conviction_multiplier 0.5..1.5              -> after strip: none  ($defs: yes, 1685 B)
PeadSignalOutput   raw risky: sentiment_score.minimum/maximum             -> after strip: none  ($defs: no,  1166 B)
```

Residual: three of the six schemas use `$defs` + internal `$ref`. The API docs
list `$ref`/`$def`/`definitions` as supported (S2); draft-07 resolves
`#/$defs/...` as a plain JSON pointer. Low risk, but it is the **first thing
the live smoke must prove** (§11 M-SCHEMA) because none of these six has ever
executed on the rail.

### 9.3 Throughput: is serialized subprocess invocation a problem?

Yes, materially, and it is under-appreciated.

- **Per-invocation latency**: median **32–64 s** for native Claude in 600+
  headless benchmark runs, April 2026 (S5). Our own code corroborates: the
  phase-60.1 note at `claude_code_client.py:471-477` records **88.9 s observed
  live 2026-06-11**, and the default subprocess timeout is 150 s
  (`settings.claude_code_timeout_s`). Direct Haiku 4.5 API calls are seconds.
- **Per-invocation context tax**: Claude Code's system prompt + tool definitions
  is *"roughly 21K tokens"* (S5). Our `disallowed_tools` default
  (`claude_code_client.py:223`) removes Bash/Edit/Write/Read/Glob/Grep/Agent
  from context (per S3, "A bare tool name removes the matching tools from
  Claude's context"), which trims it — but **it does not remove MCP tools**;
  S3 says that needs `--disallowedTools "mcp__*"` or `--strict-mcp-config`
  without `--mcp-config`.
- **We inherit the repo as cwd.** `claude_code_invoke` passes `cwd=None`
  (`:222`, `:305`) so the subprocess inherits the backend's working directory,
  and `~/Library/LaunchAgents/com.pyfinagent.backend.plist` sets
  `WorkingDirectory = /Users/ford/.openclaw/workspace/pyfinagent`. Per S1,
  without `--bare` `claude -p` "loads the same context an interactive session
  would, including anything configured in the working directory or `~/.claude`".
  Measured locally: **`CLAUDE.md` is 34 333 bytes**, `.mcp.json` declares
  **8 MCP servers** (`alpaca, bigquery, paper-search-mcp, pyfinagent-backtest,
  pyfinagent-data, pyfinagent-risk, pyfinagent-signals, playwright`), and
  `.claude/settings.json` registers **`SessionStart` + `InstructionsLoaded`
  hooks** that fire per session (the latter appends to the tracked
  `handoff/audit/instructions_loaded_audit.jsonl`).
  **Every CC-rail call today pays this.** This is a PRE-EXISTING rail defect
  (it already applies to the whole Layer-1 pipeline), not something 78.1
  creates — but 78.1 extends it to six more call sites, two of which
  (C5, C6) are **per-ticker**.
- **Per-ticker blast radius**: C5/C6 iterate `2 * paper_screen_top_n` tickers
  behind an `asyncio.Semaphore`. At 30-60 s/call, 20-40 tickers, and modest
  concurrency, that is tens of minutes added to a cycle whose hard wall-clock
  budget is `paper_cycle_max_seconds = 7200` (`settings.py:33` — already raised
  to 2 h *because of* this rail). **C5 and C6 are the two services where the
  rewire could plausibly blow the cycle budget.**

### 9.4 Prompt caching on the rail — latency, not dollars

The Max rail is flat-fee, so frame this as latency/quota. Anthropic's cache
TTL is *"roughly 5 minutes of inactivity"* (S6) and what gets cached includes
the system prompt, tool/MCP definitions, and **CLAUDE.md**. Each
`claude_code_invoke` is a fresh session (no `--continue`/`--resume`), and the
six overlays fire roughly once per daily cycle, so **every call is a cold
cache** — full re-ingest of the ~21 K-token preamble plus CLAUDE.md, every
time. Within a single cycle, back-to-back per-ticker C5/C6 calls (< 5 min
apart) plausibly DO hit the shared cache; that is the one place caching helps.
No dollar impact on Max; on the direct rail see **D7** (cache writes cost +25%,
S6).

## 10. Recency scan (2024-2026) — PERFORMED

Search-query discipline (3 variants, per `.claude/rules/research-gate.md`):
- **Current-year (2026):** `Claude Code CLI headless --json-schema structured output 2026`;
  `Claude Code headless claude -p subprocess startup latency concurrency batch workload 2026`
- **Year-less canonical:** `wrapping Claude CLI as programmatic backend production reliability`
- **Last-2-year (2025):** `Anthropic prompt caching 5 minute TTL cache misses separate CLI sessions 2025`

**Result: 4 new findings inside the 2024-2026 window that materially change
the plan.**

1. **(2026) `--json-schema` is post-hoc validated, not constrained-decoded**
   (S3/S4). Supersedes the natural assumption — carried implicitly in
   `claude_code_client.py`'s docstring, which cites the structured-outputs docs
   as if the two rails were equivalent — that `--json-schema` gives the same
   guarantee as `output_config.format`.
2. **(2026) `--bare` "will become the default for `-p` in a future release"**
   (S1). Bare mode "skips OAuth and keychain reads. Anthropic authentication
   must come from `ANTHROPIC_API_KEY`". **When that flip lands, the entire
   Max-subscription rail breaks** — and it would break toward the *metered*
   API, i.e. exactly the failure mode phase-72 diagnosed. Our code comment at
   `claude_code_client.py:258-261` already says "Do NOT add `--bare`"; it now
   also needs a *version tripwire* for the day the default inverts.
   → queued defect **Q2**.
3. **(2026, v2.1.205) invalid-schema behavior changed**: "Before v2.1.205,
   Claude Code silently ignored an invalid schema and returned unstructured
   text" (S1/S3). On an older CLI the six would run **unconstrained and
   silently**. The live smoke must record `claude --version`.
4. **(2025-2026) prompt-cache TTL regression to ~5 min** (S6, corroborated by
   GitHub issues #14628 / #46829 and The Register 2026-04-13 in the
   snippet-only set). Confirms the "always cold" analysis in §9.4 and is the
   documented basis for `llm_client.py:1467-1474`'s explicit `ttl:"1h"`.

No finding in the window contradicts the rewire itself; two (items 2 and 3)
add tripwires that did not previously exist in the plan.

## 11. Per-service rewire recipe

### 11.0 The shared edit

For each service, replace the direct construction with `make_client`, keep the
empty-key early-return, and add two config keys. Illustrated on C1
(`meta_scorer.py:220-237`); the other five are the same shape.

```python
# BEFORE (meta_scorer.py:220-225)
from backend.agents.llm_client import ClaudeClient
client = ClaudeClient(
    model_name=getattr(settings, "meta_scorer_model", "claude-haiku-4-5"),
    api_key=anthropic_key,
    enable_prompt_caching=False,
)

# AFTER
from backend.agents.llm_client import make_client
try:
    client = make_client(
        getattr(settings, "meta_scorer_model", "claude-haiku-4-5"),
        None,                 # vertex_model: unused on every claude-* branch
        settings,
    )
except Exception as e:                       # routing-breach ValueError, ImportError, ...
    logger.warning("meta_scorer: client init failed: %s", e)
    return _fallback_all(head + tail)
```

and in the `generation_config` dict add:

```python
    "_role": "meta_scorer",          # -> llm_call_log agent='cc_rail:meta_scorer'
    "system": _house_system_prompt(),  # ClaudeClient IGNORES this key; the rail uses it
```

Per-service `except` bodies (the ONLY thing that differs):

| # | file:line to change | init-failure fallback |
|---|---|---|
| C1 | `meta_scorer.py:220-225` | `return _fallback_all(head + tail)` |
| C2 | `news_screen.py:266-271` | `return {}` |
| C3 | `macro_regime.py:505-510` | `return _fallback_regime(indicators, f"client init failed: {type(e).__name__}")` |
| C4 | `pead_signal.py:278-283` | `return _fallback("client_init_failed")` |
| C5 | `analyst_narrative_scorer.py:133-142` | already wrapped — swap in place, keep `return None` |
| C6 | `call_transcript_gpr.py:111-120` | already wrapped — swap in place, keep `return None` |

`_role` values: `meta_scorer`, `news_screen`, `macro_regime`, `pead_signal`,
`analyst_narrative`, `call_transcript_gpr`.

C5/C6 additionally need `settings` in scope — they already call
`get_settings()` at `:105` / `:89`, so pass that object; their `model` stays
the function argument.

### 11.1 Two explicit decisions the executor must make (do not decide silently)

**D-KEY — the empty-key guard.** Today all six bail out when
`ANTHROPIC_API_KEY` is empty. On the rail **no API key is needed** (the CLI
authenticates via `~/.claude/` OAuth, and `claude_code_invoke:295-298`
deliberately *scrubs* `ANTHROPIC_API_KEY` from the subprocess env). So after
the rewire, a missing key still silently disables all six even though the rail
would serve them. **Recommendation: keep the guard but make it rail-aware**:

```python
_rail_on = bool(getattr(settings, "paper_use_claude_code_route", False))
if not anthropic_key and not _rail_on:
    ...existing early return...
```

Rationale: otherwise the step is cosmetic in exactly the scenario it exists to
fix (dead/absent Anthropic credits). **The guard must not be deleted outright**
— it is also what keeps the GitHub-Models branch (`llm_client.py:2148`)
unreachable, and `claude-haiku-4-5` IS in that catalog (§2.1).

**D-SYS — the house system prompt.** Nothing in the repo passes
`config["system"]`/`["system_instruction"]` to `generate_content` (grepped
across `backend/agents/` + `backend/services/`), so `--append-system-prompt` is
**never used on the rail today** — the whole Layer-1 pipeline already runs
without `_HOUSE_INSTRUCTIONS` when the flag is on. Options:
(a) add `"system"` to the six (recommended — restores parity for the six and
costs one helper); (b) skip it and file the parity gap as a queued defect for
ALL rail callers (**Q3**). Either way, say which you chose in
`experiment_results.md`; do not leave it implicit.

### 11.2 Flag-flip test shape (one per service)

```python
@pytest.mark.parametrize("flag,expected", [(True, "ClaudeCodeClient"),
                                           (False, "ClaudeClient")])
def test_meta_scorer_client_type_follows_rail_flag(monkeypatch, flag, expected):
    """phase-78.1: PAPER_USE_CLAUDE_CODE_ROUTE must actually reach this service."""
    import backend.agents.llm_client as lc
    from backend.services.meta_scorer import meta_score_candidates

    # SimpleNamespace, NOT MagicMock: a MagicMock makes every flag truthy and
    # int(mock) raises inside make_client (see brief section 7.1).
    settings_stub = SimpleNamespace(
        anthropic_api_key="sk-ant-test", meta_scorer_model="claude-haiku-4-5",
        paper_use_claude_code_route=flag, claude_code_timeout_s=150,
        paper_synthesis_integrity_enabled=False,
    )
    monkeypatch.setattr("backend.services.meta_scorer.get_settings",
                        lambda: settings_stub)

    seen = {}
    real_make = lc.make_client

    def spy(model, vertex, settings):
        client = real_make(model, vertex, settings)
        seen["type"] = type(client).__name__
        seen["cfg"] = None
        def fake_gen(prompt, cfg=None):
            seen["cfg"] = cfg
            return SimpleNamespace(text='{"candidates":[]}', stop_reason="end_turn")
        client.generate_content = fake_gen        # no subprocess, no HTTP
        return client

    monkeypatch.setattr(lc, "make_client", spy)
    asyncio.run(meta_score_candidates([_mk_cand("AAPL")]))

    assert seen["type"] == expected, (
        f"flag={flag} must select {expected}; got {seen['type']} -- "
        "PAPER_USE_CLAUDE_CODE_ROUTE does not reach this call site")
    assert seen["cfg"]["_role"] == "meta_scorer", (
        "missing _role -> llm_call_log row lands in the BARE 'cc_rail' bucket")
```

The lazy `from backend.agents.llm_client import make_client` **inside** the
function is what makes `monkeypatch.setattr(lc, "make_client", spy)` work —
keep the import lazy (it also preserves the existing import-cycle avoidance).

### 11.3 Mutations that MUST turn each guard red

Per `feedback_mutation_test_guards_and_fixtures`: mutate the STUB too, and
mutate first the guard you catch yourself defending.

| ID | Mutation | Guard that must go red |
|----|----------|------------------------|
| M1 | revert one service to `ClaudeClient(...)` | its `flag=True` parametrization (`seen["type"] == "ClaudeCodeClient"`) |
| M2 | drop `"_role"` from that service's config | the `_role` assertion → proves the agent-tag claim isn't vacuous |
| M3 | in `make_client`, force the CC branch unconditionally | the `flag=False` parametrization (`== "ClaudeClient"`) → proves the test can see BOTH rails, not just the happy one |
| M4 | change `_role` to a different service's name | a cross-service assertion (each test asserts ITS own name) → kills copy-paste |
| M5 | mutate the STUB: make `fake_gen` return `text=""` | the service's fallback assertion must still hold (proves the D5 empty-response path is exercised, not just the happy path) |
| M6 | delete the empty-key early return | a `no-key ⇒ fallback` test must go red (protects the GitHub-Models capture, §2.1) |
| M7 | (if D-SYS = option a) drop `"system"` from the config | a `seen["cfg"]["system"].startswith("You are a financial analysis AI")` assertion |
| M8 | in `claude_code_client.py`, revert `:557` to reject dicts | `backend/tests/test_phase_75_llm_rail.py::test_dict_schema_path_is_preserved_not_replaced` (already exists) |

M3 and M5 are the two most likely to be skipped and the two that matter most:
without M3 the suite cannot distinguish "the flag works" from "the CC branch
always fires"; without M5 the suite proves routing but not survival.

### 11.4 Dummy-key $0-leakage proof

Three independent legs; run all three, they fail differently.

1. **Unit (deterministic, no network).** With the flag ON, monkeypatch
   `backend.agents.llm_client._anthropic_sdk = None`. Any accidental
   direct-Anthropic construction then raises `ImportError("anthropic package
   not installed")` from `ClaudeClient._get_client()` (`llm_client.py:1363`),
   while the rail path (with `claude_code_invoke` patched) still succeeds.
   A green test under this monkeypatch is positive evidence that **no
   direct-API client was used**.
2. **Argv/env proof.** Patch `claude_code_client.subprocess.run`, invoke each
   service, and assert (a) `argv[0]` resolves to the `claude` binary,
   (b) `"--json-schema"` is in argv with the service's own sealed schema,
   (c) `env` passed to `subprocess.run` contains **neither**
   `ANTHROPIC_API_KEY` **nor** `ANTHROPIC_AUTH_TOKEN` — the scrub at
   `claude_code_client.py:295-298` is what makes CLI-side metered billing
   impossible even with a live key in the parent env.
3. **Live/BQ (post-flip).** After the next real cycle with the flag on:
   `SELECT agent, COUNT(*), SUM(input_tok+output_tok) FROM llm_call_log
   WHERE ts > <flip_ts> AND agent LIKE 'cc_rail%' GROUP BY agent`
   must show six `cc_rail:<service>` buckets, and the same window must show
   **zero** rows whose `provider='anthropic'` carry a non-`cc_rail` agent for
   these models. Cross-check against the 75.5.1 spend metric: those rows must
   contribute **$0** (`test_phase_75_5_1_spend_metric.py:180-220`).

### 11.5 Live smoke — the three things that have never been proven

Because these six have **0 rows in 30 days** (a genuine zero — `ClaudeClient`
IS instrumented at `llm_client.py:1887`), nothing about them on the rail is
empirically established. Minimum live evidence for the live_check:

- **M-SCHEMA**: one real `claude -p --json-schema` round-trip per service with
  the ACTUAL sealed schema (three of them carry `$defs`/`$ref`), recording
  `claude --version` (must be >= 2.1.205, §10 item 3) and the returned
  `subtype`.
- **M-C2**: the news_screen output ceiling. Send a deliberately large batch and
  record `stop_reason`; this is the only place the max-tokens no-op (D1) can
  bite. If `stop_reason == "max_tokens"`, the 32 K-Haiku-ceiling hypothesis is
  confirmed and C2 needs chunking before the flag is trusted.
- **M-LAT**: wall-clock per call for C5/C6 across a realistic ticker count,
  compared against `paper_cycle_max_seconds` (7200).

## 12. RISK ranking — what could break live signals

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| **R1** | **C5/C6 per-ticker latency blows the cycle budget** (30-90 s x 2*top_n serialized behind a semaphore, §9.3) | HIGH | cycle timeout ⇒ *no trades*, i.e. the exact 97 %-cash failure class | Rewire C5/C6 **last**, behind their own live_check (M-LAT); consider leaving them on direct-API until latency is measured; they are already default-OFF |
| **R2** | `make_client` raises (routing-breach `ValueError` at `:2129`, or `int(MagicMock)`-class `TypeError`) **outside** the existing try ⇒ exception escapes the service | MED | overlay dies; loop catches it non-fatally, so degraded not fatal | wrap `make_client` in try/except per §11.0 — non-optional |
| **R3** | **C2 news_screen output ceiling silently drops from 48 000 → model default** (D1) | MED | on the busiest news days the batch truncates ⇒ `return {}` ⇒ whole news overlay lost — the exact bug 69.3 fixed | check `stop_reason` (available on the rail, `:636`); chunk or lower `max_headlines`; M-C2 live proof |
| **R4** | **No house system prompt on the rail** (D3) ⇒ six financial classifiers run against Claude Code's coding-agent persona (and, per §9.3, with 34 KB of CLAUDE.md in context) | MED-HIGH | silent signal-quality drift; conviction calibration bands, cite-or-discard and no-hallucinated-news rules all disappear | D-SYS option (a): pass `"system"` (ignored by `ClaudeClient`, used by the rail) |
| **R5** | **Temperature 0.0 is unreachable** (D2) ⇒ non-deterministic scores | HIGH (certain) | run-to-run variance in conviction/regime; A/B comparisons become noisy | unavoidable; document it, and never treat an on-vs-off score delta as a defect |
| **R6** | Schema rejected by the CLI's draft-07 validator (`$defs`/`$ref`) ⇒ hard exit ⇒ all six permanently fall back | LOW (measured clean, §9.2) | six overlays silently dead — indistinguishable from today | M-SCHEMA live smoke per service, with `claude --version` recorded |
| **R7** | Rows land in the **bare `cc_rail`** bucket (D4) and 78.1 ships before 75.5.12 | MED | flat-fee tokens priced at API rates ⇒ inflated $25/day metric ⇒ false spend alarm | pass `_role` (§11.0) — removes the ordering dependency entirely |
| **R8** | `enable_prompt_caching` silently flips **True** on the direct rail (D7) | HIGH (certain) | +25 % on cache-write tokens; probably net-cheaper via hits | accept + measure; state it explicitly |
| **R9** | Empty-key guard removed ⇒ **GitHub Models** captures the call (§2.1, `claude-haiku-4-5` IS in the catalog) | LOW | silent third-party metered rail | do not touch the guard; M6 mutation |
| **R10** | `--bare` becomes the `-p` default (§10 item 2) | LOW now, CERTAIN eventually | entire Max rail flips to metered/broken auth | queued defect **Q2**: version tripwire on `claude --version` |

**Can all six be safely rewired?** Four yes, two with a condition:

- **C1 meta_scorer, C2 news_screen, C3 macro_regime, C4 pead_signal — YES**, with
  R2/R3/R4 mitigations. Single call per cycle; latency is a non-issue; all have
  fallbacks; C1 additionally has clamping.
- **C5 analyst_narrative_scorer, C6 call_transcript_gpr — YES *but not in the
  same breath as C1-C4*.** They are the only per-ticker services, they are the
  only two with no test file at all, and R1 is a cycle-killer. **Recommendation:
  rewire all six in code, but land C5/C6 behind their own live_check evidence
  (M-LAT) and note that both are default-OFF today** (`analyst_narrative_enabled`,
  `call_transcript_gpr_enabled`), so the code change is dormant until the
  operator enables them. That is an honest "C5/C6 need M-LAT first" rather than
  six uniform green lights.

### 12.1 Defects discovered out-of-scope — queue as their own masterplan steps

Per `feedback_queue_discovered_defects_in_masterplan` (write for an executor
with no memory of this discovery):

- **Q1** — Every CC-rail invocation inherits the repo as cwd (`cwd=None`,
  `claude_code_client.py:222/305`; backend `WorkingDirectory` = repo root), so
  `claude -p` loads `CLAUDE.md` (34 333 B), 8 MCP servers from `.mcp.json`, and
  the `SessionStart`/`InstructionsLoaded` hooks on **every** rail call. Proposed
  fix: pass a neutral `cwd`, add `--strict-mcp-config` (no `--mcp-config`) and
  `mcp__*` to `disallowed_tools`. Affects the whole Layer-1 rail, not just 78.1.
- **Q2** — `--bare` will become the `-p` default (Anthropic doc, S1); bare mode
  skips OAuth/keychain and requires `ANTHROPIC_API_KEY`, which would silently
  move the rail back to metered billing. Needs a `claude --version` tripwire +
  an explicit opt-out flag when one exists.
- **Q3** — No caller anywhere passes `config["system"]`, so
  `--append-system-prompt` is dead and `_HOUSE_INSTRUCTIONS` is absent from
  **every** rail call today (all of Layer-1 included). Needs a rail-wide parity
  decision, not a six-service patch.
- **Q4** — `make_client:2114` catches only `ImportError` around the
  `ClaudeCodeClient` construction; any other exception (e.g. a bad
  `claude_code_timeout_s`) escapes instead of falling through. Consider
  broadening to `Exception` with a WARN, or documenting the intent.

## 13. Research Gate Checklist

Hard blockers:
- [x] >= 5 authoritative external sources READ IN FULL via WebFetch (S1-S6 = 6;
      S7 partial and NOT counted)
- [x] 10+ unique URLs total (30 collected — see §9.0 + §13.1)
- [x] Recency scan (last 2 years) performed + reported (§10, 4 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (6 services,
      `llm_client.py`, `claude_code_client.py`, `screener.py`,
      `autonomous_loop.py`, `settings.py`, 5 test files, `.mcp.json`,
      `.claude/settings.json`, launchd plist)
- [x] Contradictions noted (S2 constrained decoding vs S3/S4 post-hoc
      validation — the central finding)
- [x] Claims cited per-claim
- [ ] **Gap (disclosed):** `backend/.env` is denied to the researcher sandbox,
      so I could NOT verify which of the six `*_enabled` flags and
      `PAPER_USE_CLAUDE_CODE_ROUTE` are actually set, nor whether
      `GITHUB_TOKEN` exists. Main must confirm before claiming live impact.
- [ ] **Gap (disclosed):** the "32K Haiku / 64K Opus / 4K Sonnet" CLI output
      ceiling is an in-repo assertion (`claude_code_client.py:275-278`) that I
      could not corroborate in Anthropic's docs. R3/M-C2 must measure it.

### 13.1 Snippet-only sources (context; do NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://amux.io/guides/claude-code-headless/ | guide | **HTTP 403** on WebFetch |
| https://earthly.dev/lunar/guardrails/claude/cli-structured-output/ | vendor guide | superseded by S3/S4 official docs |
| https://github.com/anthropics/claude-code/issues/14628 | issue | cache-TTL regression (Dec 2025), corroborates S6 |
| https://github.com/anthropics/claude-code/issues/46829 | issue | cache-TTL regression (Mar 2026), corroborates S6 |
| https://www.theregister.com/software/2026/04/13/anthropic-claude-quota-drain-not-caused-by-cache-tweaks/5222501 | press | Anthropic's counter-claim on quota drain |
| https://particula.tech/blog/anthropic-prompt-cache-ttl-5-minute-regression-debugging | blog | duplicate of S6's TTL finding |
| https://platform.claude.com/docs/en/build-with-claude/batch-processing | official doc | Batch API is a *different* rail (not applicable to a Max CLI) |
| https://avasdream.com/blog/claude-cli-agentic-wrapper | blog | year-less canonical hit; wrapper patterns already covered by S1/S3 |
| https://github.com/ChrisColeTech/claude-wrapper | repo | community wrapper; lower tier |
| https://www.buildthisnow.com/blog/guide/development/claude-code-headless-mode | blog | superseded by S1 |
| https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html | blog | CI/CD focus, not batch latency |
| https://egghead.io/enforcing-structured-output-with-json-schema-and-zod-in-claude-code-workflows~fm674 | course | Zod/draft-07 angle covered by S4 |
| https://stevekinney.com/courses/self-testing-ai-agents/structured-cli-output-as-pipeline-glue | course | ditto |
| https://www.gradually.ai/en/changelogs/claude-code/ | changelog mirror | unofficial mirror; prefer official docs |
| https://agentfactory.panaversity.org/docs/General-Agents-Foundations/claude-code-teams-cicd/claude-code-in-cicd-pipelines | course | community tier |
| https://mcpmarket.com/tools/skills/claude-headless-mode-automation | listing | marketing page |
| https://www.mindstudio.ai/blog/build-cli-claude-code-printing-press | blog | off-topic (CLI generation) |
| https://blakecrosley.com/guides/claude-code | guide | general |
| https://www.developersdigest.tech/blog/claude-api-reliability-error-handling | blog | API-side retries, not CLI |
| https://www.claudepluginhub.com/agents/andersonlimahw-cli-wrapper-plugins-cli-wrapper/agents/claude-cli | listing | community tier |
| https://introl.com/blog/claude-code-cli-comprehensive-guide-2025 | guide | 2025 canonical variant hit |
| https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching | vendor | Bedrock rail, not applicable |
| https://code.claude.com/docs/en/agent-sdk/overview | official doc | linked from S1; SDK packages out of scope |
| https://code.claude.com/docs/llms.txt | index | doc index only |

### 13.2 JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 24,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 17,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All six anchors re-confirmed. The rewire is mechanically simple (make_client(model, None, settings) + a guarded try/except) but carries nine concrete API differences between ClaudeClient and ClaudeCodeClient. The biggest: the CLI's --json-schema is POST-HOC VALIDATED WITH INTERNAL RE-PROMPTING (official CLI reference + Agent SDK docs), not constrained decoding like the API's output_config.format -- the guarantees are NOT the same, though our error path fails safe. temperature=0.0 is unreachable on the rail (no CLI flag exists), so determinism is lost. max_output_tokens is a documented no-op, which relaxes C1/C3/C4/C5/C6 harmlessly but may silently LOWER news_screen's 48K cap to a model default -- the one place a signal could get worse. No caller anywhere passes config['system'], so the house financial-analyst prompt disappears on the rail (pre-existing, rail-wide). None of the six passes _role, so rows would land in the BARE 'cc_rail' bucket; passing _role fixes attribution and removes the 75.5.12 ordering dependency. Three meta_scorer tests break (MagicMock settings make every flag truthy); C5/C6 have no tests at all and are per-ticker, so latency (32-64s/call measured externally, 88.9s observed in-repo) makes them the only genuine blocker: rewire them but gate on measured latency. Four out-of-scope defects queued (repo-cwd context tax, --bare default flip, rail-wide system-prompt gap, narrow ImportError catch).",
  "brief_path": "handoff/current/research_brief_78.1.md",
  "gate_passed": true
}
```
