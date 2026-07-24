# Research Brief — phase-78.1: Rewire six signal-overlay services (C1-C6) off direct `ClaudeClient` onto the CC rail

**Tier:** moderate. **Audit-class:** false. **Started:** 2026-07-24.
**Status:** IN PROGRESS (write-first; sections appended as sources are read).

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

_(pending)_

## 10. Recency scan (2024-2026)

_(pending)_

## 11. Per-service rewire recipe

_(pending)_

## 12. RISK ranking

_(pending)_

## 13. Research Gate Checklist + envelope

_(pending)_
