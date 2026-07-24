# Research Brief -- Masterplan step 75.5

**Step:** Audit75 S5 -- live CC-rail schema enforcement, metered-bypass guards,
model-retirement + telemetry/cost correctness
**Tier:** complex (NOT audit-class -- bounded enumerated list of 7 sub-items)
**Researcher spawn:** 2026-07-20
**Status:** COMPLETE -- `gate_passed: true`

---

## 0. Scope

Seven sub-items (a)-(g) across the live LLM/cost rail:

| id | audit id | Claim (as written in the step) | Verdict |
|----|----------|-------------------------------|---------|
| (a) | llmeng-01 | `ClaudeCodeClient.generate_content` drops Pydantic model classes via `isinstance(dict)` gate | **CONFIRMED** (stronger than stated -- 9/9 pipeline schemas are classes, so `--json-schema` is dead code) |
| (b) | llmeng-03 | `advisor_call` bypasses `_check_cost_budget` + CC-route breach guard | **CONFIRMED** (3 defects; `def` is at `:2075`, not `:2110`) |
| (c) | llmeng-04 | `LLMResponse` lacks `stop_reason`; retry ownership must be deconflicted | **PARTIAL** -- 3 corrections: `LLMResponse` is at `:653` not `:1684`; no shared JSON-parse helper exists; **TWO** retry owners exist |
| (d) | llmeng-06 | 5 hardcoded gemini-2.5 pins; need constants + retirement warning | **PARTIAL** -- date confirmed official; ~12 pins not 5; `GEMINI_DEEP_THINK` does not exist yet |
| (e) | llmeng-10 | `cost_tracker.record` double-subtracts cache tokens | **CONFIRMED** -- Anthropic doc is explicit; 66.7% under-report in the worked example |
| (f) | llmeng-11 | `OpenAIClient.generate_content` missing `log_llm_call` | **CONFIRMED** (`def` is at `:1140`, not `:1214`) |
| (g) | arch-04 | promote spend fetch into `backend/services/observability.fetch_spend()` | **CONFIRMED** -- 3 prod consumers + 1 test that will break |

**Headline corrections the contract must carry:** (i) criterion 5's
`UsageMeta(input=…, cache_read=…)` kwargs do not exist -- the real fields are
`prompt_token_count` / `cache_read_input_tokens`; (ii) (c)'s "shared JSON-parse
helper" must be created, it does not exist; (iii) (c)'s single retry owner must
be named as `llm_client.py:1656-1681` **on the `generate_content` path**, with
`multi_agent_orchestrator.py:1363-1394` declared out of scope.

---

## 1. Queries run (three-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| Current-year (2026) | `Gemini 2.5 Flash Pro retirement date October 2026 deprecated successor migration` | (d) retirement date + successors |
| Last-2-year (2025) | `LLM cost tracking double counting cached tokens billing bug 2025` | (e) adversarial / cross-tool corroboration |
| Year-less canonical | `Anthropic prompt caching input_tokens excludes cache_read_input_tokens cost calculation` | (e) canonical semantics |
| Direct-doc fetches | 6 official vendor doc URLs (see below) | (a)(c)(d)(e) primary evidence |

All three variants were run. The year-less canonical query surfaced the
authoritative Anthropic doc plus a spread of practitioner write-ups; the
2025-scoped query surfaced the adversarial cross-tool evidence (litellm /
langfuse / mlflow bug reports) that independently corroborates (e).

## 2. Sources read in full (6 -- all tier-1/tier-2: official vendor docs)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-07-20 | Official doc (Anthropic) | WebFetch | **"The `input_tokens` field represents only the tokens that come after the last cache breakpoint in your request - not all the input tokens you sent."** + `total_input_tokens = cache_read + cache_creation + input_tokens`. Decides (e). |
| 2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-07-20 | Official doc (Anthropic) | WebFetch | "`$ref`, `$def`, and `definitions` (external `$ref` not supported)"; `additionalProperties` "must be set to `false` for objects"; unsupported: recursive schemas, `minimum`/`maximum`/`multipleOf`, `minLength`/`maxLength`, array constraints beyond `minItems` 0/1 -> 400 error. Decides (a). |
| 3 | https://platform.claude.com/docs/en/api/handling-stop-reasons | 2026-07-20 | Official doc (Anthropic) | WebFetch | 7 `stop_reason` values verbatim; `max_tokens` -> "Raise `max_tokens` or continue the response"; streaming: null in `message_start`, provided in `message_delta`. Decides (c) Claude leg. |
| 4 | https://ai.google.dev/gemini-api/docs/deprecations | 2026-07-20 | Official doc (Google) | WebFetch | `gemini-2.5-pro`/`-flash`/`-flash-lite` all shutdown **October 16, 2026**; successors `gemini-3.1-pro-preview` / `gemini-3.5-flash` / `gemini-3.1-flash-lite`; dates are "earliest possible". Decides (d). |
| 5 | https://platform.claude.com/docs/en/about-claude/pricing | 2026-07-20 | Official doc (Anthropic) | WebFetch | Opus 4.8 $5/$25 per MTok; cache multipliers **1.25x (5m write) / 2x (1h write) / 0.1x (read)**; Batch = 50% off. Validates `cost_tracker.MODEL_PRICING` + the 2.0x/0.1x constants. |
| 6 | https://ai.google.dev/api/generate-content | 2026-07-20 | Official doc (Google) | WebFetch | `finishReason` enum is **UPPERCASE**: `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`. Decides (c) Gemini leg + surfaces the case-mismatch. |

## 3. Snippet-only sources (24; context, do NOT count toward the gate)

From the Gemini-retirement search: aiweekly.co (2.0-flash retirement),
gcpstudyhub.com (Gemini 2.5 agent-platform retirement guide),
docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions,
qwe.edu.pl tutorial, wpnews.pro (image-preview retirements),
discuss.ai.google.dev thread 174217 ("2.5 Flash deprecated without warning
earlier than shutdown date"), github.blog/changelog/2026-07-02 (GitHub Models
deprecating Gemini 2.5 Pro + Gemini 3 Flash).

From the prompt-caching search: mindstudio.ai, hidekazu-konishi.com,
jobsbyculture.com 2026 guide, tygartmedia.com, agentbrisk.com,
teachmeidea.com, aws.amazon.com/blogs/machine-learning (Bedrock caching),
dev.to/gabrielanhaia, portkey.ai docs.

From the cost-double-counting search **(the adversarial set -- see §4)**:
github.com/BerriAI/litellm/issues/27191, .../issues/19681,
github.com/langfuse/langfuse/issues/12306, github.com/mlflow/mlflow/issues/22606,
langfuse.com/docs/observability/features/token-and-cost-tracking,
docs.litellm.ai/docs/troubleshoot/cost_discrepancy, worklytics.co,
finopsllm.com, honeycomb.io.

## 4. Recency scan (last 2 years, 2024-2026)

**Result: 3 new findings in the window, all complementary; none supersedes the
canonical Anthropic doc.**

1. **(2026, official)** Google's deprecations page now lists concrete
   `gemini-3.x` successors for every 2.5 model. This is newer than
   `model_tiers.py:43-49`'s comment, which names the retirement date but not
   the successors. The (d) warning should name the successor, not just warn.
2. **(2026, community)** `discuss.ai.google.dev` thread 174217 reports
   `gemini-2.5-flash` being *effectively* deprecated **earlier than** the
   published shutdown date. This strengthens the case for the (d) warning
   firing at **2026-09-15**, a month ahead of 2026-10-16 -- the step's chosen
   lead time is well-founded, not arbitrary.
3. **(2025-2026, adversarial -- the highest-value recency finding)** Multiple
   independent observability tools shipped the *same class* of cached-token
   accounting bug: litellm #19681 / #27191, langfuse #12306, mlflow #22606.
   The langfuse issue is titled *"Anthropic cache tokens double-counted:
   usage.input already includes cache"* -- i.e. it asserts the **opposite** of
   the Anthropic doc.

**Adversarial reconciliation (important -- do not skip).** The langfuse title
contradicts source #1. The reconciliation is that langfuse was consuming
Anthropic usage **through the pydantic-ai OTel / genai-prices instrumentation
layer, which had already summed the buckets** before langfuse saw it -- so at
*that* layer input did include cache. The raw first-party Messages API response
(what `ClaudeClient` and the CC envelope read) follows source #1: `input_tokens`
**excludes** both buckets. The same search result states the provider asymmetry
plainly: *"OpenAI includes cache read tokens inside the reported input token
count, while Anthropic reports cache read tokens separately from non-cached
input tokens. This difference has caused many tools to miscalculate costs."*

Practical consequence for pyfinagent: (e)'s fix is correct **for the Anthropic
and CC paths only**. If OpenAI cached-token reporting is ever wired into
`UsageMeta` (it is not today -- `llm_client.py:1208-1212` sets only the three
count fields), it would need the *opposite* treatment. The contract should
record this asymmetry next to the fix so a future OpenAI cache retrofit does not
copy the Anthropic formula.

## 5. Internal evidence table (17 files inspected)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/agents/cost_tracker.py` | 134-209 (`record`), 175-181 (bug), 16-80 (pricing) | (e) subject | **BUG CONFIRMED** at `:176` |
| `backend/agents/llm_client.py` | 32 (`unwrap_secret`), 344-368 (`_ensure_additional_properties_false`), 396-456 (`_check_cost_budget`), 643-650 (`UsageMeta`), 653-679 (`LLMResponse`), 1072-1076 (Gemini umeta), 1086-1101 (Gemini log), 1140-1214 (`OpenAIClient`), 1656-1681 (retry owner #1), 1766-1772 (Claude umeta), 1776-1796 (Claude log), 2012-2021 (breach guard), 2075-2192 (`advisor_call`) | (b)(c)(e)(f)(g) subject | multiple confirmations |
| `backend/agents/claude_code_client.py` | 220-283 (argv), 264-271 (flags), 516-602 (`generate_content`), 535-537 (the gate), 593-599 (cache fields) | (a) subject | **BUG CONFIRMED** at `:536` |
| `backend/agents/multi_agent_orchestrator.py` | 1302-1428 (stop_reason dispatch), 1363-1394 (retry owner #2) | (c) second owner | out of scope, documented |
| `backend/agents/orchestrator.py` | 133,144,157,175,181 (schemas), 309 (`_parse_json_with_fallback`) | (a)(c) | Pydantic classes confirmed |
| `backend/agents/debate.py` | 45,50 (schemas), 122 (`_parse_json`) | (a)(c) | duplicate helper |
| `backend/agents/risk_debate.py` | 44,49 (schemas), 118 (`_parse_json`) | (a)(c) | byte-identical duplicate |
| `backend/agents/schemas.py` | 6 schema classes measured | (a) | 4 emit `$defs`; 0 unsupported keywords |
| `backend/config/model_tiers.py` | 43-50 (`GEMINI_WORKHORSE`), 93-97 (deep-think in dict) | (d) target | no `GEMINI_DEEP_THINK` symbol |
| `backend/config/settings.py` | 31 (`deep_think_model`), 374-380 (budget caps) | (d)(g) | pin confirmed; LLM-vs-BQ mismatch |
| `backend/agents/evaluator_agent.py` | 94 | (d) pin | confirmed |
| `backend/agents/rag_agent_runtime.py` | 57 | (d) pin | confirmed |
| `backend/agents/skill_modification_review.py` | 216 | (d) pin | confirmed |
| `backend/autonomous_loop.py` | 76 | (d) pin | confirmed |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 41, 82-115 | (g) source | BQ-spend, not LLM-spend |
| `backend/api/cost_budget_api.py` | 8, 24, 126 | (g) consumer | confirmed |
| `backend/services/observability/api_call_log.py` | 220-250 (`log_llm_call`) | (f)(g) target | signature captured |

**Environment pins measured:** `anthropic==0.96.0` (exact pin,
`backend/requirements.txt:39`; installed matches), `google-genai==1.73.1`,
Claude Code CLI **2.1.215**.

**Test-collision check:** `backend/tests/test_phase_75_llm_rail.py` does **not**
exist yet (new file, as criterion 1 states). Existing tests touching these
surfaces: `backend/tests/test_anthropic_fallback.py:56-61` constructs
`UsageMeta(...)` directly (safe -- additive `stop_reason` field with a default
will not break it), `backend/tests/test_phase_61_2_decision_integrity.py` and
`test_phase_60_1_deep_pipeline.py` reference cost_tracker, and
`tests/slack_bot/test_scheduler_wiring_phase991.py:65,150` is the one that
**will break** under (g).

### Test-breakage analysis for (e) -- MEASURED, not assumed

I read every test that touches `CostTracker.record` with cache tokens:

| Test | Shape | Breaks under (e)? |
|---|---|---|
| `tests/agents/test_cost_tracker_pricing.py:41-54` | `cache_creation=0, cache_read=0` -> takes the `else` branch at `cost_tracker.py:183` | **NO** -- untouched by the fix |
| `tests/verify_phase_25_A9.py:28-32` | **source scan** (`regex` for `cache_write_cost = cache_creation * pricing[0] * 2.0`) | **NO**, provided the implementer changes only the `regular_input` line and leaves the `cache_write_cost` line textually intact. Fragile -- warn the implementer |
| `tests/verify_phase_25_A9.py:57-61` | **tautology** (`expected` and `actual` are the same literal expression; `math_ok` compares a constant to itself) | **NO** -- it cannot fail, and it does not exercise `record()` at all |
| `tests/verify_phase_25_B9.py:132-160` | Calls `record()` twice with cache tokens; asserts on `cache_read_input_tokens` and a derived `hit_rate` -- **not** on `cost_usd` | **NO**, but see below |
| `tests/verify_phase_25_C9.py:205-252`, `verify_phase_25_D9/E9`, `verify_phase_25_S_1.py` | all pass `cache_*=0` | **NO** |

**But `verify_phase_25_B9.py`'s fixture encodes the WRONG semantics and must be
corrected in the same commit.** At `:147-151` it constructs:

```python
r1 = _fake_response(input_tokens=5050, output_tokens=120, cache_creation=5000, cache_read=0)
r2 = _fake_response(input_tokens=5050, output_tokens=130, cache_creation=0,    cache_read=5000)
```

`input_tokens=5050` alongside `cache_read=5000` only makes sense under the
**inclusive** (OpenAI-style) reading -- it is modelling "5050 total, of which
5000 cached, so 50 regular". Under the correct Anthropic semantics established
by source #1, that fixture describes 5050 *uncached* tokens **plus** 5000 cached
ones. The test still passes (it never asserts `cost_usd`), so this is a **latent
wrong-shaped fixture**, not a break -- exactly the "a fixture that couldn't
represent the failure" anti-pattern in the standing auto-memory. The contract
should require it be re-shaped to `input_tokens=50` so the codebase stops
carrying two contradictory models of the same field.

**Mutation-resistance guidance for the new test (criterion 5).** Two of the
three existing guards on this code path are a source-scan and a tautology --
neither can catch (e). The new `test_phase_75_llm_rail.py` assertion must be
**behavioural**: call `record()` and assert `entry.cost_usd` equals the
value implied by 1000 uncached input tokens. Verify by mutation: reverting
`regular_input = input_tokens` back to
`max(0, input_tokens - cache_read - cache_creation)` MUST make the new test
fail.

## 6. Per-item findings (a)-(g)

### (e) llmeng-10 -- cost_tracker double-subtract -- **CONFIRMED** (highest stakes)

**External ground truth** (Anthropic prompt-caching doc, fetched 2026-07-20):

> "The `input_tokens` field represents only the tokens that come **after the
> last cache breakpoint** in your request - not all the input tokens you sent."

> "total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens +
> input_tokens"

So Anthropic `input_tokens` **EXCLUDES** both cache buckets. The step's premise
is correct.

**Current code** -- `backend/agents/cost_tracker.py:175-181`:

```python
if cache_read > 0 or cache_creation > 0:
    regular_input = max(0, input_tokens - cache_read - cache_creation)
    cached_read_cost  = cache_read     * pricing[0] * 0.1 / 1_000_000
    cache_write_cost  = cache_creation * pricing[0] * 2.0 / 1_000_000
    regular_cost      = regular_input  * pricing[0] / 1_000_000
    output_cost       = output_tokens  * pricing[1] / 1_000_000
    cost = cached_read_cost + cache_write_cost + regular_cost + output_cost
```

`input_tokens` is read at `cost_tracker.py:157` from
`usage.prompt_token_count`, which for the Anthropic path is populated by
`ClaudeClient` as `usage.input_tokens` verbatim (see internal evidence table)
-- i.e. it is already the cache-exclusive number. Subtracting the two cache
buckets again is a **second** subtraction. CONFIRMED.

**Proposed formula:** `regular_input = input_tokens` (drop the subtraction and
the `max(0, ...)` clamp entirely for the Anthropic shape).

#### Worked example -- UsageMeta(prompt_token_count=1000, cache_read_input_tokens=5000)

Model `claude-opus-4-8`, pricing[0] = $5.00/Mtok, output 0, cache_creation 0.

| Quantity | Today | Correct |
|---|---|---|
| `regular_input` | `max(0, 1000 - 5000 - 0)` = **0** | **1000** |
| regular cost | 0 x 5.00 / 1e6 = **$0.000000** | 1000 x 5.00 / 1e6 = **$0.005000** |
| cache-read cost | 5000 x 5.00 x 0.1 / 1e6 = $0.002500 | $0.002500 (unchanged) |
| cache-write cost | $0 | $0 |
| **total input cost** | **$0.002500** | **$0.007500** |

Under-report = **$0.005000 per call, i.e. 66.7% of the true input cost is lost**
whenever `cache_read >= input_tokens` (the common steady-state case: a large
cached system prompt + a small user turn). The `max(0, ...)` clamp is what makes
the error silent -- it turns a negative into a plausible-looking zero.

Note the clamp also means the bug is *bounded* (never negative) but *always*
under-reports, never over-reports, on the Anthropic path.

#### Over-count risk check (does the fix break the Gemini path?)

**This is the real trap and the step text does not mention it.** `record()` is
shape-polymorphic: it reads `prompt_token_count` (a **Gemini/Vertex** field
name) but `cache_creation_input_tokens` / `cache_read_input_tokens`
(**Anthropic** field names) off the same object. Gemini's own semantics are the
OPPOSITE of Anthropic's -- Gemini's `prompt_token_count` **includes**
`cached_content_token_count` (cache tokens are a *subset*, not a sibling).

Mitigating fact: the normalized `UsageMeta` (`llm_client.py:643-650`) is the
only object that carries the two Anthropic cache fields, and **the Gemini
client never populates them** (`llm_client.py:1072-1076` sets only the three
count fields). So on the Gemini path `cache_read == cache_creation == 0`, the
`if` branch is never taken, and the `else` branch at `cost_tracker.py:183` is
used. The fix is therefore **safe for Gemini by construction** -- but that
safety is *incidental*, not asserted anywhere. Recommend the contract add a
guard test pinning "Gemini path never sets the Anthropic cache fields", else a
future Gemini cache-metrics retrofit silently re-introduces an OVER-count.

**SDK version check:** `anthropic==0.96.0` (exact pin,
`backend/requirements.txt:39`; installed version matches). The
`input_tokens`-excludes-cache semantics have been stable since prompt caching
went GA and are re-confirmed in the current doc fetched today, so there is no
version-skew risk on this pin.

#### Criterion-3/5 kwarg correction (contract MUST fix this)

Immutable criterion 5 says `UsageMeta(input=1000, cache_read=5000)`. **Those
kwargs do not exist.** The real dataclass (`llm_client.py:643-650`) is:

```python
@dataclass
class UsageMeta:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
```

A test written literally as `UsageMeta(input=1000, cache_read=5000)` raises
`TypeError`. The criterion is descriptive shorthand; the test must construct
`UsageMeta(prompt_token_count=1000, cache_read_input_tokens=5000)` and assert
the priced result equals 1000 uncached input tokens. Flag this in the contract
so Q/A does not read the mismatch as a criterion violation.

Also note `record()` takes a **response object**, not a `UsageMeta` -- it does
`getattr(response, "usage_metadata", None)` (`cost_tracker.py:153`). The test
must wrap: `SimpleNamespace(usage_metadata=UsageMeta(...))`.

### (b) llmeng-03 -- advisor_call bypasses both guards -- **CONFIRMED**

`advisor_call` is at `backend/agents/llm_client.py:2075` (step said 2110 -- that
is the `_anthropic.Anthropic(api_key=key)` line, 2110; the `def` is 2075).

Three defects, all confirmed:

1. **No `_check_cost_budget()`.** The function body (2101-2131) goes straight
   from arg-parsing to `client.beta.messages.create(**kwargs)` at 2131. Compare
   the three sanctioned call sites that DO gate: `llm_client.py:870`
   (Gemini), `:1142` (Claude), `:1346` (OpenAI).
2. **No routing-breach guard.** `make_client` raises a `ValueError` at
   `llm_client.py:2012-2021` when `paper_use_claude_code_route=True` and a
   direct-Anthropic client is about to be built. `advisor_call` constructs
   `_anthropic.Anthropic(...)` at :2110 with no such check -- it is a
   **metered-billing hole straight through the $0-metered boundary**.
3. **Raw `os.getenv`, not settings.** `key = api_key or _os.getenv("ANTHROPIC_API_KEY")`
   at `:2105`. Criterion 2 explicitly requires this to go away.

**Pattern to copy** -- the breach guard verbatim from `llm_client.py:2012-2021`;
the budget gate is a bare `_check_cost_budget()` call (it is module-local, no
import needed).

**SecretStr trap check (auto-memory `project_secretstr_dead_overlays`):** the
correct resolution is `unwrap_secret(getattr(settings, "anthropic_api_key", ""))`
-- exactly the idiom already used at `llm_client.py:1953`. Do **NOT** write
`settings.anthropic_api_key or ""`: a non-empty `SecretStr` is truthy, so `or`
returns the *wrapper*, and the SDK then sends a header of `SecretStr('...')`.
`unwrap_secret` is defined at `llm_client.py:32` and is a no-op for plain `str`.

### (a) llmeng-01 -- CC rail drops Pydantic schema classes -- **CONFIRMED**

The gate is `backend/agents/claude_code_client.py:535-537`:

```python
schema = config.get("response_schema")
if isinstance(schema, dict):
    json_schema = schema
```

and the surrounding comment (`:530-534`) **admits the behaviour**: "The
orchestrator may pass a Pydantic model class as response_schema. We surface no
schema to the CLI ... but flip --json-schema when the caller explicitly provides
a dict."

**Every pipeline schema is a Pydantic CLASS; zero are dicts.** Enumerated:

| Config site | `response_schema` value |
|---|---|
| `backend/agents/orchestrator.py:133` | `SynthesisReport` |
| `backend/agents/orchestrator.py:144` | `CriticVerdict` |
| `backend/agents/orchestrator.py:157` | `CriticVerdict` |
| `backend/agents/orchestrator.py:175` | `RiskJudgeVerdict` |
| `backend/agents/orchestrator.py:181` | `SynthesisReport` |
| `backend/agents/debate.py:45` | `DevilsAdvocateResult` |
| `backend/agents/debate.py:50` | `ModeratorConsensus` |
| `backend/agents/risk_debate.py:44` | `RiskAnalystArgument` |
| `backend/agents/risk_debate.py:49` | `RiskJudgeVerdict` |

So `isinstance(dict)` is **never true in production** -- `--json-schema` is dead
code on the live rail. CONFIRMED, and stronger than the step states.

**CLI flags confirmed live.** Installed CLI is **2.1.215**; `claude --help`
lists both:
- `--json-schema <schema>            JSON Schema for structured output`
- `--append-system-prompt <prompt>   Append a system prompt to the default`

Argv is built at `claude_code_client.py:264-271`:
```python
"--print", "--output-format", "json",
... if system:      args.extend(["--append-system-prompt", system])
... if json_schema: args.extend(["--json-schema", json.dumps(json_schema)])
```
So the plumbing already exists end-to-end; only the `isinstance` gate blocks it.

**The `$defs`/`$ref` trap -- MEASURED, and it is NOT a blocker.** Anthropic's
structured-outputs doc lists as supported: "`$ref`, `$def`, and `definitions`
(external `$ref` not supported)". Internal refs are fine; only `http://...`
external refs are rejected. Measured on our real schemas:

| Schema | `$defs` present | `$ref` count | Unsupported keywords |
|---|---|---|---|
| `CriticVerdict` | yes | 5 | none |
| `SynthesisReport` | yes | 3 | none |
| `RiskJudgeVerdict` | yes | 1 | none |
| `ModeratorConsensus` | yes | 2 | none |
| `DevilsAdvocateResult` | no | 0 | none |
| `RiskAnalystArgument` | no | 0 | none |

**Conclusion: no flattening/dereferencing is required.** Pass
`model_json_schema()` through `_ensure_additional_properties_false` and hand it
to `--json-schema` as-is.

**BUT -- the residual risk the step does not mention.** The same doc enumerates
keywords that produce a **400 error**: recursive schemas, complex types in
enums, `minimum`/`maximum`/`multipleOf`, `minLength`/`maxLength`, array
constraints beyond `minItems` 0/1, and `additionalProperties` set to anything
but `false`. I scanned all six schemas for those keywords and found **none
today** -- but a future `Field(ge=0, le=100)` on e.g. a confidence field would
silently emit `minimum`/`maximum` and start 400-ing the live rail. Recommend the
contract add a guard test asserting the emitted schema contains no keyword from
that unsupported set. (Caveat: the doc describes the **API**
`output_config.format` surface; the CLI `--json-schema` flag is a different
surface and its validator is not separately documented. Treat the API
constraint list as the best available proxy, not a proven identity.)

`_ensure_additional_properties_false` is at `llm_client.py:344-368`; it already
recurses into `$defs`, `definitions`, `properties.*`, `items`, `anyOf`,
`oneOf`, `allOf`, and is idempotent -- exactly the shape (a) needs.

### (d) llmeng-06 -- gemini-2.5 pins -- **PARTIAL / step undercounts**

Retirement date **CONFIRMED**: Google's official deprecations page lists
`gemini-2.5-pro`, `gemini-2.5-flash`, and `gemini-2.5-flash-lite` all with
shutdown **October 16, 2026**, with successors `gemini-3.1-pro-preview`,
`gemini-3.5-flash`, and `gemini-3.1-flash-lite` respectively. The page notes
shutdown dates "indicate the _earliest possible dates_ on which a model might be
retired."

**The "5 hardcoded pins" claim is an undercount.** A repo scan
(`grep -rn "gemini-2\.5" --include="*.py" backend/ scripts/`, excluding tests)
returns **54 matches**. Most are comments, pricing-table keys, context-window
tables and UI allowlists which are legitimately model-name literals. But the
set of *behavioural defaults* is larger than the 5 named:

| File:line | Kind | In step's list? |
|---|---|---|
| `backend/config/settings.py:31` | `deep_think_model` default `gemini-2.5-pro` | YES |
| `backend/agents/evaluator_agent.py:94` | `model_name` default | YES |
| `backend/agents/rag_agent_runtime.py:57` | `DEFAULT_QUERY_MODEL` | YES |
| `backend/agents/skill_modification_review.py:216` | `model=` kwarg | YES |
| `backend/autonomous_loop.py:76` | `evaluator_model` default | YES |
| `backend/meta_evolution/directive_review.py:159` | `model="gemini-2.5-flash"` | **NO** |
| `backend/meta_evolution/directive_rewriter.py:202` | `model="gemini-2.5-flash"` | **NO** |
| `backend/news/sentiment.py:81` | `SCORER_MODEL_GEMINI_FLASH` | **NO** |
| `backend/agents/harness_memory.py:322,503` | `model_name` defaults | **NO** |
| `backend/services/autonomous_loop.py:2648,2663` | `settings.gemini_model or "gemini-2.5-flash"` fallback | **NO** |
| `backend/api/agent_map.py:132` | `live_model` fallback | **NO** |

That is **at least 12** behavioural pins, not 5. Criterion 4 only requires the
scan to prove zero literals **across the 5 listed files**, so the step is
*satisfiable* as written -- but the contract should state honestly that the 5
are a subset, and the remaining pins are a follow-on. Per the standing
"queue discovered defects in the masterplan" rule, the extras warrant their own
step rather than silent scope creep.

**Constants already exist**: `backend/config/model_tiers.py:50`
`GEMINI_WORKHORSE = "gemini-2.5-flash"` and `:97`
`"gemini_deep_think": "gemini-2.5-pro"`. Note there is **no module-level
`GEMINI_DEEP_THINK` constant** -- the deep-think value lives inside a dict at
`:97`. The step's phrasing ("GEMINI_WORKHORSE/DEEP_THINK constants") implies a
symbol that does not yet exist; the contract must specify creating
`GEMINI_DEEP_THINK` as a sibling of `GEMINI_WORKHORSE`.

Also note `model_tiers.py:43-49` already carries a comment referencing the
retirement date -- so the warning's home is unambiguous.




### (c) llmeng-04 -- stop_reason + retry ownership -- **PARTIAL (premise needs correction)**

**External enums confirmed.**

Claude `stop_reason` (Anthropic handling-stop-reasons doc, 7 values, lowercase
snake_case): `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`,
`pause_turn`, `refusal`, `model_context_window_exceeded`. Documented handling
for `max_tokens`: "Raise `max_tokens` or continue the response."

Gemini `finishReason` (Gemini API generateContent doc, **UPPERCASE**): `STOP`,
`MAX_TOKENS`, `SAFETY`, `RECITATION`, and others. **Case and spelling differ
from Claude's** (`MAX_TOKENS` vs `max_tokens`) -- the contract must specify a
normalization policy (recommend: normalize to Claude's lowercase vocabulary,
since `ClaudeClient` already dispatches on it at `llm_client.py:1656-1712` and
`multi_agent_orchestrator.py:1302-1428`).

CC rail: the CLI envelope carries `stop_reason` -- see
`claude_code_client.py:245` (documented envelope shape) and `:364-368` (already
read via `envelope.get("stop_reason")`). So all three paths can populate it.

**CORRECTION 1 -- `LLMResponse` line number.** The step says
`llm_client.py:1684`. That line is inside the ClaudeClient max_tokens branch.
`LLMResponse` is actually declared at **`llm_client.py:653-679`**, with fields
ending at `:672` and a `__post_init__` at `:674`. Add `stop_reason:
Optional[str] = None` there.

**CORRECTION 2 -- "the shared JSON-parse helpers" do not exist as described.**
There is no shared helper. There are **three duplicated, unshared** ones, and
none of them can see a stop reason because they take a **plain string**:

| File:line | Signature |
|---|---|
| `backend/agents/debate.py:122` | `_parse_json(text: str, label: str) -> Optional[dict]` |
| `backend/agents/risk_debate.py:118` | `_parse_json(text: str, label: str) -> Optional[dict]` (byte-identical duplicate) |
| `backend/agents/orchestrator.py:309` | `_parse_json_with_fallback(json_string: str, agent_name: str) -> Optional[dict]` |

(There is also `backend/utils/json_io.py:41 parse_json_line`, a different
concern, and `skill_optimizer.py:893 _extract_json`.)

Because these receive `text`, not the `LLMResponse`, **they cannot "expose
stop_reason" without a signature change**. The contract must choose and state
one of:
 (i) change the signatures to accept an optional `stop_reason: str | None`
     supplied by the caller, or
 (ii) introduce one genuinely shared helper (e.g. in `backend/utils/json_io.py`)
     that takes the `LLMResponse` and have the three sites delegate to it.
Option (ii) also retires the debate/risk_debate duplication. Either way the
criterion-3 phrase "the shared JSON-parse helper" is currently a
**forward-looking description of something to be created**, not of existing
code -- say so explicitly in the contract so Q/A does not score it as a
mis-description of the codebase.

**CORRECTION 3 (most important) -- there are TWO max_tokens re-request owners
today, not one.**

| # | Owner | Anchor | Budget | Guard |
|---|---|---|---|---|
| 1 | `ClaudeClient.generate_content` stop_reason dispatch (phase-4.14.4 MF-26/27) | `backend/agents/llm_client.py:1656-1681`; re-request at **:1672** | `min(max_tokens * 2, 8192)` (`:1664`) | none -- single-shot by construction, comment `:1662` "Single-shot; callers own the loop" |
| 2 | `_run_agent_turn` tool loop in the Layer-2 MAS | `backend/agents/multi_agent_orchestrator.py:1363-1394`; re-request at **:1381** | `min(_max_tokens * 2, 32768)` (`:1380`) | `_mt_retried_turn == turn` (`:1371`, `:1377`) |

They sit on **different call paths** (Layer-1 pipeline via `generate_content`
vs Layer-2 MAS calling `client.messages.create` directly), so they do not stack
on a single call *today*. But the step's phrase "the SINGLE owner ... stays the
existing ClaudeClient retry" is only true of the Layer-1 path.

**The contract MUST name the owner precisely as:**
> `ClaudeClient.generate_content`'s phase-4.14.4 MF-26/27 stop_reason dispatch,
> `backend/agents/llm_client.py:1656-1681` (re-request at `:1672`) -- the sole
> owner of the max_tokens re-request **on the `generate_content` path**. The
> Layer-2 MAS loop at `multi_agent_orchestrator.py:1363-1394` is a separate,
> pre-existing owner on a separate path and is explicitly out of scope for 75.5.

Additional precision the contract needs: owner #1 only retries when the last
content block is a **`tool_use`** tail (`llm_client.py:1659`). A plain **text**
truncation takes the `else` at `:1682-1685` and merely logs -- **no retry at
all**. So "the sole max_tokens re-request owner" is conditional, and a test
asserting "the parse helper adds no retry" must not accidentally assert that
owner #1 always retries.

**Out-of-scope defect noted (already queued):** owner #2's guard at
`multi_agent_orchestrator.py:1371` reads `locals().get("_mt_retried_turn") ==
turn`, and `continue` at `:1394` advances the loop -- so on the next `turn` the
guard no longer matches and a second retry can fire, contradicting the `:1368`
comment "bounds this to a single retry per turn per contract". This matches the
known phase-73 anchor (`:1363-1394`). **Do not fix it here** -- per the standing
"queue discovered defects in the masterplan" rule it belongs in its own
research-gated step.

### (f) llmeng-11 -- OpenAIClient missing log_llm_call -- **CONFIRMED**

`OpenAIClient.generate_content` is at **`backend/agents/llm_client.py:1140`**
(the step's `:1214` is the `return LLMResponse(...)` line). It calls
`_check_cost_budget()` at `:1142` but writes **no** `log_llm_call` row -- it
goes straight from `client.chat.completions.create(**kwargs)` at `:1204` to
`return` at `:1214`.

Every sibling client has the writer:

| Client | log_llm_call anchor |
|---|---|
| `GeminiClient` (phase-35.2) | `llm_client.py:1086-1101` |
| `ClaudeClient` (phase-6.7) | `llm_client.py:1776-1796` |
| `ClaudeCodeClient` (phase-60.4 AW-7) | `claude_code_client.py:496-514` |
| `advisor_call` (phase-26.2) | `llm_client.py:2179-2192` |
| **`OpenAIClient`** | **MISSING** |

**Reference shape to copy = the GeminiClient block at `llm_client.py:1086-1101`**
(the step names phase-35.2, correct). Its comment at `:1088-1092` even records
the same defect class ("c7801712 had 0 llm_call_log rows because
GeminiClient.generate_content lacked the same telemetry write"). Signature is
`backend/services/observability/api_call_log.py:220-235`.

Notes for the implementer: `OpenAIClient` currently has **no `_t0` latency
timer** -- one must be added around `:1204` to populate `latency_ms`. Provider
string should be conditional: the same class serves both direct OpenAI and
GitHub Models (routed at `llm_client.py:2035-2039` with a `base_url`), so
`provider="openai" if not self._base_url else "github_models"` is the honest
tag. Cache kwargs are `0` (OpenAI's cached-token reporting is not read here).

### (g) arch-04 -- promote fetch_spend into observability -- **CONFIRMED, with a scope caveat**

`_default_fetch_spend` is defined at
**`backend/slack_bot/jobs/cost_budget_watcher.py:82-115`**.

**Full consumer list (complete; grep over the repo excluding `.venv`):**

| Consumer | Anchor | Notes |
|---|---|---|
| `llm_client._check_cost_budget` | `backend/agents/llm_client.py:427` (import), `:431` (call) | function-local import -- the hard-block |
| `api/cost_budget_api` | `backend/api/cost_budget_api.py:24` (import), `:126` (call via `asyncio.to_thread`) | also referenced in the module docstring at `:8` |
| `slack_bot/jobs/cost_budget_watcher` | `:41` (`fetch_fn or _default_fetch_spend`), `:82` (def) | the owning module |
| **`tests/slack_bot/test_scheduler_wiring_phase991.py`** | **`:65`** (direct call), **`:150`** (`monkeypatch.setattr(cost_budget_watcher, "_default_fetch_spend", ...)`) | **WILL BREAK** -- see below |

**This is the "breaks a prior phase's verifier" shape the caller asked about,
and it is real.** `tests/slack_bot/test_scheduler_wiring_phase991.py:150`
monkeypatches the symbol **on the `cost_budget_watcher` module**. If (g) moves
the implementation into `backend/services/observability` and
`cost_budget_watcher` merely re-imports it, the monkeypatch will still bind a
name on `cost_budget_watcher` -- but `llm_client` and `cost_budget_api` would by
then resolve from `observability`, so the patch would no longer affect them.
Mitigation the contract should mandate: keep a **backwards-compatible alias**
`_default_fetch_spend = fetch_spend` in `cost_budget_watcher`, and update that
test to patch the new canonical location. Note this test lives under the
top-level `tests/` tree, **not** `backend/tests/`, so it is outside the
verification command's path -- the implementer must run it separately or it
will regress silently.

Target package `backend/services/observability/` already exists with
`__init__.py`, `alerting.py`, `api_call_log.py`, `log_redaction.py`,
`rainbow_canary.py`, `rate_limit.py`, `retry.py`; `__init__.py` re-exports via
`__all__` at `:32`. `alerting.py` is the natural home for the "count + alert
when the guard degrades" seam required by criterion 6.

**Scope caveat / discovered defect (do NOT silently fix here).**
`_default_fetch_spend` queries
`INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at `$6.25/TiB` -- it
measures **BigQuery** spend, not LLM spend. But
`backend/config/settings.py:379` documents `cost_budget_daily_usd` as
"Daily **LLM-spend** cap across all cycles (USD)", and `_check_cost_budget`'s
docstring (`llm_client.py:397-408`) says "Reads today's BQ spend" while the
whole mechanism is presented as the LLM circuit-breaker. So the "$25/day hard
cap" that CLAUDE.md and settings call the real LLM circuit breaker is in fact
gating on BigQuery bytes-billed. Promoting the function unchanged (as (g)
specifies) **preserves** this mismatch. That is the right call for 75.5 --
(g) is a refactor, and changing the metric would be a live-money behaviour
change -- but the contract should record it and it warrants its own masterplan
step.

---

## 7. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Criterion 5's literal `UsageMeta(input=..., cache_read=...)` kwargs do not exist -> test raises `TypeError` | HIGH | Use `UsageMeta(prompt_token_count=1000, cache_read_input_tokens=5000)` wrapped in `SimpleNamespace(usage_metadata=...)`; state the shorthand-vs-actual mapping in the contract |
| R2 | (e) changes **live cost accounting** on the CC rail (`claude_code_client.py:593-599` populates both cache fields) | HIGH | Behaviour is report-only (cost_tracker feeds the Cost tab + the *nominal* `max_analysis_cost_usd` warning, not the hard block). Costs will correctly go UP in reporting. Flag in the contract so the operator is not alarmed by a step-change in the Cost tab. No dark flag needed -- it is telemetry, not an order path |
| R3 | (e) would OVER-count if any Gemini path ever populated the Anthropic cache fields | MED | Verified today it does not (`llm_client.py:1072-1076` omits them; `:1091-1092` hardcodes `cache_*_tok=0`). Add a guard test pinning this, or the fix silently inverts later |
| R4 | (a) enabling `--json-schema` sends a real schema to the live CLI for the first time; an unsupported keyword yields HTTP 400 | MED | Measured: all 6 pipeline schemas are clean today. Add a test asserting no keyword from Anthropic's unsupported set. Consider shipping (a) behind a dark flag since the CC rail is the LIVE decision rail |
| R5 | The CLI `--json-schema` validator is not separately documented; API constraints are a proxy, not a proven identity | MED | Criterion 1 only requires **argv construction** (no CLI spawn), so the test is safe. But do not claim live CLI acceptance without a live_check |
| R6 | (g) breaks `tests/slack_bot/test_scheduler_wiring_phase991.py:150` monkeypatch, which is OUTSIDE the verification command's path | MED | Keep a `_default_fetch_spend` alias in `cost_budget_watcher`; update + run that test explicitly in the same commit |
| R7 | (c) as written implies helpers/symbols that do not exist (`LLMResponse:1684`, "the shared JSON-parse helper", single retry owner) | MED | Contract must restate all three with the corrected anchors above, and name owner #1 verbatim |
| R8 | (d) "5 pins" undercounts the true ~12 behavioural pins | LOW | Criterion 4 is scoped to the 5 listed files and is satisfiable; disclose the remainder and queue a follow-on step |
| R9 | (b) resolving the key via `settings.anthropic_api_key or ""` would return a truthy `SecretStr` wrapper | MED | Use `unwrap_secret(getattr(settings, "anthropic_api_key", ""))` per `llm_client.py:1953` |
| R10 | Date-frozen test for the 2.5-retirement warning may be flaky if it reads real `datetime.now()` | LOW | Inject a clock or monkeypatch; criterion 4 explicitly says "frozen >=2026-09-15 date" |
| R11 | Adding `_check_cost_budget()` to `advisor_call` could newly raise `BudgetBreachError` on a path that previously always proceeded | LOW | That is the intended behaviour, and the gate is fail-open on error (`llm_client.py:432-437`). `COST_BUDGET_HARD_BLOCK_DISABLED=1` is the test escape hatch (`:411`) |

**Live-money / dark-flag assessment:** none of (a)-(g) touches order placement,
sizing, or the risk gates. (a) is the only one that changes what the **live
decision rail** sends to the model -- recommend a dark flag there. (e), (f) and
the (g) refactor are telemetry/accounting. (b) and (d) are guards that can only
*block* spend or *warn*, never place a trade.

## 8. JSON envelope

```json
{
  "tier": "complex",
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
  "summary": "All 7 sub-items verified against current source. (a) CONFIRMED and stronger than stated -- all 9 pipeline response_schema values are Pydantic CLASSES, zero dicts, so the isinstance(dict) gate at claude_code_client.py:536 makes --json-schema dead code; CLI 2.1.215 has both flags; $defs/$ref are supported so no flattening needed. (b) CONFIRMED -- advisor_call (llm_client.py:2075) has neither _check_cost_budget nor the routing-breach guard and reads os.getenv at :2105. (c) PARTIAL -- LLMResponse is at :653 not :1684; there is no shared JSON-parse helper (three duplicated ones taking plain str); and TWO max_tokens retry owners exist (llm_client.py:1656-1681 and multi_agent_orchestrator.py:1363-1394). (d) PARTIAL -- retirement 2026-10-16 confirmed official, but ~12 pins exist not 5, and GEMINI_DEEP_THINK does not yet exist. (e) CONFIRMED -- Anthropic doc states input_tokens EXCLUDES both cache buckets, so max(0, input - cache_read - cache_creation) double-subtracts; worked example shows $0.0025 charged vs $0.0075 true. (f) CONFIRMED. (g) CONFIRMED, and it will break a monkeypatch in tests/slack_bot/test_scheduler_wiring_phase991.py:150.",
  "brief_path": "handoff/current/research_brief_75.5.md",
  "gate_passed": true
}
```
