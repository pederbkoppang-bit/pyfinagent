# Research Brief — phase-78.16

**Tier:** moderate (caller-specified). Not audit-class.
**Question:** After 78.1 rewired six signal-overlay services from direct
`ClaudeClient(..., enable_prompt_caching=False)` onto `make_client(...)`,
prompt caching flipped ON where it was explicitly OFF. Decide between
(a) `make_client` forwards `enable_prompt_caching`, (b) preserve caller
intent another way, (c) accept the change + fix the doc — but only if
measured harmless.

**Status:** COMPLETE — `gate_passed: true` (7 sources read in full, 21 URLs,
recency scan performed, 16 internal files inspected).
**Recommendation: (a) now for revert fidelity; queue (c) behind a measurement.**

---

## A. Internal code inventory

### A0. Headline internal findings (details below)

1. **The recorded rationale for `enable_prompt_caching=False` EXISTS and is
   now FACTUALLY OBSOLETE.** It was "the prompt differs per ticker so caching
   provides no benefit" (phase-23.1.2 research brief, Apr 2026). At that time
   the system prompt was ~10–400 tokens of boilerplate. **phase-25.B9 later
   introduced `_HOUSE_INSTRUCTIONS`** — a deliberately padded 19,026-char
   (~5.4K-token) STABLE prefix — expressly *to make caching register*. The
   two decisions were never reconciled.
2. **The cacheable block is byte-identical across all six services**, and
   indeed across every `ClaudeClient` JSON-mode caller in the repo (proof in
   A4). This is the opposite of the 23.1.2 premise.
3. **`ClaudeCodeClient` has NO `enable_prompt_caching` notion** — the kwarg
   only ever affects the **metered/direct** path, which is exactly the revert
   path (`PAPER_USE_CLAUDE_CODE_ROUTE=false`). The blocker is real and its
   blast radius is exactly the revert path.
4. There is **one** `ClaudeClient(...)` construction site left in production
   code (`llm_client.py:2139`, inside `make_client`). Adding a kwarg is a
   pure-additive, zero-caller-break change.

---

### A1. Git archaeology — why was `enable_prompt_caching=False` set?

`git log -S 'enable_prompt_caching=False' -- backend/services/` returns 7
commits. The **originating** one is:

```
743d65e5  phase-23.1.1: macro regime filter (LLM-as-judge over FRED) + screener
          conviction multiplier      Mon Apr 27 00:06:59 2026 +0200
```

The commit message says **nothing** about prompt caching — the kwarg entered
as part of the first overlay's boilerplate (`backend/services/macro_regime.py`,
diff hunk `+ enable_prompt_caching=False,`). The five later commits
(`5a6a6e17` 23.1.2 PEAD, `76d89aa4` 23.1.3 news, `35ff8f59` 23.1.5 meta-scorer,
`ac5a5b3c` 28.11 analyst-narrative, `6e88f91a` 28.13 transcript-GPR) each
**copy** the macro_regime idiom.

**The rationale IS recorded — once — and it is a deliberate cost decision, not
a cargo-cult** (though every subsequent site *is* a copy of it):

> `handoff/archive/phase-23.1.2/phase-23.1.2-research-brief.md:301`
> **`enable_prompt_caching=False`** (line 206 of macro_regime.py): The PEAD
> prompt will be different per-ticker per-quarter so caching provides no
> benefit. Use `enable_prompt_caching=False`.

and restated in that brief's implementation plan:

> `…phase-23.1.2-research-brief.md:535`
> ### Step 6 — Claude Haiku 4.5 call
> Max 512 output tokens, temperature=0.0, `enable_prompt_caching=False`.

**The premise of that rationale was true when written and is false now.** The
brief reasons about *the prompt* (the per-ticker user message). Prompt caching
in this codebase is applied **only to the system block** (`llm_client.py:1475-1484`),
never to the user message. At the time (Apr 27 2026) the system block really
was tiny — corroborated by the phase-4.10 platform audit:

> `handoff/audit/phase-4.10/platform_overview.md:75`
> **Prompt caching min-token threshold probably not met.** System prompt in
> `ClaudeClient.generate_content` is ~20 chars of boilerplate. Under the 4096
> min for Opus 4.7 / Haiku 4.5, so cache entries never get created.
> `enable_prompt_caching=True` is silently a no-op on those models.

…and by the compliance audit:

> `docs/audits/compliance-caching-context.md:64`
> **Status:** Non-compliant for the ClaudeClient path. The
> `enable_prompt_caching=True` default is misleading — caching never actually
> activates.

**phase-25.B9 then fixed exactly that**, and the fix's own comment states the
intent (`backend/agents/llm_client.py:46-53`):

```
# phase-25.B9: substantive "house instructions" block prepended to every
# ClaudeClient system prompt so the block exceeds the per-model cache
# write threshold (Opus 4.7 = 4096, Sonnet 4.6 = 2048, Haiku 4.5 = 4096
# tokens). Without this prefix the system prompt is ~10-400 tokens and
# `cache_control={"type":"ephemeral","ttl":"1h"}` silently no-ops --
# cache_creation_input_tokens stays at 0 and the 90% cache_read discount
# never materializes. Closes phase-24.9 F-2.
```

**Verdict on archaeology:** the rationale is recorded, was correct in April
2026, was invalidated by 25.B9 in May 2026, and the six services were never
revisited. That is a *stale* decision, not an unexplained one — which is the
finding that makes option (c) live.

---

### A2. Every `make_client` caller (signature-compat risk)

Production call sites (all positional-or-keyword, none use `**kwargs`
splatting, none subclass or monkey-patch the signature):

| # | File:line | Call |
|---|-----------|------|
| 1 | `backend/agents/orchestrator.py:652` | `make_client(settings.gemini_model, _general_vertex, settings)` |
| 2 | `backend/agents/orchestrator.py:653` | `make_client(deep_model_name, _dt_vertex, settings)` |
| 3 | `backend/agents/orchestrator.py:654` | `make_client(deep_model_name, _synth_vertex, settings)` |
| 4 | `backend/agents/orchestrator.py:659` | `make_client(settings.gemini_model, _quant_exec_vertex, settings)` |
| 5 | `backend/backtest/quant_optimizer.py:478` | `make_client(settings.gemini_model, _bundle, settings)` |
| 6 | `backend/services/autonomous_loop.py:2696` | `make_client(model_name, vertex_model=None, settings=settings)` |
| 7 | `backend/services/autonomous_loop.py:2932` | `make_client(settings.gemini_model, None, settings)` |
| 8 | `backend/services/meta_scorer.py:230` | `make_client(<meta_scorer_model>, None, settings)` |
| 9 | `backend/services/macro_regime.py:515` | `make_client(<macro_regime_model>, None, settings)` |
| 10 | `backend/services/news_screen.py:276` | `make_client(<news_screen_model>, None, settings)` |
| 11 | `backend/services/pead_signal.py:288` | `make_client(<pead_signal_model>, None, settings)` |
| 12 | `backend/services/analyst_narrative_scorer.py:144` | `make_client(model, None, settings)` |
| 13 | `backend/services/call_transcript_gpr.py:123` | `make_client(model, None, settings)` |

Test call sites that would see a new signature:
`backend/tests/test_phase_78_1_c_block_rail.py:170-171` (`make_client(model, None, on/off)`),
`backend/tests/test_phase_31_1_fixes.py:49,86` (patches `backend.agents.llm_client.make_client`),
`backend/tests/test_phase_75_llm_rail.py:301` (tripwire monkeypatch).

**Signature-compat risk of adding `enable_prompt_caching: bool | None = None`
as a 4th keyword-with-default parameter: NONE.** Every caller passes exactly 3
args; the monkeypatch/tripwire sites replace the symbol wholesale and don't
introspect arity.

### A3. Remaining direct `ClaudeClient(` construction sites

| File:line | Passes the kwarg? | Notes |
|---|---|---|
| `backend/agents/llm_client.py:2139` | **NO** | The only production site. Inside `make_client`: `return ClaudeClient(model_name=model_name, api_key=anthropic_key)` → default `True`. **This is the whole bug.** |
| `backend/tests/test_phase_75_prompt_contracts.py:152-153` | yes (`False`) | see A6 |
| `backend/tests/test_claude_request_shapes.py:77` | no | default True; wire-shape seam (A6) |
| `backend/tests/test_phase_75_llm_rail.py:217` | no | |
| `backend/tests/test_phase_51_1_secretstr.py:44,52,77` | no | SecretStr unwrap tests |
| `tests/verify_phase_25_B9.py:213` | yes (`True`) | asserts the attribute round-trips |
| `tests/verify_phase_25_D9.py:135,264` | yes (both) | |
| `tests/verify_phase_25_E9.py:162,196,219` | yes (`False`) | |

Zero production code outside `make_client` constructs a `ClaudeClient` — 78.1
removed the last six. So **`make_client:2139` is now the single chokepoint for
this decision**, which is what makes option (a) a one-line change.

### A4. What the flag actually changes on the wire

`backend/agents/llm_client.py:1437-1492`:

- `:1453` `system_prompt = _HOUSE_INSTRUCTIONS` — **unconditional**. `ClaudeClient`
  **ignores `config["system"]` entirely** (grep for `config.get("system")` in
  `llm_client.py` returns nothing; only `:1490 "system": system_arg` and the
  unrelated OpenAI `:1211` / advisor `:2270` paths).
- `:1454-1461` schema suffix. Two branches:
  - schema is a **Pydantic class** (`hasattr(schema, "model_json_schema")`) →
    appends the full JSON schema (per-service, variable).
  - **else** → appends the fixed 47-char sentence
    `"You MUST respond with a valid JSON object only."` (+2 newlines = 49 chars).
- `:1475-1484` the only behavioural difference:

```python
if self.enable_prompt_caching:
    system_arg = [{"type": "text", "text": system_prompt,
                   "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
else:
    system_arg = system_prompt          # plain str
```

**Nothing else in the class branches on the flag.** No `anthropic-beta` header
is added (grep: the only `betas=[...]` use is the Files-API path at
`:1518+`, unrelated). `_cache_hits` / `_cache_misses` /
`_total_cache_read_tokens` / `_total_cache_creation_tokens` (`:1354-1357`) are
incremented from the *response* usage fields, so they are flag-independent
accounting, not flag-gated behaviour.

**Measured arithmetic confirming the 78.1 wire capture** (run in-venv):

```
len(_HOUSE_INSTRUCTIONS)                     = 19026
len("\n\nYou MUST respond with a valid JSON object only.") = 49
total                                        = 19075   ← exactly the captured value
```

**Therefore the six services hit the `else` branch** — they pass
`_strip_unsupported_schema_keys(Model.model_json_schema())`, i.e. a plain
**dict**, which has no `.model_json_schema` attribute. **Consequence: the
cacheable system block is byte-identical (19,075 chars, ~5.0–5.4K tokens)
across all six services and across every other `ClaudeClient` JSON-mode
caller in the repo.** The per-service schema never enters the cached block.

This is the decisive fact. The 23.1.2 premise ("prompt differs per ticker")
was about the *user message*, which is never cached on either setting.

### A5. Does the CC rail care? — No.

`backend/agents/claude_code_client.py`:
- No `enable_prompt_caching` parameter anywhere in the module (grep: 0 hits).
- `:524` `system = config.get("system") or config.get("system_instruction")` →
  `:595 system=system` → `:269-270 args.extend(["--append-system-prompt", system])`.
  Caching is entirely the CLI's own affair.
- `:620-621` merely **reads back** `cache_read_input_tokens` /
  `cache_creation_input_tokens` from the CLI envelope for accounting
  (`:641-642` into `UsageMeta`).

**So `enable_prompt_caching` is a no-op whenever `PAPER_USE_CLAUDE_CODE_ROUTE=true`
and only bites on the metered fallback** — i.e. precisely the state that
78.1's success criterion 5 (`=false`) restores. The blocker's blast radius is
exactly the documented revert path and nothing else.

Corollary worth stating: **the rewire also silently DROPPED the house prompt
on the rail** before `fe5476f2` (ClaudeClient ignores `config["system"]`, so
the six had always received `_HOUSE_INSTRUCTIONS`; `ClaudeCodeClient` reads
`config["system"]` and got `None`). That was fixed by passing
`"system": _HOUSE_INSTRUCTIONS` explicitly at each of the six call sites
(e.g. `backend/services/meta_scorer.py:243`) — inert on the metered path.

### A6. Test surface + the wire-kwarg capture seam

**The seam to reuse is `backend/tests/test_claude_request_shapes.py:52-80`.**
It is the established, purpose-built wire-kwarg capture in this repo:

- `:52-58` `_CaptureMessages.create(**kwargs)` appends to a list and returns a
  fake response.
- `:61-65` `_fake_client(captured)` exposes both `.messages` and
  `.beta.messages` (so Files-API/beta paths capture too).
- `:68-80` `_shape(monkeypatch, model, config)` monkeypatches
  `ClaudeClient._get_client`, neutralises `observability.log_llm_call`, builds
  the client, calls `generate_content`, returns `captured[0]`.
- `:26` sets `COST_BUDGET_HARD_BLOCK_DISABLED=1` at import time — required,
  because `generate_content:1439` calls `_check_cost_budget()` first.

Caveat for 78.16: `_shape` constructs `ClaudeClient(...)` **directly**
(`:77`), so it exercises the *class* default, not the *`make_client`* default.
A 78.16 test asserting the **revert-path request shape** must drive
`make_client(model, None, settings_with_rail_off)` and capture at the same
seam, otherwise it tests the wrong constructor and passes vacuously. The
existing rail test already builds both settings objects —
`backend/tests/test_phase_78_1_c_block_rail.py:149-171` — so combining
`_CaptureMessages` with that settings pair is the minimal honest test.

Tests that would observe a change to the default:
- `backend/tests/test_phase_75_prompt_contracts.py:152-153` — constructs with
  `enable_prompt_caching=False` explicitly, so it is **unaffected by a
  `make_client` change** and will keep passing either way (i.e. it does NOT
  cover the regression; do not cite it as coverage).
- `backend/tests/test_claude_request_shapes.py` — all `_shape` tests assert on
  `thinking`/`temperature`/`output_config` keys, never on `system`, so they are
  green under both settings today.
- `tests/verify_phase_25_B9.py:213-215` — asserts `cc.enable_prompt_caching is
  True` after explicit construction; unaffected.
- `backend/tests/test_phase_78_1_c_block_rail.py` — AST + client-type checks
  only; blind to the kwarg.

**Net: there is currently ZERO test coverage of the `make_client` → caching
default.** That is why 78.1 shipped the regression.

### A7. Do the six repeat calls within a 1h TTL? (does the cache ever HIT)

| Service | Call pattern | Calls per cycle | Cache-hit potential |
|---|---|---|---|
| `meta_scorer.py:233` | ONE batched call over all candidates (head/tail split at `:266-289`) | 1 (rarely 2) | low alone |
| `macro_regime.py:526` | daily, behind a 24h file cache | 1/day | none alone |
| `news_screen.py:287-290` | `for _attempt in range(2)` retry loop | 1–2 | low alone |
| `pead_signal.py:293` | per-ticker, `asyncio.gather(*(_one(t) for t in tickers))` `:371` | N tickers | **high** |
| `analyst_narrative_scorer.py:162` | per-ticker `gather` `:227` | N tickers | **high** |
| `call_transcript_gpr.py:141` | per-ticker `gather` + `asyncio.Semaphore` `:199` | N tickers | **high** |

**But because the cached block is identical across all six (A4), the correct
unit of analysis is the FLEET, not the service.** Within one autonomous cycle
all six run, so calls 2..N across the whole fleet share one cache entry —
provided the block clears the per-model minimum and the writes are not all
racing concurrently (see external §B on parallel-request cache population).

---

## B. External sources — read in full

| # | URL | Accessed | Tier | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| E1 | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-07-25 | 2 official docs | WebFetch, full page | **Haiku 4.5 minimum = 4,096 tokens**; under-minimum = silent no-op; concurrency caveat; invalidation table |
| E2 | https://platform.claude.com/docs/en/about-claude/pricing | 2026-07-25 | 2 official docs | WebFetch, full page | Haiku 4.5 $1 / $1.25 (5m) / $2 (1h) / $0.10 (hit) / $5 out; break-even stated verbatim |

### B1. Minimum cacheable prompt length — Haiku 4.5 is **4,096 tokens**

Verbatim from E1 (the current list; note it is per-model, not per-family):

> * 512 tokens for Claude Opus 5, Claude Fable 5, and Claude Mythos 5
> * 2,048 tokens for Claude Mythos Preview and Claude Opus 4.7
> * 4,096 tokens for Claude Opus 4.6 and Claude Opus 4.5
> * 1,024 tokens for Claude Opus 4.8, Claude Sonnet 5, Claude Sonnet 4.6, Claude Sonnet 4.5, …
> * **4,096 tokens for Claude Haiku 4.5**
> * 2,048 tokens for Claude Haiku 3.5 (retired…)

This **corrects the search-snippet claim of 2,048 for Haiku 4.5** (that number belongs to Haiku 3.5) and **confirms the internal comment** at `backend/agents/llm_client.py:48` ("Haiku 4.5 = 4096").

Under-minimum behaviour (verbatim):

> "Shorter prompts cannot be cached, even if marked with `cache_control`. Any requests to cache fewer than this number of tokens will be processed without caching, **and no error is returned**. To verify whether a prompt was cached, check the response usage fields: if both `cache_creation_input_tokens` and `cache_read_input_tokens` are 0, the prompt was not cached…"

**Where pyfinagent's block lands.** `_HOUSE_INSTRUCTIONS` + JSON suffix = **19,075 chars**.
Haiku 4.5 predates the new tokenizer (E2: "Claude 4.7 and later models … use a newer
tokenizer… Claude Sonnet 4.6 and earlier models use the previous tokenizer"), so the
right heuristic is Anthropic's own "1 token is approximately 4 characters" (E2 FAQ):

| Heuristic | Estimated tokens | vs 4,096 floor |
|---|---|---|
| 3.5 chars/token (25.B9's assumption) | ~5,450 | +33 % headroom |
| 4.0 chars/token (Anthropic FAQ) | ~4,769 | **+16 % headroom** |
| 4.5 chars/token (conservative) | ~4,239 | +3.5 % headroom |
| 5.0 chars/token (pessimistic) | ~3,815 | **BELOW the floor → silent no-op** |

**So caching on Haiku 4.5 in this codebase is near the threshold, not comfortably
over it.** It is almost certainly *above* it (the 4-chars/token estimate is the
documented one) but this is an ESTIMATE — the only authoritative check is
`cache_creation_input_tokens > 0` on a real Haiku response. See §G open gaps.

### B2. Pricing + the exact break-even

Verbatim from E2, Haiku 4.5 row:

| Model | Base Input | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | $1 / MTok | $1.25 / MTok | **$2 / MTok** | **$0.10 / MTok** | $5 / MTok |

Multipliers (verbatim, E1 and E2 agree):

> * 5-minute cache write tokens are 1.25 times the base input tokens price
> * **1-hour cache write tokens are 2 times the base input tokens price**
> * Cache read tokens are 0.1 times the base input tokens price

And the break-even, stated by Anthropic itself (E2, verbatim):

> "A cache hit costs 10% of the standard input price, which means caching pays off
> after just one cache read for the 5-minute duration (1.25x write), or **after two
> cache reads for the 1-hour duration (2x write)**."

**Derivation for pyfinagent's `ttl:"1h"`, T = cached tokens, N = calls sharing the entry:**

```
uncached total  = N · 1.0 · T
1h-cached total = 2.0 · T + (N-1) · 0.1 · T
cheaper iff      2.0 + 0.1(N-1) < N   →   1.9 < 0.9N   →   N > 2.11   →   N >= 3
```

**N = 1 (cache never hits) is a pure 2x cost increase on the cached block.**
At T ≈ 4,769 tokens and $1/MTok that is **+$0.0048 per call** — the whole-fleet
per-cycle magnitude is single-digit cents either way (see §F).

### B3. Does `ttl:"1h"` need a beta header? — **No, it is GA.**

E1 verbatim syntax:

```json
"cache_control": { "type": "ephemeral", "ttl": "1h" }
```

> "The 1-hour cache duration is available on the Claude API, Amazon Bedrock, …
> Google Cloud, and Microsoft Foundry."

No beta header is mentioned anywhere in the current page. The legacy
`extended-cache-ttl-2025-04-11` header appears only in older third-party writeups
(see §C). **Cross-check against the repo's pinned SDK** (`backend/requirements.txt:41`
→ `anthropic==0.96.0`, installed version confirmed `0.96.0`): the SDK types the field
as first-class, not beta —

```python
# anthropic/types/cache_control_ephemeral_param.py  (v0.96.0, read locally)
class CacheControlEphemeralParam(TypedDict, total=False):
    type: Required[Literal["ephemeral"]]
    ttl: Literal["5m", "1h"]
    """… Defaults to `5m`."""
```

and `Messages.create` has **no `betas` parameter** (checked via `inspect.signature`),
confirming the 1h TTL rides the plain (non-beta) endpoint. **The note at
`backend/agents/llm_client.py:1467-1474` is accurate**: the 2026-03-06 default drop
to 5m is real, explicit `ttl:"1h"` is the correct restoration, and the "silent no-op"
warning refers to the *minimum-length* rule (B1), not to a missing header. Both halves
of that comment check out.

### B4. Cache invalidation semantics

From E1 (verbatim table + hierarchy):

> "the cache follows the hierarchy: `tools` → `system` → `messages`. Changes at each
> level invalidate that level and all subsequent levels."

| Change | Impact |
|---|---|
| Tool definitions | invalidates entire cache (tools, system, messages) |
| Images added/removed anywhere | affects message blocks |
| Thinking parameters | **always invalidates message blocks** |
| `output_config.effort` | always invalidates message blocks |
| `tool_choice` | affects message blocks only |

Relevant to pyfinagent: the six send **no tools and no images**, and the cached
breakpoint is on the **system** block only — so the per-ticker user message varying
does **not** invalidate the system-level entry. Haiku 4.5 carries no `output_config`
(pinned by `backend/tests/test_claude_request_shapes.py:149`), so the effort-invalidation
row does not bite. Model identity does scope the cache — all six default to
`claude-haiku-4-5`, but `analyst_narrative_scorer` / `call_transcript_gpr` take a
caller-supplied `model=` arg, so a caller passing a different model splits the entry.

### B5. The concurrency trap (decisive for the per-ticker services)

Verbatim, E1:

> "For concurrent requests, note that a cache entry only becomes available after the
> first response begins. If you need cache hits for parallel requests, wait for the
> first response before sending subsequent requests."

`pead_signal.py:371`, `analyst_narrative_scorer.py:227` and `call_transcript_gpr.py:199`
all fan out with `asyncio.gather` (the last with a `Semaphore`, which bounds but does
not serialise). **If those N per-ticker calls are the first to touch the prefix in the
window, they all MISS and all pay the 2x write** — the exact failure mode the caller
flagged. They only benefit if an earlier, sequential call in the same cycle (e.g.
`macro_regime` or `meta_scorer`, both single-shot) has already populated the identical
block within the hour.

### B6. Additional sources read in full

| # | URL | Accessed | Tier | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| E3 | https://tianpan.co/blog/2026-04-17-prompt-cache-break-even-math | 2026-07-25 | 5 practitioner blog | WebFetch, full page | break-even formula; **parallel-request race**: "All 100 pay the write overhead; none can read from cache" |
| E4 | https://claude.com/blog/prompt-caching (308 from anthropic.com/news/prompt-caching) | 2026-07-25 | 2 official (vendor announcement) | WebFetch, full page | canonical GA announcement, Dec 17 2024 GA / Aug 14 2025 rev; -90% cost / -85% latency claims; **no minimum-length figure stated at announcement** |
| E5 | https://www.mager.co/blog/2026-04-29-claude-prompt-caching/ | 2026-07-25 | 5 practitioner blog | WebFetch, full page | prefix match is **byte-for-byte exact, not semantic**; hierarchy tools→system→messages |
| E6 | https://dev.to/whoffagents/claudes-prompt-cache-ttl-silently-dropped-from-1-hour-to-5-minutes-heres-what-to-do-13co | 2026-07-25 | 5 community | WebFetch, full page | dates the 2026-03-06 default drop 1h→5m; **claims a beta header is still required — CONTRADICTED, see below** |
| E7 | https://spring.io/blog/2025/10/27/spring-ai-anthropic-prompt-caching-blog/ | 2026-07-25 | 3 authoritative vendor blog (VMware/Spring) | WebFetch, full page | **independently corroborates Haiku 4.5 = 4,096 tokens**; "1-hour cache writes cost 2x vs 5-minute writes"; avoid caching when "cache miss rate >50%" |

**Conflict, and how it resolves.** E6 (Apr 14 2026) says the 1h TTL is "opt-in via
a beta header" and cites `anthropic-beta: prompt-caching-2024-07-31` — which is the
*original GA* header, not the extended-TTL one, so the post is imprecise on its own
terms. It is overruled by three independent lines of evidence: (1) E1, the current
official page (accessed 2026-07-25), documents `ttl:"1h"` with no beta header and lists
it as generally available across five platforms; (2) the repo's pinned
`anthropic==0.96.0` types `ttl: Literal["5m","1h"]` as a **first-class** field on
`CacheControlEphemeralParam`; (3) `Messages.create` in that SDK exposes **no `betas`
parameter** at all. **Conclusion: `llm_client.py:1475-1484` needs no beta header and
is correct as written.**

**Corroboration on the Haiku floor.** E1 (official, 2026-07-25) and E7 (Spring AI,
2025-10-27) independently state **4,096 tokens for Claude Haiku 4.5**. The
"2,048 for Haiku" figure that surfaces in search snippets belongs to Haiku 3.5.

### B7. Local token measurement of the cached block (three heuristics straddle the floor)

Run in-venv on the actual constant:

| Method | Result | vs 4,096 floor |
|---|---|---|
| chars = 19,075, words = 2,908 | — | — |
| words / 0.75 (Anthropic FAQ "0.75 words per token") | **3,877** | **BELOW — would be a silent no-op** |
| `tiktoken` cl100k_base (approximation; NOT Claude's tokenizer) | **4,551** | above |
| chars / 4 (Anthropic FAQ "1 token ≈ 4 characters") | **4,769** | above |

**This is the single most important open question in the step and it cannot be
closed from the sandbox.** Two of three heuristics clear the floor and one does not;
the spread is ±20 % around a hard threshold. Per E1, being under the floor is a
**silent no-op with no error**, so the only authoritative check is a real response's
`cache_creation_input_tokens`.

**The instrument already exists** — no new plumbing needed:
- `backend/agents/llm_client.py:1871-1874` logs
  `"[ClaudeClient] Cache MISS (created): {n} tokens cached"` at DEBUG when
  `cache_creation > 0`.
- `backend/agents/llm_client.py:1902` writes `cache_creation_tok=cache_creation` into
  the BQ `llm_call_log` row via `backend/services/observability.log_llm_call`.
- Pricing is already correct for the 1h rate:
  `backend/agents/cost_tracker.py:165-172` documents and applies the 2.0x write
  multiplier (phase-25.A9), and `MODEL_PRICING["claude-haiku-4-5"] = (1.00, 5.00)`
  at `cost_tracker.py:36` matches E2's $1/$5.
  *(Minor drift: that comment's anchor "llm_client.py:773-779" is stale — the
  `ttl:"1h"` site is now `:1475-1484`.)*

### B8. Intra-cycle ordering (does anything warm the shared block first?)

`backend/services/autonomous_loop.py`: `macro_regime` `:426` (single-shot) →
`pead_signal` `:440` (`gather` fan-out) → `news_screen` `:449` →
`call_transcript_gpr` `:736` (`gather`) → `analyst_narrative_scorer` `:760`
(`gather`). `meta_scorer` is invoked outside this block.

The ordering *looks* favourable — a single-shot call precedes the fan-outs and would
warm the shared 19,075-char block. **But `macro_regime` sits behind a 24h file cache**
(introduced in the same commit `743d65e5`: "24h file cache prevents LLM re-bill"), so
on most cycles it returns from disk and makes **no LLM call at all** — leaving
`pead_signal`'s concurrent `gather` as the first toucher, which per E1/E3 means every
one of those N calls misses and pays the 2x write. **The warm-up is not reliable.**

---

## C. Snippet-only sources (context; do NOT count toward the gate)

| URL | Tier | Why not fetched in full |
|---|---|---|
| https://docs.claude.com/en/docs/build-with-claude/prompt-caching | 2 | same content as E1 (mirror host) |
| https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching | 2 | Vertex partner mirror; pyfinagent uses the first-party API, and cloud.google.com reference pages are JS-rendered (WebFetch returns the nav tree only) |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching | 2 | same, duplicate of the above |
| https://www.anthropic.com/claude/haiku | 2 | marketing page; pricing already sourced authoritatively from E2 |
| https://platform.claude.com/cookbook/misc-prompt-caching | 2 | cookbook examples; no threshold/pricing facts not already in E1 |
| https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages | 2 | not applicable — the six are single-turn |
| https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching/ | 2 | Bedrock-only; confirms 1h GA on a platform we don't use |
| https://www.digitalapplied.com/blog/prompt-caching-2026-cut-llm-costs-engineering-guide | 5 | superseded by E3 on the same math |
| https://www.digitalocean.com/community/tutorials/prompt-caching-cost-break-even | 4 | duplicate of E3's break-even treatment |
| https://gu-log.vercel.app/en/posts/en-sp-112-20260313-anthropic-prompt-caching-2026-update | 5 | 2026 automatic-caching writeup; the automatic-caching feature is confirmed directly in E1/E2 |
| https://technspire.com/en/blog/prompt-caching-2026-real-cost-wins | 5 | cross-provider comparison, out of scope |
| https://hidekazu-konishi.com/entry/anthropic_claude_api_prompt_caching_and_token_efficiency.html | 5 | breakpoint guide; no new facts |
| https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025 | 5 | 2025 infra overview; snippet supplied the ~30 % hit-rate heuristic |
| https://reference.langchain.com/javascript/langchain/index/anthropicPromptCachingMiddleware | 4 | JS middleware, not our stack |

**Unique URLs collected: 21** (7 read in full + 14 snippet-only).

## D. Recency scan (last 2 years, 2024-07 → 2026-07)

**Performed. Result: 4 findings in the window that materially change the picture,
none of which supersede the official docs.**

1. **2026-03-06 — Anthropic silently dropped the default ephemeral TTL from 1 hour to
   5 minutes** (E6; dates it precisely). pyfinagent already handles this: the explicit
   `"ttl": "1h"` at `llm_client.py:1480` and the comment at `:1467-1470` both predate
   this brief and are correct.
2. **1-hour TTL went GA — no beta header** (E1, accessed 2026-07-25; corroborated by
   the pinned SDK's type definitions). This **supersedes** the 2025-era guidance in
   E7 and the 2026-04 claim in E6 that a beta header is required.
3. **Automatic caching (single top-level `cache_control`) now exists** as an
   alternative to explicit per-block breakpoints (E1/E2, 2026). pyfinagent uses the
   explicit-breakpoint form, which remains fully supported — **no migration needed**,
   but it is the modern idiom if the block layout is ever revisited.
4. **The Haiku 4.5 minimum is 4,096 tokens, not the 2,048 that older/family-level
   summaries imply** (E1 + E7). This is the number the whole 78.16 decision turns on.

No 2024–2026 source contradicts the pricing multipliers (1.25x / 2x / 0.1x), the
byte-exact prefix-match semantics, or the concurrency caveat.

## E. Queries run (3-variant discipline)

| Variant | Query | Yield |
|---|---|---|
| **Year-less canonical** | `prompt caching Anthropic Claude` | E4, E5, Vertex mirrors, cookbook |
| **Year-less canonical (targeted)** | `Anthropic prompt caching minimum cacheable prompt length Haiku 4.5 2048 tokens` (domain-scoped to anthropic/claude docs) | E1, E2 — and exposed the 2,048-vs-4,096 snippet conflict that E1 resolved |
| **Last-2-year (2025)** | `prompt caching cost break-even cache write premium 2025 LLM inference` | E3, E7, introl 2025 |
| **Current-year (2026)** | `prompt caching 1 hour TTL extended-cache-ttl-2025-04-11 beta header required 2026` | E6, Bedrock 1h GA, digitalapplied |
| **Current-year (2026)** | `Anthropic prompt caching changes 2026 cache_control automatic caching top-level breakpoint` | automatic-caching finding, gu-log, technspire |

## F. Recommendation

### **Ship (a) — `make_client` accepts and forwards `enable_prompt_caching` — now. Queue the (c) question as a separate, MEASURED step.**

**Why (a) is right today, in order of decisiveness:**

1. **78.1's own success criterion 5 is a correctness property, and only (a) restores
   it.** The documented one-flag revert must return the metered path to its pre-78.1
   wire shape. Right now it does not: `system` goes out as a 1-block list carrying
   `cache_control` where it used to be a plain `str`. Whether that is *cheap* is a
   separate question from whether the revert is *faithful*. A revert lever that
   quietly changes the request shape is a broken lever.
2. **It is a one-line change at a single chokepoint with zero caller-break risk.**
   `llm_client.py:2139` is the only production `ClaudeClient(...)` site (A3), and all
   13 `make_client` callers pass exactly 3 positional args (A2). A 4th
   keyword-with-default breaks nothing.
3. **(c) requires "measured harmless" and the measurement is currently impossible
   from this sandbox** — worse, it is impossible *in principle* while the rail is on,
   because the flag only bites on the metered path (A5). Recommending (c) on the
   strength of a token estimate that straddles the threshold by ±20 % (B7) would be
   asserting rather than measuring.
4. **The plausible downside of leaving it ON is small but strictly one-directional in
   the bad case.** If the block clears 4,096 and the concurrent `gather`s all miss
   (E3's race; B8 shows the warm-up is unreliable), every call pays 2x on ~4.8K
   tokens ≈ **+$0.0048/call** — a pure loss with no offsetting benefit. If it clears
   4,096 *and* gets ≥2 reads per write, it's a net win of similar magnitude. If it
   does **not** clear 4,096, it is exactly $0.00 either way. Fleet magnitude is
   single-digit cents per cycle in all three cases, so **cost is not the deciding
   factor — revert fidelity is.**

**Concrete shape:**

```python
def make_client(model_name: str, vertex_model, settings: "Settings",
                enable_prompt_caching: bool = True) -> LLMClient:
    ...
    return ClaudeClient(model_name=model_name, api_key=anthropic_key,
                        enable_prompt_caching=enable_prompt_caching)
```

with the six passing `enable_prompt_caching=False` at their `make_client(...)` call
sites (A2 rows 8-13). Inert on the CC rail (A5), so it is a metered-path-only change.

**Then queue a separate step** whose live_check is the measurement — because the
23.1.2 rationale IS stale (A1) and the answer is probably "turn it on fleet-wide":

> Run one metered `claude-haiku-4-5` call through `ClaudeClient` with caching ON and
> record `cache_creation_input_tokens` from the response (surfaced at
> `llm_client.py:1871-1874` and persisted to `llm_call_log.cache_creation_tok` via
> `:1902`). `> 0` ⇒ the block clears the 4,096 floor; `== 0` ⇒ documented silent
> no-op and option (c) is trivially harmless.

### What would have to be TRUE for the other options

**(b) — preserve caller intent some other way** (e.g. a `settings.llm_prompt_caching_enabled`
flag, or per-service settings keys) is correct **iff** the intent is a *global policy*
rather than a *per-call-site* one, i.e. iff more consumers than these six need to
express it. Today they don't: `make_client` is the sole ctor site and only 6 of 13
callers care. (b) buys the same behaviour for more machinery and more drift surface.
If the measured answer later turns out to be "caching should be OFF everywhere on
Haiku", (b) becomes the better shape and (a)'s parameter is the natural place to hang
the default.

**(c) — the six stop caring, correct the doc** is correct **iff** measurement shows
either:
- **(c-i)** `cache_creation_input_tokens == 0` on a real Haiku 4.5 call ⇒ the block is
  under the 4,096 floor, `cache_control` is a documented silent no-op (E1), the wire
  difference is cosmetic, and correcting the doc IS the whole fix; **or**
- **(c-ii)** `cache_creation_input_tokens > 0` **and** `llm_call_log` shows ≥2 cache
  reads per write within the hour across the fleet ⇒ caching is a net win, the 23.1.2
  premise is confirmed obsolete, and the right move is to *delete* the `False` intent
  rather than preserve it.

Note (c-i) and (c-ii) are mutually exclusive and both are cheap to test with the
existing instrument. **Nothing in the literature can settle this — only the
measurement can.** That is precisely why (a) ships first: it makes the revert lever
honest *while* the measurement is pending, and it costs one line to undo if (c-ii)
wins.

### The exact test seam for wire-kwarg capture

**Reuse `backend/tests/test_claude_request_shapes.py:52-80`** — `_CaptureMessages`
(`:52-58`) + `_fake_client` (`:61-65`) + `_shape` (`:68-80`). It is the established
purpose-built seam: it monkeypatches `ClaudeClient._get_client`, neutralises
`observability.log_llm_call`, and returns `captured[0]` = the literal kwargs dict
passed to `messages.create`. It also sets `COST_BUDGET_HARD_BLOCK_DISABLED=1` at
import (`:26`), which is mandatory because `generate_content:1439` calls
`_check_cost_budget()` first.

**Two mutation-resistance requirements** (both are ways this test could pass
vacuously):

1. **Drive `make_client`, not `ClaudeClient` directly.** `_shape:77` constructs the
   class directly, so a copy-paste of `_shape` would exercise the *class* default and
   stay green even if the forwarding were never added. The test must build the client
   via `make_client(model, None, settings_rail_off)` — reuse the settings pair already
   built at `backend/tests/test_phase_78_1_c_block_rail.py:149-171` — and only then
   call `generate_content` against the patched `_get_client`.
2. **Assert on the `system` value, not on the presence of the key.**
   `"system" in kwargs` is true on both settings. The revert-path assertion must be
   `isinstance(k["system"], str)` **and** `"cache_control" not in json.dumps(k["system"])`;
   pair it with the caching-ON case asserting the 1-block list carries
   `{"type":"ephemeral","ttl":"1h"}`. Deleting the forwarding must flip the first
   assertion — verify that by actually mutating it.

For the rail-ON side there is no wire to capture (`ClaudeCodeClient` shells out to the
CLI), so assert the **client type** there, as `test_phase_78_1_c_block_rail.py`
already does — do not fabricate a rail wire-shape assertion.

## G. Open gaps (could not close from this sandbox)

1. **The decisive one: the true Haiku 4.5 token count of the 19,075-char block.**
   Three heuristics give 3,877 / 4,551 / 4,769 against a hard 4,096 floor (B7).
   Closing it needs a real API response (`cache_creation_input_tokens`) or the
   `count_tokens` endpoint — both require live Anthropic credits, whose current state
   I could not verify. **Do not let anyone claim "caching is a no-op on Haiku" or
   "caching is saving money" without this number.**
2. **No historical `llm_call_log` query run.** `SELECT model, SUM(cache_creation_tok),
   SUM(cache_read_tok) FROM llm_call_log WHERE model LIKE 'claude-haiku%' …` (bounded
   + date-filtered) would answer gap 1 from existing data if any metered Haiku call
   has run since phase-25.B9. I did not run it: `execute-query` is approval-gated and
   this is a brief-only task. **This is the cheapest path to closing gap 1 — try it
   before spending a live call.**
3. **`backend/.env` is sandbox-denied to this agent**, so I could not confirm the
   current `PAPER_USE_CLAUDE_CODE_ROUTE` value, i.e. whether the metered path (the
   only place the flag bites) is even live right now.
4. **Real-world hit rate is unknown.** B8 argues the warm-up is unreliable
   (`macro_regime`'s 24h file cache usually suppresses the single-shot call that would
   warm the block), but I have not observed an actual cycle's call sequence or
   timings. The ≥2-reads-per-write break-even is therefore projected, not measured.
5. **Not investigated:** whether `analyst_narrative_scorer` / `call_transcript_gpr`
   callers ever pass a non-Haiku `model=`, which would split the cache entry
   (settings defaults are `claude-haiku-4-5` at `autonomous_loop.py:743,767`, but I
   did not audit every caller).

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 14,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The enable_prompt_caching=False rationale IS recorded (phase-23.1.2 brief:301 -- 'prompt differs per ticker so caching provides no benefit') and is now factually obsolete: caching applies only to the SYSTEM block, and phase-25.B9 later added _HOUSE_INSTRUCTIONS, a deliberately padded stable prefix. Measured: the cached block is 19,075 chars = _HOUSE_INSTRUCTIONS (19,026) + a fixed 49-char JSON suffix, byte-identical across all six services (they pass dict schemas, so the per-service schema branch never fires). ClaudeCodeClient has NO caching notion, so the kwarg bites ONLY the metered path -- exactly 78.1's revert path. make_client:2139 is the sole production ClaudeClient site and all 13 callers pass 3 args, so adding a kwarg is zero-risk. Official docs: Haiku 4.5 minimum is 4,096 tokens (corroborated by Spring AI), under-minimum is a SILENT no-op, 1h write = 2x / read = 0.1x (break-even at 2 reads), ttl:'1h' is GA with no beta header (confirmed against the pinned anthropic==0.96.0 types). Token estimates straddle the floor (3,877 / 4,551 / 4,769), so option (c) cannot be called harmless without measurement. RECOMMEND (a) now for revert fidelity; queue the flip-to-True question with cache_creation_input_tokens as the live_check.",
  "brief_path": "handoff/current/research_brief_78.16.md",
  "gate_passed": true
}
```
