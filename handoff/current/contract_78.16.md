# Contract — phase-78.16

**Step id:** 78.16
**Phase:** phase-78 (Anthropic Max-rail LLM routing — full audit + fix)
**Priority:** P1 · `harness_required: true`
**Executor tag (masterplan):** opus-4.8/xhigh
**Boundary (masterplan, binding):** `llm_client.py` `make_client` + tests
**Date:** 2026-07-25 · Cycle 164

---

## 1. Why this step exists

Step 78.1 rewired six signal-overlay services off direct `ClaudeClient(...)`
construction onto `make_client(...)`, so that `PAPER_USE_CLAUDE_CODE_ROUTE`
governs them. All six previously constructed
`ClaudeClient(..., enable_prompt_caching=False)` — 6/6 explicitly OFF.
`make_client` constructs `ClaudeClient(model_name=..., api_key=...)` with no
such kwarg, against a constructor default of `True`.

**Consequence:** the documented one-flag revert (`PAPER_USE_CLAUDE_CODE_ROUTE=false`
— which is also 78.1's own criterion 5) no longer returns the six to their
pre-78.1 behaviour. Prompt caching is now ON where it was explicitly OFF, on the
metered money path, reached by a flag the operator is told is a safe revert.

The six cannot fix this themselves: `make_client` owns the construction. That is
why this is its own step and not part of 78.1.

---

## 2. Research-gate summary

**Brief:** `handoff/current/research_brief_78.16.md` (researcher subagent,
tier `moderate`).

> **GATE VERDICT — `gate_passed: true`** (envelope transcribed verbatim from the
> brief): `external_sources_read_in_full: 7`, `snippet_only_sources: 14`,
> `urls_collected: 21`, `recency_scan_performed: true`,
> `internal_files_inspected: 16`. Cleared to proceed to GENERATE.
>
> The brief's own recommendation is **option (a)**, reached independently of this
> contract: *"RECOMMEND (a) now for revert fidelity; queue the flip-to-True
> question with cache_creation_input_tokens as the live_check."* That matches the
> hypothesis in §3 and the queued step in §7.

Findings that drive the plan (file:line anchors are the brief's; I re-derived the
ones marked ✔ independently before writing this contract):

| # | Finding | Consequence for the plan |
|---|---------|--------------------------|
| R1 | The `enable_prompt_caching=False` rationale IS recorded, once: *"The PEAD prompt will be different per-ticker per-quarter so caching provides no benefit"* (`handoff/archive/phase-23.1.2/phase-23.1.2-research-brief.md:301`, Apr 2026). The five later services copy the idiom. | It was a deliberate decision, not a cargo-cult. Silently discarding it is a real defect, not a cleanup. |
| R2 | That rationale reasons about the **user message**; caching in this codebase is applied **only to the system block** (`llm_client.py:1475-1484`). phase-25.B9 later added `_HOUSE_INSTRUCTIONS` — a 19,026-char STABLE prefix — expressly so the block would clear the cache-write minimum. The two decisions were never reconciled. | The original premise is now **stale**. That makes a *future* "turn caching ON for the six" a legitimate, measurable question — but it is a **different decision** from this step. Queued separately (§7). |
| R3 ✔ | `ClaudeCodeClient` has no `enable_prompt_caching` notion. The kwarg only ever affects the **metered/direct** path — which is exactly the revert path. | Blast radius is precisely the revert path. No rail behaviour changes. |
| R4 ✔ | Exactly **one** production `ClaudeClient(` construction site remains: `llm_client.py:2139`, inside `make_client`. 13 production `make_client` callers, all 3 positional args, none introspect arity. | A 4th keyword param with a default is a pure-additive, zero-caller-break change. |
| R5 | Haiku 4.5 minimum cacheable prompt = **4,096 tokens** (Anthropic prompt-caching docs, verbatim). Under-minimum is a **silent no-op**, no error. The block is 19,075 chars ≈ **4,769 tokens** at Anthropic's own documented 4-chars/token heuristic — i.e. only ~16 % headroom. | Whether caching even *engages* for the six is **near the threshold and unproven**. Another reason not to opportunistically leave it ON. |
| R6 | 1h cache write = **2×** base input; cache read = **0.1×**. Anthropic states 1h caching "pays off after two cache reads". Derivation: cheaper iff N ≥ 3 calls share the entry. **N = 1 is a pure 2× cost increase.** | The divergence is not merely cosmetic; on a miss it *costs money*. |
| R7 | Concurrency trap, verbatim from the docs: *"a cache entry only becomes available after the first response begins."* `pead_signal.py:371`, `analyst_narrative_scorer.py:227`, `call_transcript_gpr.py:199` all fan out with `asyncio.gather`. | The three per-ticker services are exactly the ones that would all MISS and all pay the 2× write. The regression is worst where the fan-out is widest. |
| R8 | `ttl:"1h"` is GA — no beta header required on `anthropic==0.96.0` (`CacheControlEphemeralParam` types `ttl` first-class; `Messages.create` has no `betas` param). | The existing comment at `llm_client.py:1467-1474` is accurate; nothing to fix there. |
| R9 | Wire-kwarg capture seam already exists: `backend/tests/test_phase_75_prompt_contracts.py:147` `_claude_kwargs(config)`. | Reuse it. Do not invent a second capture idiom. |

**Magnitude, stated honestly:** at ~4,769 cached tokens and $1/MTok, a pure miss
costs **+$0.0048 per call**. Whole-fleet per-cycle this is single-digit cents.
This step is not justified by dollars — it is justified by *"a flag the operator
is told is a safe revert must actually be one."*

### Independent measurements I made before writing this contract

- **Wire shape, captured** (scratchpad probe driving the real
  `ClaudeClient.generate_content` assembly path with the exact config the six
  pass, SDK intercepted before any network call):
  - `enable_prompt_caching=False` → `system` is a **plain `str`**, len **19075**
  - `enable_prompt_caching=True` → `system` is a **1-block list**, `block0.text`
    len 19075, `cache_control = {'type': 'ephemeral', 'ttl': '1h'}`
  - All other kwargs identical (`model`, `max_tokens`, `temperature`,
    `messages`, `output_config`).
  This reproduces the 78.1 Q/A's captured divergence exactly.
- **Production telemetry** (`pyfinagent_data.llm_call_log`, 60d, BigQuery MCP):
  every `provider='anthropic'` row carrying `cache_creation_tok > 0` is an
  `agent='cc_rail'` row (sonnet-4-6: 681 of 3406; opus-4-7: 63 of 546). There is
  **no** production evidence of the *metered* `ClaudeClient` path ever
  registering a cache write for the house block — consistent with R5's
  "near the threshold, unproven".
- `model='claude-haiku-4-5'` has **zero** `provider='anthropic'` rows in 60 days
  (only 37 rows under the dated id `claude-haiku-4-5-20251001`, all with zero
  cache tokens). So today the divergence is **latent**: it goes live the moment
  the direct-API credits are restored (owed operator action 79.3).

---

## 3. Hypothesis

`make_client` silently discarding a caller's explicit construction intent is the
defect — independently of whether the resulting behaviour happens to be better or
worse. Therefore:

> Give `make_client` an `enable_prompt_caching` parameter that it forwards to
> `ClaudeClient`, and have the six state their intent explicitly again. The
> flag-OFF path then reproduces the pre-78.1 request shape **byte-for-byte**,
> and criterion 1 is satisfied in its **strong** form (exact restoration) rather
> than its escape-hatch form ("measured, justified and documented divergence").

**Explicitly rejected: option (c)** ("the six stop caring, correct the doc").
The criterion permits it only "if measured to be harmless", and R5+R6+R7 say it
is *not* measurably harmless: caching may not even engage (R5), a miss is a 2×
charge (R6), and the three widest-fan-out services are structurally the ones that
miss (R7). Changing money-path behaviour inside a step whose entire purpose is
"make the advertised revert honest" would be the wrong place to spend that
uncertainty. The *stale rationale* (R2) is a real finding and gets its own
research-gated step (§7) — not a silent ride-along here.

---

## 4. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "Flipping PAPER_USE_CLAUDE_CODE_ROUTE=false returns the six to their PRE-78.1 request shape, proven by captured wire kwargs (system as a plain str, no cache_control) -- or the divergence is measured, justified and documented as intended"
2. "A test asserts the revert-path request shape, so this cannot regress silently again"
3. "MUTATION: drop the caching intent again -> that test goes red"

**Verification command (immutable):**
```
.venv/bin/python -m pytest backend/tests/ -q -k 'llm_client or make_client or prompt_caching'
```

**live_check (immutable):**
`handoff/current/live_check_78.16.md`: captured pre/post wire kwargs for one
service on the flag-OFF path, and the mutation.

---

## 5. Plan

1. **`make_client` gains the parameter.**
   `def make_client(model_name, vertex_model, settings, *, enable_prompt_caching: bool | None = None)`.
   Keyword-only, default `None` meaning *"caller expressed no preference — keep
   the `ClaudeClient` class default"*. `None` is deliberately not `True`: it keeps
   the 7 non-C-block callers on today's behaviour with zero risk, and it makes
   "no opinion" distinguishable from "explicitly wants caching".
   Forward it at the single construction site (`llm_client.py:2139`) only when not
   `None`. The CC-rail branch ignores it (R3) — documented inline, since a reader
   will ask.

2. **The six restate their intent.** Each of the six passes
   `enable_prompt_caching=False` to `make_client`, with a comment pointing at the
   phase-23.1.2 rationale (R1) *and* at R2 (that the rationale is stale and
   revisiting it is queued), so the next reader is not left to re-derive it.

3. **Tests** in a new `backend/tests/test_phase_78_16_caching_intent.py`, built on
   the existing `_claude_kwargs` seam (R9):
   - **Revert-path shape, per service** (criterion 2): for each of the six, with
     `paper_use_claude_code_route=False`, the client `make_client` returns must
     produce `system` as a **plain `str`** with **no** `cache_control` — asserted
     on captured wire kwargs, not on the constructor argument.
   - **The parameter is actually honoured**: `enable_prompt_caching=True` through
     `make_client` produces the 1-block-list + `cache_control` shape. Without this
     the first test could pass because the parameter is ignored *in the other
     direction*.
   - **Default-`None` callers are unchanged**: a `make_client` call that omits the
     parameter still yields `enable_prompt_caching is True` (today's behaviour for
     the other 7 callers) — this is the regression guard for the blast radius.
   - **Rail path is unaffected** (R3): with the flag ON, `make_client` returns a
     `ClaudeCodeClient` regardless of the parameter, and passing it raises nothing.

4. **Mutation matrix** (criterion 3) — run AFTER the executor work is complete, per
   `feedback_executor_sees_mutation_transients`. Each mutation applied, suite run,
   reverted, and the revert SHA-verified. `find . -name '__pycache__' -prune`
   bytecode purge between mutations per step 78.14's finding:
   - M1: drop `enable_prompt_caching=False` from one service → revert-shape test RED.
   - M2: `make_client` accepts the param but drops it on the way to `ClaudeClient`
     (the exact 78.1 defect, re-injected) → revert-shape test RED.
   - M3: invert the default from `None` to `False` → the default-`None` blast-radius
     test RED.
   A mutation that leaves every test green is a **vacuous guard** and the test gets
   rewritten, not the mutation retired (`feedback_mutation_test_guards_and_fixtures`).

5. **Artifacts:** `experiment_results_78.16.md`, `live_check_78.16.md`, a fresh
   **Q/A** verdict transcribed verbatim into `evaluator_critique_78.16.md`,
   `harness_log.md` append, then the masterplan status flip (log before flip).

---

## 6. What this step does NOT do (scope fence)

- Does **not** change whether caching is ultimately *right* for the six. It
  restores the pre-78.1 posture; the re-decision is queued (§7).
- Does **not** touch the CC-rail path, `ClaudeCodeClient`, or the CLI's own
  caching (R3).
- Does **not** touch the other 7 `make_client` callers' behaviour (default `None`).
- Does **not** re-open 78.1's other criteria; 78.1 closes separately once this and
  78.2 have landed.

---

## 7. Defect queued out of this step

Per `feedback_queue_discovered_defects_in_masterplan` — R2 is a real, out-of-scope
finding and gets its own research-gated masterplan step rather than a prose
disclosure:

> **The `enable_prompt_caching=False` rationale for the six overlays is
> factually stale.** It was recorded in Apr 2026 on the premise that "the prompt
> differs per ticker so caching provides no benefit"; phase-25.B9 (May 2026) then
> introduced a 19,026-char byte-identical system prefix precisely so the block
> would clear the cache-write minimum. Nobody revisited the six. Deciding this
> needs a *measurement* the current outage blocks: whether the block clears Haiku
> 4.5's 4,096-token floor at all (R5 says ~16 % headroom on an estimate;
> `cache_creation_input_tokens > 0` on a real Haiku response is the only
> authoritative check), and whether the three `asyncio.gather` services can be
> made to serialise their first call (R7).

---

## 8. References

- `handoff/current/research_brief_78.16.md` — research gate (this step)
- `handoff/current/research_brief_78.1.md`, `contract_78.1.md` — the parent step
- `handoff/archive/phase-23.1.2/phase-23.1.2-research-brief.md:301` — original rationale
- `backend/agents/llm_client.py:46-53` (25.B9 house block), `:1345` (ctor default), `:1467-1484` (cache_control), `:2139` (the construction site)
- `backend/tests/test_phase_75_prompt_contracts.py:147` — the wire-kwarg seam
- Anthropic prompt caching + pricing docs (E1/E2 in the brief)
