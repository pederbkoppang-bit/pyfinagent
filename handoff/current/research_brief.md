# Research Brief — Step 59.1: Fable 5 model adoption (both layers, quality-first)

**Tier:** moderate-complex. **Phase-59.** Operator-directed 2026-06-11 (8 in-session pre-approvals).
**Researcher snapshot:** OLD (this session's own qa/researcher run on the pre-59.1 pins — fine; pins don't change protocol).
**Claude Code version (local):** **2.1.172** — VERIFIED via `claude --version`. This is >= the 2.1.170 floor required for Fable 5 and >= the 2.1.154 floor for Opus 4.8. The `fable` alias is supported on this version.
**This topic is <1 week old** (Fable 5 GA = 2026-06-09); year-less canonical prior-art on "Fable 5" is necessarily thin and I say so explicitly in the query log. Canonical prior-art DOES exist for the surrounding mechanics (subagent frontmatter, effort levels, Claude Code model aliases) and is cited.

---

## TL;DR — the load-bearing answers

1. **Frontmatter `model:` value = `fable`** (the alias). Officially documented in BOTH the Claude Code
   sub-agents frontmatter reference AND model-config doc. Use `model: fable` in `.claude/agents/*.md`,
   mirroring the existing `model: opus` idiom. Full model id `claude-fable-5` is also accepted but the
   alias is preferred (auto-tracks the recommended version, matches the existing `opus`-alias convention).
2. **`effort: max` REMAINS valid for Fable 5.** The effort table lists Fable 5 levels as
   `low, medium, high, xhigh, max` — identical superset to Opus 4.8. `max` is documented as accepted
   subagent frontmatter `effort` (session-only at the API layer, but valid as a per-subagent override).
3. **`maxTurns` is a documented subagent frontmatter field**: "Maximum number of agentic turns before
   the subagent stops." Raising it (qa 12->30, researcher 30->40) is the correct lever for the
   stall-mid-work signature.
4. **THE ECONOMICS CHANGE.** Fable 5 is the FIRST model whose Max adoption is NOT flat-fee-unlimited:
   free on Max/Pro/Team only **June 9 -> June 22, 2026**; "On June 23, we'll remove Fable 5 from
   those plans. Using it after that will require usage credits." This directly **invalidates the
   phase-29.2 "Max subscription -> flat-fee, no per-token ceiling" rationale** in CLAUDE.md and both
   agent-file comment blocks. The rationale must be rewritten to the new model: quality-first +
   rare-event firing (cost contained by *frequency*, not by a flat fee).
5. **Effort guidance for Fable 5 DIFFERS from Opus 4.8.** Opus doc says "start with `xhigh` for
   coding/agentic." Fable doc says "**Start with `high`, the default, for most tasks**, use `xhigh`
   for the most capability-sensitive workloads." For our quality-first rare-event roles, keeping
   `effort: max` is defensible (these ARE the capability-sensitive workloads and fire rarely), but
   the file comments should note that Fable's recommended baseline is `high`, not `xhigh`, so `max`
   is a deliberate over-spec — and the Fable doc's own caveat "Lower effort settings on Claude Fable 5
   still perform well and often exceed `xhigh` performance on prior models" means `high` would already
   beat our previous Opus-`max` config. (See the "effort decision" section for the recommendation.)

---

## A) EXTERNAL RESEARCH

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://code.claude.com/docs/en/model-config | 2026-06-11 | official doc | WebFetch (full) | `fable` alias = "Uses Claude Fable 5 for your hardest and longest-running tasks"; effort table lists Fable 5 = low/medium/high/xhigh/max; "**Fable 5 requires Claude Code v2.1.170 or later**"; "Fable 5 is not the default model... Select it with `/model fable`"; default effort = `high` on Fable 5 |
| https://code.claude.com/docs/en/sub-agents | 2026-06-11 | official doc | WebFetch (full) | Frontmatter `model` field: "`sonnet`, `opus`, `haiku`, `fable`, a full model ID (for example, `claude-opus-4-8`), or `inherit`. Defaults to `inherit`"; `maxTurns` = "Maximum number of agentic turns before the subagent stops"; `effort` options "`low, medium, high, xhigh, max`; available levels depend on the model" |
| https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5 | 2026-06-11 | official doc | WebFetch (full) | API id `claude-fable-5`; **1M token context window by default, up to 128k output tokens per request**; $10/$50 per Mtok; refusals return `stop_reason: "refusal"` as HTTP 200 (not error); "not billed for a request that is refused before any output is generated"; supports effort, vision, memory tool, compaction |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-06-11 | official doc | WebFetch (full) | **Fable 5 guidance (verbatim):** "**Start with `high`, the default, for most tasks**, use `xhigh` for the most capability-sensitive workloads, and step down to `medium` or `low` for routine work. Lower effort settings on Claude Fable 5 still perform well and often exceed `xhigh` performance on prior models." `max` and `xhigh` both available on Fable 5 |
| https://www.anthropic.com/claude/fable | 2026-06-11 | official (vendor) | WebFetch (full) | "Run Claude Fable 5 in an agent harness like Claude Code... it can work for days at a time: planning across stages, delegating to sub-agents, and checking its own work"; "Claude Fable 5 is thorough, proactive, and tests its own work." NO explicit subagent-tier guidance on this page |
| https://www.anthropic.com/news/claude-fable-5-mythos-5 | 2026-06-11 | official (vendor) | WebFetch (full, prior turn) | "$10 per million input tokens and $50 per million output tokens"; "From today through June 22, Fable 5 is included on Pro, Max, Team... at no extra cost"; "On June 23, we'll remove Fable 5 from those plans. Using it after that will require usage credits"; "At the highest effort, Claude Fable 5 reflects on and validates its own work"; classifier fallback to Opus 4.8 on cyber/bio/chem/distillation |
| https://explainx.ai/blog/is-fable-5-available-on-claude-code-2026 | 2026-06-11 | industry blog | WebFetch (full) | Independently confirms "Update to Claude Code v2.1.170 or later"; advises "skipping Fable 5 for quick one-off edits or bug fixes" due to cost; documents multiple-install version-conflict caveat |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://openrouter.ai/anthropic/claude-fable-5 | aggregator | Pricing confirmed via search snippet ($10/$50, cached input $1/Mtok = 90% off) — redundant with official |
| https://artificialanalysis.ai/models/claude-fable-5 | benchmark site | Independent perf/price analysis; snippet sufficient for the adversarial point (slow TTFT) |
| https://www.developersdigest.tech/blog/claude-usage-limits-fable-5-explained | industry blog | Burn-rate detail ("~2x faster than Opus" official; steeper in practice) captured via snippet — ADVERSARIAL evidence on the credit economics |
| https://www.coderabbit.ai/blog/fable-5-model-review | industry review | Corroborates "overkill for routine tasks; reserve for hard subtasks" |
| https://decrypt.co/370688/internet-furious-anthropic-claude-mythos-fable-5 | news | Community reaction to the post-free-period credit change; context only |
| https://lushbinary.com/blog/build-long-horizon-ai-agents-claude-fable-5-guide/ | industry blog | "Fable parent, Sonnet/Haiku subagents is the cost-sane configuration" — informs the harness-design note |
| https://www.truefoundry.com/blog/claude-fable-5-api-benchmarks-pricing-how-to-use-it | industry blog | Pricing/benchmark recap; redundant |
| https://www.cloudzero.com/blog/claude-mythos-pricing/ | industry blog | Pricing recap across the family; context |
| https://github.com/VoltAgent/awesome-claude-code-subagents | community repo | Prior-art on subagent frontmatter conventions (year-less canonical hit) |
| https://gist.github.com/danielrosehill/96dd15d1313a9bd426f7f12f5375a092 | community gist | Year-less canonical "Claude Code subagent frontmatter" reference |

**Sources read in full: 7. Unique URLs collected: 17. Floor (>=5 full, >=10 URLs): CLEARED.**

### Recency scan (last 2 years / 2024-2026)

Searched the 2026 frontier explicitly (Fable 5 GA 2026-06-09, so the *entire* corpus is last-2-weeks).
**Result: the canonical sources ARE the recency window** — there is no older prior-art on "Fable 5"
specifically because the model is 2 days old. New findings that materially shape the step:

1. **Post-June-22 credit economics** (announcement + developersdigest) — supersedes the phase-29.2
   flat-fee assumption. NEW, load-bearing.
2. **Fable-5-specific effort baseline is `high`, not `xhigh`** (effort doc) — differs from the
   Opus 4.8 guidance currently baked into `model_tiers.py` comments. NEW.
3. **v2.1.170 floor** (model-config + explainx) — our 2.1.172 clears it. NEW, verified locally.
4. **Burn-rate ~2x Opus official / steeper in agentic practice** (developersdigest, artificialanalysis)
   — ADVERSARIAL; informs the "metered-by-frequency not flat-fee" framing. NEW.

For the *surrounding mechanics* (subagent frontmatter schema, effort levels, model aliases), the
canonical year-less prior-art exists and is cited (sub-agents doc, model-config doc, community gists).

### Query log (3-variant discipline)

- **Current-year frontier:** "Claude Fable 5 model pricing context window max output tokens API";
  "Anthropic Fable 5 long-running agents harness subagent model selection engineering guidance";
  "Claude Fable 5 downsides slow expensive subagent overkill when not to use 2026" (adversarial).
- **Last-2-year window:** subsumed by frontier (model is 2 days old; 2024/2025 returns nothing on
  "Fable 5" by construction — stated explicitly per research-gate rule).
- **Year-less canonical:** "Claude Code subagent frontmatter model fable alias agent.md configuration"
  — surfaced the model-config doc + VoltAgent repo + danielrosehill gist (prior-art on the *schema*,
  which is what's actually load-bearing here; the schema predates Fable 5 and the `fable` value was
  simply added to the existing `model:` enum).

### Consensus vs debate (external)

- **Consensus:** `fable` is the correct alias; v2.1.170+ required; 1M context / 128K output; effort
  supported with `high` default; Fable 5 is purpose-built for long-horizon agentic harnesses that
  "validate their own work" — exactly the Researcher/Q-A evaluator-gate profile.
- **Debate / adversarial (the honest counterpoint):** Multiple independent industry sources call Fable 5
  "**overkill and overpriced for routine, latency-sensitive, or high-volume tasks**" and recommend the
  "Fable parent, cheaper subagents" topology. Burn rate is ~2x Opus officially and reportedly steeper
  in unbounded agentic sessions (one Max user "exhausted their full 5-hour window in 8 minutes").
  TTFT is high (~108s) because of heavy pre-answer reasoning.
  **Why this does NOT block the step:** our harness Researcher + Q-A are the precise opposite of
  "routine/high-volume" — they are rare-event (once per masterplan step), quality-critical, long-horizon
  reasoning roles. They are the canonical Fable 5 use case. The adversarial evidence is correctly
  *honored* by (a) keeping per-ticker/metered roles OFF Fable, and (b) rewriting the economics to
  "contained by frequency," and (c) flagging the post-June-22 credit cliff for the operator.

### Pitfalls (from literature) — applied

- **P1 — credit cliff (June 23):** after June 22, Fable usage draws Max usage credits. The agent-file
  comments and CLAUDE.md MUST record this so a future reader doesn't assume flat-fee. (announcement)
- **P2 — unbounded agentic burn:** the `maxTurns` raise is *also* a burn-control: it bounds how long a
  Fable subagent can run. Pair the raise with the cost note. (developersdigest)
- **P3 — effort over-spec:** Fable's recommended baseline is `high`; `max` is over-spec. Defensible for
  rare-event quality-first roles, but document it as a deliberate choice, not a default. (effort doc)
- **P4 — classifier fallback:** Fable refusals on cyber/bio/chem/distillation reroute to Opus 4.8
  (>95% sessions unaffected; finance unaffected). No code change needed — Claude Code handles it — but
  worth a one-line note. (announcement + model-config)
- **P5 — TTFT latency:** Fable is slow to first token. Irrelevant for rare-event batch roles; would
  matter for per-ticker/latency-sensitive roles — another reason to leave those OFF Fable. (artificialanalysis)

---

## B) INTERNAL CODE INVENTORY

### Layer-3 harness agent files

| File | Lines | Role | Current pins | Status |
|------|-------|------|--------------|--------|
| `.claude/agents/researcher.md` | 5-16 | Layer-3 Researcher | `model: opus` (L5), `maxTurns: 30` (L6), `effort: max` (L15) | needs: model->fable, maxTurns->40, comment block (L7-14) economics rewrite |
| `.claude/agents/qa.md` | 5-15 | Layer-3 Q/A | `model: opus` (L5), `maxTurns: 12` (L6), `effort: max` (L15) | needs: model->fable, maxTurns->30, comment block (L7-14) economics rewrite |

Exact current frontmatter keys (verified): `name, description, tools, model, maxTurns, effort, memory, color, permissionMode` (qa also has `skills:`). All keys are valid per the sub-agents frontmatter reference. The phase-29.2 comment blocks at researcher.md:7-14 and qa.md:7-13 contain the "Max-subscription flat-fee removes per-token ceiling" claim that is now stale.

### Layer-2 in-app MAS pin table — `backend/config/model_tiers.py`

| Element | Lines | Detail |
|---------|-------|--------|
| `_BUILD_TIER` dict | 42-74 | The pin table. `mas_main` = `claude-opus-4-8` (**L49**), `mas_qa` = `claude-opus-4-8` (**L51**), `autoresearch_strategic` = `claude-opus-4-8` (**L60**). `mas_communication`/`mas_research` = sonnet-4-6; `autoresearch_fast` = haiku; `autoresearch_smart` = sonnet; `gemini_*`/`layer1_swappable` = Gemini (locked/out-of-scope) |
| `_LIVE_TIER` | 81-82 | All-sentinel placeholder; not in scope (COST_TIER=build is live) |
| `EFFORT_SUPPORTED_MODELS` | 183-191 | tuple of effort-capable model-id prefixes. **Does NOT include `claude-fable-5`** -> `model_supports_effort()` returns False for Fable, so `llm_client` would silently DROP effort on a Fable route. MUST add `"claude-fable-5"` here if any role is repinned to Fable |
| `EFFORT_DEFAULTS` | 221-231 | per-role effort. `mas_main`/`mas_qa`/`mas_communication`/`mas_research` all = `"max"` (step-scoped 23.2.2 override, never reverted); `autoresearch_strategic` = `"high"` |
| `MODEL_EFFORT_FALLBACK` | 233-242 | tuple of `(model-id-prefix, effort)`. **No `claude-fable-5` entry.** Per the effort doc, the correct Fable default is `"xhigh"` (it's the only Opus-class member that also accepts xhigh). Add `("claude-fable-5", "xhigh")` |
| `resolve_model()` | 101-155 | single lookup point; honors `apply_model_to_all_agents` override (L134-137) for non-Gemini-locked roles |
| `resolve_effort()` / `resolve_effort_by_model()` / `model_supports_effort()` | 245-283 | effort resolution; the latter two are the gate that would drop Fable effort if the prefix list isn't updated |

**Consumers of the pin table (non-test) — every resolution path the unit test must cover:**
- `backend/agents/agent_definitions.py:181` `model=resolve_model("mas_main")`, `:229` `resolve_model("mas_qa")`, `:129`/`:275` (communication/research), `:60` strategic via run_memo
- `scripts/autoresearch/run_memo.py:180` `STRATEGIC_LLM": f"anthropic:{resolve_model('autoresearch_strategic')}"`
- `backend/api/agent_map.py:61` `"main"->"mas_main"`, `:65-66` skill_optimizer/directive_rewriter -> autoresearch_strategic, `:73-74` orchestrator/planner -> mas_main; `_inject_live_model()` calls `resolve_model()` per node
- `backend/agents/llm_client.py:1465-1481` resolves effort via `resolve_effort(role_hint)` / `resolve_effort_by_model(model_id)` and gates on `model_supports_effort(model_id)` (L1481) — **this is where a missing Fable prefix silently drops effort**
- `backend/api/settings_api.py`, `backend/config/settings.py` — also import resolve_model

### Rare-event vs metered classification (grounded in code, not vibes)

| Role | Invocation path (code anchor) | Cadence | In scope for Fable? |
|------|------------------------------|---------|---------------------|
| `mas_main` | `agent_definitions.py:181` "Ford (Slack Orchestrator)"; `multi_agent_orchestrator.py` (Layer-2 Slack/iMessage routing, NOT the per-ticker Gemini pipeline) | **operator-paced** (fires on inbound Slack/iMessage, human-initiated) — RARE | **YES** (quality-first, rare) |
| `mas_qa` | `agent_definitions.py:229` "Analyst (Q&A Agent)"; same Layer-2 Slack-routed orchestrator | operator-paced per Slack analytical request — NOT per-ticker in the Gemini sense (the per-ticker pipeline is the 28 Gemini agents in `orchestrator.py`, which are Gemini-locked & out of scope) | **YES** (quality-first, rare) — but see CAVEAT below |
| `autoresearch_strategic` | `run_memo.py:180` STRATEGIC_LLM; nightly cron `backend/autoresearch/cron.py:25` `"0 2 * * *"` (2am ET) | **1x/night** — RARE | **YES** (strategic synthesis, rare) |
| `mas_communication` / `mas_research` | `agent_definitions.py:129`/`:275` | sonnet-tier helper roles | NO (keep sonnet — cost discipline; research is the cheap fan-out leg) |
| `autoresearch_fast` / `autoresearch_smart` | run_memo fast/smart legs | haiku/sonnet high-volume | NO (keep — cost discipline) |
| `gemini_*` / `layer1_swappable` | per-ticker Gemini analysis (`orchestrator.py`) | **per-ticker, HIGH volume**; Gemini-locked (Vertex APIs) | NO (out of scope, locked) |

**CAVEAT on `mas_qa` / the step's "BOTH layers" instruction:** The step says "pin Fable 5 on the
QUALITY-FIRST rare-event roles in both MAS layers" and "Per-ticker/metered roles MUST keep their
current models." `mas_main`, `mas_qa`, and `autoresearch_strategic` are all rare-event/operator-paced
by the code evidence above — they are NOT the per-ticker firing path (that's the Gemini pipeline). So
repinning these three to Fable is consistent with the cost-discipline constraint. The genuinely
per-ticker roles (`gemini_*`) are Gemini-locked and untouched. **Recommendation: repin `mas_main`,
`mas_qa`, `autoresearch_strategic` -> `claude-fable-5`; leave `mas_communication`, `mas_research`,
`autoresearch_fast`, `autoresearch_smart` unchanged.** (Confirm the exact role set with the operator's
"both layers, quality-first" intent — but note the immutable criteria only NAME the Layer-3 agent
files explicitly; the Layer-2 `model_tiers.py` repin is the "both layers" half and should follow the
quality-first/rare-event rule above.)

### Ticket-queue agent map — `backend/services/ticket_queue_processor.py`

`agent_model_map` at **L165-169**: `main`/`q-and-a` = `claude-opus-4-8`, `research` = `claude-sonnet-4-6`
(consumed at L171 `model_name = agent_model_map.get(...)`, used at L235 `client.messages.create(model=model_name)`).

**Cost math for ticket agents:** tickets are operator-initiated Slack/iMessage messages — ~1-2/day,
human-paced (the project memory confirms ~1-2 tickets/day). Each ticket call is `max_tokens=1000`
(L233). Worst case 2 tickets/day x (say) 4K input + 1K output on Fable = `2 x (4000 x $10 + 1000 x $50)/1e6`
= `2 x ($0.04 + $0.05)` = **~$0.18/day** if both main+q-and-a fire on every ticket. Negligible in
absolute terms. **Recommendation:** repinning `main`/`q-and-a` here to `claude-fable-5` is defensible
on cost (these are the same quality-first operator-paced roles), BUT this map uses the **direct SDK
rail** (`client.messages.create`) — it is NOT routed through `resolve_model`/`model_supports_effort`,
so it carries no effort param and is unaffected by the `EFFORT_SUPPORTED_MODELS` gap. Note also the
56.2 CLI-rail flag (`paper_use_claude_code_route`): when set, tickets spawn via the Claude Code CLI
(`_spawn_real_agent`), which inherits the session/agent model, not this map. **Decision is the
operator's**; the immutable criteria for 59.1 do NOT name this file, so changing it is optional. If
changed, update the inline comments (L166-168) to say "fable-5" and the rationale.

### `.claude/settings.json`

- `effortLevel: "xhigh"` (L2) — the persistent main-session default.
- **NO `model` key** (verified: `grep -c '"model"' = 0`). The MAIN session model is therefore
  **user-default**, set by the operator's `/model fable` choice (which, per the model-config doc, v2.1.153+
  writes to *user* settings, not this repo file). **Nothing in the repo contradicts the operator's
  Fable default** — confirmed. The current session already runs `claude-fable-5[1m]`.
- `permissions.defaultMode: "bypassPermissions"` (L141) — unchanged by this step.

### Hardcoded-model-id test inventory (CRITICAL for the verification command)

Searched all of `backend/tests/`. Tests that hardcode model ids relevant to this change:

| Test file | Line | Assertion | Breaks on mas_main->fable? |
|-----------|------|-----------|----------------------------|
| `test_agent_map_live_model.py` | `test_endpoint_injects_live_model_field` | `main_node.live_model == "claude-opus-4-8"` | **YES — WILL FAIL.** This resolves `mas_main` through the build tier and asserts the literal. Step MUST update it to `"claude-fable-5"` |
| `test_apply_model_to_all_agents.py` | 52 | `resolve_model("mas_main") == _BUILD_TIER["mas_main"]` | NO (dynamic — reads the dict, not a literal) |
| `test_apply_model_to_all_agents.py` | 79 | override test: patches `gemini_model="claude-opus-4-7"`, asserts all roles resolve to that **override value** | NO (the literal is the override stand-in, not the build pin) |
| `test_phase_56_2_ops_fixes.py` | 184-213 | ticket CLI-rail tests; spawn `"main"` but assert on the RAIL (CLI vs SDK), not the model id | NO |
| `test_phase_39_1_autoresearch_env.py` | 27-33 | asserts `resolve_model('autoresearch_strategic')` is *interpolated* into env strings (dynamic) | NO |
| `test_phase_37_2_default_alignment.py` | 66 | `s.deep_think_model == "gemini-2.5-pro"` (Gemini, unrelated) | NO |
| `test_claude_code_client.py` | 155,172 | constructs client with `"claude-sonnet-4-6"` (unrelated) | NO |
| `test_phase_31_1_fixes.py` | 32,73 | `gemini_model="claude-sonnet-4-6"` stubs (unrelated) | NO |

**THE ONE BREAKING TEST: `test_agent_map_live_model.py::test_endpoint_injects_live_model_field`.**
It currently passes on the opus-4-8 baseline (verified: `1 passed`). After `mas_main -> claude-fable-5`
it asserts the wrong literal and fails. This is the exact analogue of the 56.2 `4-7->4-8` stale-assertion
update — follow that pattern: change the literal to `"claude-fable-5"` (the comment at
test_agent_map_live_model.py:60-61 already documents the 56.2 precedent).

### **VERIFICATION-COMMAND COVERAGE GAP (must flag to implementer + Q/A)**

The step's verification command is:
```
python -m pytest backend/tests -k 'fable or model_tiers or phase_59' -q && test -f handoff/current/live_check_59.1.md
```
I ran the `-k` collection LIVE. It currently collects **exactly 1 test**
(`test_phase_37_2_default_alignment.py::...gemini_deep_think...` — it matches the substring `model_tiers`
in its name). **It does NOT collect `test_agent_map_live_model.py`** (verified: ">>> NOT caught <<<").

Consequences for the implementer:
1. The `-k` net will NOT exercise the one test that breaks (`test_agent_map_live_model`). If the
   implementer fixes `model_tiers.py` but forgets to update that test, the verification command will
   still **exit 0 (false green)** while the real suite is red.
2. There is currently **no test matching `fable` or `phase_59`.** The step's criteria require a NEW
   test (it says "the unit test covers real resolution"). The implementer MUST add a test whose NAME
   contains `fable` or `phase_59` (e.g. `test_phase_59_1_fable_pins.py`) so the `-k` pattern catches it.
3. **Recommendation:** the new `test_phase_59_1_*.py` should assert the full real-resolution path —
   `resolve_model("mas_main") == "claude-fable-5"`, same for `mas_qa` and `autoresearch_strategic`;
   `"claude-fable-5" in EFFORT_SUPPORTED_MODELS`; `resolve_effort_by_model("claude-fable-5") == "xhigh"`;
   `model_supports_effort("claude-fable-5") is True`; and the unchanged roles still resolve to their
   prior pins. AND the implementer must update `test_agent_map_live_model.py` even though `-k` won't
   force it — Q/A should run the FULL `backend/tests` suite (not just `-k`) as part of EVALUATE to
   catch the out-of-net breakage. (Flag this in `experiment_results.md`.)

### CLAUDE.md effort-policy edit plan

The "Effort policy (Layer-3 harness MAS...)" bullet spans **L56-62**. Specific edits:
- **L56:** "owner is on a **Max subscription** (flat-fee, no per-token ceiling on Claude Code
  first-party usage)" -> rewrite. New economics: Max included Fable free only June 9-22 2026; from
  June 23 Fable draws usage credits, so the rationale is no longer "flat fee" but "**rare-event firing
  contains cost** (Researcher/Q-A fire once per masterplan step, not per ticker), plus the operator's
  explicit 2026-06-11 quality-first pre-approval accepting credit burn on these gate roles."
- **L57:** "Switch sessions to 4.8 via `/model claude-opus-4-8`" -> "`/model fable`" (and note v2.1.170+
  / 2.1.172 installed). Update "Opus 4.8 as of 2026-05-28" to reflect Fable as the session default.
- **L58 (Q/A):** `model: opus` -> note `model: fable`; effort note: Fable baseline is `high`, we keep
  `max` deliberately; maxTurns 12->30; add restart caveat + 2026-06-11 operator pre-approval.
- **L59 (Researcher):** `model: opus` -> `model: fable`; maxTurns 30->40; same credit-economics +
  restart note; the "Max plan auto-includes Opus 1M context" line -> Fable runs 1M by default on API.
- **L62:** the "default to the Anthropic-recommended pairing... unless Max-subscription + rare-event
  rationale applies" guidance -> update the rationale name to "rare-event + quality-first
  (credit-metered post-June-22)".

Also touch the **agent-file comment blocks** (researcher.md:7-14, qa.md:7-13): replace the
"Max-subscription flat-fee removes per-token ceiling" sentence with the rare-event + June-22-credit
framing, add the v2.1.170 floor + 2026-06-11 operator pre-approval + session-restart caveat.

---

## Application to pyfinagent (external -> internal mapping)

| External finding | Internal action (file:line) |
|------------------|------------------------------|
| `fable` is the valid frontmatter alias (sub-agents doc) | `researcher.md:5` + `qa.md:5`: `model: opus` -> `model: fable` |
| `maxTurns` bounds agentic turns (sub-agents doc) + burn-control (developersdigest) | `researcher.md:6` 30->40; `qa.md:6` 12->30 |
| Fable not in our effort-capability lists | `model_tiers.py:183-191` add `"claude-fable-5"` to `EFFORT_SUPPORTED_MODELS`; `:233-242` add `("claude-fable-5", "xhigh")` to `MODEL_EFFORT_FALLBACK` |
| Fable is quality-first/rare-event role match (anthropic/claude/fable + effort doc) | `model_tiers.py:49,51,60`: `mas_main`/`mas_qa`/`autoresearch_strategic` -> `claude-fable-5` |
| Stale literal assertion (56.2 precedent) | `test_agent_map_live_model.py`: `claude-opus-4-8` -> `claude-fable-5` |
| Verification `-k` net + "real resolution" criterion | NEW `backend/tests/test_phase_59_1_fable_pins.py` (name matches `phase_59`) |
| June-22 credit cliff supersedes flat-fee rationale (announcement) | `CLAUDE.md:56-62` + `researcher.md:7-14` + `qa.md:7-13` comment rewrites |
| Fable effort baseline = `high`, not `xhigh` (effort doc) | document the deliberate `max` over-spec in the comment blocks; EFFORT_DEFAULTS already `max` (no change needed, but note the rationale) |
| Ticket agents cost ~$0.18/day on Fable (cost math) | `ticket_queue_processor.py:165-169` — OPTIONAL repin (not named in 59.1 criteria); operator's call |
| Session model is user-default, repo has no `model` key | `.claude/settings.json` — NO change needed; operator `/model fable` already in effect |

### Effort decision (recommendation)

Keep `effort: max` on the Layer-3 Researcher + Q/A agent files. Justification: (a) Fable's effort table
accepts `max`; (b) these are the rare-event, quality-critical gate roles Fable's `max` ("reflects on and
validates its own work") is designed for; (c) the operator's 2026-06-11 directive is explicitly
quality-first. BUT document in the comments that Fable's *recommended* baseline is `high` (not Opus's
`xhigh`), and that per the effort doc "lower effort on Fable 5 often exceeds `xhigh` on prior models" —
so even `high` would already beat the previous Opus-`max` config. This is a deliberate, documented
over-spec, not an unexamined carry-over. For Layer-2 `EFFORT_DEFAULTS`, the roles repinned to Fable are
already `"max"` (the 23.2.2 step-scoped override) — no change required, but the same comment note applies.

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read in full: 5 official + 2 vendor/industry)
- [x] 10+ unique URLs total (17 collected)
- [x] Recency scan (last 2 years) performed + reported (entire corpus is the last-2-week window; stated explicitly)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (agent files, model_tiers, ticket_queue, settings.json, all 8 model-id tests, CLAUDE.md, consumers)
- [x] Contradictions / consensus noted (adversarial "overkill for routine" counterpoint surfaced + reconciled)
- [x] All claims cited per-claim with URL/file:line

---

```json
{
  "tier": "moderate-complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
