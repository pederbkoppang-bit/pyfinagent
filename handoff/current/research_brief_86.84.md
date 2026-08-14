# Research Brief — step 86.84

**Topic:** Remedy design for Workflow `agent({schema})` drops:
`agent({schema}): subagent completed without calling StructuredOutput (after in-conversation nudge)`.
Root cause taken as GIVEN from the caller's measurement (NOT re-derived here): the
subagent exhausts its `.md` frontmatter `maxTurns` budget with no turn left to emit
the schema call.

**Tier:** complex · **Audit-class:** false · **Accessed:** 2026-08-14
**Runtime under test:** Claude Code **2.1.232** (`/Users/ford/.local/share/claude/versions/2.1.232`)

```json
{
  "brief_status": "COMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 11,
  "snippet_only_sources": 8,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "Turn semantics are doc-confirmed: max_turns counts tool-use turns only, and the StructuredOutput emission is itself a tool call, so the terminal action needs a spare tool-use turn -- the remedy is arithmetic, not heuristic. The documented default for an absent maxTurns is literally 'No limit' (agent-loop Turns-and-budget table), which is why the uncapped agent types never drop. Decompiling the 2.1.232 runtime shows: (a) the local workflow agent() reads only model/schema/isolation/label/agentType/effort from opts -- there is NO per-call turn budget, so 'reserve the last turn' is NOT expressible today; (b) on max_turns_reached the subagent runner logs and BREAKS, giving no terminal-action opportunity; (c) the schema branch THROWS when structured is undefined while the non-schema branch returns ot.text unconditionally -- so graceful degradation exists on the non-schema path only; (d) the remote-isolation path reads resultSubtype and names the cause, the local path does not, so the error string pyfinagent sees is cause-blind by construction. Raising the cap is what phase-59.1 already did (12->30, 30->40) and it recurred, because a cap cannot be sized from a distribution the cap itself right-censors. Recommended: remove the cap (or move to an uncapped agentType) and bound cost with the budget/token levers instead, keeping write-first durability and the existing in-script retry.",
  "brief_path": "handoff/current/research_brief_86.84.md",
  "gate_passed": true
}
```

## Search-query composition (3-variant discipline)

| Variant | Query run |
|---|---|
| Year-less canonical | `Claude Code subagent maxTurns frontmatter semantics documentation` |
| Year-less canonical | `"subagent completed without calling StructuredOutput" claude code` |
| Year-less canonical | `Anthropic "harness design for long-running apps" agent turn budget bound the loop` |
| Year-less canonical | `agent loop turn limit reserve final action structured output reliability` |
| Year-less canonical | `claude code workflow agent schema StructuredOutput not called github issue maxTurns subagent` |
| Current-year frontier (2026) | `LLM agent step budget early stopping "budget forcing" terminal action reserve 2026` |
| Last-2-year window (2025) | `agent max turns exhausted no final answer truncated tool loop 2025 mitigation` |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| 1 | https://code.claude.com/docs/en/agent-sdk/agent-loop | Official doc (tier 2) | WebFetch, full | "A turn is one round trip inside the loop"; "`max_turns` / `maxTurns`, which **counts tool-use turns only**"; Turns-and-budget table: Max turns default **"No limit"**; `error_max_turns` → "`result` field available? **No**". |
| 2 | https://code.claude.com/docs/en/sub-agents | Official doc (tier 2) | WebFetch, full | Frontmatter table `:291`: "`maxTurns` — Maximum number of agentic turns before the subagent stops". **No default stated.** Partial-output-on-cutoff is scoped to "a rate limit, overload, or server error". |
| 3 | https://code.claude.com/docs/en/agent-sdk/subagents | Official doc (tier 2) | WebFetch, full | `AgentDefinition.maxTurns` = "Maximum number of agentic turns before the agent stops". Depth/concurrency/spend caps documented; **no per-invocation turn cap**. |
| 4 | https://code.claude.com/docs/en/workflows | Official doc (tier 2) | WebFetch, full | "An `agent()` call resolves to `null` if you stop it mid-run or it hits an unrecoverable API error." "The subagents the workflow spawns always run in `acceptEdits` mode and **inherit your tool allowlist**". Cache prefix keyed on "model, effort level, agent type, tools, output schema, and working directory". |
| 5 | https://code.claude.com/docs/en/agent-sdk/structured-outputs | Official doc (tier 2) | WebFetch, full | "the SDK validates the output against it, **re-prompting on mismatch**"; `error_max_structured_output_retries`; "A result can also end with subtype `success` but no `structured_output` value ... **Treat that case as a failure as well.**" |
| 6 | https://github.com/anthropics/claude-code/issues/65500 | Vendor issue (tier 2) | WebFetch, full | **[ADVERSARIAL to the local retry]** OPEN, v2.1.162. Error quoted with "(after **2** in-conversation nudges)". Reporter: "This error is **not catchable at the script level**... the harness escalates this to a workflow-fatal failure that bypasses the script's own `.catch`". ~3.5M tokens, zero output. |
| 7 | https://github.com/anthropics/claude-code/issues/41143 | Vendor issue (tier 2) | WebFetch, full | **[ADVERSARIAL to "just raise maxTurns"]** **Closed as not planned**, v2.1.84. `maxTurns: 10` produced 72–75 tool calls, "Status: SUCCESS (not PARTIAL — agent was never stopped)". No maintainer comment. |
| 8 | https://github.com/anthropics/claude-code/issues/20625 | Vendor issue (tier 2) | WebFetch, full | **Closed as not planned** + `stale`. Request to declare `structured_output` in subagent frontmatter with "automatic retry on validation failure". Not implemented, no roadmap. |
| 9 | https://github.com/NousResearch/hermes-agent/issues/36239 | Practitioner issue (tier 4) | WebFetch, full | Closest prior art for forcing a terminal action. Soft user-role stop prompt is inadequate; proposed 3-layer hard stop: `tool_choice="none"`, **system**-role prefill, and scrub dangling tool calls → fixed fallback string. "soft prompts *do* work in short, stable contexts; the gap is specifically **long-context + heavy-tool-use**." |
| 10 | https://openai.github.io/openai-agents-python/running_agents/ | Official doc, adjacent vendor (tier 2) | WebFetch, full | Cross-domain: "raise[s] a `MaxTurnsExceeded` exception"; "Pass `max_turns=None` to **disable this turn limit**"; documented `on_max_turns` error handler "returning a controlled fallback instead of propagating the exception". |
| 11 | https://arxiv.org/html/2606.00198v1 (BAGEN, May 29 2026) | Peer-review-track preprint (tier 1) | WebFetch, full | **[ADVERSARIAL to "reserve a turn"]** "early stopping saves between 28% and 64% of tokens on failed trajectories" for 1.6–4.2pp success loss; "models predict feasibility above 70% even after 60% of the budget is consumed; **the alarm fires only in the final 20%**"; r≈0.35 between task success and interval hit rate. Does **not** recommend reserving budget for a final action. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://code.claude.com/docs/en/agent-sdk/typescript | Official doc | **Attempted; the page truncated before the Workflow section.** The `agent()` opts question was answered from the runtime instead (F-6). |
| https://code.claude.com/docs/llms.txt | Doc index | Index used to resolve exact page URLs; not a source. |
| https://github.com/anthropics/claude-code/issues/65731 | Vendor issue | `/deep-research` rate-limit banner; different failure mode. |
| https://github.com/anthropics/claude-code/issues/8501 | Vendor issue | Asks for authoritative frontmatter docs; superseded by the live table. |
| https://github.com/anthropics/claude-agent-sdk-python/issues/502 | Vendor issue | StructuredOutput wraps payload in an `output` key — a *malformed*, not *absent*, output. |
| https://github.com/anthropics/claude-agent-sdk-python/issues/571 | Vendor issue | Duplicate class of #502. |
| https://github.com/openai/openai-agents-python/issues/844 | Vendor issue | "Max turns exceeded" — same class, covered by source 10. |
| https://platform.claude.com/docs/en/build-with-claude/structured-outputs | Official doc | Referenced by source 5 for JSON-Schema keyword limitations (already recorded in project doctrine). |

## Runtime evidence (Claude Code 2.1.232, decompiled)

Not a web source, so not counted toward the gate — but it is the **authoritative**
answer to the three questions the docs do not address. Method: located the code
occurrence of the throw at byte offset **279,879,313** of the 306,111,312-byte
binary, dumped a 12 KB window, then a 16 MB region (`skip=266M count=16M`) for
symbol searches.

**R-1 — the exact string, singular nudge.** The 2.1.232 binary contains, verbatim:
```
if(Ue){if(Tt===void 0)throw Error("agent({schema}): subagent completed without
calling StructuredOutput (after in-conversation nudge)");
```
`Ue` = "a schema was requested". So the throw fires when the agent **completed**
— not stalled, not API-errored, not skipped — with `ot.structured === undefined`.
Note **"nudge", singular**: issue #65500 (v2.1.162) quotes "(after **2**
in-conversation nudges)". The message changed between versions. Telemetry symbols
`nudged`, `nudge_exhausted`, `nudge_ignored`, `thinking_only_retry` are present.

**R-2 — a stall-retry loop exists, and the drop does not enter it.** The runtime
already retries *stalled* agents (`for(let ln=1; ot.stalled && !Lt && ln<=$wf; ln++)`),
and that path even reports `structuredOutputAttempts` validation failures. The
"completed with no structured output" case is **not** `stalled`, so it bypasses the
runtime's own retry and goes straight to the throw. This is why an in-script retry
(pyfinagent's `agentRetryingDrops`) is the only retry that fires for this failure.

**R-3 — `max_turns_reached` BREAKS the loop.**
```
if(tr.attachment.type==="max_turns_reached"){w(`[Agent: ${e.agentType}] Reached
max turns limit (${tr.attachment.maxTurns})`);break}
```
There is no terminal-action hook on that branch. The loop simply ends.

**R-4 — the schema branch throws; the non-schema branch returns text.** In the same
function the schema branch ends `throw ... / return m(ln)`, and the final statement
for a schema-less call is `return ot.text`. So a **non-schema** workflow agent
returns whatever text accumulated even after the loop broke; the **schema** agent
returns nothing. This is the concrete asymmetry behind the caller's question 1e.

**R-5 — the remote path names the cause; the local path cannot.** `agent({isolation:'remote', schema})`
reads `resultSubtype` and throws one of three distinct messages, including
"the cloud agent turn ended with result subtype '<subtype>'" — which would surface
`error_max_turns`. The local path's result object exposes `stalled`, `apiError`,
`skipped`, `structured`, `structuredOutputAttempts`, `text`, `tokens`, `toolCalls`,
`agentMessages` — **no `resultSubtype`**. The local error message is therefore
**cause-blind by construction**: it says "never called StructuredOutput" whether the
model ended in prose or the runtime cut the loop at `maxTurns`.

**R-6 — no per-call turn budget on workflow `agent()`.** The workflow agent
implementation reads `ke.model`, `ke.schema`, `ke.isolation` from opts, plus
`label` / `phase` / `agentType` / `effort` at the call sites. All **16** occurrences
of `maxTurns` in the 16 MB workflow region are something else: internal one-shot
forks (`maxTurns:1` for rename / agent_summary / away_summary / side_question), a
built-in agent definition (`agentType:"comment-thread-analyst" ... maxTurns:6`), the
SDK `query()` option surface (`maxTurns`, `maxBudgetUsd`, `taskBudget`, `jsonSchema`),
an eval-harness zod schema, and the `max_turns_reached` attachment handler.
**Scope caveat:** this is a 16 MB window, not a whole-binary proof; combined with the
docs listing no such option, treat it as strong but not absolute.

**R-7 — a workflow-level token budget does exist.** `parallel()` maps a
`WorkflowBudgetExceededError` to `null` and logs `parallel: N slots dropped — token
budget exceeded`. `parallel()`/`pipeline()` also `Promise.allSettled` each slot and
convert a throwing `agent()` into `null` + a log line — so **in 2.1.232 a
StructuredOutput throw inside `parallel()` does not abort the run**, contradicting
#65500's report on v2.1.162. Consistent with pyfinagent's in-script `catch` working
today.

## Key findings

**F-1. A turn is a tool-use round trip, and the cap counts only those.** "A turn is
one round trip inside the loop: Claude produces output that includes tool calls, the
SDK executes those tools, and the results feed back to Claude automatically" and
"You can cap the loop with `max_turns` / `maxTurns`, which counts tool-use turns
only" (source 1).

**F-2. The StructuredOutput emission is itself a tool call, so it needs a spare
tool-use turn.** The runtime wraps the schema as a tool the model must call (R-1
shows the harness checking for the resulting value; #65500's analysis names it a
"fake tool definition"). With F-1 this makes the remedy **arithmetic**: the budget
must be `work_turns + 1`. A cap sized to the work is a cap that cannot terminate
cleanly.

**F-3. The documented default for an absent `maxTurns` is literally "No limit".**
Source 1's Turns-and-budget table. That is an *absence of a cap*, not a high default
— which is exactly the configuration of the two agent types the caller measured at
0 drops (`general-purpose` 0/252, `Explore` 0/263).

**F-4. At `maxTurns` there is nothing to salvage on the schema path.** `error_max_turns`
→ "`result` field available? No" (source 1). The one documented partial-return is for
"a rate limit, overload, or server error" (source 2), i.e. API errors, not turn
limits. R-4 refines this: the *non-schema* workflow path does return `ot.text`, so
the honest answer to question 1e is **"partially yes, and only off the schema path"**.

**F-5. Structured outputs have their own retry ladder — for the wrong failure.**
"re-prompting on mismatch ... If validation does not succeed within the retry limit,
the result is an error" → `error_max_structured_output_retries` (source 5). That
handles **malformed** output. pyfinagent's failure is **absent** output, a different
branch that the docs explicitly say to treat as a failure.

**F-6. There is no way to force the schema call, and the feature request to make one
was declined.** Issue #20625 asked for `structured_output` in subagent frontmatter
with automatic retry; **closed as not planned, labelled stale**. No documented
primitive makes the runtime emit a schema call at exhaustion.

**F-7. The nudge is undocumented; whether it consumes a turn is unknown.** No
official page fetched mentions it. Only #65500 describes it publicly, and its count
disagrees with the live binary (2 vs 1). **INFERENCE, not measured:** an
in-conversation nudge is a message that requires another assistant response, i.e.
another round trip — so a nudge cannot be answered by an agent whose turn budget is
already spent. That is consistent with the caller's exact-30/exact-40 pattern, but it
is reasoning about a mechanism I could not observe, and it must not be cited as fact.

**F-8. `maxTurns` enforcement has a public contradiction that is probably a
path/version difference.** #41143 reports `maxTurns:10` → 72–75 tool calls with status
SUCCESS on the **Agent-tool** path at v2.1.84 (closed as not planned). The caller
measured exact stops on the **Workflow** path at 2.1.2xx, and R-3 shows the
enforcement branch in the current binary. Both can hold. The consequence: `maxTurns`
is a load-bearing control with a history of *silent non-enforcement*, so any fix that
depends on its exact numeric value should be paired with a check that does not.

**F-9. Anthropic's own remedy for hitting the cap is "resume with a higher limit",
which the workflow rail cannot do.** Source 1's worked example comments
`maxTurns: 30, // Prevent runaway sessions` and then handles `error_max_turns` with
"Agent ran out of turns. Resume with a higher limit." Workflow `agent()` exposes no
session/agent handle to resume (R-6), so that documented escape hatch is unavailable
on this rail. (Noteworthy: **30 is Anthropic's own example number**, and it is
exactly pyfinagent's qa cap.)

**F-10. Cross-domain: the sanctioned pattern is an exhaustion HANDLER, not a bigger
number.** OpenAI Agents SDK raises `MaxTurnsExceeded`, supports `max_turns=None` to
disable, and documents `on_max_turns` "returning a controlled fallback instead of
propagating the exception" (source 10). Hermes-agent #36239 goes further: force the
terminal shape with `tool_choice="none"` + a **system**-role prefill, then *validate*
it and substitute a fixed fallback string if the model still emits tool calls — with
the measured caveat that soft prompts work in short contexts and fail in
"long-context + heavy-tool-use", which is precisely the Q/A and Researcher regime.

**F-11. [ADVERSARIAL] The literature does not endorse "reserve the last turn".**
BAGEN (source 11) recommends **early stopping on infeasibility**, not reserving
terminal budget, and its central negative result undercuts any prompt-level
self-rationing instruction: "models predict feasibility above 70% even after 60% of
the budget is consumed; the alarm fires only in the final 20%". Telling an agent
"save your last turn for the schema call" asks it to do the thing this paper measures
it cannot do. Budget awareness also "decouples from task performance" (r≈0.35), so a
capable agent is not thereby a budget-aware one.

**F-12. Right-censoring is why phase-59.1 recurred, and it is the sizing rule.** Every
run that "used exactly 30 turns" is a **censored observation** — the cap truncated it,
so it tells you the true requirement was `>=30`, never that 30 sufficed. A cap sized
from capped runs is sized from a distribution the cap itself created. 12→30 was fit to
censored data and recurred at 30; 30→40 recurred at 40. The uncapped types reaching 63
and 56 turns are the only uncensored evidence available, and both exceed 40.

## Consensus vs debate (external)

**Consensus.** (a) Turn caps are a *cost/runaway* control, not a correctness control —
Anthropic ("Setting a budget is a good default for production agents"), OpenAI, and
the practitioner sources all frame them this way. (b) Every framework surveyed makes
exhaustion an **explicit, handled outcome** (`error_max_turns`, `MaxTurnsExceeded`,
`handle_max_iterations`) rather than a silent truncation. (c) Disabling the limit is a
first-class documented option (Anthropic default "No limit"; OpenAI `max_turns=None`).

**Debate.** *How* to end a budget-exhausted run splits. Hermes-agent argues for a
**hard forced terminal action** (`tool_choice="none"` + system prefill + validation +
fixed fallback). BAGEN argues for **early stopping** and is explicitly sceptical that
the model can manage its own remaining budget. Anthropic's docs take a third line —
**resume with a higher limit** — which presumes a resumable handle the workflow rail
does not expose. Nothing in any source advocates the "reserve one turn by prompting"
approach, and F-11 is direct evidence against it.

## Pitfalls (from literature and the runtime)

1. **Reading the error message as the cause.** R-5: the local message is cause-blind.
   Do not let it become evidence that the model "chose prose".
2. **Sizing the next cap from capped runs.** F-12.
3. **Trusting the agent to self-ration.** F-11.
4. **Assuming the runtime retry covers this.** R-2: the drop is not `stalled`, so the
   runtime's own retry never fires for it.
5. **Assuming #65500's "not catchable" still holds.** R-7 shows the in-script catch
   working in 2.1.232; the project's own retry logs corroborate. Version-check before
   citing that issue as current behaviour.
6. **Quoting the nudge count.** The binary says one; the issue says two.
7. **A probe that matches its own subject.** Already burned this project once — the
   run record embeds the workflow SOURCE, and both workflow files quote the drop
   string (`scripts/qa/rail_drop_rate.py:44-56`, "THE SELF-MATCH TRAP"). Any new
   measurement here must read a named field, never the blob.

## Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/agents/qa.md` | `:6` `maxTurns: 30` | Q/A frontmatter cap | LIVE |
| `.claude/agents/qa.md` | `:15-16` | Comment: "maxTurns 30 (phase-59.1): the old 12 cap caused mid-evaluation stalls (20-26 tool-uses per evaluation); 30 gives headroom." | **This is the prior instance of the same failure class** |
| `.claude/agents/qa.md` | `:593-599` | "Verification budget ... your real bound is maxTurns" — the prompt tells Q/A the cap is its bound while giving it no way to observe remaining turns | LIVE |
| `.claude/agents/researcher.md` | `:6` `maxTurns: 40` | Researcher frontmatter cap | LIVE |
| `.claude/agents/researcher.md` | `:16-17` | Comment: "maxTurns 40 (phase-59.1): complex briefs hit the old 30 cap mid-write; 40 gives headroom." | **Same class, same commit series** |
| `.claude/workflows/qa-verdict.js` | `:436-451` `agentRetryingDrops(prompt, opts, maxAttempts = 2)` | Retries ONLY on `'without calling StructuredOutput'`; any other error rethrows | LIVE |
| `.claude/workflows/qa-verdict.js` | `:452-459` | The spawn: `{label, phase, schema, agentType:'qa', model:'opus', effort:'max'}` — **no turn option, consistent with R-6** | LIVE |
| `.claude/workflows/qa-verdict.js` | `:264-273` | agentType 'qa' pinned (was `general-purpose`) to CONSTRAIN the surface: probe `wf_9277ada4-390` showed general-purpose carries "Edit/Write/Bash + 7 loaded MCP tools + the full deferred MCP surface (incl. playwright)" | **Load-bearing constraint on option C below** |
| `.claude/workflows/qa-verdict.js` | `:400-406` | "The mechanism is UNPROVEN. Four hypotheses were tested ... and REFUTED — prompt/run size, wall clock, effort, and Anthropic's documented preamble-suppression trigger" | **Now supersedable** — the caller's turn measurement is a fifth hypothesis that these four never tested |
| `.claude/workflows/research-gate.js` | `:709-710` `STAGE1_MAX_ATTEMPTS = 3` | Stage-1 retry loop around the gate spawn | LIVE |
| `.claude/workflows/research-gate.js` | `:717` `agentType: 'researcher'` | Stage-1 pin | LIVE |
| `.claude/workflows/research-gate.js` | `:43-46` | "qa-verdict.js pins agentType 'qa' to RESTRICT the surface (Q/A is read-only) ... agentType 'researcher' gets Write via its `memory: project` injection" | **Corrects the caller's framing — see option C** |
| `.claude/workflows/research-gate.js` | `:750-751` `STAGE2_MAX_ATTEMPTS = 2`, `:792` `agentType: 'Explore'` | **Stage 2 already runs on an UNCAPPED built-in type** | LIVE — existing precedent |
| `.claude/workflows/research-gate.js` | `:663-694` | "the mechanism is UNPROVEN"; maxAttempts=3 justified by loss cost, not rate | LIVE |
| `scripts/qa/rail_drop_rate.py` | `:44-56` | Self-match trap; predicates read `error` / `logs`, never the blob | LIVE |
| `scripts/qa/rail_drop_rate.py` | `:36-42` | "`logs` is EMPTY on 44 of 44 dropped runs" — an exhausted run cannot be distinguished from one whose retry never ran | LIVE constraint on measuring any fix |

## Application to pyfinagent

**Direct answers to the caller's questions.**

| # | Question | Answer |
|---|---|---|
| 1a | `maxTurns` semantics; default when absent | A turn = one tool-use round trip; the cap counts tool-use turns only (source 1). Absent ⇒ **"No limit"** (source 1's table) — genuinely uncapped. Enforcement is real in 2.1.232 (R-3), historically unreliable on the Agent-tool path (#41143). |
| 1b | Per-call turn budget in `agent()` opts? | **No.** Not in the docs (sources 3, 4), and not read from opts by the runtime (R-6). `opts` carries `schema`, `label`, `phase`, `agentType`, `model`, `effort`, `isolation`. |
| 1c | Force the schema call / make it emit at exhaustion? | **No documented mechanism**, and #20625 asked for one and was closed as not planned. The runtime's re-prompt ladder covers *invalid* output only (F-5), and its stall-retry never sees this failure (R-2). |
| 1d | Is the nudge documented? Does it cost a turn? | Undocumented. Count disagrees between #65500 (2) and the 2.1.232 binary (1). Turn cost: **unknown**; F-7 gives the inference and labels it as such. |
| 1e | Does the non-schema Agent path degrade gracefully? | **Docs say no** for `error_max_turns` (no `result` field), and the documented partial-return covers API errors only. **The runtime says partially yes** for the workflow non-schema branch, which returns `ot.text` unconditionally (R-4). Prefer R-4: it is the code on the path in question. |
| 2 | Known issues | #65500 (open, exact string, not-catchable claim — now version-stale per R-7), #41143 (closed not planned, non-enforcement), #20625 (closed not planned, schema-in-frontmatter). **No public issue links maxTurns exhaustion to StructuredOutput drops** — the caller's measurement appears novel and is worth filing on #65500. |
| 3 | Sizing guidance | F-9, F-10, F-11, F-12. The published guidance is: caps are cost controls; exhaustion needs an explicit handler; disabling is first-class; and (BAGEN) don't ask the model to ration itself. |

**Is raising `maxTurns` sanctioned, or a smell?** It is *sanctioned* (Anthropic's own
`error_max_turns` guidance is "Resume with a higher limit") but it is **the move
phase-59.1 already made twice**, and F-12 explains why it recurs: the cap censors the
data you would use to size the next cap. The token trade-off is smaller than it looks
— a higher cap does not spend more turns, it only *permits* more; the marginal cost is
paid only by runs that would otherwise have been cut, and those runs are currently
paying full price for zero output. On a shared weekly Max pool, the drop is the
expensive outcome: research-gate.js:690 puts a lost brief at ~190K tokens, and the
caller's own figures put drops at 39/290 (qa) and 8/89 (researcher) on opus-5[1m].

**Is "reserve the last turn" expressible today?** **No** — not as a mechanism. There is
no per-call budget (R-6), no reserve primitive, no way to force the terminal call
(F-6), and the runtime `break`s without a terminal hook (R-3). The only expressible
approximations are: (i) make the cap large enough that the terminal turn is never the
one cut; (ii) remove the cap; (iii) prompt-level self-rationing — **refuted as
unreliable** by F-11 and the hermes long-context finding; (iv) durability so
exhaustion loses the envelope and not the work, which this project already does
(researcher write-first; `verdict_wip`).

**Three remedy options, with the trade-offs that decide between them.**

- **A — Remove the caps (delete the `maxTurns` frontmatter keys).** Turns the two roles
  into the same uncapped configuration as `general-purpose`/`Explore`, which the
  caller measured at 0 drops across 515 spawns. Doc-supported (F-3). Cost is then
  bounded by the runtime's own limits and by workflow token budget (R-7) rather than
  by turns. **Risk:** loses the runaway guard entirely; a pathological run could spend
  a lot before finishing. Mitigate with the workflow/spend budget levers, not with turns.
- **B — Raise the caps with headroom derived from uncensored data.** The only
  uncensored observations available are the uncapped types at 63 and 56 turns, both
  **above** the current 40. A defensible cap is therefore ≳2× the observed uncapped
  maximum for the role, not a small bump — and it must be paired with monitoring for
  runs landing exactly on the cap (that is the recurrence alarm). **Risk:** still fits
  a number to a censored distribution; recurrence is deferred, not removed.
- **C — Move off the custom `agentType` onto an uncapped type, passing the `.md` body
  as the prompt.** Documented as equivalent: "Use `prompt` for the system prompt,
  equivalent to the markdown body in file-based subagents" (source 2). **The caller's
  stated blocker is over-stated in one direction and under-stated in another:**
  `research-gate.js:46` records that the researcher gets `Write` from its
  `memory: project` injection rather than from the tools list, and workflow subagents
  "inherit your tool allowlist" (source 4) — so an uncapped default type would have
  `Write`. But `qa-verdict.js:264-273` records the *real* cost: `general-purpose`
  re-expands the surface to "Edit/Write/Bash + 7 loaded MCP tools + the full deferred
  MCP surface (incl. playwright)", which is precisely what phase-75.20 pinned away
  from for the read-only Q/A. So C is **cheap for the Researcher** (which already
  needs Write, and whose stage 2 *already runs on the uncapped built-in `Explore`* —
  `research-gate.js:792`) and **expensive for the Q/A** (it would undo a deliberate
  confinement). Incidental benefits of C: the `.md` body becomes a prompt string, so
  **deletions from `qa.md` take effect immediately** instead of waiting for a session
  restart. Incidental cost: agent type is part of the prompt-cache prefix key
  (source 4), so the change invalidates the shared prefix once.

**Recommendation.** For the **Researcher**: option A or C — remove `maxTurns: 40`, and
note that the same script already trusts an uncapped built-in type at stage 2, so this
is consistency, not a new risk. For the **Q/A**: option A — remove `maxTurns: 30` and
keep `agentType: 'qa'`, because the surface confinement is worth more than the cap and
the two are independent settings. In both cases keep the in-script retry (it is the
only retry that fires, R-2) and keep write-first durability (it is the only thing that
survives an exhausted retry). Whatever is chosen, the acceptance test must be a
**measured drop rate**, not a green run: `python3 scripts/qa/rail_drop_rate.py`, with
the standing caveat at `rail_drop_rate.py:36-42` that `logs` is empty on all dropped
runs, so a fix must be judged on the EXHAUSTED count and needs enough post-change runs
to separate from an 8–11% base rate.

**One caveat the step should carry:** F-8 means a fix that *depends* on `maxTurns`
being honoured is exposed to a control with a history of silent non-enforcement.
Removing the key (A) is immune to that; raising it (B) is not.

## Recency scan (2024-2026)

**Performed.** Queries: `...2026` frontier, `...2025` window, and year-less canonical
(table above). Findings **from within the last two years**, all of which post-date the
phase-59.1 decision this step revisits:

1. **BAGEN, arXiv:2606.00198v1, 29 May 2026** (source 11) — the newest and most
   consequential. Its late-alarm result (feasibility predicted >70% after 60% of budget
   spent) is a direct, current-literature argument against any prompt-level
   self-rationing remedy. Nothing older in the corpus says this.
2. **Issue #65500, v2.1.162, 2026** (source 6) — the only public description of this
   exact error, and **already stale** in two respects: nudge count (2 → 1) and
   catchability (R-7 shows `parallel()` containing the throw in 2.1.232).
3. **Issue #41143, v2.1.84, 2026** (source 7) — closed as not planned; establishes that
   `maxTurns` non-enforcement was a real, unfixed state within the last year.
4. **Hermes-agent #36239, session dated 2026-06-01** (source 9) — current practitioner
   state of the art for forcing a terminal action, with a measured long-context caveat.

**Supersession judgment:** the 2024-2025 canonical material (LangChain `max_iterations`,
LangGraph `recursion_limit`, generic loop-guardrail advice) is **complementary, not
superseded** — it establishes that hard iteration caps are a cost control everyone
ships. What the last-2-year window adds, and what changes this step's answer, is (a)
the evidence that agents cannot self-manage the tail of their own budget (BAGEN), and
(b) the runtime-behaviour facts in #65500/#41143 that are already version-drifting and
must be re-checked against the installed binary rather than cited.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **11**
- [x] 10+ unique URLs total (incl. snippet-only) — **19**
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set; the one truncation
      (`agent-sdk/typescript`) is disclosed in the snippet-only table and the gap it
      left was closed from the runtime instead
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions noted (#65500 vs R-7; #41143 vs the caller's measurement;
      BAGEN vs the "reserve a turn" framing; docs vs runtime on partial return)
- [x] All claims cited per-claim
- [ ] `coverage.dry` — not applicable (step is not audit-class); loop-until-dry not run
