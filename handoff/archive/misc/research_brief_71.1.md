# Research Brief — Step 71.1: Codify Workflow structured-output Q/A/Researcher launch

**Status:** COMPLETE — gate_passed=true (write-first; appended incrementally; envelope is authoritative)
**Researcher:** Layer-3 research gate (Workflow structured-output, run wf_7f99e5e1-1a7)
**Date:** 2026-07-17 (sources accessed 2026-07-17)
**Tier:** moderate (harness-protocol edit; grounding validation + de-risk)

## Objective

Validate the grounding for step 71.1: codify the Claude Code **Workflow** structured-output
path (`agent(prompt, {schema})` returns the validated object as the captured RETURN VALUE,
immune to the Agent-tool end-flush stall) as the FIRST-CLASS unattended launch for the
Layer-3 Q/A (and Researcher) role, with the Agent-tool subagent path retained as a documented
FALLBACK. Save a reusable `.claude/workflows/` script. Update per-step-protocol.md + qa.md +
researcher.md + CLAUDE.md. GUARDRAIL: Main must transcribe the returned verdict VERBATIM into
evaluator_critique.md. Preserve single-Q/A-per-step + file-based fresh-respawn cycle-2 +
exactly-3-agents.

## Search-query plan (3 variants per topic; year-less canonical included)

- Topic A — Claude Code Workflow tool: "claude code workflow tool 2026" / "claude code
  workflow agent structured output 2025" / "claude code workflows" (year-less)
- Topic B — Structured outputs GA: "anthropic structured outputs 2026" / "claude constrained
  decoding 2025" / "anthropic structured outputs" (year-less)
- Topic C — Multi-agent/harness design: "anthropic multi-agent research system" (year-less) /
  "harness design long-running apps 2026" / "anthropic subagent lifecycle 2025"
- Topic D — Subagent structured output stall / GitHub #20625: "claude code github issue 20625"
  / "claude code subagent structured output 2026" / "claude code subagent end flush stall"

---

## Sources read in full (WebFetch)

### S1 — Claude Code Dynamic Workflows docs (OFFICIAL, tier-2) — READ IN FULL
URL: https://code.claude.com/docs/en/workflows — accessed 2026-07-17
**All four required confirmations hold, quoted verbatim:**

(a) **agent(prompt,{schema}) returns the validated structured object.** The doc's
worked example:
> `const found = await agent('List every .ts file under src/routes/.', { schema: { type: 'object', required: ['files'], properties: { files: { type: 'array', items: { type: 'string' } } } } })`
> ... then uses `found.files` directly.
i.e. the return value IS the validated object. Search-corroborated: "the `agent()`
function spawns one subagent, and without options it returns the agent's final
text — the schema option ... forces the subagent to return validated structured
data." "`agent()` spawns one subagent and `pipeline()` runs one per item in a list."

(b) **Runtime tracks each agent result; resumable in-session.** Verbatim:
> "The runtime tracks each agent's result as the run progresses, which is what makes
> a run resumable within the same session."
> "If you stop a run, you can resume it: agents that already completed return their
> cached results, and the rest run live."
> CAVEAT (verbatim): "Resume works within the same Claude Code session. If you exit
> Claude Code while a workflow is running, the next session starts the workflow fresh."

(c) **Scripts saved under .claude/workflows/, named + re-runnable.** Verbatim:
> "`.claude/workflows/` in your project: shared with everyone who clones the repo"
> "The workflow runs as `/<name>` in future sessions from either location."
> Saved file "holds a `meta` block followed by a script body" — `meta = { name, description }`.
> Save via `/workflows` → select run → press `s`.

(d) **Model inheritance = session model unless overridden.** Verbatim:
> "Every agent in a workflow uses your session's model unless the script routes a
> stage to a different one or the `CLAUDE_CODE_SUBAGENT_MODEL` environment variable
> is set, which overrides both."

**Load-bearing extras for the design:**
- Version floor: "Dynamic workflows require Claude Code v2.1.154 or later" — local is
  v2.1.205 (per qa.md:29), so satisfied.
- Unattended launch works: permission table — "Bypass permissions, `claude -p`, Agent
  SDK | Never [prompted]. The run starts immediately." So a SAVED `/name` workflow runs
  headless. BUT the `ultracode` KEYWORD auto-trigger does NOT fire from "a prompt passed
  with `-p`, a scheduled task prompt, a webhook payload" (v2.1.210+). => Main must
  EXPLICITLY invoke the saved workflow; do not rely on keyword auto-fire in cron.
- Isolation: "No direct filesystem or shell access from the workflow itself — Agents
  read, write, and run commands. The script coordinates the agents." => the WORKFLOW
  SCRIPT cannot Write evaluator_critique.md; only Main (who receives the return value)
  can. This is exactly what forces the verbatim-transcription guardrail.
- Subagent permission: "The subagents the workflow spawns always run in `acceptEdits`
  mode and inherit your tool allowlist." => the Q/A prompt must self-enforce read-only
  (it cannot rely on permissionMode:plan from qa.md — see open question O1 below).
- Caps: "Up to 16 concurrent agents" / "1,000 agents total per run" — irrelevant for a
  single-Q/A workflow but confirms a 1-agent script is well within bounds.

## Snippet-only sources

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| alexop.dev/posts/claude-code-workflows-deterministic-orchestration | blog (tier-3) | community corroboration of Workflow-as-tool |
| mindstudio.ai/blog/claude-code-agentic-workflow-patterns | blog (tier-4) | pattern overview only |
| claudefast.st/blog/guide/development/dynamic-workflows | blog (tier-4) | secondary |
| medium.com/@danushidk507 workflows-in-agentic-ai | blog (tier-5) | low weight |
| docs.cloud.google.com/.../claude/structured-outputs | vendor doc | Bedrock/GCP variant, not Claude Code |
| kenhuangus.substack.com/p/chapter-15-structured-output | blog | community |

### S2 — "Get structured output from agents" (Claude Code Agent SDK, OFFICIAL tier-2) — READ IN FULL
URL: https://code.claude.com/docs/en/agent-sdk/structured-outputs — accessed 2026-07-17
This is the mechanism BEHIND the workflow `agent(prompt,{schema})` return value.
> "Define a JSON Schema for the structure you need, and the SDK validates the output
> against it, re-prompting on mismatch. If validation does not succeed within the retry
> limit, the result is an error instead of structured data."
> "the result message includes a `structured_output` field with validated data matching
> your schema." (subtype `success`)
**Failure modes the design MUST handle (why FALLBACK matters):**
> subtype `error_max_structured_output_retries` = "No valid output remained after
> multiple attempts (validation failures, or a model-fallback retraction with no
> successful retry)."
> "a model fallback can retract an already-completed output mid-stream, and if no retry
> replaces it the run ends with the same error."
=> The Workflow path is robust but NOT infallible: a schema-unsatisfiable / retracted
run yields an ERROR, not a verdict. Main must treat an errored workflow as "NO VERDICT
— do NOT auto-PASS" and fall back to the Agent-tool subagent path. This is the
anti-rubber-stamp tie-in (mirrors qa.md L287-292 "no auto-PASS path").
**Version note (matches local v2.1.205):**
> "A schema that isn't valid JSON Schema fails the run at startup with an error naming
> the problem. Before v2.1.205, an invalid schema was silently ignored and the agent
> returned unstructured text." + "The `format` keyword ... is accepted as an annotation
> and isn't enforced ... Before v2.1.205, any schema containing `format` was treated as
> invalid."
=> VERDICT_SCHEMA should use only basic types + `enum` + `required` (no `format`) to be
maximally portable; at 2.1.205 an invalid schema fails LOUDLY (good — no silent
unstructured fallback).

### S3 — Structured Outputs (Claude Platform, OFFICIAL tier-2) — READ IN FULL
URL: https://platform.claude.com/docs/en/build-with-claude/structured-outputs — accessed 2026-07-17
Answers task source #2 (why the return value is trustworthy):
> "Structured outputs guarantee schema-compliant responses through constrained decoding:
> Always valid: No more JSON.parse() errors; Type safe: Guaranteed field types and
> required fields; Reliable: No retries needed for schema violations."
> vs just asking for JSON: "Without structured outputs, Claude can generate malformed
> JSON ... Parsing errors ... Missing required fields ... Inconsistent data types."
Caveat (bounded): "While structured outputs guarantee schema compliance in most cases,
there are scenarios where the output may not match your schema: Refusals
(stop_reason: refusal) ... Token limit reached (stop_reason: max_tokens)."
**GA + model support (our model is covered):** "Structured outputs are generally
available on the Claude API for Claude Fable 5, Claude Mythos 5, **Claude Opus 4.8**,
... Claude Sonnet 5, ..." — Opus 4.8 (Q/A + Researcher pin) is GA. This is the
grounding for "the returned object is schema-guaranteed, hence trustworthy to transcribe."

### S4 — "How we built our multi-agent research system" (Anthropic Engineering, OFFICIAL tier-2) — READ IN FULL
URL: https://www.anthropic.com/engineering/multi-agent-research-system — accessed 2026-07-17
Confirms the doer/judge + file-based + resume principles SURVIVE the Workflow migration:
> "a multi-agent architecture with an orchestrator-worker pattern, where a lead agent
> coordinates the process while delegating to specialized subagents." (separation of duties)
> "Each subagent needs an objective, an output format, guidance on the tools and sources
> to use, and clear task boundaries." (matches the workflow prompt shape)
> "The LeadResearcher synthesizes these results and decides whether more research is
> needed—if so, it can create additional subagents." (cycle-2 fresh-respawn basis)
> "we built systems that can resume from where the agent was when the errors occurred."
> "agents can spawn fresh subagents with clean contexts while maintaining continuity
> through careful handoffs." (file-based fresh-respawn)
Note: the blog's "Direct subagent outputs can bypass the main coordinator" is an
OPTION, not a mandate; pyfinagent deliberately keeps Main in the loop (Main transcribes
the verdict) for auditability — consistent with the guardrail.

### S5 — GitHub issue #20625 (VERIFIED via WebSearch; page fetch pending) — topic-D anchor
"[FEATURE] Support structured output schemas for Claude Code subagents that directly use
the structured output API" · anthropics/claude-code#20625. Opened **2026-01-24**, later
**closed as not planned**. Problem it names: "The CLI can enforce structured output only
at the top level via flags ... Subagents defined in Markdown or settings.json ... cannot
declare a structured-output contract." => Confirms the design's premise: there is NO
native Agent-tool subagent structured-output path (the qa.md subagent CANNOT declare a
schema in frontmatter), so the Workflow `agent(prompt,{schema})` path is the ONLY
first-class way to get a schema-guaranteed verdict as a return value. Justifies choosing
Workflow over an Agent-tool structured path. (Page fetch to confirm state verbatim: next.)

---

## Internal findings (grep/read on HEAD b0c29946)

**`.claude/workflows/` does NOT exist yet** (confirmed: `ls` → "No such file or
directory"). New dir to be created by GENERATE. No existing reusable workflow scripts
to mirror; the shape must come from the official doc's `meta`+body pattern (S1).

**`.claude/agents/qa.md` (15239 B, 321 lines):**
- Frontmatter `tools: Read, Bash, Glob, Grep, SendMessage`; `model: opus`; `effort: max`;
  `permissionMode: plan`; `maxTurns: 30`; skills: code-review-trading-domain (L4-51).
  The Fable comments L7-44 are HISTORY (reverted to opus per 67.4) — leave untouched.
- Section header "# Q/A Agent (merged qa-evaluator + harness-verifier)" (L53).
- "You run ONCE per cycle (not in a parallel pair anymore). The 3-agent MAS is: Main
  (orchestrator) + Researcher + Q/A." (L69-71) — the single-Q/A + exactly-3 invariant.
- Output format single-JSON block (L213-251): fields ok/verdict/reason/violated_criteria/
  violation_details/certified_fallback/checks_run. THIS is the canonical verdict shape
  the VERDICT_SCHEMA must mirror.
- "Never second-opinion-shop -- but fresh-respawn on changed evidence is the documented
  pattern." (L293-299) — cycle-2 pattern, must be preserved.
- **No Workflow mention anywhere** and **no verbatim-transcription language** — both are
  NET-NEW additions. There is NO "how Q/A is launched/invoked" section today; the doc
  is written as the subagent's own system prompt (assumes Agent-tool spawn). Best
  insertion point for a "## Launch (Workflow-first-class; Agent-tool fallback)" section:
  after L71 (the 3-agent-MAS paragraph) or just before "## Verification order" (L73).

**`.claude/agents/researcher.md` (15283 B, 322 lines):**
- Frontmatter `tools: Read, Grep, Glob, Bash, WebSearch, WebFetch, SendMessage`;
  `model: opus`; `effort: max`; `maxTurns: 40`; `permissionMode: plan` (L1-47).
- "## Write-first (non-negotiable)" (L84-92) — the brief file created in first tool call.
- "## Output JSON envelope (ALWAYS EMIT)" (L287-304) — tier/external_sources_read_in_full/
  snippet_only_sources/urls_collected/recency_scan_performed/internal_files_inspected/
  report_md/gate_passed. THIS is the researcher return schema the workflow can carry.
- No Workflow mention. Insertion point for a Launch note: after the "## When invoked"
  block (L75-82) or adjacent to the envelope section (L287).

**`docs/runbooks/per-step-protocol.md`:** EVALUATE section + how Q/A is spawned today —
_pending read (next batch)_.

**`CLAUDE.md`:** Single-Q/A rule + cycle-2 flow + "Workflow Q/A when subagents stall"
memory ref — _pending grep (next batch)_.

**Auto-memory basis (feedback_workflow_qa_when_subagents_stall, via MEMORY.md):**
"Agent-tool harness subagents stalled 6x on end-flush 2026-07-11 (intermittent, NOT
model-specific — Opus too); run the qa role via Workflow structured-output (verdict =
captured return value, immune to file-write stall + retry); Main persists it from the
return. Watch transcript mtime for hangs." => the empirical justification is on file.

---

### S6 — GitHub issue #20625 page (OFFICIAL repo, tier-2) — READ IN FULL
URL: https://github.com/anthropics/claude-code/issues/20625 — accessed 2026-07-17
Title (verbatim): "[FEATURE] Support structured output schemas for Claude Code
subagents that directly use the structured output API". **State: CLOSED as not
planned.** Opened 2026-01-24. Request: add a `structured_output` field to
`.claude/agents/*.md` frontmatter / settings.json so subagents can declare a JSON-schema
contract "without having to reimplement a separate coordinator using the low-level SDK."
=> CONFIRMS the design's premise verbatim: the Agent-tool subagent path has NO native
structured-output contract (feature requested + declined). The Workflow `agent(prompt,
{schema})` IS the sanctioned path to a schema-guaranteed verdict-as-return-value. This is
the direct justification for choosing Workflow over an Agent-tool structured path.

### S7 — "A harness for every task: dynamic workflows in Claude Code" (Anthropic/Claude blog, OFFICIAL tier-2) — READ IN FULL
URL: https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code — accessed 2026-07-17
> "Claude can now write its own harness on the fly, custom-built for the task at hand."
> "For each spawned agent, run a separate spawned agent to adversarially verify its
> output against a rubric or criteria." (the separate-evaluator pattern — but see
> Rider-trap R1: the *self-revising* variant is what we must NOT adopt)
> "If a workflow is interrupted ... resuming the session will allow the workflow to pick
> up where it left off." (resumable in-session — corroborates S1)
> "You can save workflows by pressing 's' ... check these into `~/.claude/workflows` or
> distribute them via a skill."
Frames the Workflow as a harness — directly on-point for codifying it as our launch.

---

## Recency scan (last 2 years)

Searched the 2025-2026 window explicitly ("Claude Code dynamic workflows subagent
structured output verdict evaluator 2025") plus year-less canonical Anthropic sources
(harness-design, multi-agent-research-system have no year in the query). Findings:

- **The Workflow feature itself is 2026-current** (requires v2.1.154+; local v2.1.205).
  There is NO pre-2024 prior art to supersede — this is genuinely new scaffolding, so
  the year-less canonical for the *mechanism* is the Workflows doc (S1) + the structured-
  outputs GA doc (S3). The canonical *principles* (doer/judge, file-based, resume) are
  the year-less Anthropic engineering posts (S4 2024, S5 2025) — still authoritative.
- **New-in-window finding that MATTERS (rider-trap source):** multiple 2026 write-ups
  describe a "grading layer ... each subagent's result is graded in its own context by a
  separate evaluator, and a failure sends the subagent back to revise until the result
  meets the rubric" (recency search + S7). This *self-revising grader* is exactly
  Rider-trap R1 — it collapses the doer/judge separation. FLAGGED below; do NOT adopt.
- **#20625 (2026-01-24, closed not-planned)** is the freshest datapoint confirming the
  Agent-tool path has no native schema contract — supersedes any assumption that a
  frontmatter `structured_output:` field might exist. It does not.
- No 2025-2026 source contradicts the single-Q/A / verbatim / fresh-respawn doctrine;
  they reinforce it.

---

## Insertion points & current wording (exact, HEAD b0c29946)

### 1. `.claude/agents/qa.md` (321 lines) — NET-NEW launch section
- **No "how Q/A is launched" section exists**; no "Workflow" mention; no verbatim-
  transcription language. All net-new.
- **Insert a `## Launch (how this agent is invoked)` section AFTER L71** ("There is no
  separate harness-verifier.") and **BEFORE L73** (`## Verification order`). Content to
  add: (a) Workflow-first — Main runs the saved `.claude/workflows/qa-verdict` script,
  which calls `agent(<Q/A task context>, {schema: VERDICT_SCHEMA})`; the return value IS
  the verdict object, immune to the Agent-tool end-flush stall; (b) Agent-tool spawn =
  documented FALLBACK; (c) the existing **Output format block L213-251 IS the canonical
  `VERDICT_SCHEMA`** — the workflow schema mirrors it; (d) read-only self-enforcement:
  workflow subagents run in `acceptEdits` + inherit the tool allowlist (NOT
  `permissionMode: plan`), so the "NEVER Edit/Write" constraint (L273-276) holds by
  INSTRUCTION, not by permission mode — restate it; (e) Main transcribes the return
  VERBATIM.
- Frontmatter `model: opus` + `effort: max` (L5, L45) govern the **Agent-tool** path;
  the **Workflow** path inherits the SESSION model/effort — see open question O2.
- Leave the Fable history comments (L7-44) untouched (already reverted to opus per 67.4).

### 2. `.claude/agents/researcher.md` (322 lines) — NET-NEW launch note
- No "Workflow" mention. **Insert a `## Launch` note AFTER the `## When invoked` block
  (ends L82) and BEFORE `## Write-first (non-negotiable)` (L84).** Content: Workflow-first
  option — saved `.claude/workflows/research-gate` calls `agent(<researcher context>,
  {schema: ENVELOPE_SCHEMA})`; the envelope is the return value; **write-first STILL holds
  because workflow agents CAN write files** ("Agents read, write, and run commands" — S1),
  so the incrementally-written brief on disk remains the deliverable and the envelope is
  the return; Agent-tool spawn = fallback. The **envelope block L293-304 IS the
  `ENVELOPE_SCHEMA`**.

### 3. `docs/runbooks/per-step-protocol.md` (298 lines)
- **§4 EVALUATE, insert AFTER L113** ("...Spawn the Q/A agent ONCE (no more pair spawn).")
  a `**Launch: Workflow-first-class (Agent-tool fallback).**` paragraph + the verbatim-
  transcription requirement (Main copies the returned verdict object into
  `evaluator_critique.md` with no editorial edits; an errored/absent return = NO VERDICT,
  never auto-PASS, fall back to Agent-tool).
- **§1 RESEARCH (L57-78):** add the researcher Workflow launch option (parallel wording).
- **"Subagent runtime semantics" (L264-279):** add a "Workflow path" bullet — isolated
  env, intermediate results in script variables, **resumable in-session only** (a fresh
  Claude Code session restarts the workflow), return-value immune to end-flush stall;
  unattended `-p`/scheduled runs must EXPLICITLY invoke the saved `/qa-verdict` — the
  `ultracode` keyword does NOT auto-fire from `-p`/scheduled/webhook (v2.1.210+, S1).
- **References (L288-298):** add the Workflows doc + structured-outputs doc.

### 4. `CLAUDE.md` (352 lines)
- **Primary home: the `### Single-Q/A rule (was: dual-evaluator)` section (L229).** Insert
  a `**Launch — Workflow-first-class (Agent-tool fallback):**` paragraph immediately AFTER
  the `Returns {ok, verdict, ...}` block (L240) and BEFORE the cycle-2 block (L242).
  Content: prefer the saved `.claude/workflows/qa-verdict` (verdict = captured RETURN
  VALUE, immune to the 6x end-flush stall per `feedback_workflow_qa_when_subagents_stall`);
  Agent-tool spawn = FALLBACK; **Main MUST transcribe the returned verdict VERBATIM** into
  `evaluator_critique.md`; an errored/absent workflow result (`error_max_structured_output
  _retries`, refusal, max_tokens) = NO VERDICT → never auto-PASS → Agent-tool fallback;
  workflow inherits the SESSION model, so route the Q/A stage to `opus` explicitly (O2).
- The existing "Historical note on `SendMessage`" (L272-279) is the precedent-style block;
  the new note complements it (why Workflow beats both SendMessage and plain Agent-tool).
- OPTIONAL (mirror existing bullet style): one Critical-Rules bullet near the MAS-harness
  bullet pointing to the Single-Q/A section. Minimal-edit path can skip it; the Single-Q/A
  section is the required home.

---

## Recommended `.claude/workflows/` script shape + VERDICT_SCHEMA

`.claude/workflows/` does NOT exist yet — GENERATE creates it. Mirror the official
`meta` + top-level-`await` body shape (S1). A single-agent EVALUATE workflow:

```javascript
export const meta = {
  name: 'qa-verdict',
  description: 'Run the pyfinagent Layer-3 Q/A evaluator once and return a schema-validated verdict.',
}
// args = { step_id, criteria, handoff_dir }  (S1: saved workflows read a global `args`)
const VERDICT_SCHEMA = {
  type: 'object',
  required: ['ok','verdict','reason','violated_criteria','violation_details','certified_fallback','checks_run'],
  properties: {
    ok: { type: 'boolean' },
    verdict: { type: 'string', enum: ['PASS','CONDITIONAL','FAIL'] },
    reason: { type: 'string' },
    violated_criteria: { type: 'array', items: { type: 'string' } },
    violation_details: { type: 'array', items: { type: 'object',
      properties: {
        violation_type: { type: 'string', enum: ['Missing_Assumption','Invalid_Precondition',
          'Unjustified_Inference','Circular_Reasoning','Contradiction','Overgeneralization','Threshold_Not_Met'] },
        action: { type: 'string' }, state: { type: 'string' }, constraint: { type: 'string' } } } },
    certified_fallback: { type: 'boolean' },
    checks_run: { type: 'array', items: { type: 'string' } },
  },
}
const verdict = await agent(
  `You are the pyfinagent Layer-3 Q/A evaluator. Follow .claude/agents/qa.md EXACTLY
   (deterministic checks first, read the handoff files under ${args.handoff_dir}, never
   Edit/Write, no auto-PASS). Evaluate masterplan step ${args.step_id}. Return the verdict object.`,
  { schema: VERDICT_SCHEMA, model: 'opus', label: `qa-${args.step_id}` },
)
return verdict   // Main transcribes this VERBATIM into evaluator_critique.md
```

Notes / rationale:
- **Schema mirrors qa.md L215-251 exactly** — single source of truth stays qa.md.
- **No `format` keyword** (S2: `format` unenforced/portability-safe; at v2.1.205 an
  invalid schema fails loudly anyway).
- **`enum:[PASS,CONDITIONAL,FAIL]`** — deliberately excludes the qa.md `verdict:null`
  loop-prevention path (L287-292): that path is a Stop-hook/Agent-tool concern and does
  not apply to a Workflow launch. If fidelity is wanted, make `verdict` type
  `['string','null']`; otherwise document the exclusion. (Minor schema decision for GENERATE.)
- **`model:'opus'` per-agent** guards against session-model drift (O2).
- A parallel `research-gate` workflow carries the researcher `ENVELOPE_SCHEMA`
  (researcher.md L293-304); the researcher agent still WRITES the brief file, returns the
  envelope.

### Open questions for GENERATE (resolve, don't paper over)
- **O1 — does `agent()` target a named `.claude/agents/*.md` type?** The Workflows doc
  shows `agent(prompt, {schema,label,model})`; it does NOT document a `subagent_type`/
  `agents` option that binds qa.md's frontmatter (tools/permissionMode/skills). Safest:
  the prompt POINTS to qa.md and the agent inherits the session tool allowlist. If the
  Agent-SDK reference (`/en/agent-sdk/typescript` "Workflow tool entry") exposes an agent-
  type binding, prefer it so the code-review-trading-domain skill + read-only tools bind
  automatically. VERIFY during GENERATE.
- **O2 — effort inheritance.** Workflow agents inherit the SESSION model/effort (S1); the
  docs show per-agent `model` routing but NOT per-agent `effort`. Main runs `xhigh` (Opus
  4.8). qa.md pins `effort: max` for the Agent-tool path. If `max` is strictly required on
  the Workflow path, either confirm a per-agent effort option exists or accept `xhigh`
  (session) on the Workflow path and keep the Agent-tool fallback as the `max` guarantee.
  Document whichever is true — do not claim `max` if the mechanism only gives session effort.

---

## Drift from 71.0 design

**No material drift.** Masterplan step 71.1 + `design_harness_mas_71.md §71.1` + this
task brief agree on: Workflow-first + Agent-tool fallback; reusable `.claude/workflows/`
script; one `VERDICT_SCHEMA`/`ENVELOPE_SCHEMA`; verbatim-transcription guardrail;
preserve single-Q/A + file-based fresh-respawn cycle-2 + exactly-3-agents. The design
EXPLICITLY names the same rider-traps to reject (R1/R4/R11) — matches the task.
Two design nuances worth carrying into the contract (not drift, refinements):
1. The design says "Pin `model: opus` (model-inheritance nuance — inherits Main's model
   unless overridden)" → concretely, route the Q/A stage to `opus` in the agent() call
   (O2). 
2. `design §Binding constraints` + step verification note #4: qa.md/researcher.md edits
   take effect at NEXT session start (roster snapshot) AND require a Peder-review note in
   `harness_log.md` (separation of duties) before a later step depends on the new wording.
   The Workflow-path launch does NOT dodge the snapshot caveat — a saved workflow script
   is read at invocation, but the qa.md/researcher.md CONTENT the script points to is what
   the agent reads at runtime, so content edits are live immediately for the Workflow path
   (unlike the Agent-tool roster snapshot). Worth stating explicitly.

---

## Rider-trap watch (must NOT sneak into GENERATE)

- **R1 — `/goal`-to-PASS auto-fix loop / self-revising grader. FLAGGED, do NOT adopt.**
  The 2026 "grading layer that sends the subagent back to revise until it meets the
  rubric" (S7 + recency search) collapses the doer/judge separation that S5 calls the
  "strong lever" against LLM self-leniency. The Q/A workflow must return a verdict and
  STOP; Main (a separate agent) owns the fix and spawns a FRESH Q/A on changed evidence
  (cycle-2). The workflow must NOT loop fix→re-grade→PASS internally.
- **R4 — model-swap on stall. FLAGGED, do NOT adopt.** The end-flush stall is
  model-agnostic (auto-memory: "NOT model-specific — Opus too"). The fix is the
  return-value path, not a cheaper/different model. `settings.json fallbackModel` is
  OVERLOAD-only (per-step-protocol L277-279). Also: routing Q/A to a non-Opus model would
  violate the effort/model policy. Keep Q/A on `opus`.
- **R11 — Monitor/transcript-mtime watchdog. FLAGGED, do NOT adopt.** Adding a Monitor-
  based mtime poller contradicts the standing do-not-poll rule (per-step-protocol L266-267
  "Do not busy-wait or poll transcripts") and adds new scaffolding the Workflow return-
  value path makes unnecessary. The memory's "watch transcript mtime" is a MANUAL Main
  check on a stalled Agent-tool spawn, not an automated component.

Also guard the **anti-rubber-stamp** line: an errored/empty workflow return
(`error_max_structured_output_retries` / refusal / max_tokens per S2) is NOT a PASS — it
is NO VERDICT. Main must fall back to the Agent-tool path, never auto-PASS (mirrors qa.md
L287-292 "an evaluator must have no auto-PASS path").

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 13,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

`gate_passed: true` — 7 sources (all OFFICIAL tier-1/tier-2 Anthropic/Claude Code docs +
the GitHub issue) read in full via WebFetch (floor is 5); recency scan performed and
reported; >10 URLs collected; every hard-blocker satisfied; internal anchors captured
with file:line for all 4 GENERATE targets.
