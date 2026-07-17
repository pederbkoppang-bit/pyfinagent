# Contract — step 71.1 (codify Workflow structured-output as the first-class Q/A + Researcher launch)

**Phase:** phase-71 | **Step:** 71.1 | **Priority:** P1 | harness_required: true | depends_on: 71.0 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** harness-infrastructure (docs + a reusable .claude/workflows/ script).
live_check: none (no UI; no live-loop behavior change). $0, local-only.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0), run wf_7f99e5e1-1a7. Envelope: **gate_passed=true**,
tier=moderate, **7 external sources read in full** (>=5 floor), 13 snippet-only, 20 URLs, recency scan performed,
6 internal files inspected. Brief: `research_brief_71.1.md`. Grounding HOLDS: (S1) Claude Code Workflow docs
confirm `agent(prompt,{schema})` returns the validated object as the captured RETURN VALUE, the runtime tracks
each result (resumable in-session), scripts save under `.claude/workflows/` with a `meta{name,description}` block,
and workflow agents inherit the SESSION model unless routed; (S3) structured-outputs GA on Opus 4.8 via
constrained decoding is WHY the return value is schema-guaranteed; (S4/S5) doer/judge separation + file-based
fresh-respawn + resume-not-restart still support single-Q/A + verbatim + exactly-3-agents; GitHub #20625
CONFIRMED closed-as-not-planned — Agent-tool subagents have NO native structured-output contract, so the Workflow
path is the sanctioned schema-guaranteed launch. `.claude/workflows/` does not exist yet. No material drift.

## Hypothesis / plan

Codify the Workflow structured-output path (already used 6+ times this session, $0 on the Opus Max rail,
stall-immune) as the FIRST-CLASS unattended launch for the Layer-3 Q/A (and Researcher) role, with the Agent-tool
subagent path retained as a documented FALLBACK. Concretely:

1. **Create `.claude/workflows/qa-verdict.js`** — a reusable, parameterized (`args`) single-agent Workflow script
   that runs the Q/A role and RETURNS the canonical qa.md verdict schema as the captured return value (immune to
   the Agent-tool end-flush stall). Design decisions from the research:
   - `agentType: 'general-purpose'` + the prompt instructs the agent to **`Read .claude/agents/qa.md` in full and
     follow it** as the single source of truth. This makes any `qa.md` edit **live-from-disk immediately** on the
     Workflow path (no roster snapshot) — the snapshot caveat binds only the Agent-tool `qa` type.
   - `model: 'opus'` set **explicitly** on the `agent()` stage (the workflow otherwise inherits the session model;
     routing off Opus would violate the effort/model policy — rider-trap R4).
   - `effort: 'max'`. VERDICT_SCHEMA mirrors qa.md's Output-format block (L215-251): `ok, verdict,
     violated_criteria, violation_details[{violation_type,action,state,constraint}], certified_fallback,
     checks_run` (+ a free-text `notes`). No `format` keyword (that's the Messages-API structured-outputs field,
     not the Workflow `schema` param).
   - **Read-only self-enforce:** the prompt forbids Edit/Write to production files (Bash only for verification).
   - **NO auto-PASS on an errored/empty return** (error_max_structured_output_retries / refusal / max_tokens = NO
     VERDICT → Main falls back to the Agent-tool path; never PASS). Mirrors qa.md L287-292.
2. **Document the Workflow-first-class path** in `docs/runbooks/per-step-protocol.md` §4 EVALUATE + the
   Subagent-runtime-semantics section, `.claude/agents/qa.md` (new `## Launch` section between the intro and
   Verification order), `.claude/agents/researcher.md` (a `## Launch` note), and `CLAUDE.md` Single-Q/A section —
   each stating: Workflow = primary unattended launch; Agent-tool = fallback; **Main transcribes the returned
   verdict VERBATIM into `evaluator_critique.md`** (no editorial edits) so the no-self-eval guarantee stays
   airtight; single-Q/A-per-step preserved; file-based fresh-respawn cycle-2 preserved; harness stays exactly 3.
3. **Rider-traps NOT adopted:** R1 (self-revising grader / goal-to-PASS auto-fix loop — the Q/A returns a verdict
   and STOPS; Main owns the fix + spawns a FRESH Q/A on changed evidence), R4 (model-swap-on-stall — the stall is
   model-agnostic; keep Q/A on opus), R11 (Monitor/transcript-mtime watchdog — contradicts do-not-poll), and
   auto-PASS-on-errored-return.

## Immutable success criteria (verbatim from masterplan.json 71.1)

1. A saved, reusable .claude/workflows/ script runs the Q/A (and optionally Researcher) role and returns a
   structured verdict; a dry-run or a real step shows the verdict captured as the workflow return value (not
   dependent on a file-write flush)
2. per-step-protocol.md + qa.md + CLAUDE.md document the Workflow structured-output path as the first-class
   unattended launch, with the Agent-tool path explicitly retained as fallback and the verbatim-transcription
   guardrail spelled out
3. The single-Q/A-per-step rule and the no-second-opinion-shopping / file-based fresh-respawn cycle-2 pattern are
   preserved unchanged; harness stays exactly 3 agents
4. NOTE the roster-snapshot + separation-of-duties caveat: qa.md/researcher.md edits take effect at NEXT session
   start; the harness_log requests review before a step depends on the new wording

Verification command (immutable):
`bash -c 'ls .claude/workflows/ 2>/dev/null | grep -Eqi "qa|eval|verdict" && grep -Eqi "workflow|structured.?output" .claude/agents/qa.md docs/runbooks/per-step-protocol.md'`

## Plan
2 (this contract). 3. GENERATE: write `.claude/workflows/qa-verdict.js`; edit qa.md, researcher.md,
per-step-protocol.md, CLAUDE.md; **prove the script runs** (invoke via Workflow with real args → structured
verdict captured as the return value). 4. experiment_results.md. 5. Q/A (fresh, via the NEW script — dogfood).
6. LOG (incl. the separation-of-duties + verify_qa_roster_live.sh note). 7. FLIP.

## Boundaries (binding)
$0; local-only; no production/live-loop behavior change (harness-infra + docs only); exactly-3-agents at L3
(no fourth agent; adversarial checks stay WITHIN Q/A); no self-eval (fresh independent Q/A; verdict transcribed
VERBATIM); single-Q/A-per-step + file-based fresh-respawn cycle-2 PRESERVED unchanged; the 4 rider-traps
(R1/R4/R11 + auto-PASS-on-error) NOT introduced; historical_macro FROZEN. **Separation of duties / roster
snapshot:** this step edits `.claude/agents/qa.md` + `researcher.md` → harness_log requests Peder review before a
LATER step depends on the new wording, and notes `scripts/qa/verify_qa_roster_live.sh` must confirm the roster
next session. The Workflow launch reads qa.md from disk at runtime (live); only the Agent-tool `qa` type snapshots
at session start. The 71.1 Q/A itself is INDEPENDENT (fresh general-purpose instance evaluating the artifacts;
it does not run under a tampered evaluator prompt).

## References
research_brief_71.1.md; design_harness_mas_71.md §71.1; harness_proposals.json (#1/#2/#3/#10);
Claude Code Workflow docs; structured-outputs GA; multi-agent-research + harness-design; GitHub #20625.
