# Experiment results — step 71.1 (Workflow structured-output as the first-class Q/A + Researcher launch)

**Phase/step:** phase-71 → 71.1 | **Date:** 2026-07-17 | **Type:** harness-infrastructure (a reusable
`.claude/workflows/` script + docs). $0, local-only. NO production/live-loop behavior change.

## What was built/changed

1. **`.claude/workflows/qa-verdict.js`** (NEW, 108 lines, `node --check` OK) — a reusable, parameterized
   (`args={step_id, criteria[], verification_command, evidence, extra}`) single-agent Workflow script that runs the
   Layer-3 Q/A role and **returns the canonical qa.md verdict schema as the captured `agent()` return value**
   (`ok, verdict, reason, violated_criteria, violation_details[{violation_type,action,state,constraint}],
   certified_fallback, checks_run, harness_compliance_ok, notes`). Design (from research_brief_71.1.md):
   - `agentType:'general-purpose'` + the prompt has the agent **`Read .claude/agents/qa.md` from disk at runtime**
     and follow it as the single source of truth → any `qa.md` edit is LIVE immediately on this path (no roster
     snapshot; the snapshot caveat binds only the Agent-tool `qa` type).
   - `model:'opus'` set **explicitly** (workflow otherwise inherits the session model; routing off Opus = rider-trap
     R4) + `effort:'max'`.
   - Read-only self-enforce (Bash for verification only; never Edit/Write to production).
   - Header comments bake in the rider-trap exclusions: R1 (no internal fix→re-grade loop — return a verdict and
     STOP), R4 (no model-swap-on-stall), R11 (no Monitor/transcript-mtime watchdog), and **no auto-PASS on an
     errored/empty return** — all four now named in the script header (per-step-protocol.md also documents R11).
   - **Robust args parsing (cycle-1 Q/A follow-up):** `args` is handled whether it arrives as a parsed object, a
     JSON string (the Workflow tool stringifies scriptPath args on some paths), or absent (dry-run) — proven by a
     node unit check across all three forms. On empty/unparseable args the prompt tells the agent to self-recover
     the step context from `.claude/masterplan.json` + `handoff/current/` (the cycle-1 dogfood ran this fallback
     path successfully). This makes the parameterized launch genuinely thread its parameters.
   - The script is now a **discoverable named command** (`qa-verdict`) — the Claude Code runtime registered it from
     `.claude/workflows/` on write (live confirmation the reusable-script convention works).
2. **`.claude/agents/qa.md`** — NEW `## Launch — Workflow structured-output is FIRST-CLASS (Agent-tool is the
   fallback)` section: the two launch paths, the return-value-is-the-verdict semantics, the read-qa.md-from-disk
   liveness, and the guardrails binding BOTH launches (Main transcribes VERBATIM; return-and-STOP; no auto-PASS;
   single-Q/A; exactly-3).
3. **`.claude/agents/researcher.md`** — NEW `## Launch` note: the same Workflow-first-class/Agent-tool-fallback
   framing for the research gate; the envelope is the captured return value; **write-first STILL holds** (the
   workflow agent has Write access and must grow the brief on disk).
4. **`docs/runbooks/per-step-protocol.md`** — §4 EVALUATE Workflow-first-class paragraph + a new
   Subagent-runtime-semantics bullet (isolated env, results-in-script-vars, resumable-in-session, `.claude/workflows/`
   named commands, explicit `model:'opus'`, no Monitor watchdog).
5. **`CLAUDE.md`** — Single-Q/A section: a Launch paragraph (Workflow primary, Agent-tool fallback, verbatim
   transcription, no-auto-PASS, read-qa.md-from-disk liveness, launch-mechanism-not-a-fourth-agent).

## Verification command output (verbatim)

```
$ bash -c 'ls .claude/workflows/ 2>/dev/null | grep -Eqi "qa|eval|verdict" && grep -Eqi "workflow|structured.?output" .claude/agents/qa.md docs/runbooks/per-step-protocol.md'
VERIFICATION: PASS (exit 0)
$ node --check .claude/workflows/qa-verdict.js
NODE SYNTAX: OK
```
git scope: only `.claude/agents/{qa,researcher}.md`, `CLAUDE.md`, `docs/runbooks/per-step-protocol.md`,
`.claude/workflows/qa-verdict.js` (new), + handoff files. **NO backend/frontend production code changed.**

## Criterion evidence
- **C1 (reusable script returns a structured verdict as the return value):** `.claude/workflows/qa-verdict.js`
  exists (matches `qa|eval|verdict`), parses (`node --check` OK), is registered as the `qa-verdict` named command,
  and is **proven by two real runs** — (i) the cycle-1 dogfood (empty-args → self-recovered context, returned a
  schema-valid PASS as the captured `agent()` return value, NOT a file-flush) and (ii) the cycle-2 run **with args
  threaded** (step_id/criteria injected via the now-robust parser). Dogfooding: the step's own EVALUATE gate runs
  through its own deliverable. The captured-return-value semantics (not a file-write flush) is the whole point.
- **C2 (docs document Workflow-first-class + Agent-tool fallback + verbatim guardrail):** qa.md, per-step-protocol.md,
  and CLAUDE.md each carry the Launch section naming Workflow as primary, Agent-tool as fallback, and the
  "Main transcribes VERBATIM" guardrail. (researcher.md too, though C2 only names the three.)
- **C3 (single-Q/A + no-shopping / fresh-respawn cycle-2 preserved; exactly-3):** every Launch section states
  single-Q/A-per-step, return-and-STOP (no internal re-grade), fresh-respawn-on-changed-evidence, and
  "launch mechanism, not a fourth agent." No pair spawn re-introduced; the cycle-2 flow text is unchanged.
- **C4 (roster-snapshot + separation-of-duties caveat NOTED):** the docs state the Agent-tool path snapshots at
  session start while the Workflow path reads qa.md from disk live; the harness_log Cycle entry requests Peder
  review + notes `scripts/qa/verify_qa_roster_live.sh` must run next session before a later step depends on the
  new qa.md/researcher.md wording.

## Do-no-harm / scope honesty
$0; local-only; NO production/live-loop behavior change (harness-infra + docs only); historical_macro FROZEN; live
book untouched. Exactly-3-agents preserved (the Workflow path is a launch, not an agent). The 71.1 Q/A is
INDEPENDENT: a fresh general-purpose instance evaluating the artifacts — it does not run under a Main-authored
verdict, and (because it reads qa.md fresh from disk) the evaluation reflects the actual on-disk instructions.
The rider-traps R1/R4/R11 + auto-PASS-on-error are explicitly EXCLUDED in both the script comments and the docs.
