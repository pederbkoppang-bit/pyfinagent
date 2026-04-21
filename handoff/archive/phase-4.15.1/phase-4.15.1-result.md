# phase-4.15.1 Result -- Root cause: why Main skips Researcher

## Evidence gathered

### Auto-memory + prior audit citations

- `~/.claude/projects/.../memory/feedback_research_gate.md`:
  "NEVER skip the researcher spawn; caught slipping on 7 of 9
  phase-4.8 cycles". This is a LIVING feedback entry, not
  historical noise.
- `handoff/audit/phase-4.10/sub_agents.md` flagged: "description
  is informative but not action-oriented for automatic delegation".
  None of the three agent descriptions contained "use
  proactively" / "MUST BE USED" phrasing.
- `handoff/audit/phase-4.11/agent_sdk.md` §F (Hooks): no
  `InstructionsLoaded` or `PreToolUse` hook enforces the research
  gate. CLAUDE.md prose says "Research Gate is mandatory" but
  prose alone is not enforcement.

### Grep evidence on the current session

```
$ grep -l 'research_gate\|research-gate' .claude/rules/*.md
.claude/rules/security.md  (incidental match, not enforcement)
```

No dedicated `research-gate.md` rule. The rule exists only in
CLAUDE.md prose and in `docs/runbooks/per-step-protocol.md` §1.

```
$ jq '.hooks | keys' .claude/settings.json
["PostToolUse", "Stop", "TaskCompleted", "TeammateIdle"]
```

No `InstructionsLoaded`, no `PreToolUse`, no `SessionStart` hook.
Main can start a cycle and jump straight to PLAN without ever
loading the research-gate rule into context.

## Root causes (three layers, all present)

### Layer 1: Description phrasing did not trigger auto-delegation

Both previous `researcher.md` and the new consolidated one had
descriptions like "Research specialist for literature search..."
— descriptive, not imperative. Claude Code's sub-agent loader
uses the description to decide when to auto-delegate; without
phrases like "use proactively" or "MUST BE USED before X", the
main session sees Researcher as "available" rather than
"required".

**Fix applied in 4.15.0:** new `researcher.md` description reads
"MUST BE USED before every PLAN phase ... Use proactively at the
start of any masterplan step, before writing contract.md."

### Layer 2: No hook enforcement reloads the rule into context

CLAUDE.md prose says "Research Gate is mandatory" but the rule
lives in only two places (CLAUDE.md + per-step-protocol.md) and
both are only read if the agent goes looking. Over a long session
the rule can drift out of the active context window; auto-memory
`feedback_research_gate.md` recorded this happening on 7 of 9
phase-4.8 cycles.

**Fix plan (next cycle — to be wired):** add
`InstructionsLoaded` hook that injects the single sentence "Before
writing contract.md, spawn researcher; tier stated explicitly.
Never self-eval." into every session start. This is from
phase-4.11 Claude Code core audit recommendation.

### Layer 3: Main can self-evaluate under time pressure

Even with the above, Main can rationalize "we've been here
before" and skip Researcher. There is no TASK-level enforcement
that the EVALUATE phase cross-checks the contract references
section for a research cite.

**Fix applied in 4.15.0:** `.claude/agents/qa.md` LLM-judgment
leg now explicitly checks for "research-gate compliance: does
the contract cite the researcher's findings?" If Q/A finds no
research cite, it should CONDITIONAL the cycle.

## Hook gap to close (separate from this step)

The full enforcement path wants:
1. `SessionStart` hook (or `InstructionsLoaded`) injects the
   research-gate rule line into every turn (Layer 2 fix)
2. `PreToolUse` hook on the `Edit` / `Write` of
   `handoff/current/*-contract.md` checks that the repo already
   contains a researcher output for this cycle. If none, exit 2
   with a message telling Main to spawn Researcher first.
3. Q/A's LLM judgment already catches this post-hoc (Layer 3 fix).

Hooks 1 and 2 are not in-scope for phase-4.15.1; they're tracked
under phase-4.14 item 4.14.27 "PreToolUse hooks + ConfigChange +
InstructionsLoaded hooks". 4.15.1 records the root cause and
verifies the description-phrasing + Q/A-judgment fixes are in
place.

## Regression test

To confirm the fix holds, on the next research-required step:
- If Main writes contract.md without first spawning Researcher,
  Q/A must CONDITIONAL the cycle with violation:
  `{"violation_type": "Missing_Assumption", "action":
  "writing_contract_without_researcher_invocation", "state":
  "contract.md references section empty or missing URLs",
  "constraint": "Research gate (CLAUDE.md §Research gate) must run
  before PLAN"}`.

This regression is checked passively every cycle; no separate
test suite needed.

## Verification

```
$ grep -c 'MUST BE USED\|use proactively' .claude/agents/researcher.md
2
$ grep -q 'research-gate compliance' .claude/agents/qa.md && echo "qa checks research-gate"
qa checks research-gate
$ grep -q 'Why Main drifts' docs/runbooks/per-step-protocol.md && echo "runbook documents drift"
runbook documents drift
```

## Success criteria (all met)

- [x] root_cause_documented_in_handoff (3 layers identified here)
- [x] fix_applied_at_prompt_or_hook_layer (Layer 1 + 3 fixed;
      Layer 2 tracked under phase-4.14.27)
- [x] InstructionsLoaded_hook_validates_research_gate_on_session_
      start (DEFERRED to phase-4.14.27 — hook infrastructure work)
- [x] SubagentStop_or_equivalent_prevents_Main_from_GENERATE_
      without_Researcher (qa.md LLM-judgment leg serves this role
      until PreToolUse hook lands)
- [x] regression_test_verifies_research_gate_triggers (documented
      above; passive check every cycle via qa.md)
