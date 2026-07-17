# Contract — step 71.6 (report-only self-audit workflow + subagent context-hygiene + prune/keep the dead driver)

**Phase:** phase-71 | **Step:** 71.6 | **Priority:** P3 | harness_required: true | depends_on: 71.0 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** harness-infra (a saved report-only workflow) + agent-file context-hygiene
+ a dead-driver decision. $0, local-only, NO production/live-loop change; historical_macro FROZEN; live book untouched.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0), run wf_8aec9e9d-b8b. Envelope: **gate_passed=true**,
tier=complex, **6 external sources read in full**, 8 snippet-only, 14 URLs, recency scan, 12 internal files.
Brief: `research_brief_71.6.md`. Grounded in Anthropic harness-design (stress-test doctrine) + multi-agent-research
(lightweight references) + Microsoft-Security-2026 read/write tool separation + SkillScope (least-privilege).
**preserves_three_agents=true.** KEY findings:
- **Enforcement-safe schedule = STRUCTURAL tool-restriction, NOT a "report-only" prompt** (the resumption-risk memory
  is correct: `claude -p`/bypass never prompts). A checked-in `.claude/workflows/harness-self-audit.js` whose fan-out
  auditors get READ-ONLY tools (no Edit/Write/git/launchctl) structurally cannot push/flip; the 62.0 PreToolUse guard
  (`pre-tool-use-danger.sh`) blocks `git push`/`launchctl` as defense-in-depth. Claude Code has NO native recurring
  trigger — scheduling is external (cron/launchd), so **`schedule_needs_operator=true`**: the recurring *agentic*
  weekly run is an operator infra decision.
- **Dead driver:** `scripts/mas_harness/{cycle_prompt.md,run_cycle.sh}` are DEAD as a driver (no live plist; absent
  from `launchctl list`) BUT are LIVE test fixtures — 3 consumers (`revert_hygiene_drill.py`, `test_phase_47_9`,
  `smoke_test_4_17_11`) + `cron_dashboard_api.py::_LAUNCHD_JOBS`. Naive `rm` REDs them → the 71.0 "just delete" is a
  RIDER-TRAP. → **KEEP-WITH-REASON** (criterion 3 explicitly allows this).
- Context-hygiene: `researcher.md` already clears the grep richly; strengthen the envelope (add a ≤200-word summary +
  "don't return the full `report_md` through Main's context"). `qa.md` matches only via "return" — add one
  compact-envelope sentence. Both edit agent files → separation-of-duties + roster note.

## Plan

### A. `.claude/workflows/harness-self-audit.js` (NEW — re-runnable, structurally REPORT-ONLY) [criterion 1a]
Mirrors `qa-verdict.js`. Fan-out finder agents over the harness+MAS surface (`agentType:'Explore'` = READ-ONLY: no
Edit/Write/Agent) → a synthesis that RETURNS ranked findings (the register shape of `harness_proposals.json`). The
SCRIPT has no fs/shell/git access; the agents have no Write/Edit; nothing flips masterplan or pushes. The RETURN
VALUE is the findings report — the caller persists it to `handoff/self_audit/<date>-harness-audit.md`. Lands the
`ls .claude/workflows/ | grep audit|self|stress` check.

### B. Context-hygiene (agent files → separation-of-duties note) [criterion 2]
- `researcher.md`: strengthen the envelope — add a `summary` (≤200-word) instruction + "return `brief_path` + the
  ≤200-word summary; do NOT return the full brief text through Main's context — Main reads it from `brief_path`."
- `qa.md`: add a compact-envelope sentence — "Return a COMPACT verdict envelope (verdict + one-sentence reason
  summary + violated_criteria); the full critique prose lives at the `evaluator_critique.md` file path — never
  paste the full critique through Main's context." (Adds envelope/summary/compact/file-path to qa.md.)

### C. Dead-driver: KEEP-WITH-REASON [criterion 3]
Add a documented KEEP note (in the harness_log + a header comment on `cycle_prompt.md`) — the driver is neutralized
(no live plist; safety intent met) AND the files are LIVE fixtures for passing safety drills (`revert_hygiene_drill.py`
executes `run_cycle.sh`); deleting them REDs 3 consumers. Do NOT touch `run_harness.py::_default_spawn_researcher`
(the live spawn path). Harness stays exactly 3 agents.

### D. Schedule ACTIVATION — operator-gated (`schedule_needs_operator=true`)
Document (in the workflow header + the harness_log + experiment_results) how to schedule the report-only audit
weekly (external cron/launchd invoking the saved workflow, OR a deterministic-Python report writer on the
`register_meta_evolution_cron` APScheduler pattern). Because a recurring *agentic* run is precisely the
background-agent-resumption risk the operator flagged, ACTIVATION is the operator's call — flagged as the token owed.

## Immutable success criteria (verbatim from masterplan.json 71.6)

1. The harness/MAS self-audit is a saved, re-runnable workflow scheduled REPORT-ONLY on a weekly cadence (local);
   it never auto-applies changes -- it writes a findings report the operator reviews (honors the background-agent
   unauthorized-action memory)
2. researcher.md/qa.md instruct the subagent to return a compact envelope (summary + verdict + handoff path) rather
   than the full brief/critique through Main's context
3. Any dormant/dead self-evaluating driver path confirmed unused by the stress-test is pruned (or explicitly kept
   with a reason); harness stays exactly 3 agents

Verification command (immutable):
`bash -c 'ls .claude/workflows/ 2>/dev/null | grep -Eqi "audit|self|stress" && grep -Eqi "envelope|summary|file path|return" .claude/agents/researcher.md .claude/agents/qa.md'`

## Boundaries (binding)
$0; local-only; NO production/live-loop change (harness-infra + agent-file docs + a dead-driver KEEP note). The saved
audit workflow is STRUCTURALLY report-only (read-only auditor tools; script has no fs/git; nothing flips/pushes) —
enforcement by tool-restriction, not prompting. **Schedule ACTIVATION is operator-gated** (`schedule_needs_operator=
true`; the recurring agentic run is the resumption-risk category — I build + document the safe mechanism but do NOT
create a recurring autonomous run). Dead driver KEPT-with-reason (deletion is a rider-trap that REDs 3 live fixtures).
Do NOT touch `run_harness.py::_default_spawn_researcher`. Harness stays exactly 3 agents (a report is a routine, not a
4th agent). historical_macro FROZEN. Agent-file edits → separation-of-duties + verify_qa_roster_live.sh note.

## References
research_brief_71.6.md; design_harness_mas_71.md §71.6 (note: the "just delete" was a rider-trap); harness_proposals.json;
qa-verdict.js (workflow shape); backend/meta_evolution/cron.py (safe APScheduler pattern); Anthropic harness-design.
