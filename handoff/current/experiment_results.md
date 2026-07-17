# Experiment results — step 71.6 (report-only self-audit workflow + context-hygiene + dead-driver keep)

**Phase/step:** phase-71 → 71.6 | **Date:** 2026-07-17 | **Type:** harness-infra (a saved report-only workflow) +
agent-file context-hygiene + a dead-driver decision. $0, local-only, NO production/live-loop change; historical_macro
FROZEN; live book untouched.

## What was changed

### `.claude/workflows/harness-self-audit.js` (NEW — STRUCTURALLY report-only) [criterion 1, saved re-runnable]
A re-runnable self-audit workflow (mirrors `qa-verdict.js`): fan-out finder agents over 4 harness+MAS dimensions
(harness-protocol / layer2-mas / layer4-meta / capabilities-drift) → adversarial verify (pipeline, no barrier) →
**RETURNS ranked confirmed findings** (the register shape of `harness_proposals.json`). **Enforcement = TOOL-
RESTRICTION, not a prompt:** every auditor is `agentType:'Explore'` (READ-ONLY — no Edit/Write/Agent); the workflow
SCRIPT has no fs/shell/git access; nothing writes files, commits, pushes, or flips the masterplan. Registered as the
discoverable `harness-self-audit` command (confirmed live). Lands the `ls .claude/workflows/ | grep audit|self|stress`
check. Backstopped by the 62.0 PreToolUse guard (blocks `git push`/`launchctl`) as defense-in-depth.

### `.claude/agents/researcher.md` + `.claude/agents/qa.md` (context-hygiene) [criterion 2]
- researcher.md: retired the full-brief `report_md` envelope field → a `<=200-word` `summary` + `brief_path` +
  an explicit "return the compact envelope, NOT the full brief through Main's context; Main reads it from
  brief_path" instruction (Anthropic "lightweight references").
- qa.md: added a "Context hygiene" clause — the return IS a **compact verdict envelope** (verdict + one-sentence
  reason summary + violated_criteria); the full critique prose lives at the `evaluator_critique.md` **file path**,
  never pasted through Main's context. (Adds envelope/summary/file-path to qa.md, matching the grep for BOTH files.)

### Dead-driver: KEEP-WITH-REASON [criterion 3] — no file edit
`scripts/mas_harness/{cycle_prompt.md,run_cycle.sh}` are DEAD as a driver (no live plist; absent from `launchctl
list`; the mas-harness label cannot fire) — the safety intent (neutralize the dangerous `claude -p
--dangerously-skip-permissions` driver) is already MET. But they are **LIVE test fixtures**: `revert_hygiene_drill.py`
reads both + executes `run_cycle.sh` (a passing dirty-tree-refusal drill), `test_phase_47_9` asserts the model pin
from `run_cycle.sh`, `smoke_test_4_17_11` references it, and `cron_dashboard_api.py::_LAUNCHD_JOBS` lists the label.
A naive `rm` REDs 3 consumers — the 71.0 design's "just delete" is a RIDER-TRAP the research caught. → **KEPT with
reason** (documented here + in the harness_log); the files are NOT edited (editing risks breaking the live drills).
`run_harness.py::_default_spawn_researcher` (the live spawn path) NOT touched. Harness stays exactly 3 agents.

### Schedule ACTIVATION — operator-gated (`schedule_needs_operator=true`) [criterion 1, "scheduled" part]
Claude Code has NO native recurring trigger; scheduling is external. Per the research + the
background-agent-resumption-risk memory ("review-only prompts are NOT enforcement"), a recurring *agentic* weekly run
is precisely the resumption-risk category → **ACTIVATION is the operator's call.** The safe mechanisms are documented
(the workflow header + here): (a) an external cron/launchd invoking the saved report-only workflow, or (b) a
deterministic-Python report writer registered on the `register_meta_evolution_cron` weekly APScheduler pattern (no
LLM = no agency). The saved workflow is structurally report-only and re-runnable NOW (manually or once activated).

## Verification command output (verbatim)
```
$ bash -c 'ls .claude/workflows/ 2>/dev/null | grep -Eqi "audit|self|stress" && grep -Eqi "envelope|summary|file path|return" .claude/agents/researcher.md .claude/agents/qa.md'
VERIFICATION: PASS (exit 0)
$ node --check .claude/workflows/harness-self-audit.js   -> OK
```
git scope: `.claude/workflows/harness-self-audit.js` (new), `.claude/agents/{researcher,qa}.md`, handoff. NO
backend/frontend/production code changed; the dead-driver files + `run_harness.py` are UNTOUCHED.

## Criterion evidence
- **C1** — saved, re-runnable `.claude/workflows/harness-self-audit.js`, STRUCTURALLY report-only (read-only Explore
  auditors; script has no fs/git; returns findings, never applies). The weekly SCHEDULE is documented + its
  safe mechanisms specified; **activation is operator-gated** (`schedule_needs_operator=true`; honors the
  background-agent memory — enforcement by tool-restriction, not a "report-only" prompt).
- **C2** — researcher.md + qa.md now both instruct the subagent to return a COMPACT envelope (summary + verdict +
  file path) rather than the full brief/critique through Main's context.
- **C3** — the dead self-evaluating driver is EXPLICITLY KEPT WITH A REASON (neutralized driver + live drill
  fixtures; deletion is a rider-trap); harness stays exactly 3 agents; `run_harness.py` untouched.

## Do-no-harm / scope honesty
$0; local-only; NO production/live-loop change. The audit workflow is structurally report-only (tool-restriction, not
prompt). The dead driver is KEPT (deleting it would RED 3 live drills — a real regression the research caught). The
weekly-schedule ACTIVATION is transparently flagged operator-gated (the recurring agentic run is the resumption-risk
category the operator flagged) — I built + documented the safe mechanism but did NOT create a recurring autonomous
run. historical_macro FROZEN; live book untouched; harness stays 3 agents. Agent-file edits → separation-of-duties +
verify_qa_roster_live.sh note in the harness_log.
