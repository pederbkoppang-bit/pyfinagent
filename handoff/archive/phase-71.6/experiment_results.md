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

## Cycle 2 — deterministic weekly REPORT-ONLY scheduler BUILT (resolves the C1 CONDITIONAL)

**Why Cycle-1 parked C1, and why that was too conservative.** Cycle-1 treated ALL scheduling as the
agentic-resumption category and deferred activation to the operator. But criterion 1 demands BOTH "scheduled
REPORT-ONLY on a weekly cadence (local)" AND "honors the background-agent unauthorized-action memory." A scheduled
*agentic* audit VIOLATES that memory — so the only criterion-satisfying implementation is a scheduled **deterministic
(non-agentic) report-only** job. The 71.6 research brief already RANKED this as safe option 2: "a DETERMINISTIC Python
report writer on the proven register_meta_evolution_cron weekly APScheduler pattern — greps the harness invariants
(3-agent roster, dead-driver absence...)" — "zero-agency," no resumption risk. Cycle-2 builds it.

**Files (4):**
- `backend/harness_self_audit_report.py` (NEW): `register_harness_self_audit_cron(scheduler)` (weekly Sun 03:00 ET
  cron, `replace_existing=True`, fail-open — mirrors `backend/meta_evolution/cron.py`) +
  `run_harness_self_audit_report()` — DETERMINISTIC, report-only: greps roster integrity (exactly Researcher+Q/A; a
  re-split guard for explore.md/harness-verifier.md), saved workflows present, 5-file protocol presence+age,
  deep-audit staleness (harness_proposals.json age > 14d → nudge). Writes ONE file
  `handoff/self_audit/<date>-harness-health.md`. NO LLM / NO agent / NO subprocess / NO git / NO network / NO BQ / NO
  trade / NO risk touch / NO masterplan flip. Each sub-check fail-open.
- `backend/config/settings.py`: `harness_self_audit_report_enabled: bool = Field(True, ...)`. Default **True** —
  criterion 1 needs the weekly cadence ACTUALLY scheduled (not flag-conditional), and this is report-only
  observability CATEGORICALLY OUTSIDE the do-no-harm set (kill-switch/stops/caps/DSR/PBO byte-untouched). Activates on
  the next backend restart; set False to disable.
- `backend/main.py`: registers the job on the live `AsyncIOScheduler` inside the existing paper-trading scheduler
  block (after `_register_cron_scheduler("main", scheduler)`), gated by the flag, wrapped fail-open so it can never
  break startup.
- `backend/tests/test_phase_71_6_self_audit_cron.py` (NEW, 8 tests): weekly-cron shape, fail-open register,
  report-written+status, determinism, re-split guard → ATTENTION, stale/missing deep-audit → ATTENTION, and a
  report-ONLY assertion (the writer touches ONLY handoff/self_audit/).

**Verification (verbatim):**
- `ast.parse` 4 files → `ast OK`.
- IMMUTABLE cmd `ls .claude/workflows | grep -Eqi "audit|self|stress" && grep -Eqi "envelope|summary|file path|return" researcher.md qa.md` → **exit 0 (PASS)**.
- `uvx ruff check` (qa.md §1a lint gate) → **All checks passed** (exit 0).
- `pytest test_phase_71_6_self_audit_cron.py` → **8 passed**. Regression
  `test_phase_71_2/71_3/71_4/71_6/59_1` → **48 passed** (the effort assertion untouched by 71.6).
- **DOGFOOD** against the REAL repo: `run_harness_self_audit_report()` → status **OK**, wrote
  `handoff/self_audit/2026-07-17-harness-health.md`, roster.ok=True (no re-split), workflows present, deep_audit
  age_days=1, attention=[], errors=[].
- `get_settings().harness_self_audit_report_enabled` → **True**.

**Scope / do-no-harm.** $0; deterministic (no LLM/agents); report-only observability. Live book untouched; no risk
threshold moved; historical_macro FROZEN. The DEEP *agentic* audit stays MANUAL / operator-scheduled (that recurring
agentic run is the resumption-risk category — unchanged). What is now live-on-restart is only the benign deterministic
weekly HEALTH report. No operator token required to satisfy criterion 1; if the operator prefers DARK, flip
`harness_self_audit_report_enabled=False`. Cycle-1 agent-file edits (researcher.md/qa.md) still carry the
separation-of-duties + `verify_qa_roster_live.sh` note.

**Artifact shape:** `handoff/self_audit/<date>-harness-health.md`. Return dict: `{status, report_path, roster,
workflows, five_file, deep_audit, attention[], errors[]}`.
