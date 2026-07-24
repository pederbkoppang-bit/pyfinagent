# Contract — Step 75.20.1 (P2: subagent loader injects Write+Edit past the qa allowlist)

Date: 2026-07-24 | Cycle: 155 | Executor: MAIN-on-Fable (opus-tagged step, tiering precedent) | Gates: opus/max via Workflow | Claude Code 2.1.218

## Research-gate summary (gate PASSED — Workflow wf_9f6c22cf-24c, tier=complex)

`handoff/current/research_brief_75.20.1.md` — 6 external sources read in full, 17
snippet-only, 23 URLs, recency scan performed, 8 internal files, gate_passed=true.
Load-bearing findings (every one re-derived from primary text, not asserted):

1. **The injection is DOCUMENTED, intended upstream behavior — not a bug.** qa.md:25
   `memory: project` triggers it; sub-agents doc verbatim: "Read, Write, and Edit tools
   are automatically enabled so the subagent can manage its memory files." Q/A's memory
   is real + load-bearing (.claude/agent-memory/qa/ = MEMORY.md + 6 curated files), so
   the tools allowlist CANNOT drop Write/Edit without killing Q/A memory curation.
2. **Upstream-wait is dead**: issues #57507 / #57118 / #40140 all closed
   "not planned"/stale; no changelog entry reverts the memory auto-enable.
3. **qa.md:27 `permissionMode: plan` is INERT** (parent defaultMode=bypassPermissions
   takes precedence per sub-agents doc; corroborated by #57118).
4. **DECIDING FACT — enforcement is a PreToolUse hook**: hooks doc L623-628 documents
   `agent_id` + `agent_type` in the PreToolUse common input "when the hook fires inside
   a subagent call"; for custom subagents `agent_type` = the frontmatter `name` (= "qa").
   PreToolUse hooks run before the permission prompt and block even under
   bypassPermissions (permissions doc L411/L419/L58; empirically corroborated: our
   pre-tool-use-danger.sh fires today under bypassPermissions). `Agent(name)` permission
   rules gate SPAWN only — no settings construct can deny a tool to a specific acting
   subagent. So the hook is the ONLY provable per-acting-agent block.
5. **Honest limitation** (permissions doc L272): Write/Edit hooks do NOT stop Bash
   subprocess writes; Q/A has Bash → the Main-side post-verdict git-status backstop
   (work item d) is the covering control for that path, not the hook.
6. **The 7a Glob/Grep "drop" is a self-report artifact** (both are on the sub-agents
   doc's background-KEEP list L336) → the probe must measure EXECUTION, not self-report.

## Hypothesis

A path-aware PreToolUse hook (deny Write|Edit when `agent_type=="qa"` unless the target
is under `.claude/agent-memory/qa/`) converts Q/A's read-only-on-file-contents property
from prose to enforcement, without breaking Q/A memory curation or Main's own tools;
the git-status cleanliness rule codified in per-step-protocol §4 covers the Bash-write
path the hook structurally cannot.

## Immutable success criteria (verbatim from .claude/masterplan.json 75.20.1)

1. "A re-runnable probe artifact (script or Workflow) measures the qa agent's RUNTIME tool surface and its output is recorded verbatim; the Write/Edit-injection claim is either reproduced (with version) or found fixed (with version), never asserted from memory"
2. "The injection source is identified with evidence (upstream docs/changelog citation or a local config culprit), or an upstream issue is filed/referenced with its URL recorded"
3. "If a hook-level enforcement ships, it is proven to block a qa-type Write/Edit attempt in a live probe AND proven to NOT block Main's own Write/Edit (both recorded); if no enforcement ships, the step records WHY with the evidence"
4. "per-step-protocol.md section 4 EVALUATE carries the post-verdict git-status cleanliness rule (verdict inadmissible on evaluator-authored tree changes), and the rule text names the fresh-Q/A consequence"
5. "Each new behavioral guard is mutation-tested with evidence recorded in the step's live_check, including at least one fixture/stub mutation"

Immutable verification command (verbatim):
`.venv/bin/python -m pytest backend/tests/test_phase_75_20_1_qa_write_injection.py -q`

## Plan

1. **Hook** `.claude/hooks/qa-write-guard.sh` (NEW): PreToolUse, registered in
   `.claude/settings.json` with matcher `Write|Edit`. Reads stdin JSON. ALWAYS appends
   `{ts, agent_type, tool_name, file_path}` to `handoff/logs/qa_write_guard.log`
   (the log IS the empirical field-shape confirmation on 2.1.218 — the researcher's
   log-only recommendation, fused in). BLOCKS (exit 2 + stderr reason) iff
   `agent_type == "qa"` AND `tool_name in {Write, Edit}` AND resolved `file_path` is
   NOT under `.claude/agent-memory/qa/`. FAIL-OPEN (exit 0) on missing fields, parse
   errors, or absent jq/python — consistent with project hook discipline. ASCII only.
2. **Probe** `.claude/workflows/probe-qa-tool-surface.js` (NEW, re-runnable): spawns
   `agentType:'qa'` (haiku — trivial probe, tiering) instructing it to ATTEMPT each
   tool against scratch paths — Write(outside memory), Edit(outside), Write(inside
   memory dir is NOT probed against the real memory — use report-only), Glob, Grep,
   Read — and return per-tool {attempted, succeeded, error_verbatim} JSON. Criterion 1's
   instrument; behavioral, not self-disclosure.
3. **Runbook** `docs/runbooks/per-step-protocol.md` §4 EVALUATE, new subsection
   inserted after L135 (before "Q/A runs deterministic-first" at L137): POST-VERDICT
   CLEANLINESS — after every Q/A return, Main runs `git status --short`; any tree
   change not authored by Main renders the verdict INADMISSIBLE and a FRESH Q/A is
   required after reverting/reconciling; names the Bash-write hook gap (L272) as why
   this backstop exists even with the qa-write-guard live.
4. **Tests** `backend/tests/test_phase_75_20_1_qa_write_injection.py` (NEW): drive the
   REAL hook script via subprocess with JSON fixtures — (t1) qa+Write+outside → exit 2;
   (t2) qa+Edit+outside → exit 2; (t3) qa+Write+inside .claude/agent-memory/qa/ →
   exit 0; (t4) absent agent_type (Main shape) → exit 0; (t5) agent_type=other → exit 0;
   (t6) malformed JSON → exit 0 (fail-open); (t7) settings.json registers the hook
   under PreToolUse matcher Write|Edit; (t8) per-step-protocol §4 carries the rule
   (content-bearing asserts: git status --short + INADMISSIBLE + fresh Q/A + Bash-gap
   mention); (t9) qa.md:25 still has `memory: project` (pin-with-message: the hook's
   raison d'etre — if memory is ever dropped, re-evaluate the hook, see brief).
5. **Mutation matrix (Main, after GENERATE)**: M1 remove the agent_type=="qa"
   condition (hook blocks Main too) → t4 red; M2 remove the memory-dir exception →
   t3 red; M3 exit 2 → exit 0 (block becomes allow) → t1 red; M4 unregister hook from
   settings.json → t7 red; M5 **FIXTURE mutation**: rename the fixture field
   `agent_type` → `agentType` in t1's payload → t1 red (fail-open on the wrong shape
   proves the fixture speaks the real schema); M6 **STUB mutation**: neuter the
   test's `_run_hook` helper to skip the subprocess and return exit 2 constant →
   t3/t4 red.
6. **Live probes (Main)**: (P1) Main's own Edit still works post-registration
   (recorded); (P2) run the probe workflow → verbatim per-tool results; expect the
   qa Write attempt BLOCKED if hooks hot-load on 2.1.218, else record
   "hook binds at next session start" honestly and the live block-proof becomes
   restart-gated (75.20 roster precedent) — MEASURE, don't assume, either way the
   guard-log shows whether the hook fired and with which fields; (P3) the guard-log
   lines quoted verbatim (field-shape confirmation on 2.1.218).
7. Q/A via qa-verdict Workflow → log Cycle 155 → **COMMIT LOCAL, HOLD flip+push for
   operator review** (75.20 precedent: this step edits .claude/settings.json hook
   machinery — separation-of-duties review requested in harness_log).

## Boundaries

- Harness/permission config only: `.claude/hooks/`, `.claude/settings.json` (hook
  registration ONLY — no permission rules touched), `.claude/workflows/`,
  `docs/runbooks/per-step-protocol.md`, `backend/tests/`. NO product code.
- qa.md UNTOUCHED (memory stays; the hook is the fix, not an allowlist change).
- The hook must be fail-open and scoped to agent_type=="qa" — it must be structurally
  unable to block Main (t4 + M1 prove it).
- No upstream issue FILED (injection is documented behavior, not a bug — criterion 2's
  "or referenced" arm: #57507/#57118/#40140 URLs recorded in brief + results).

## References

- Brief: handoff/current/research_brief_75.20.1.md (doc-verbatim quotes + issue URLs)
- Live reproduction this session: handoff/archive/misc/live_check_75.20.md §7a
  (haiku qa probe self-disclosed Write+Edit present on the live roster)
- Precedents: 75.20 (held-commit review flow), existing pre-tool-use-danger.sh
  (PreToolUse under bypassPermissions, empirical)
