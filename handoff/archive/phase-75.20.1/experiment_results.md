# Experiment results — Step 75.20.1 (loader injects Write+Edit past the qa allowlist)

Date: 2026-07-24 | Cycle: 155 | Execution: MAIN-on-Fable GENERATE (opus-tagged step) | Claude Code 2.1.218

## What was built

### 1. Root cause — IDENTIFIED, DOCUMENTED UPSTREAM BEHAVIOR (criterion 2)

qa.md:25 `memory: project` is the injection source. Sub-agents doc verbatim: "Read,
Write, and Edit tools are automatically enabled so the subagent can manage its memory
files." Q/A's memory is real + load-bearing (`.claude/agent-memory/qa/` = MEMORY.md +
6 curated files), so the tools allowlist CANNOT exclude Write/Edit without destroying
memory curation. NOT a bug → no new upstream issue filed; the three adjacent issues are
referenced with URLs in the brief (#57507 memory/allowlist, #57118 permissionMode
inert, #40140 agent_type-in-hooks feature request — all closed not-planned/stale; the
capability #40140 asked for is NOW documented in the hooks doc, which is what makes the
enforcement below possible). qa.md:27 `permissionMode: plan` proven INERT (parent
defaultMode=bypassPermissions precedence).

### 2. Enforcement — `.claude/hooks/qa-write-guard.sh` (NEW) + settings registration (criterion 3)

Path-aware PreToolUse hook, matcher `Write|Edit` (second `hooks.PreToolUse` entry in
`.claude/settings.json`):

- BLOCKS (exit 2) iff `agent_type == "qa"` AND `tool_name in {Write, Edit}` AND the
  normpath-normalized `file_path` is NOT under `.claude/agent-memory/qa/` (normpath
  collapses `../` so a traversal path that merely CONTAINS the memory segment cannot
  escape — t3b).
- ALLOWS everything else; FAIL-OPEN on missing fields / malformed JSON / python absent
  (a broken guard must never brick the session — same discipline as
  pre-tool-use-danger.sh).
- ALWAYS appends `{ts, agent_type, tool_name, file_path}` to
  `handoff/logs/qa_write_guard.log` — the log doubles as empirical confirmation of the
  hook-input fields 2.1.218 actually populates (the researcher's log-only leg, fused
  into the enforcing hook).

Smoke matrix (verbatim, all six branches):

```
qa Write outside       -> rc=2 + "qa-write-guard: BLOCKED -- the qa evaluator is read-only on file contents"
qa Write inside memory -> rc=0
traversal escape       -> rc=2
Main (no agent_type)   -> rc=0
other agent_type       -> rc=0
malformed payload      -> rc=0 (fail-open)
```

KNOWN GAP, disclosed and covered (permissions doc L272): Write/Edit hooks do not
intercept Bash subprocess writes; Q/A holds Bash for verification commands → the §4
POST-VERDICT CLEANLINESS rule is the covering control; the hook comments and the
runbook text both say so explicitly.

### 3. Re-runnable behavioral probe — `.claude/workflows/probe-qa-tool-surface.js` (NEW) (criterion 1)

Spawns `agentType:'qa'` (haiku) that ATTEMPTS Write/Edit/Glob/Grep/Read against scratch
paths and returns per-tool `{attempted, succeeded, outcome_verbatim}` — execution, not
self-disclosure. Ran live this session (wf_90a3a61d-4a4): verbatim results in
live_check_75.20.1.md §3 — headline: the qa Write attempt was BLOCKED by the new hook
LIVE, and Glob/Grep are genuinely absent from the qa runtime surface at execution level
(upgrading 75.20 §7a's self-report from artifact-suspect to measured fact on the
Workflow path).

### 4. Runbook — per-step-protocol.md §4 POST-VERDICT CLEANLINESS (criterion 4)

New mandatory subsection (inserted before "Q/A runs deterministic-first"): after every
Q/A return Main runs `git status --short`; any tree change not authored by Main renders
the verdict INADMISSIBLE → revert/reconcile → fresh Q/A (never patch or partially trust
the verdict). Names the Bash-write hook gap as why the rule coexists with the hook, and
the memory-dir as the sole exempt evaluator write path. (This session already practiced
the rule on both 76.9 verdicts before codifying it.)

### 5. Tests — backend/tests/test_phase_75_20_1_qa_write_injection.py (NEW, 11 tests)

t1/t2 block outside (Write/Edit), t3 memory allowed, t3b traversal still blocked, t4
Main allowed, t5 other-agent allowed, t6/t6b fail-open, t7 registration + matcher
scoped exactly `Write|Edit`, t8 runbook content asserts (whitespace-normalized after a
real wrap-split red run — fixed the strong way, assertions kept), t9 the
`memory: project` root-cause pin with re-evaluation instructions.

## Verification (verbatim)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_20_1_qa_write_injection.py -q
11 passed in 0.49s

$ { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | xargs uvx ruff check --select F821,F401,F811
All checks passed!
lint exit=0

$ bash -n .claude/hooks/qa-write-guard.sh   # exit 0
$ python3 -c "import json; json.load(open('.claude/settings.json'))"   # valid JSON
```

## Files changed

- `.claude/hooks/qa-write-guard.sh` (NEW)
- `.claude/settings.json` (+1 PreToolUse entry, matcher Write|Edit — restrictive only,
  no permission rules touched)
- `.claude/workflows/probe-qa-tool-surface.js` (NEW)
- `docs/runbooks/per-step-protocol.md` (§4 POST-VERDICT CLEANLINESS subsection)
- `backend/tests/test_phase_75_20_1_qa_write_injection.py` (NEW, 11 tests)
- qa.md UNTOUCHED (boundary held — memory stays; the hook is the fix)

## Live probes + mutation matrix

Verbatim in `handoff/current/live_check_75.20.1.md`: probe output (block message
verbatim), guard-log lines incl. the runtime-populated `agent_type:"qa"` at 11:59:29Z,
hook-liveness finding (binds immediately for new subagent spawns; main-session binding
lands at next session start — Main's writes provably unimpeded either way), and the
6/6 mutation matrix with pre/post SHA-256 identity.

## Post-verdict correction (N1, Q/A cycle-1 catch — disclosure, not a rewrite)

The "FAIL-OPEN on missing fields" line above is imprecise in one sub-case the evaluator
measured: a payload with agent_type=="qa" and tool_name Write/Edit but a MISSING or
empty file_path BLOCKS (empty string normalizes to "." which is outside the memory
dir) — i.e. that sub-case fails CLOSED, the safe direction. Fail-open remains true for
malformed JSON, absent agent_type, absent tool_name, and python failures. Behavior
unchanged; prose corrected here.
