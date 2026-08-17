# live_check -- step 86.71 (captured 2026-08-17, operator-attended session)

## 1. Criterion-1 re-derivation (command + population rule + output)

Population rule: every parseable `*/workflows/wf_*.json` run record under the
project's `~/.claude/projects` tree; step_id recovered from args (object, or
json.loads of string args, or a regex salvage of the head); records with no
recoverable step_id counted separately, never silently dropped. A run is a
REPEAT iff an earlier-timestamped record carries the same step_id.

```
runs with step_id: 481  (no step_id recoverable: 99)
repeats (same step_id seen before, any role): 320  = 66.5%
  qa: total=390 repeats=302 (77.4%)
  researcher: total=90 repeats=18 (20.0%)
max runs one step: 36.8 -> {'qa': 9}
```

DISAGREEMENT with the audit_basis (58.4%): reported, not adopted -- the corpus
grew from 513 to 580+ records between the two measurements.

## 2. Criterion-2: no-runtime-caller, positive-controlled

```
control (the pattern must hit a file KNOWN to reference the module):
$ grep -rln "attempt_budget" backend/tests/ | head -1
backend/tests/test_phase_86_32_attempt_budget.py            <- NON-ZERO: control passes
runtime surfaces:
$ grep -rln "attempt_budget" scripts/harness/ backend/ .claude/hooks/ | grep -v attempt_budget.py | grep -v attempt_gate | grep -v test_
(no output)                                                  <- zero runtime callers before this step
```

*(Probe honesty: the first control grepped the module for its own filename and
returned 0 -- a failed control, disclosed and replaced rather than deleted.)*

## 3. Criterion-4: the live drives, verbatim

**At-ceiling, a REAL Workflow tool call (step 999.2 seeded to 5/5):**

```
PreToolUse:Workflow hook error: [python3 "${CLAUDE_PROJECT_DIR:-$(pwd)}/scripts/harness/attempt_gate.py"]:
[attempt-gate] DENIED: step 999.2 has used 5/5 attempts (cumulative, cross-session;
increments on ATTEMPT, not outcome). This launch was stopped BEFORE any tokens were
spent. A denial is not a verdict. Operator escalation written to
handoff/current/escalation_attempt_budget_999.2.md; to authorize another attempt run:
python3 scripts/harness/attempt_gate.py --operator-extend 999.2 --by 1 --reason "..."
```

The tool call errored at the seam -- no run record, no spawn, no tokens. The
escalation file exists at the named path with the module's summary.

**Below-ceiling, the next REAL launch (86.85 cycle-7 respawn) -- allowed and
COUNTED (the gate appended this row from the hook process):**

```
$ tail -1 handoff/audit/attempt_budget_audit.jsonl
{"ts": "2026-08-17T10:33:04Z", "type": "attempt", "step_id": "86.85",
 "workflow": "qa-verdict.js", "tool_use_id": "...", 
 "session_id": "e6b8ec06-f72e-41c5-8700-bceb15df4c5f",
 "attempt_number_inclusive": 1, "note": "recorded at launch (PreToolUse); outcome unknown at this seam"}
```

**Wiring registration** (validated by the skill's own jq check):

```
$ jq -e '.hooks.PreToolUse[] | select(.matcher == "Workflow") | .hooks[] | select(.type == "command") | .command' .claude/settings.json
"python3 \"${CLAUDE_PROJECT_DIR:-$(pwd)}/scripts/harness/attempt_gate.py\""
```

## 4. Criterion-3: cross-session persistence

Six separate python invocations (the pipe-test) wrote rows 1-5 and were denied
on the 6th; each invocation re-read the count from disk; the live hook process
then read the same file. Self-test additionally proves re-read-after-write and
that an operator-extension row re-opens exactly one attempt:

```
$ python3 scripts/harness/attempt_gate.py --self-test
  ok    fresh step -> allow
  ok    at ceiling (5) -> deny
  ok    count survives re-read from disk
  ok    operator extension re-opens exactly one attempt
  ok    extension consumed -> deny again
  ok    verdict-ledger PASS -> allow (re-grades never budget-blocked)
  ok    corrupt row counts as an attempt (over-count is the safe direction)
  ok    deny path emits no verdict artifact (no such key exists)
  ok    no step_id -> not attributed
  ok    string args attribute correctly
  ok    malformed string args salvage the step id
  ok    hostile step id refused
SELF-TEST PASSED                                                    (exit 0)
```

## 5. Criterion-8: the mutation matrix (control caught a real bug first)

```
$ python3 scripts/qa/mutation_matrix_86_71.py --verify
CONTROL green: all 6 behavioural checks hold (below rc=0 rows=1; at-ceiling rc=2)
  G1  KILLED     deny branch removed -- exhaustion silently allows
  G2  KILLED     attempt row write dropped at the CALL SITE -- launches stop being counted
  G3  KILLED     step-id extraction neutered -- every launch reads as unattributable
  G4  KILLED     corrupt ledger row silently skipped -- the count can only shrink
  G5  KILLED     deny demoted to allow -- exit 2 becomes exit 0 with the message kept
  G6  KILLED     ceiling comparison bypassed -- disposition read but ignored
BYTE-IDENTICAL RESTORE: ok
cells=6  killed=6  real survivors=0  errors=0
VERIFY: PASS -- control green, 0 survivors, 0 errors, tree unchanged.       (exit 0)
```

The matrix's FIRST control run was RED: the deny message's unconditional
`relative_to(REPO)` raised under the overridden escalation dir, fell into the
fail-open handler, and converted a deny into an allow. Fixed
(`is_relative_to` guard, story in the code comment); a control that catches
the subject before any mutant runs is the control working.

## 6. Module guards after the docstring correction

```
$ python -m pytest backend/tests/test_phase_86_32_attempt_budget.py -q
15 passed                                                            (exit 0)
$ python3 scripts/qa/mutation_matrix_86_32.py    # exit 0
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/harness/attempt_budget.py\").read()); print(\"parses\")"'
parses                                                               (immutable command, exit 0)
```
