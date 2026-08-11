# live_check -- step 86.33

Captured 2026-08-11 by Main. All output verbatim from execution.

## 0. Criterion 1 -- THE CENSUS (required by verification.live_check)

> **ADDED after the cycle-2 CONDITIONAL.** This file carried the probe, the
> researcher decisions and the mutation cells but **no census** -- and the
> masterplan's own verification.live_check names it FIRST. It was never here, in
> either committed revision; the cycle-1 remediation did not remove it.
>
> **My error: I built a NEW derivation script without checking whether an existing
> one already satisfied the criterion.** census_qa_write_guard_log_86_31.py already
> produces the --before cutoff, the excluded-row count, the outside-memory-dir
> counts and the breach recall that criterion 1 demands.
> derive_agent_type_population_86_33.py produces none of them. The covering evidence
> existed the whole time; my handoff record pointed at the wrong script.

```
$ python scripts/qa/census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z
log            : /Users/ford/.openclaw/workspace/pyfinagent/handoff/logs/qa_write_guard.log
cutoff         : 2026-08-10T09:30:00Z
rows counted   : 3012
rows excluded  : 6867   (at or after the cutoff)
rows unparsed  : 0

qa-ROLE identities that are NOT exactly 'qa': 27
    qa-36-12-cycle6
    qa-36-7-80-40
    qa-36-7-80-40-cycle2-retry
    qa-75-5-12-c3
    qa-76-9-2-c2
    qa-80-1
    qa-80-1-c2
    qa-80-2
    qa-80-2-c2
    qa-80-27
    qa-80-27-c2
    qa-80-3
    qa-80-3-c2
    qa-80-3-c3
    qa-80-3-c4
    qa-80-3-c5
    qa-80-3-c6
    qa-80-3-c7
    qa-80-31
    qa-80-31-c2
    qa-80-31-c3
    qa-80-4-cycle3
    qa-80-40-cycle3
    qa-80-5-cycle2
    qa-80-5-cycle3
    qa-82-0-c6
    qa-85-5-c3

Write/Edit events from those identities            : 113
  ... targeting paths OUTSIDE .claude/agent-memory/qa/: 69   -> ALLOWED by the old predicate

NO-SELF-EVAL BREACHES -- DERIVED over the class, not a hand-picked path list:
  20 events across 10 identities
    qa-80-2          Write handoff/current/evaluator_critique.md
    qa-80-2-c2       Edit  frontend/src/lib/api.ts
    qa-80-2-c2       Edit  frontend/src/lib/api.ts
    qa-80-2-c2       Edit  handoff/current/evaluator_critique.md
    qa-80-2-c2       Edit  handoff/current/evaluator_critique.md
    qa-80-2-c2       Edit  handoff/current/evaluator_critique.md
    qa-80-1          Write handoff/current/evaluator_critique_80.1.md
    qa-80-1          Write handoff/current/evaluator_critique_80.1.md
    qa-80-1-c2       Edit  handoff/current/evaluator_critique_80.1.md
    qa-80-27         Write handoff/current/evaluator_critique_80.27.md
    qa-80-27         Write handoff/current/evaluator_critique_80.27.md
    qa-80-31         Write handoff/current/evaluator_critique_80.31.md
    qa-80-31         Write handoff/current/evaluator_critique_80.31.md
    qa-80-31-c2      Edit  handoff/current/evaluator_critique_80.31.md
    qa-80-5-cycle2   Write frontend/src/components/QA805Probe.test.tsx
    qa-80-5-cycle2   Write frontend/src/components/QA805Probe2.test.tsx
    qa-80-5-cycle3   Write frontend/src/components/QaProbe805.test.tsx
    qa-82-0-c6       Write handoff/current/evaluator_critique_82.0_cycle6.md
    qa-82-0-c6       Write handoff/current/evaluator_critique_82.0_cycle6.md
    qa-82-0-c6       Write handoff/current/evaluator_critique_82.0_cycle6.md

  The class rule, stated so it can be re-checked: an evaluator write is a
  breach when the path contains 'evaluator_critique' (Main is its verbatim
  scribe) or starts with any of ('backend/', 'frontend/src/', 'scripts/', 'docs/'). Counting a
  hand-picked path list instead of this rule is what produced the earlier
  'six events' figure -- accurate for the list, wrong for the class.

DISCLOSED RESIDUAL -- NOT matched by is_qa_role(), queued as step 86.33:
    workflow-subagent      80 events -- indistinguishable from a legitimate researcher write
    general-purpose        22 events -- indistinguishable from a legitimate researcher write

For reference, the identities the qa-role predicate now covers are matched by
`agent_type` == 'qa' or lowercased-prefix 'qa-' / 'qa_'.
```

**Headline numbers, bound to the cutoff that produced them:**

| quantity | value |
|---|---|
| rows counted (before 2026-08-10T09:30:00Z) | **3,012** |
| rows EXCLUDED by the cutoff | **6,867** |
| rows unparsed | 0 |
| qa-role identities not exactly "qa" | **27** |
| Write/Edit events from those identities | **113** |
| ...targeting paths OUTSIDE .claude/agent-memory/qa/ | **69** (allowed by the old predicate) |
| NO-SELF-EVAL BREACHES, derived over the class | **20 events across 10 identities** |

**The breach class is a RULE, not a path list**: a path containing
"evaluator_critique" (Main is its verbatim scribe) or starting with backend/,
frontend/src/, scripts/ or docs/. The script states this itself and records that
counting a hand-picked list instead produced an earlier "six events" figure --
accurate for the list, wrong for the class.

**Perishability, demonstrated between two runs an hour apart:** the cycle-2 Q/A
measured rows-excluded = **6,866**; this run reads **6,867**. The log is live and
gitignored, so every figure here is bound to its cutoff AND its run, and the script
-- not this file -- is the source of truth.

## 1. Immutable verification command
```
$ bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'
guard-parses
exit=0
```

## 2. Criterion 3 -- every researcher spelling, driven against the real hook
```
==================================================================================
phase-86.33 criterion 3 -- researcher rail, every spelling, driven
==================================================================================

  population rule startswith('research')          -> 31 spellings
  population rule startswith('research'|'res-')  -> 34 spellings  [USED]

  driving 34 identities against handoff/current/research_brief_86.33.md
  ... (34 rows, all ALLOW) ...
  CONTROL  qa -> backend/main.py: BLOCK (rc=2)

==================================================================================
  OK -- all 34 researcher spellings still write, and the control
       is still blocked, so both directions are exercised.
==================================================================================
```

## 3. Criterion 4 + 6 -- mutation matrix
```
[control]
    guard parses (bash -n)           GREEN
    qa write-separation prover       GREEN
    researcher rail prover           GREEN
    payload key set recorded         GREEN
[M1-exact-qa-match] KILLED  (revert the 86.31 widening: match ONLY the literal agent_type 'qa')
    expected red: ['separation']
    actual  red: ['separation']
[M2-drop-payload-keys] KILLED  (stop recording the payload key set (criterion 2's measurement))
    expected red: ['keyset']
    actual  red: ['keyset']
[M3-apostrophe-trap] KILLED  (CRITERION 6: inject ONE apostrophe into the single-quoted python body)
    expected red: ['liveness']
    actual  red: ['liveness', 'separation', 'researcher', 'keyset']
[restore] byte-identical: True
  KILLED           M1-exact-qa-match
  KILLED           M2-drop-payload-keys
  KILLED           M3-apostrophe-trap
post-restore all green: True
RESULT: all 3 cells KILLED, control green, guard restored.
```

## 4. Criterion 2 -- ANSWERED by a REAL spawn

My synthetic probe echoed my own input back (8 keys) and I did not present it as the
answer. The cycle-1 Q/A's own `Write` drove the real hook. From
`handoff/logs/qa_write_guard.log`, rows `2026-08-11T12:59:29.967328Z` and
`13:06:06.080792Z`:

```
REAL SUBAGENT WRITE -- 12 keys:
  agent_id, agent_type, cwd, effort, hook_event_name, permission_mode,
  prompt_id, session_id, tool_input, tool_name, tool_use_id, transcript_path
  agent_type='qa'   agent_id='afd21026f4056c9e0'

MAIN-SHAPED WRITE -- 10 keys, agent_type and agent_id BOTH ABSENT
```

**ANSWER: NO.** One caller-chosen role field plus one opaque instance id. Nothing
separates TYPE from NAME. The four keys the synthetic probe lacked -- `effort`,
`prompt_id`, `tool_use_id`, `transcript_path` -- are none of them role attributes.

## 5. Criterion 5 -- no fail-closed change shipped (ASK #6)

The only guard edit is log-only. `git diff` on the behavioural predicate:
```
+try:
+    payload_keys = sorted(d.keys()) if isinstance(d, dict) else []
+except Exception:
+    payload_keys = []
+
-                       "tool_name": tool_name, "file_path": file_path})
+                       "tool_name": tool_name, "file_path": file_path,
+                       "payload_keys": payload_keys})
```
