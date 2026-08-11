# live_check -- step 86.33

Captured 2026-08-11 by Main. All output verbatim from execution.

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

## 4. Criterion 2 -- the key-set mechanism works; the REAL answer is pending
```
synthetic payload -> payload_keys recorded:
  ["agent_id","agent_type","cwd","hook_event_name","permission_mode","session_id","tool_input","tool_name"]
```

**That is MY input echoed back.** The installed platform's key set requires a
REAL spawn. The Q/A grading this step produces the first genuine one; read it
from handoff/logs/qa_write_guard.log rather than from this file.

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
