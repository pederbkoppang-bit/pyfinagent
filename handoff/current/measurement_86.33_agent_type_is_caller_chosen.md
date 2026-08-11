# 86.33 -- `agent_type` is CALLER-CHOSEN, measured

**Not a contract.** The research gate (`wf_883e1c4a-56a`) is in flight. This is the
measurement it will be cross-checked against, recorded now so it is not re-derived.

## The central fact for criterion 4

**`agent_type` in the PreToolUse payload is whatever the caller passed as
`agentType` in the spawn.** It is not assigned by the platform.

Evidence, from my own workflow scripts:

```
.claude/workflows/qa-verdict.js:232      agentType: 'qa',
.claude/workflows/research-gate.js:670   agentType: 'researcher',
.claude/workflows/research-gate.js:734   agentType: 'Explore',
```

And the values those spawns actually produced today, after the 86.33 P0 commit
(`8a9a4293`, UTC 2026-08-11T07:55:14):

```
rows after the P0 commit: 168
  ''            76 rows,  0 with agent_id     <- main-session writes
  'qa'          68 rows, 44 with agent_id
  'researcher'  24 rows, 18 with agent_id
```

**Zero `workflow-subagent` and zero `general-purpose` today** -- because my current
scripts pin `qa` and `researcher`.

## `workflow-subagent` and `general-purpose` are NOT platform types

The step text describes these two as the remaining unmatched values, left alone
because they looked "indistinguishable from legitimate". The measurement says they
are the same kind of thing as the 27 named `qa-*` identities the 86.31 fix closed:

- **`general-purpose` was a PREVIOUS PIN IN MY OWN SCRIPT.**
  `qa-verdict.js:207` reads: *"phase-75.20: agentType 'qa' (was 'general-purpose')"*.
  So those 24 historical rows are Q/A spawns from before that change -- a caller
  choosing a different label, which is exactly the bypass shape.
- **`workflow-subagent`** (82 rows) is the value present when no `agentType` is
  pinned -- a default, still caller-controlled by omission.

**Consequence: every value in `agent_type` is self-asserted.** A guard keyed on it
cannot be an authorization boundary, only a convention check. That is the same
conclusion the 86.31 cycle-1 Q/A reached about the named identities, generalised.

## All-time distribution (criterion 1's population)

```
''                        2055
'qa'                      1473
'qa-80-2-c2'               617
'qa-36-12-cycle6'          612
'qa-85-5-c3'               611
'qa_86_31'                 611
'QA-80-2'                  611
'researcher'               606
'workflow-subagent'         82
'res-78-1'                  38
'researcher-80-4-death'     36
'research-82-0'             36
'qa-86-34-c2'               26
'general-purpose'           24
```

Derived by script from `handoff/logs/qa_write_guard.log`, not transcribed from the
step text -- criterion 1 requires exactly that.

## AN OPEN QUESTION I AM NOT GUESSING AT

**24 of the 68 post-P0 `qa` rows carry no `agent_id`.** The field is not uniformly
populated even within a single agent type. I do not know why, and the difference
matters: if `agent_id` is to be load-bearing for any future rule, an
intermittently-absent field fails differently from a consistently-absent one.

Recorded as an open question for the gate/contract rather than resolved by
inference. **Do not build a predicate on `agent_id` until this is explained.**

## Carried forward from this morning

The runtime **does** populate `agent_id` (23 of 33 role-typed rows at the time of
that measurement), the key is **absent** rather than empty when unsupplied, and one
sampled value (`ab3ff92edf47e42e5`) joins to a real transcript on disk -- so it is a
genuine runtime identity, not an opaque token. See
`measurement_86.33_agent_id_runtime.md`.
