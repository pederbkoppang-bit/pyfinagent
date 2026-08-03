---
name: verdict-gate-ignores-per-cycle-json
description: The phase-81.2 verdict gate resolves only evaluator_critique_<sid>.json or the rolling evaluator_critique.json -- per-cycle _cycleN.json files do NOT match, so a step evaluated over many cycles can reach the flip with the gate reading no_input (fail-open)
metadata:
  type: project
---

`.claude/hooks/lib/verdict_gate.py::resolve_verdict_source(step_id, handoff_root)`
tries, in order: `current/evaluator_critique_<sid>.json`,
`current/evaluator_critique.json`, `archive/phase-<sid>/...`, `archive/misc/...`.
It does **not** glob per-cycle filenames. So persisting verdicts only as
`evaluator_critique_<sid>_cycle<N>.json` leaves the gate with **no input**:

```
resolve_verdict_source('82.0','handoff')     -> (None, 'none')
gate_decision_with_source('82.0','handoff')  -> ('no_input', 'none')   # fail-open
```

**Why:** measured on 2026-08-03 during the 82.0 cycle-6 evaluation. 82.0 ran six
cycles and Main persisted `_cycle1..4.json` plus `_cycle5_ERRORED.json`; the
rolling `handoff/current/evaluator_critique.json` did not exist at all (only a
stale `.md` from phase-80.2). Phase-81.2 had just been closed specifically to
repair this gate after it sat dead through 13 step closes -- so a step could have
flipped with the freshly-repaired gate still reading nothing. Fail-open is the
gate's designed behaviour, so nothing errors and nothing warns visibly
(`handoff/logs` is gitignored).

**How to apply:** on any multi-cycle step, check the resolver's decision before
the status flip, not just that *some* verdict file exists. The final verdict must
land at `handoff/current/evaluator_critique_<sid>.json` carrying `step_id`,
`verdict`, and `ok`. Q/A is read-only and never writes it -- name it as an
explicit pre-flip action for Main in the critique. Note the gate also fails open
on a mismatched `step_id`, so a stale rolling file from a previous step does not
block, it just silently does not gate. Related:
[[structural-fix-needs-a-mechanism]], [[stepid-grep-escape-dot]].
