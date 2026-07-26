---
name: enumerate-every-position-at-a-recidivist-call-site
description: At a call site that has produced repeat mutation survivors, enumerate EVERY mutable position mechanically (each argument separately, the assign target, the branch test, the body, AND the adjacent line that publishes the same value) -- the untested slots were arg#1 and the neighbouring publish line
metadata:
  type: feedback
---

When one call site has produced a mutation survivor in several consecutive cycles
(phase-36.12: `halt_reason = cycle_halt_reason(ks_check, _ks_state().is_paused())`
survived five times running), stop reasoning about "is the slot closed" and
mechanically enumerate its positions: **each argument independently**, the assign
target, the `if` test, the branch body, and the **adjacent line that publishes the
same value**. Main's five prior fixes had covered the inline literal, the predicate
result, the body, and arg#2 twice (`False`, `True`) -- nobody had ever mutated
**arg#1**, and nobody had mutated the neighbouring `summary["kill_switch"] = ks_check`.

Measured 2026-07-26: arg#1 -> `{}` and arg#1 -> `{"triggered": ks_check.get("triggered")}`
(strip only the new key) were both KILLED, so the halt really was closed -- but the
adjacent publish line SURVIVED (25 passed), stripping `blocked`/`block_reason`/`pre_armed`
out of the operator-facing cycle summary with every test still green.

**Why:** the survivor family here is *relocation* -- each fix moves the hole one
position sideways, so the only reliable coverage is a mechanical enumeration of
positions rather than another judgment about the last one.
**How to apply:** before grading "the slot is closed", write the position list, run
each as a DISK mutation with the WHOLE test file as selector, and classify a
survivor on the publish/reporting line as WARN (no criterion depends on it) rather
than as a hole in the guard. See [[survivor-needs-behavioural-differential]] and
[[mutate-the-flag-read-not-just-the-guard]].
