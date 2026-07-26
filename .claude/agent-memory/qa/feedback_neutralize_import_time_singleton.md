---
name: neutralize-import-time-singleton
description: sys.modules mutation injection is NOT side-effect-free when the module builds a singleton at import; neutralize its real file path BEFORE exec or you write to production
metadata:
  type: feedback
---

Before injecting a mutation that ADDS a side effect (a write, an append, a network
call), first neutralize the module-level singleton's real path in the mutated source.
A control run does NOT protect you: the control has no injected write, so it is clean
by construction and tells you nothing about the mutant.

**Why:** 36.7 cycle 5. `backend/services/kill_switch.py:480` is `_state = KillSwitchState()`
— a module-level singleton whose `__init__` calls `_load_from_audit()`. My mutation
injected `_append_audit("peak_reset", ...)` into that restore path. `exec(compile(src))`
built the singleton against the REAL `_AUDIT_PATH` **before any pytest fixture could
redirect it**, and appended 54 rows to the live `handoff/kill_switch_audit.jsonl` in ~2ms.
Worse, `peak_reset` replays as an authoritative ASSIGNMENT with the newest `ts` winning
the sort, so a read-only restart simulation then returned `peak_nav: None` — I had
disarmed the trailing leg on the next restart. The test fixtures were correctly
isolated; the leak was entirely my harness's import-time exec.

**How to apply:** in the mutated source string, rewrite the path constant first —
`re.sub(r'^_AUDIT_PATH = .*$', f'_AUDIT_PATH = Path(r"{tmpdir}/x.jsonl")', src, flags=re.M)`
— and assert the substitution matched exactly once. Verified working on the same step:
the criterion-1 pre-fix replay ran with md5 identical before and after.

Diagnostics that identify this signature: many rows sharing one millisecond = ONE
replay loop at import, not per-test writes; and the md5 changing on the FIRST mutant of
a batch then staying stable. Re-check the md5 after EVERY mutant, not just at the end —
attributing the write to a specific mutant is what makes the disclosure precise.

Read-only means you report the damage and hand the operator the exact restore
(`git checkout -- <file>` when `git diff` shows only your additions); you never run it
yourself. See [[restore-mutations-from-worktree-backup]] and
[[mutate-without-touching-the-tree]].
