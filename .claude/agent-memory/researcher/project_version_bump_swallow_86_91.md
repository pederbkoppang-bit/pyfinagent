---
name: project-version-bump-swallow-86-91
description: 86.91 changelog flip detector -- the never-raise marker has NEVER fired (0 hits in 976KB), stderr is redirected to a gitignored log, the replay script mirrors the same defect, and 86.68's own contract predicted this step
metadata:
  type: project
---

The `_flip_magnitude` under-count (step 86.91) is NOT an error path. Four things
that are not visible from the step's `audit_basis`:

**1. The never-raise marker has never fired.** `grep -c "flip-detect FAILED"
handoff/logs/auto-push.log` = **0** over 976,895 bytes. So the frozen version is
100% the silent `[]` from the `None` exclusion, not from exceptions. Do not spend
a cycle hardening the `except` branch -- it is already correct and already unused.

**2. The stderr channel is BROKEN, so "add a marker" is not a fix.**
`post-commit-changelog.sh` is NOT a git hook -- `.git/hooks/` contains only
`pre-commit`. It runs two ways, both from `.claude/settings.json`:
`PostToolUse/Bash` directly, and `PostToolUse/Write|Edit` ->
`auto-commit-and-push.sh:396`, which invokes it as
`bash "$CHANGELOG_HOOK" >> "$LOG_FILE" 2>&1` -- stderr lands in a **gitignored**
`handoff/logs/auto-push.log`. Git's documented "stderr is forwarded to the user"
guarantee does not apply. A stderr-only marker is as invisible as the silence it
replaces.

**3. `scripts/qa/replay_changelog_rule_86_68.py:54` carries the SAME predicate.**
Fix the hook alone and criterion 3's "three numbers" compares a fixed hook against
a stale baseline. Both files must change.

**4. 86.68's own contract predicted this step**, at
`handoff/archive/phase-86.68/contract.md:106-107`: *"A silent failure mode in a
never-raise detector is worth a criterion of its own if it proves true; file it,
do not absorb it here."* Cite it -- it is in-repo prior art.

**Measured 2026-08-16** over 621 commits since 2026-08-11 (0 parse errors): **2**
commits contain a created-and-closed step (swallowed); **5** contain a normal
transition (counted). The increase criterion 3 demands be accounted for commit by
commit is therefore at most 2 commits. The union of the two sets was NOT computed
-- re-derive it.

**Why:** the class is ABSENT-vs-UNCHANGED conflation -- `before.get(sid)` returns
`None` for both "step did not exist" and, to a reader, "nothing to compare". The
docstring's intent was "not a transition"; the extension is "ignore steps created
this commit", which is the file-it-and-fix-it workflow this project runs
constantly.

**How to apply:** prefer key-space membership (`sid not in before`) over a
sentinel default -- it needs no discipline at the call site. Sibling of
[[project_ingress_falsy_zero_86_86]] (producer-side collapse) and
[[feedback_operations_that_cannot_fail_loudly]].
