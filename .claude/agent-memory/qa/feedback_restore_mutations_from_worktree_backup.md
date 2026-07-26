---
name: restore-mutations-from-worktree-backup
description: Back up working-tree copies before mutating — `git checkout -- <path>` on an uncommitted step reverts the FIX, not just my mutation, and the suite goes green for the wrong reason
metadata:
  type: feedback
---

Before running a mutation battery, `cp` each target file to the scratchpad and
restore from THAT copy. Never `git checkout -- <path>` / `git show HEAD:<path>`
to undo a mutation.

```bash
SP=<scratchpad>/wt_backup; mkdir -p $SP
for f in <targets>; do cp "$f" "$SP/$(echo $f | tr '/' '_')"; done
md5 <targets>          # record the restore targets
# ...mutate, run, then:
cp "$SP/<mangled>" "$f"
md5 <targets>          # MUST match the recorded values
```

**Why:** the code under evaluation is almost always UNCOMMITTED (Q/A runs before
the status flip and before the auto-commit hook). `HEAD` is therefore the
**pre-fix** version. Restoring from HEAD silently reverts the step's entire fix,
and the next mutation runs against pre-fix code — the mutation "survives" or
"kills" for reasons that have nothing to do with the guard being tested, and
every later measurement in the session is against the wrong tree. The one file
where HEAD *is* the right source is a data file the tests must not have touched
(`handoff/kill_switch_audit.jsonl`), because there the goal is the opposite:
discard any write.

**How to apply:** at the start of any cycle where I plan to mutate source.
Verify restores by md5 against the values recorded BEFORE the first mutation,
not by `git diff` (which compares to HEAD and will always look "dirty" on an
uncommitted step — a useless signal here). See
[[derived-scope-lint-use-xargs]] for the sibling class of "my harness measured
something other than what I claimed".
