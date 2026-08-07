---
name: derived-scope-misses-untracked-files
description: git diff --name-only HEAD returns EMPTY for a new-file step, so qa.md's mandatory empty-set guard aborts the lint gate instead of linting; union in git ls-files --others
metadata:
  type: feedback
---

`git diff --name-only HEAD -- '*.py'` is NOT the authority on "changed .py files"
when the step's only code change is a NEW file. New files are UNTRACKED, so that
command returns the empty string, and qa.md section 1a's mandatory empty-set guard
then ABORTS the lint gate ("EMPTY FILE SET -- gate FAILED") on a step that does
have a lintable diff. Always resolve the scope as the UNION:

```
PYF=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u )
test -n "$PYF" || exit 1
echo "$PYF" | tr '\n' '\0' | xargs -0 uvx ruff check --select F821,F401,F811
```

**Why:** phase-83.0.3 (2026-08-07) shipped exactly one code change --
`backend/tests/test_phase_83_0_3_pbo_false_pass.py`, a brand-new file. The
qa.md-literal command resolved to nothing. This is the mirror image of
[[derived-scope-lint-use-xargs]]: there the variable expanded to one bogus arg and
ruff measured NOTHING while printing "All checks passed"; here the resolver
resolves nothing and the guard mis-reports a gate failure. Both defects are
"the scope resolver is wrong", opposite signs.

**How to apply:** any PROOF/test-only step, any step whose diff is a new module,
and any step where `git status --short` shows `??` lines. Check `git status --short`
for `??` BEFORE trusting `git diff --name-only`. Same union applies to the
"no unintended production change" check -- `git diff` alone will not show a
newly-added production file.

**Companion gap in the same file:** qa.md's 3rd-CONDITIONAL rule says to count
prior CONDITIONALs by grepping `handoff/harness_log.md`. That grep returns ZERO
mid-cycle **by design**, because log-last defers the harness_log append until
after the verdict. On 83.0.3 two CONDITIONALs existed and the log showed none.
Count from `handoff/current/evaluator_critique_<sid>.md` and
`handoff/current/qa_returns/*.output.json` instead; treat the harness_log grep as
a lower bound only. See [[verdict-gate-ignores-per-cycle-json]] for the adjacent
resolver defect.
