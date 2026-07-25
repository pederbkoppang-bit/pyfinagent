---
name: derived-scope-lint-use-xargs
description: Lint a git-derived file scope with xargs, never an unquoted $VAR — zsh doesn't word-split, so ruff lints ZERO files and reports "All checks passed" exit 0
metadata:
  type: feedback
---

When running the `qa.md` §1a Python lint gate on a git-derived scope, pipe the
file list through `xargs`. Never pass an unquoted shell variable.

```bash
{ git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } \
  | sort -u > /tmp/scope.txt
N=$(wc -l < /tmp/scope.txt); test "$N" -gt 0 || { echo "EMPTY SET -- GATE FAILED"; exit 1; }
xargs uvx ruff check --select F821,F401,F811 < /tmp/scope.txt
```

**Why:** this shell is **zsh**, which does NOT word-split unquoted variables
(unlike bash). `uvx ruff check $FILES` therefore passes ONE argument — the whole
newline-joined blob — and ruff prints
`Failed to lint <blob>: No such file or directory` followed by
**`All checks passed!` with exit 0**. A pure false pass. I hit this myself on
phase-80.2 cycle 2 (2026-07-25) while checking condition C3, and only caught it
because the printed blob looked wrong; the exit code alone said PASS.

**How to apply:** every time the gate runs. Note the non-empty guard on the
FILE COUNT is not sufficient — it proves the derivation found files, not that
the linter received them. Verify the finding count or the printed file paths,
not just the exit code. This is vacuity shape #9 in `qa.md` §4c
(executor-environment non-reproducibility), which names the trap but not the
remedy; see also [[verbatim-paste-drift-arithmetic]] and
[[stepid-grep-escape-dot]] for the same class of "the command didn't measure
what you think it measured".
