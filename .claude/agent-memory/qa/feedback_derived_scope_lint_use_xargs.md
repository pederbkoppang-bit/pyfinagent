---
name: derived-scope-lint-use-xargs
description: NEVER pass an unquoted $VAR as a multi-file argument in this zsh — the tool receives one blob and silently measures nothing (hit on ruff lint AND a grep disclosure audit)
metadata:
  type: feedback
---

**Generalized rule: in this shell, an unquoted `$VAR` holding a newline- or
space-joined file list is ONE argument, so any tool you hand it measures NOTHING
— and most such tools report success.** Applies to every multi-file command I
run, not just the lint gate: use `xargs`, an explicit argument list, or a
`while read` loop.

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

**Recurrence outside the lint gate (36.7/80.40, 2026-07-26):** auditing which
changed files the handoff disclosed, I ran
`grep -lF "$b" $ART` with `ART` holding six artifact paths. `ugrep` warned
`"<all six paths joined>: No such file or directory"` and my loop scored
**14 of 14 files "UNDISCLOSED"** — a fabricated finding I nearly reported. The
real answer was 4. Same shape, different tool: the warning went to stderr while
my arithmetic ran on the exit code.

**How to apply:** every time the gate runs, and every time I build a
multi-file command. Note the non-empty guard on the
FILE COUNT is not sufficient — it proves the derivation found files, not that
the linter received them. Verify the finding count or the printed file paths,
not just the exit code. **A result that is uniformly 0% or uniformly 100% across
a derived set is the signature of this bug — re-run one member by hand before
believing it.** This is vacuity shape #9 in `qa.md` §4c
(executor-environment non-reproducibility), which names the trap but not the
remedy; see also [[verbatim-paste-drift-arithmetic]] and
[[stepid-grep-escape-dot]] for the same class of "the command didn't measure
what you think it measured".
