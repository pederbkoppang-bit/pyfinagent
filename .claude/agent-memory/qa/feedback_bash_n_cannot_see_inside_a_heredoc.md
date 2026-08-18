---
name: bash-n-cannot-see-inside-a-heredoc
description: A `bash -n` buildability gate is structurally blind to the Python inside a quoted heredoc, so every heredoc-side mutant passes the UNSCORABLE arm and a SyntaxError scores as a KILL
metadata:
  type: feedback
---

When a guard implements "a mutant that does not BUILD is UNSCORABLE, never a kill"
using `bash -n`, check WHERE the mutants land. `bash -n` does not parse a quoted
heredoc (`python3 - << 'PYEOF'`) -- the body is opaque text to bash.

MEASURED (86.97, cycle 1): a mutant introducing an unbalanced paren inside the
detector heredoc gave `buildable() -> True`, `compile() -> SyntaxError`, driven
`rc=1` with an empty log, and the scoring rule `m_log.strip()==""` scored it
**KILLED**. Both shipped cells were heredoc-side, so the UNSCORABLE arm could not
fire for 2 of 2 cells. The mechanism was present and correctly wired; only its
oracle was blind -- the same structural-blindness defect the step existed to close,
reproduced inside the guard's own scoring.

Second half of the same probe: the "neuter-the-write" cell replaced the log path
with `os.devnull` while `os` was imported ZERO times in the heredoc, so it killed
via `NameError`, swallowed by a broad `except`. A correct kill, a wrong mechanism.

**Why:** an empty-effect assertion cannot distinguish "the guard observed the
missing effect" from "the mutant broke the program". Both produce no output.

**How to apply:** for any mutation matrix, (a) run the mutants' own language
compiler, not the wrapper's, and (b) require the mutant run to exit 0 alongside the
effect assertion. Then re-derive each cell's stated mechanism by reading the
mutant's stderr -- a kill credited to the wrong cause is [[a-correct-observation-can-credit-the-wrong-mechanism]]
and vacuity shape 11. Related: [[a-mutant-that-cannot-build-scores-as-a-kill]].
