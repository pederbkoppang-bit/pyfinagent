---
name: verify-under-the-production-interpreter
description: When the production caller is a plist/cron/launchd job, re-run the changed script under THAT interpreter and PATH, not the venv -- .venv is 3.14 while launchd's python3 is /usr/bin/python3 3.9.6
metadata:
  type: feedback
---

A green `.venv/bin/python -m pytest` says nothing about the environment that
actually invokes the code. When the caller is a launchd plist, a cron entry,
or any `#!/usr/bin/env bash` script run outside the venv, re-run the changed
module under the REAL interpreter and a stripped PATH before grading:

```bash
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOME" /bin/bash -c \
  "python3 '$PWD/scripts/.../mod.py' --args ..."
/usr/bin/python3 -c "import ast; ast.parse(open('scripts/.../mod.py').read())"
```

**Why:** phase-85.3 extracted auth-derivation from a bash heredoc into
`scripts/away_ops/auth_state.py`. The venv is **Python 3.14**; the
away-watchdog plist runs `/bin/bash healthcheck.sh`, whose `python3`
resolves under launchd's minimal PATH to **/usr/bin/python3 = 3.9.6**. The
seam uses `tuple[str, str]` annotations -- legal on 3.9 ONLY because
`from __future__ import annotations` is present. Had that import been
dropped, every scheduled run would have raised, hit the fail-open wrapper,
printed `unknown probe_error` forever, and the latch would never clear --
silently, with a fully green test suite. Nobody in two cycles had checked
it; I did, and it passed.

**How to apply:** any step that moves logic INTO a file invoked by a plist,
cron, or non-venv shell script. Read the plist's `ProgramArguments` to learn
the real entry point, then smoke the module under that interpreter. A
fail-open wrapper makes this class INVISIBLE at runtime -- the alarm just
goes quiet -- so the interpreter check is the only place it can be caught.
Related: [[feedback_run_probes_against_head_to_classify]],
[[feedback_measure_the_capture_you_didnt_take]].
