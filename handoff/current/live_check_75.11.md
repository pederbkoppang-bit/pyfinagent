# live_check -- Step 75.11 (sre-ops-01/02/04/05/07/09, pysvc-05)

Date: 2026-07-24. Verbatim captures; exit codes via rc=$? immediately.

## 1. Immutable verification command (exit 0) -- Main's independent re-run

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_sre_ops.py -q
25 passed, 1 warning in 2.16s
pytest_exit=0
```

## 2. Change surface (code-scoped; whole-tree stats drift with ambient audit appends per 75.10 Q/A Note-1)

```
$ git diff --stat HEAD -- backend/ scripts/ .claude/hooks/ .claude/masterplan.json | tail -1
 8 files changed, 138 insertions(+), 24 deletions(-)
```
Plus 7 NEW files: scripts/ops/{rotate_logs.sh, frontend_start.sh, run_ablation.sh,
com.pyfinagent.{logrotate,frontend,ablation}.plist.template} +
backend/tests/test_phase_75_sre_ops.py.

## 3. Danger-hook behavioral transcript (Main-driven, REAL script, zero real kills)

```
$ CLAUDE_TOOL_NAME=Bash CLAUDE_TOOL_INPUT='{"command":"pkill -9 uvicorn"}' bash .claude/hooks/pre-tool-use-danger.sh
pre-tool-use-danger blocked this call: pkill/killall targeting a pyfinagent service process (python|uvicorn|next|slack_bot) -- use 'launchctl kickstart -k gui/$UID/com.pyfinagent.<svc>' instead (sre-ops-05)
To override (one-shot): re-run with CLAUDE_ALLOW_DANGER=1 in env.
pkill-uvicorn exit=2
$ CLAUDE_TOOL_NAME=Bash CLAUDE_TOOL_INPUT='{"command":"pkill -9 SomeRandomApp"}' bash .claude/hooks/pre-tool-use-danger.sh
pkill-unrelated exit=0
$ ...same blocked command WITH CLAUDE_ALLOW_DANGER=1
escape exit=0
```

## 4. Full-suite regression (Main's independent re-run, symmetric diff)

```
FAIL SET IDENTICAL TO BASELINE
10 failed, 1416 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 87.75s (0:01:27)
```

## 5. Boundary evidence

backend/.env sha256 IDENTICAL before/after (executor-measured, recorded in
the draft): 2df26087...fb28015, size 6128 both. Zero launchd bootstraps
(machine actions = OPS-ROTATE-BOOTSTRAP operator token). Templates: Main's
independent 30+-char-literal grep found NO secret-shaped literals.

## 6. Flag-gated / UI note

No flag-gated live-loop behavior; every machine-visible change is either
operator-token-deferred (rotation, frontend authority, ablation wrapper),
next-restart (formatter), or hook-invocation-scoped (danger rail -- proven
live in section 3). No UI surface -> no Playwright capture.
