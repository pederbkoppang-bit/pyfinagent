# live_check 75.2 -- verbatim evidence (2026-07-20)

## Immutable verification command -- exit 0

```
$ python3 -c "import os,glob,py_compile; dead=['self_update','assistant_handler','governance','mcp_tools','streaming_handler','context_management']; assert not any(os.path.exists('backend/slack_bot/%s.py'%d) for d in dead), 'dead control-plane module still present'; txt=''.join(open(p).read() for p in glob.glob('backend/slack_bot/*.py')); assert all(('slack_bot.%s'%d not in txt and 'import %s'%d not in txt) for d in dead), 'residual import of deleted module'; c=open('backend/slack_bot/commands.py').read(); assert 'slack_operator_user_id' in c and 'to_thread' in c, 'reaction handler ungated/unthreaded'; o=open('backend/slack_bot/operator_tokens.py').read(); assert 'operator_user_id' in o, 'token-sink identity check missing'; h=open('backend/slack_bot/app_home.py').read(); assert 'slack_operator_user_id' in h, 'app-home model actions ungated'; si=open('backend/slack_bot/streaming_integration.py').read(); assert 'deploy commands are disabled' in si and 'assistant_audit' in si, 'deploy refusal or audit writer missing'; [py_compile.compile('backend/slack_bot/'+f, doraise=True) for f in ['commands.py','app_home.py','streaming_integration.py','operator_tokens.py','app.py']]"
VERIFICATION EXIT 0
```

## git diff --stat (change surface)

Deletions (staged):
```
 backend/slack_bot/assistant_handler.py      | 785 -----------------
 backend/slack_bot/self_update.py            | 467 -----------
 backend/slack_bot/governance.py             | 315 --------
 backend/slack_bot/context_management.py     | 249 --------
 backend/slack_bot/mcp_tools.py              | 247 --------
 backend/slack_bot/streaming_handler.py      | 243 --------
 scripts/go_live_drills/smoke_test_4_17_9.py |  96 --------
 7 files changed, 2402 deletions(-)
```

Modifications + new files:
```
 backend/slack_bot/commands.py                      | 67 +++++++++++++-----
 backend/slack_bot/operator_tokens.py               | 55 +++++++++++---
 backend/slack_bot/streaming_integration.py         | 35 ++++++++++
 backend/slack_bot/app_home.py                      | gate + UI label
 backend/tests/test_phase_62_2_operator_tokens.py   | 24 ++++---
 scripts/qa/sweep_ascii_logger_v3.py                |  2 --
?? backend/slack_bot/assistant_guards.py                   (new)
?? backend/tests/test_phase_75_2_slack_control_plane.py    (new, 32 tests)
```

## Behavioral tests (what the substring command cannot prove)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_2_slack_control_plane.py backend/tests/test_phase_62_2_operator_tokens.py -q
80 passed in 0.15s        # cycle 2 (was 61; +19 deploy-parity contract tests)
```

Covering, per criterion:
- c1: non-operator reaction -> NO push; unset operator -> NO push (fail-closed);
      untracked ts -> NO push; operator on tracked ts -> exactly 1 push, replay -> still 1;
      push observed going through asyncio.to_thread (spy on cmd.asyncio.to_thread)
- c2: all six dead modules raise ModuleNotFoundError on import
- c3: ALL 21 surfaces of the DELETED matcher (recovered via
      `git show HEAD:backend/slack_bot/self_update.py::handle_deploy_command`) refuse,
      including bare "deploy" and the old startswith("deploy") catch-all; 5 legitimate
      queries still pass (incl. "deployment history question" and "tell me what changed
      in the portfolio" -- no over-refusal); refusal fires with si.get_orchestrator
      patched to RAISE if called -> proves refusal precedes any LLM path.
      CYCLE-2 FIX: the cycle-1 list was 7 substrings built from memory and let 12
      surfaces through to the LLM (Q/A wf_160a3771-7b7). Parity is now measured:
        deleted-matcher surface covered: 21 / 21 | misses: []
        legit queries still answerable:   5 / 5  | over-refused: []
- c4: rate limit blocks at the 20/60s budget, is per-user, recovers after 2 quiet windows;
      audit writes exactly one JSONL line per call, writer="assistant_audit",
      64-char text_sha256, and raw message text is NOT present in the file
- c6: sink refuses wrong user / wrong channel / unset operator even when the matcher is
      bypassed; authorized path still writes line 1

## Lint gate + import smoke

```
$ uvx ruff check --select F821,F401,F811 backend/slack_bot/{commands,operator_tokens,streaming_integration,app_home,assistant_guards}.py backend/tests/test_phase_75_2_slack_control_plane.py backend/tests/test_phase_62_2_operator_tokens.py scripts/qa/sweep_ascii_logger_v3.py
All checks passed!

$ .venv/bin/python -c "<import every backend/slack_bot module>"
imported OK: 12 modules -> app, app_home, assistant_guards, assistant_lifecycle, commands,
digest_test, direct_responder, formatters, job_runtime, operator_tokens, scheduler, streaming_integration
```

## Residual-reference sweep (criterion 2)

```
slack_bot.self_update: 0        slack_bot.mcp_tools: 0
slack_bot.assistant_handler: 0  slack_bot.streaming_handler: 0
slack_bot.governance: 0         slack_bot.context_management: 0
```
(`backend.governance.*` is a DIFFERENT live package -- untouched.)

## $0 metered confirmation

No LLM call was added, removed, or repointed. The refusal branch REMOVES an LLM call for
deploy-verb messages (previously classified by the Communication agent); every other path
is byte-identical in its LLM usage.

## OPERATOR ESCALATION (BLOCKER-1) -- see experiment_results_75.2.md

Deleting the modules breaks three already-done steps' IMMUTABLE verification commands:
phases[26].steps[4] (4.14.4), phases[26].steps[23] (4.14.24), phases[29].steps[8] (4.17.9).
NOTHING was edited -- immutable criteria are not amendable. 4.17.9 was ALREADY unrunnable
before this step (it names scripts/go_live_drills/self_update_audit_test.py, which does not
exist on disk). Operator decision requested.
