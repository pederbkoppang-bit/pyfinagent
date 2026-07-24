# live_check -- masterplan step 75.7 (cycle 3)

Date: 2026-07-23 | Findings: pysvc-01, pysvc-02, gap1-06, gap1-02
Verdict history: cycle-1 CONDITIONAL (wf_8b63c4cd-b25), cycle-2 CONDITIONAL (wf_6f5f82ce-fad).
All blockers to date are artifact-accuracy; the code meets all 6 criteria every cycle.

GENERATED, not hand-edited. Exit codes captured directly (rc=$?), not via zsh
PIPESTATUS (1-indexed -- a cycle-2 lesson). No UI surface; no live bot / iMessage.

## 1. Immutable verification command (exit 0 required)
```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_slack_streaming.py -q
............                                                             [100%]
12 passed in 2.08s
EXIT=0
```

## 2. py_compile on the 4 changed source files (criterion 6)
```
  backend/slack_bot/streaming_integration.py OK
  backend/slack_bot/app_home.py OK
  backend/slack_bot/commands.py OK
  backend/slack_bot/scheduler.py OK
```

## 3. Lint gate -- EXPLICIT file args over the full change surface incl. the new test

Cycles 1 & 2 both mis-scoped this. FACTUALLY: the cycle-1 command scoped to TRACKED
files only and excluded the untracked new test file -- where the one F401 (unused
QueryComplexity) lived -- so it never linted it. (I also gave a wrong zsh
word-splitting explanation; dropping the mechanism claim and running explicit args.)
The F401 is removed; authoritative scan over ALL six changed+new files:
```
$ uvx ruff check --select F821,F401,F811 backend/config/settings.py backend/slack_bot/{app_home,commands,scheduler,streaming_integration}.py backend/tests/test_phase_75_slack_streaming.py
All checks passed!
EXIT=0
```

## 4. Criterion 5 -- no phone literal in scheduler.py
```
$ grep -c '+4794810537' backend/slack_bot/scheduler.py  (expect 0)
0
581:    escalation_phone_e164: str = Field(
```

## 5. Mutation matrix -- 7/7 killed (re-run cycle 3; M1 kill = the completion assertion, RuntimeWarning leg inert)
```
M1  await streamer.append -> sync   -> KILLED (kinds=[chat_stream,stop]; "append" in kinds fails)
M2  restore ThreadPool fan-out      -> KILLED (AST assert)
M3a _get_live_data -> bare sync     -> KILLED
M3b _read_status  -> bare sync      -> KILLED
M4  pager ignores returncode        -> KILLED
M5  restore phone literal           -> KILLED
M6  drop except-path Slack fallback  -> KILLED
```

## 6. Full-suite regression (re-measured cycle 3)
```
$ .venv/bin/python -m pytest backend/tests/ -q --timeout=300  (tail)
10 failed, 1305 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 72.59s (0:01:12)
```
Failing files = standing live-environment red set (23.2.x log-scrapers, 57.1 x3, 60.1,
60.3, portfolio_swap); NONE reference the change surface. 0 regressions; +12 are mine.

## 7. change surface
```
 backend/config/settings.py                 |   4 +
 backend/slack_bot/app_home.py              |   6 +-
 backend/slack_bot/commands.py              |   4 +-
 backend/slack_bot/scheduler.py             |  52 ++++++++---
 backend/slack_bot/streaming_integration.py | 138 +++++++++++++++--------------
 5 files changed, 126 insertions(+), 78 deletions(-)
(new) backend/tests/test_phase_75_slack_streaming.py (12 tests)
```
