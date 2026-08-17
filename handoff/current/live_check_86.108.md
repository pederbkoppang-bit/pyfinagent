# live_check -- step 86.108 (2026-08-17; exits unpiped)

**PARTIAL: criterion 1 only.** Criteria 2-6 are NOT addressed here and the step
is NOT closed. This file records what the census delivers so the next session
starts from a re-runnable tool rather than from prose.

## 1. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/agents/orchestrator.py\").read()); print(\"parses\")"'
parses
EXIT=0
```

## 2. Criterion 1 -- the census, re-derived by a COMMITTED script

`scripts/qa/census_invalid_json_86_108.py`. It prints the glob and the match
rule beside every count, and refuses to print a failure rate.

**The filed total reproduces EXACTLY on the gate's own corpus:**

```
$ python scripts/qa/census_invalid_json_86_108.py --rotated-only
TOTAL matching LINES = 2859
  compact              2371    82.9% of lines
  json                  488    17.1% of lines
```

2,859 is right *for the rotated corpus*. The live `backend.log` adds 13, so
the current total is **2,872**. The gate's census stopped at 2026-08-14 because
it globbed only `backend.log.*.gz`; `--rotated-only` reproduces that boundary
so the delta is attributable rather than a drift.

## 3. The four corrections, now encoded in the tool

**C-a mixed format.** 2371 compact / 488 json. A `"module":`-keyed parser sees
**17.1%** and looks complete.

**C-b the agent labels were a match-rule artifact.** Full-corpus buckets, with
the formatter prefix stripped:

```
   606  Critic                 315  Risk Judge
   368  Moderator              310  Neutral Analyst
   342  Devil's Advocate       309  Conservative Analyst
   264  Synthesis-Final        307  Aggressive Analyst
    53  Critic-Retry
```

**There is no agent called "Analyst".** The filed 926 is
310 + 309 + 307 = 926, three distinct analysts. "Advocate" is Devil's Advocate
and "Judge" is Risk Judge.

*Recorded because it is the same error one level down:* this script's own first
draft took "the last three capitalised words" and produced buckets like
`W orchestrator Critic` beside a bare `Critic`, splitting one agent in two. The
extractor now strips the `HH:MM:SS L [module]` prefix and takes the whole
phrase, and `test_agent_phrase_does_not_leak_the_formatter_prefix` pins it.

**C-c the total counts LINES, not events.** The Critic double-logs; both
wordings are counted separately so the fold is visible.

**C-d no rate is derivable.** No synthesis-attempt denominator exists in the
corpus. The filed 9.2% is a composition SHARE of lines.

## 4. Criterion 1's rail split -- NOT derivable, stated rather than fabricated

A JSON marker record's entire field set is `timestamp/level/module/message`.
No line carries a rail, provider or model, so a per-event
`claude_code`-vs-`gemini` split would be invented. The contract's P2 (era-bucket
on `paper_use_claude_code_route`, labelled as such) stands.

The four emit sites, re-derived at exactly the researched line numbers:

```
backend/agents/debate.py:127        logger.warning(f"{label} returned invalid JSON, using raw text")
backend/agents/risk_debate.py:123   logger.warning(f"{label} returned invalid JSON, using raw text")
backend/agents/llm_parse.py:149     logger.warning("%s returned invalid JSON, using raw text", label)
backend/agents/orchestrator.py:315  logger.warning(f"{agent_name} returned invalid JSON")
```

All four are rail-agnostic -- they sit above the client, which is why the emit
site cannot identify the transport.

## 5. Guards, mutation-tested (control green first, byte-identical restore)

```
CONTROL: exit=0  7 passed
M1 agent_of -> last token (the original filing's rule):   exit=1  1 failed  -> KILLED
M2 compact prefix NOT stripped:                           exit=1  4 failed  -> KILLED
M3 JSON lines skipped (the 17%-looks-complete defect):    exit=1  1 failed  -> KILLED
restore verified: True     SURVIVORS (this matrix): none
```

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_108_invalid_json_census.py -q
7 passed in 0.03s
$ uvx ruff check --select F821,F401,F811 <both files>
All checks passed!   EXIT=0
```

## 6. What is NOT done -- this step remains OPEN

Criteria 2 (transport guarantees, cited), 3 (loud degradation at the record
level), 4 (the dark-flag observability route), 5 (no promotion) and 6
(mutations on the new guards) are untouched. In particular **no production
behaviour changed**: nothing under `backend/agents/` or `backend/api/` was
modified, so nothing here needs a restart and nothing is pending.

C4's blind spot is already MEASURED against the running process (pid 41635):
`GET /api/settings/` exposes 45 keys and **none** of the 5 integrity or 2
diversity flags.
