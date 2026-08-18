---
name: structured-output-transports-86-108
description: Census 2859 reproduces but is LOG LINES not events (mixed-format corpus + Critic double-log); logs carry NO rail; the CC --json-schema rail is post-hoc+re-prompt, not constrained decoding
metadata:
  type: project
---

Step 86.108 research gate (2026-08-17). Findings that cost real measurement and
are NOT re-derivable by reading code.

**The `backend.log.*.gz` census is a LINE count, not an event count.**
`gzcat handoff/logs/backend.log.*.gz | grep -c "returned invalid JSON"` = **2859**
across 7 rotated files, and the per-agent split reproduces the filed numbers
exactly (Analyst 926 / Critic 602 / Moderator 359 / Advocate 342 / Judge 314 /
Synthesis-Final 264 / Critic-Retry 52). Four corrections that a naive re-count
will miss:

- **Mixed-format corpus.** Only **488** of 2859 are `JsonFormatter` records; the
  other **2371** are ANSI-coloured `CompactFormatter` lines. A parser keyed on
  `"module":` returns 488 and *looks complete*. Formatter is chosen by `DEBUG`.
- **The agent labels are a match-rule artefact.** "Analyst" is not one agent — it
  is Conservative+Neutral+Aggressive Analyst. "Judge"=Risk Judge,
  "Advocate"=Devil's Advocate. Taking the last token before the phrase collapses them.
- **The Critic path double-logs**: 274 bare + 274 `…, treating as PASS with draft.`
  The second wording now survives ONLY as a negative assertion in
  `backend/tests/test_phase_75_skill_delivery.py` (phase-75 removed it), so the
  corpus spans multiple code generations.
- **"9.2%" is a composition share, not a rate**: 264/2859 = 9.23%. No
  synthesis-attempt denominator exists in the corpus (`Synthesis complete`,
  `Running Synthesis`, `Analysis complete` all return 0).

Also: there is **no live uncompressed `backend.log`** — the corpus ends
2026-08-14T15:53Z, so no "current rate" claim is supportable from it.

**No log line carries a rail.** The full JSON record is `timestamp/level/module/
message`; `grep -c '"model"'` = 0. `pyfinagent_data.llm_call_log` DOES carry
`provider`/`model` NOT NULL (per `scripts/migrations/add_llm_call_log.py`) but
(a) the warning lines have no `request_id`, so any join is time-proximity, and
(b) `ok` is "true on 2xx" and an invalid-JSON body IS a 2xx — so llm_call_log can
never identify a parse failure by itself. Only an **era-bucketed** split (via
`paper_use_claude_code_route`) is honest.

**Emit surface is FOUR sites**, not one: `orchestrator.py:315` (bare, returns
None), `debate.py:127`, `risk_debate.py:123`, `llm_parse.py:149`.

**Why:** the step text supplied the census as fact; every headline number
reproduced while every *interpretation* of it needed correcting.

**How to apply:** when re-counting, state the glob, the match rule, AND the
denominator; check whether the log corpus is single-format before trusting a
field-keyed parse. See [[reference-structured-output-guarantee-classes-86-108]].
