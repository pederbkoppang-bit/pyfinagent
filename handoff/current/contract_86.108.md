# Contract -- step 86.108

**Step:** 86.108 -- 2,859 structured-output parse failures across ALL pipeline
agents; the CC rail is handed a Gemini-shaped schema contract it honours
nothing of, and every agent's invalid JSON degrades inputs silently. **P1,
money-adjacent.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_8581f683-d24`; 15 sources read in full, 35 distinct URLs,
audit-class dry after 12 rounds; brief `research_brief_86.108.md`, 51,994
chars; all 15 claimed URLs confirmed present in the brief).

**The headline census REPRODUCES EXACTLY -- and four corrections change what
it means.** 2,859 total, and the per-agent split, both reproduce from 7 rotated
`backend.log.*.gz` under a fixed-string grep. But:

1. **The corpus is MIXED-FORMAT.** 2,371 lines are `CompactFormatter` (ANSI
   colour), 488 are `JsonFormatter`. A `"module":`-keyed parse returns 488 and
   *looks complete* -- 17% of the truth presented as the whole.
2. **The agent labels are a match-rule artefact.** There is no agent called
   "Analyst": the 926 is `Conservative Analyst` (309) + `Neutral Analyst` (310)
   + `Aggressive Analyst` (307). Likewise "Advocate" = `Devil's Advocate`,
   "Judge" = `Risk Judge`.
3. **2,859 counts LOG LINES, not events.** The Critic path double-logs:
   `Critic returned invalid JSON` (274) and `...treating as PASS with draft.`
   (274) are one failure emitting two lines. 274+274+54 = 602 = the filed
   "Critic 602". And the second wording was REMOVED by phase-75, so the corpus
   spans multiple code generations -- a rate over all of it mixes builds.
4. **`9.2%` is a COMPOSITION SHARE, not a rate.** 264/2859 = 9.23% is
   Synthesis-Final's share of invalid-JSON *lines*. No synthesis-attempt
   denominator exists in the corpus (`Synthesis complete` / `Running Synthesis`
   / `Analysis complete` all return 0). Quoting it as a failure rate would be
   unreproducible.

**C1 as written is NOT SATISFIABLE, and the contract says so up front rather
than letting GENERATE discover it.** A JSON marker record's entire field set is
`timestamp, level, module, message`; `grep -c '"model"'` returns **0**. No log
line carries a rail, provider or model, so a per-event `claude_code` vs
`gemini` split would be **fabricated**. `pyfinagent_data.llm_call_log` does
carry `provider`/`model` NOT NULL, but the warning lines carry **no
`request_id`**, so any join is a time-proximity heuristic; and its `ok` column
is "true on 2xx", while an invalid-JSON body **is a 2xx** -- so it can supply a
rail for a WINDOW but can never identify a parse failure. **The honest
deliverable is an ERA-BUCKETED split keyed on `paper_use_claude_code_route`,
labelled as such.**

**The strongest empirical result is already in hand and refutes the obvious
fix:** 359 Moderator failures occurred **with Gemini's `response_schema` in
force**. Any plan premised on "the schema makes this unreachable" is dead on
arrival. The three guarantee classes, from the docs read in full: the Anthropic
API constrains; the Claude Code `--json-schema` path validates **post-hoc** and
re-prompts (a success with no `structured_output` is to be treated as a
failure); Gemini never claims an absolute guarantee.

**The LOUD mechanism already exists -- extend it, do not invent one.**
`_parse_failed` is already computed, persisted, and escalated to `_degraded`.
Three measured gaps: the escalation is gated on
`paper_synthesis_integrity_enabled`, which is **invisible in the settings API**;
the four emit sites only `logger.warning` and do not feed it; and
`_judge_parse_fail_fallback` **fabricates a default verdict** -- with the flag
OFF a garbled judge response silently becomes `APPROVE_REDUCED at 3% NAV`.

**The emit surface is FOUR sites, not one** (`debate.py`, `risk_debate.py`,
`llm_parse.py`, `orchestrator.py`), all rail-agnostic because they sit above
the client.

**`GET /api/settings/` answers the step's question with a clean NO:** none of
the 5 integrity or 2 diversity flags is exposed, and the `_FIELD_TO_ENV` rows
are unreachable dead code whose neighbouring comment asserts the opposite.

## Hypothesis

The 2,859 failures are real but have been mis-framed as a single-rail,
single-site, rate-shaped defect. They are four emit sites across multiple code
generations and both transports, none of which records enough to attribute a
rail. The tractable fix is **observability, not suppression**: make an
unparseable output a marked, countable record at the point it happens; expose
the flags that gate the existing marking so "committed is not in force" can be
checked from the running process; and change no risk behaviour.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the per-agent parse-failure rates are re-derived with the population rule and command stated, split by rail (claude_code vs gemini) so the transport attribution is measured, not inherited from 86.69's brief
2. the remedy is chosen from EVIDENCE about what each transport actually supports: state what the CC rail's structured-output path guarantees and what it does not, with the doc or measurement cited, before any schema/prompt change is designed
3. whatever is built, an agent's invalid-JSON output must degrade LOUDLY at the record level (marked, countable) rather than silently -- and the fix must not loosen any gate or fabricate any default verdict
4. the dark-flag observability gap is closed or explicitly re-queued: a read-only route (or equivalent) exposes the LIVE value of operator-gated flags so committed-is-not-in-force checks can read the running process
5. NO flag is promoted and NO .env is written by this step; operator-gated changes are recorded as numbered asks
6. mutation-test every new guard with the control observed GREEN first and a byte-identical restore

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/agents/orchestrator.py\").read()); print(\"parses\")"'`

**Immutable live_check:** `live_check_86.108.md` with the per-agent/per-rail
failure table, the transport-guarantee evidence, and the loud-degradation
demonstration.

## Plan

**P1 -- the census, re-derived with its rule stated (criterion 1).** A
committed script emits the table: exact glob, exact match rule, both formatter
shapes, agents de-collapsed to their real names, the Critic double-log counted
once, and every rate printed **with its denominator or not at all**.

**P2 -- the rail split, delivered as an ERA BUCKET and labelled (criterion 1).**
Bucket events on `paper_use_claude_code_route` over time. **The artifact must
state plainly that a per-event split is not derivable and why** (no rail field;
no `request_id` to join on; `ok` is 2xx-shaped and blind to invalid bodies).
Criterion 1 says "so the transport attribution is measured, not inherited" --
an era bucket is measured; a fabricated per-event split would satisfy the words
and violate the intent. **Recorded as a deviation for the evaluator to judge.**
Before depending on `llm_call_log`, verify rows exist for the census window --
the brief deliberately did not query it.

**P3 -- transport guarantees, cited (criterion 2).** A written comparison of
the three classes with doc citations, plus the local refutation (359 Moderator
failures under `response_schema`). This lands **before** any schema change is
designed, as the criterion requires.

**P4 -- loud degradation at the record level (criterion 3).** Extend the
existing `_parse_failed` path so the four emit sites feed a countable marker
distinguishing at minimum `parse_failed` (syntactic),
`schema_valid_but_rejected_downstream`, and `truncated` (via finish reason).
**No gate is loosened and no default verdict is fabricated.** Specifically:
`_judge_parse_fail_fallback`'s `APPROVE_REDUCED at 3% NAV` is **MARKED, not
changed** -- flipping a risk default is a behaviour change that needs its own
step and its own operator decision. It is filed, not fixed here.

**P5 -- flag observability (criterion 4).** Add the 7 missing keys to the
read-only settings response and make the dead `_FIELD_TO_ENV` rows reachable.
Keep the existing `*_key_configured` boolean idiom -- **no secret values in the
payload**. Note the `settings:full` cache can serve a stale value, which for an
observability endpoint is itself a silent-failure mode, and handle or disclose
it.

**P6 -- mutations (criterion 6)** with the control observed GREEN first,
byte-identical restore, each cell scored, UNSCORABLE if its control was not
green.

## Scope honesty -- what this step does NOT do

- **No repair-retry loop is built.** If one is later proposed it must be
  attempt-capped (the literature records death-loops and a rising token curve),
  consistent with the F1b doctrine. Not in scope here.
- **No risk default is flipped**, including the fabricated
  `APPROVE_REDUCED at 3%`. Filed as its own queued defect.
- **No flag promoted, no `.env` written** (criterion 5); operator-gated changes
  are numbered asks.
- **The census is blind after 2026-08-14T15:53Z** -- no live uncompressed
  `backend.log` was in the researcher's corpus. If a CURRENT rate is wanted,
  locating the live log is a prerequisite, not an assumption. (Main notes the
  live log DOES exist at repo root `backend.log`, 28MB, actively written --
  GENERATE should include it and say what it adds.)
- **Overlap with 86.60 is stated, not silently shared:** 86.60's news-screen
  empty-response diagnosis is the same failure class at a different site. This
  step owns the pipeline-wide marking; 86.60 owns the news-screen entry path.
  Neither may claim the other's fix.

## References

`research_brief_86.108.md` (the four census corrections, the C1 impossibility
finding, the four emit sites, the three guarantee classes, the settings
enumeration); `research_brief_86.69.md` (the origin of the 2,859 figure);
`research_brief_86.60.md` (the adjacent empty-response case);
`experiment_results_86.69.md` (the phase-61.2 integrity machinery this extends).
