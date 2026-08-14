# evaluator_critique — step 86.84

## 0. Verdict — TRANSCRIBED VERBATIM, cycle 1

Q/A launched via the **Agent-tool `qa` fallback** (operator instruction
2026-08-14: rail 0-for-4, Agent-tool 3-for-3 — and the Workflow rail is the
subject under repair). Scope as tasked: **the DIAGNOSIS ONLY**. Step stays
`pending`; no fix applied.

**VERDICT: CONDITIONAL.** *"Diagnosis sound and mutation-resistant; F1+F2 are a
real overclaim in an artifact whose own standard is full re-runnability, and F2
sits in the audit_basis a future executor is told to re-derive against."*

Full write-first record, verbatim, at
`.claude/agent-memory/qa/verdicts/verdict_wip_86.84__20260814T170906Z.md`
(6,488 bytes, COMPLETED 2026-08-14T17:23:11Z). Reproduced here in full:

```
HEAD at spawn: 6dcc56df / 577adcdf (day report) / c1797888 (the diagnosis).
Sequence: attempt_number 1 (source_present true, identity_checked true);
ledger no_rows_for_step (0). Consistent -- first attempt, no staleness signal.

## 1. Reproduction
`python3 scripts/qa/rail_turn_cap.py --verify` -> EXIT 0, 4.2s.
Table matches the docstring's stated figures exactly: 572 records / 1325 spawns /
0 missing; qa cap30 n302 drop39 @cap39 >cap0; researcher cap40 n93 drop9 @cap9
>cap0; Explore 0/263, None 0/414, general-purpose 0/252, claude-code-guide 0/1.
Model x type cells reproduce (opus-5[1m] qa 39/290, researcher 8/89, uncapped
0/417; opus-4-8[1m] 258 spawns of which gp 223).

## 2. Probe soundness
2a count_turns: 0 of 23,178 assistant lines across all 395 capped transcripts
   lack a requestId; min assistant_lines/turns ratio 1.0; 0 spawns with
   assistant lines and 0 turns. No undercount hole. Over-count would show as
   >cap and does not (0/395).
2b parse_cap: returns qa=30, researcher=40, general-purpose=None, None=None.
   Frontmatter-only by construction (re.match anchored at pos 0, only group(1)
   scanned). qa.md:596 body prose "your real bound is maxTurns" carries no
   `: <digits>` so it cannot match even unrestricted. Files are LF, no CRLF
   hazard. Failure mode is LOUD, not silent -- see mutant M3.
2c drop predicate: `status = rec.get("status")` (:220) / `status ==
   FAILED_STATUS` (:237). Named field, no blob scan. f88f8190 trap not reopened.
   Independently: all 46 failed runs carry the StructuredOutput error string in
   `error` -- the named-status predicate is a clean 46/46 proxy for THIS error.
2d MUTATION MATRIX (in-memory via importlib, module-attr patch; zero tree
   writes -- git diff --stat empty, git status clean, md5 3a3b5763... unchanged)
   CONTROL unmutated              -> verify_ok=True   (observed GREEN first)
   M1 parse_cap -> cap+1          -> KILLED (claim broken, @cap 39->0, 9->0)
   M2 count_turns -> const 7      -> KILLED (claim broken)
   M2b count_turns -> const 30    -> KILLED via the researcher row only
       (adversarial: matches the qa cap exactly; the two-role corpus is what
        kills it -- a single-role corpus would let this one survive)
   M3 parse_cap -> None for qa    -> KILLED twice (claim broken + 39 uncapped
       drops) = the 2b failure mode fails loud
   0 survivors.

## 3. Controls
C1 non-vacuous in the direction checked but weak alone (all-1s would pass);
   closed by the requestId census + M2.
C2 genuinely independent as the docstring claims: computed over the 347
   not-dropped capped spawns, about which the hypothesis is silent; detects
   counter INFLATION. Stated limit: cannot detect UNDER-counting (closed
   separately above).
Detector control real: 1257/1277 vs 1/48.
NON-TRIVIALITY: only 6/263 (2.3%) completed qa and 3/84 (3.6%) completed
   researcher sit at cap, vs 100% of drops. Not saturation.
FREE NEGATIVE CONTROL the author did not claim: the 6 `killed` runs
   ("Workflow aborted") land at 6/3/5/4/16/2/2/1/1 turns -- nowhere near a cap,
   exactly what non-exhaustion terminations should look like.

## 4. Findings (all in claim accounting, none overturn the conclusion)
F1 OVERGENERALIZATION live_check:4 "Re-runnable form of every number below:
   rail_turn_cap.py --verify". FALSE for the SS0 tail-shape numbers and the
   quoted 11.4/3.0/0.0 -- the script computes NO tail-shape figure (grepped).
F2 CONTRADICTION "393 of 394 successful qa/researcher transcripts end on a
   tool_result" does not reproduce. Measured: 347 completed qa/researcher
   spawns, 347/347 end on tool_result. Denominator and the "1 exception" both
   unreproducible. Propagated to masterplan audit_basis and to the day report
   with the "qa/researcher" qualifier DROPPED (broader, more wrong). Direction
   is AGAINST the author's own case.
F3 OVERGENERALIZATION the drop tool-breakdown lists 47 of 48 (Bash 37, Edit 4,
   Write 2, Read 2, WebFetch 1, WebSearch 1) and omits the 48th, whose last
   tool_use WAS StructuredOutput (wf_d4e2e794-567) -- the counterexample to
   that sentence's own contrast. Disclosed elsewhere as "1 of 48".
F4 MISSING_ASSUMPTION a third run status exists (`killed`, 6 runs / 10 spawns)
   and is bucketed as "completed"/"ok". `not dropped` != completed.
   Contaminates the 1277 denominator and the ok p50/max/@cap columns. Does not
   affect the claim; no killed run carries the drop error.
F5 MISSING_ASSUMPTION agent-level exhaustion exists OUTSIDE the 48: two
   COMPLETED research-gate runs (wf_a6ea31e7-9b9, wf_078f4125-57a) each contain
   a researcher spawn at exactly 40/40 that never emitted StructuredOutput --
   the 86.81 retry absorbing an exhaustion. The at-cap non-emitter population is
   50, not 48. STRENGTHENS the mechanism; undisclosed.
NOTE-A "0 drops in 930 uncapped" is true but only 50 of 930 ever exceeded 30
   turns (25 exceeded 40). At-risk subset is 50, not 930. Still strong (0/50 vs
   a 12.2% capped rate) -- just not 930-strong.
NOTE-B the 930 is attributed to three named types summing to 929; the 930th is
   claude-code-guide (1), unnamed. Off by one in attribution, not in total.

## 5. Scope (git show --stat)
c1797888: masterplan.json + live_check_86.84.md + rail_turn_cap.py (new). ONLY.
577adcdf: day_report + goal_next + research_brief_86.84.md.
No .claude/agents/*.md, no cap change, no gate change. None of the peer
session's files (backend/api/sovereign_api.py, frontend/src/*) swept in.

## 6. Honesty positives
Necessary-not-sufficient stated; right-censoring stated and used to REFUSE to
size a cap; the graceful-degradation corollary RETRACTED (struck + replacement
claim) and carried retracted into the day report -- no un-retracted survivor
found by sweep; brief disclosed as INCOMPLETE / 7 sources / gate_passed false
while its findings are used; "no Q/A verdict obtained" stated; stale claims in
rail_drop_rate.py + both workflow files disclosed as deliberately not edited.

VERDICT RETURNED: CONDITIONAL. Diagnosis sound and mutation-resistant; F1+F2
are a real overclaim in an artifact whose own standard is full re-runnability,
and F2 sits in the audit_basis a future executor is told to re-derive against.
```

## 1. Main's response — F2 is a real error and I have found its cause

**F2 is correct and the error is mine.** My "393 of 394" came from a
mis-scoped enumeration: my ad-hoc script selected *runs* containing a
qa/researcher agent and then globbed **every** `agent-*.jsonl` in that run
directory — so it swept in the stage-2 `Explore` spawns that `research-gate.js`
launches alongside the researcher. The Q/A's 347/347 counts qa/researcher
spawns only and is the correct figure. The "1 exception" was one of those other
agents, not a qa/researcher spawn.

This matters more than the arithmetic: the sentence was load-bearing in my
argument (*"the tail shape is not diagnostic, because successes end the same
way"*). **The corrected number makes that argument stronger, not weaker** —
347/347, with no exception at all. F2's own note that the direction is *against*
my case is accurate, and is exactly why the correction is not optional.

**F5 and the free negative control are gifts I did not earn.** The at-cap
non-emitter population is **50, not 48** — two exhaustions were absorbed by the
86.81 retry inside runs that ultimately completed. And the 6 `killed` runs land
at 1–16 turns, nowhere near any cap, which is precisely what a non-exhaustion
termination should look like. Both strengthen the mechanism.

**NOTE-A is the correction I most needed.** "0 drops in 930 uncapped" is
rhetorically inflated: only **50** of those 930 spawns ever exceeded 30 turns
and only 25 exceeded 40, so the at-risk denominator is 50. The honest form is
**0/50 against a 12.2% capped rate** — still decisive, not 930-strong. I will
not restate the 930 figure without that qualifier.

## 2. Blockers to clear before a fresh Q/A

Per the canonical cycle-2 flow, a fresh Q/A may only be spawned after these are
fixed **and the handoff files updated** — not on unchanged evidence.

1. **F2** — replace "393 of 394" with **347/347 completed qa/researcher spawns**
   in `live_check_86.84.md` §0, the masterplan `86.84` audit_basis, and the day
   report (where the `qa/researcher` qualifier was additionally dropped).
2. **F1** — narrow the `live_check.md:4` claim: `rail_turn_cap.py --verify` is
   the re-runnable form of the **table and the cross-tab**, not of the
   tail-shape figures or the 11.4/3.0/0.0 rates, which come from
   `rail_drop_rate.py` and an ad-hoc scan.
3. **F3** — state the tool breakdown as 47 of 48 and name the 48th
   (`wf_d4e2e794-567`, whose last `tool_use` *was* `StructuredOutput`) at the
   point of the contrast, not only in the separate "1 of 48" disclosure.
4. **NOTE-A / NOTE-B** — requalify 0/930 as 0/50 at-risk; attribute the 930th
   spawn (`claude-code-guide`).
5. **F4** — `killed` is a third status; stop bucketing it as "completed" in the
   script's `ok*` columns and the 1277 denominator.
6. **F5** — disclose the 50-vs-48 at-cap non-emitter population.

## 3. Not done before freeze

**None of the above is fixed yet**, and no fresh Q/A has been spawned. The
verdict landed at 17:23:11Z against a 19:30 local freeze. 86.84 stays
`pending`; this is cycle 1 of 1, CONDITIONAL, no escalation pressure.

**Disclosure of a freeze-the-tree breach by me:** I edited
`live_check_86.84.md` and the day report at ~17:10Z to land the
graceful-degradation retraction — *after* spawning this Q/A at 17:09:06Z. The
Q/A recorded HEAD `6dcc56df / 577adcdf / c1797888` at spawn and its §6 reads the
retraction as present, so it appears to have picked up the newer tree; but its
verdict should be read against what it recorded, not against `ddc08396`. The
gap noticed mid-evaluation belonged in the next cycle and I put it into the tree
being graded.
