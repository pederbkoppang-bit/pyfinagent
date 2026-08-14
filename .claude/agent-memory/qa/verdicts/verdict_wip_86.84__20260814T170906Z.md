STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-14T17:09:06Z

SCOPE AS TASKED: evaluate the DIAGNOSIS ONLY of 86.84. Step stays pending; no
fix applied. Grade soundness of the probe + honesty of the artifacts.

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

COMPLETED: 2026-08-14T17:23:11Z
