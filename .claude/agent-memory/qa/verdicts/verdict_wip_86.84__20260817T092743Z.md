STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T09:27:43Z

# Q/A cycle 4

## A. Harness compliance -- CLEAN
- research_brief 2026-08-14T17:15Z < contract 17:17Z < experiment_results 2026-08-17T09:26Z. OK
- masterplan 86.84 status = "pending" (not flipped). LOG-last OK.
- spawn-prompt criteria == masterplan success_criteria verbatim.
- evidence CHANGED since cycle 3 (rail_turn_cap.py +71/-5, uncommitted, mtime 09:23:38Z) ->
  fresh-respawn on changed evidence, NOT verdict-shopping.
- qa_wip.py --spawned-at: source_present true, attempt_number 4 (status ok), prior_attempts 3,
  3 prior records 2026-08-14 17:09:06Z / 17:38:24Z / 18:01:25Z.
- verdict_history_86_21.py --evidence-only: status **no_rows_for_step**. attempt_number(4) >
  ledger count(0) => **LEDGER IS STALE**; sequence not establishable from the ledger.
  Main's advisory [C,C,C] corroborated by three verbatim transcriptions at
  evaluator_critique_86.84.md:10, :225, :524. Carried as reported, not adopted as established.

## B. Deterministic -- ALL GREEN
- `python3 scripts/qa/rail_turn_cap.py --verify` EXIT 0 on BOTH interpreters:
  .venv python3 -> "cap parser used : yaml (PyYAML)"      VERIFY: PASS
  /usr/bin/python3 -> "cap parser used : fallback"         VERIFY: PASS
- D1 REPRODUCES: `git show HEAD:...` run unmodified -> "VERIFY: FAIL / - no agent type carries
  a maxTurns cap; nothing to test". Fixed version PASSes. Real defect, real fix.
- INDEPENDENT re-derivation from raw transcripts reproduces the committed re-measurement EXACTLY:
  qa n=36 dropped=0 non_emitters=0 p50=40 p90=54 max=60 >30=32 (>40=17)
  researcher n=11 dropped=0 non_emitters=0 p50=19 p90=36 max=38 >40=0
  6 post-removal sessions of 41; model on all 47 = claude-opus-5[1m].
- Totals reconcile: uncapped 930->880 = 833 other-role + 47 post-removal;
  at-risk 50->86 = 50 + 32 qa + 4 researcher. Arithmetic internally consistent.
- ruff F821,F401,F811 on DERIVED non-empty scope (sovereign_api.py, rail_turn_cap.py) -> exit 0
- tsc --noEmit exit 0 ; eslint src/ exit 0 (0 errors, 55 warnings)
- backend import backend.api.sovereign_api OK ; /api/health 200 ; red-line?window=1y 200
- mutate_rail_turn_cap.py --verify exit 0; control enforced (harness returns 1 if red);
  byte-identical restore 3 md5s; killed = (not ok) AND problems -> no wrong-reason kills.
- verify_rail_retry.mjs exit 0 (38/38) ; verify_research_gate_workflow.mjs exit 0 (124)

## C. Criteria
1 MET  2 MET  3 MET  4 MET  5 MET  6 MET  7 MET  8 PARTIAL/NOT MET (capping)

## FINDING 1 (WARN, illusory-guard) -- cycle-4 re-measurement is UNGUARDED
verify() has NO assertion over remediation.post_removal_turns. Scratchpad-copy mutations,
repo md5 unchanged before/after (baed6162861ff2d1265eacc40370fb2a), CONTROL observed GREEN first:
  QM1 _q() always 0                        SURVIVED rc=0
  QM2 roles renamed -> n=0, block vanishes SURVIVED rc=0
  QM3 past_old_cap INVERTED (32->2)        SURVIVED rc=0
  QM4 max_turns := min (60->26)            SURVIVED rc=0
  QM5 dropped/non_emitters hardcoded 0     SURVIVED rc=0
  QM7 one post-removal non-emitter         SURVIVED rc=0
  QM8 ALL 36 non-emitters ("fix did nothing") SURVIVED rc=0, still prints VERIFY: PASS
FAIRNESS -- named coexisting behavioural guard:
  QM6 one post-removal qa spawn dropped -> KILLED rc=1 by THREE assertions
      (uncapped_at_risk_drops, every_drop_is_at_its_cap, drops_on_uncapped_types).
So the DROP axis is genuinely guarded; the NON-EMITTER axis -- the one the 86.81 retry can
mask, as the script itself documents for the historical corpus -- is not.
NAMED FIX: floor on post_removal_turns[*].non_emitters == 0 plus a cardinality floor on n.

## FINDING 2 (capping, criterion 8) -- no cell covers the cycle-4 code
All 22 cells target parse_cap shapes, CAP_EDIT_AT/HISTORICAL_CAPS/session_is_post_removal, or
file deletion. ZERO cells over analyse()'s cycle-4 additions. D1 has an executed before/after
(reproduced) but no cell.

## FINDING 3 (WARN) -- M14 annotation stale; survivor count is a LABEL count
M14 annotated "EQUIVALENT. Every run in the corpus already precedes the boundary" -- false:
59 spawns post-date it and M14 KILLED in my run (C2 FAILED: 32 capped spawns exceed cap).
expected_survivors counts cells with expect != "KILL", so "known/equivalent survivors=3" prints
while only M6+M6b survived.

## FINDING 4 (WARN) -- superseded prose sitting BESIDE its correction
- live_check §4b "HONEST STATUS: NOT YET A VERIFICATION ... n = 2 ... none has occurred yet"
  -- false since the n=47 re-measurement; unmarked.
- live_check 2026-08-15 section: "does not discharge ... NO VERDICT ... was not run today"
  -- discharged; unmarked.
- live_check ~165/~173 still name retired constant CAP_REMOVED_AT as if live (code has
  CAP_EDIT_AT; the old name survives only inside a comment explaining its retirement).
- experiment_results lines 28-66 "Verification command output, verbatim": undated, unmarked,
  does NOT reproduce (qa 302 vs 338; uncapped 0/930 vs 0/880; at-risk 0/50 vs 0/86;
  detector 1257/1267 vs 1207/1217; "caps removed at 2026-08-15T00:00:00Z" = the calendar
  constant F-E replaced).
Same defect class this step's own criterion 5 forbids at three OTHER sources.

## FINDING 5 (NOTE, for Main at commit time) -- unrelated dirty tree
6 unrelated working-tree files (1y-window feature: backend/api/sovereign_api.py,
frontend/src/app/page.tsx + 4 components, backend/services/experiments/perf_results.tsv)
would ship under a phase-86.84 `git add -A` auto-commit. Disclosed by the cycle-2 Q/A but not
restated in cycle-4 artifacts.

## NOTE -- consequence framing present in the evidence
evaluator_critique §8 and live_check §10 both state which verdict outcomes are "admissible".
That is consequence framing inside the material I grade (the channel phase-86.78 closed on
tool output). Disregarded: I graded on merits; escalation is the caller's to compute.
I did NOT self-apply any escalation rule -- qa.md binds me to report the sequence as evidence.

COMPLETED: 2026-08-17T09:52:11Z
