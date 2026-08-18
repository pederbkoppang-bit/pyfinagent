STATUS: INCOMPLETE -- not a verdict
STEP: 86.84
WRITTEN: 2026-08-17T10:21:42Z

# Q/A write-first record -- step 86.84, cycle 6 (per spawn prompt: attempt_number 6)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: `python3 scripts/qa/rail_turn_cap.py --verify` exit code; git status scope; ruff; syntax
C. LLM judgment vs the 8 immutable criteria read VERBATIM from .claude/masterplan.json
D. Independent mutation testing of the NEW guards (cycle-6 cross-derivation additions)

## Findings log (appended as established)
- [init] qa.md read in full at 2026-08-17T10:21Z. Frontmatter now shows `maxTurns` REMOVED
  with an explicit "DELIBERATE REMOVAL, NOT AN OMISSION" comment block (lines 6-8, 17-52).
  That is the artifact under test for this step.

## B1. IMMUTABLE COMMAND -- `python3 scripts/qa/rail_turn_cap.py --verify`
EXIT=0. `VERIFY: PASS -- controls green, turn-exhaustion claim holds.`
Key figures RE-DERIVED BY ME (my own run, 2026-08-17T10:2xZ):
  run records 587 / agent spawns 1285 / transcripts missing 0
  qa cap=30 n=342 drop=39 @cap=39 >cap=0 | researcher cap=40 n=107 drop=9 @cap=9 >cap=0
  uncapped: Explore 0/277, None 0/415, general-purpose 0/144
  dropped turn sets: qa=[30], researcher=[40]
  C1 1285 turns>0, 0 zero-with-lines; C2 0 exceed cap; detector 1217/1227 vs 1/48;
  C3 killed runs at [1,1,2,2,2,3,4,5,6,16], 0 at a cap
  AT-RISK ratio 0/90 uncapped vs 12.2% capped
  at-a-cap 57, of which 49 never emitted; 2 inside COMPLETED runs (wf_078f4125-57a, wf_a6ea31e7-9b9)
  REMEDIATION: caps removed 2026-08-14T17:37:50Z (85127353); first uncapped run 2026-08-14T19:35:25.339Z
  REALISED UNCAPPED: qa n=40 dropped=0 non-emitters=0 p50=40 p90=55 max=60 >old-cap(30)=36
                     researcher n=14 dropped=0 non-emitters=0 p50=19 p90=35 max=38 >old-cap(40)=0
  live now {'qa': None, 'researcher': None}; all pins removed True; cap parser: fallback
DISAGREEMENT vs the artifacts (corpus-grows-daily, DISCLOSED by them): cycle-4 quoted
qa n=36/p90=54/past=32; cycle-5 n=38/p90=55/past=34; I measure n=40/p90=55/past=36.
Monotone growth, consistent. NOTE ALSO: total spawns FELL 1325 -> 1285 and
general-purpose fell 252 -> 144 while qa rose 302 -> 342: the corpus both grows AND
loses old session dirs. The artifacts disclose growth but not shrinkage. TO CHECK.

## B2. PRIOR-ATTEMPT / VERDICT EVIDENCE
qa_wip.py 86.84 --spawned-at 2026-08-17T10:21:42Z:
  source_present: True, attempt_number: 6 (status ok, is_lower_bound True),
  prior_attempts: 5, records_retained: 6 (gauge), identity_checked: True
verdict_history_86_21.py --step 86.84 --evidence-only:
  status ok; 5 verdicts: CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL
CROSS-CHECK: prior_attempts 5 == ledger 5 rows. Sources AGREE; no staleness signal.
(Cycle-4 GENERATE §5 said the ledger returned [] i.e. stale; cycle-5 §6 backfilled 4
rows + wrote cycle-4's. That is now consistent with qa_wip.)
