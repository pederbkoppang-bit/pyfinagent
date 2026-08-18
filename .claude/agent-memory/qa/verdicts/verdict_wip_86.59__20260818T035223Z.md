STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.59
WRITTEN: 2026-08-18T03:52:23Z

# Q/A write-first record -- step 86.59

## Attempt / sequence evidence
- `qa_wip.py 86.59 --spawned-at 2026-08-18T03:52:23Z`: source_present=true,
  attempt_number=3 (status ok, is_lower_bound true), prior_attempts=2,
  records_retained=3 (gauge), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.59 --evidence-only`: status=ok,
  "2 verdict(s) from the ledger", verdicts = CONDITIONAL -> CONDITIONAL.
- Cross-check: prior_attempts (2) == ledger rows (2) -> ledger NOT stale.
  evaluator_critique_86.59.md's own ledger table agrees (wf_5a3bc88c-4e1,
  wf_d1d01d57-0f6). Any threshold over that sequence is the caller's, not mine.

## A. Harness compliance -- 5/5 CLEAN
1. research-gate-before-contract: research_brief_86.59_rerun.md, envelope
   gate_passed: true, 8 sources read in full (>=5), 54 URLs (>=10), recency scan
   performed, brief_status COMPLETE.
2. contract-before-generate: brief 08-14 02:43 < contract 08-17 20:56 < scripts
   08-18 05:41/05:49 < experiment_results 05:51. All 7 criteria byte-verbatim in
   contract (checked programmatically against masterplan).
3. experiment_results_86.59.md + live_check_86.59.md + evaluator_critique present.
4. log-last: masterplan 86.59 status="pending"; the ONLY
   `phase=86.59 result=` row in harness_log is the 2026-08-12 GATE-FAILED
   research-gate row (line 34347). No EVALUATE result row for this cycle.
5. no-verdict-shopping: evidence CHANGED (commits fb6f8a67, a4a5765c; both
   scripts + all three artifacts modified since cycle 2).

## B. Deterministic
- IMMUTABLE COMMAND: ast.parse(backend/tools/screener.py) -> "parses", exit=0.
- Step commits 15a817cc / 3e75c2d6 / fb6f8a67 / a4a5765c: ZERO files under
  backend/ or frontend/, no .env, no settings.py. Only masterplan.json (adds
  86.116 + 86.117; the one edit to an existing entry is 86.117's OWN audit_basis
  sigma correction -- 86.59's criteria untouched), handoff artifacts, and the two
  scripts/qa/ files. I checked all FOUR commits, not the two the artifact cites.
- ruff F821,F401,F811 over a DERIVED scope (git diff HEAD + 15a817cc~1..HEAD +
  untracked, 4 files, non-empty asserted): "All checks passed!", exit=0.
- sha256(rank_stability_86_59.py)=be0565ff3c9615da == the sha in live_check s8.
  AST census: 23 guards, coverage ok=True, uncovered=[], 21 cells + 1 negative
  control = 22. Matches the published "23 guards / 22 cells" exactly.
- Peer-session working-tree edits to backend/services/autonomous_loop.py are at
  ~:3638 (_persist_analysis); _min_k_sector_slice is at :176 and is NOT in that
  diff -- the min_k arm is not contaminated by the shared tree. CHECKED, CLEAR.
- backend/tools/screener.py contains 0 `settings.` references -- Main's
  blast-radius mitigation reproduces.
- ASK-3's "-0.166 long-only Sharpe" traces to phase-51.2's 2026-06-01 replay
  (harness_log:26104, live_check_51.2.md:33). Borrowed number CORRECTLY cited.
- HEAD 8b945814 (2026-08-18 03:51:42Z) unchanged across the whole evaluation;
  scripts/qa/ clean, 0 MUTANT markers, md5 identical pre- and post-matrix.

## C. Reproduction of EVERY published number -- ALL EXACT
- `--cycles 20`: rho 0.9622 mean / 0.9319 min; top-10 turnover 15.8%/day; top-5
  15.8%/day; 3 of 19 zero-turnover; 12 distinct {DD,DDOG,DELL,DVA,FTNT,HPE,HPQ,
  HUM,MU,PANW,SNDK,ZBRA}; IT 72.0%; counts {Industrials 20, IT 72, Health Care 8};
  fidelity 80%; 18 live distinct; dedup 47,880/200,875 (23.8%); 10 split bars;
  507 screened; "OK: all 72 invariants hold", exit=0.
- `--flags --cycles 20`: 15.8 / 28.4 / 22.1 / 17.9; distinct 12/22/17/14;
  IT 72% / Industrials 20% / IT 40% / IT 60%; deltas +12.6 / +6.3 / +2.1pp.
- `--dispersion --cycles 20`: sigmas 10.646 / 19.850 / 30.441, ratio 2.86x;
  effective shares 22.6 / 37.0 / 40.4; gaps -17.4 / +2.0 / +15.4pp;
  50 of 10139 (0.493%), 5 cycles exactly identical.
- `mutation_86_59.py`: control --verify/--dispersion/--flags all rc=0 GREEN
  FIRST; coverage 23 guards / 23 covered; KILLED 22/22, SURVIVED 0, UNSCORABLE 0;
  restore verified sha256 unchanged (be0565ff3c9615da). Every s8 line reproduces.
NOTHING failed to reproduce.

## D. *** SURVIVING MUTANTS -- CAPPING FINDING ***

Mutations applied IN MEMORY ONLY (source exec'd with `__file__` pinned to the
real path so `REPO = parents[2]` resolves); the file on disk was never written.
CONTROL (null mutant through the same exec path) observed GREEN FIRST and
reproduced the published table to the digit -- so these measure the SUBJECT.

**THE STRUCTURAL POINT: `delta = arm - baseline`. The cycle-3 fix guards the
`base` call at :806, which feeds ONLY the `min_k_sectors=3` ARM. The row every
delta is subtracted FROM -- `arms["baseline"]`, produced by the separate
`replay_session(...)` inside `for name, kw in FLAG_ARMS` at :799-802 -- has NO
behavioural guard at all. Two structurally identical call sites; one guarded.**

### A1 -- tilt injected into the FLAG_ARMS loop only (:799-802), `base` untouched
`momentum_52wh_tilt=True, momentum_52wh_tilt_k=0.2` added to the arms-loop call.
RESULT: **SURVIVED**. All six criterion-4 guards ran, none fired.

| row | control (published) | A1 mutant |
|---|---|---|
| baseline | 15.8% / 12 / IT 72% | 20.0% / 13 / IT 70% |
| sector_neutral | 28.4% / 22 / Industrials 20% | 46.3% / 29 / **Financials 15%** |
| soft_diversity_w0.30 | 22.1% / 17 / IT 40% | 23.2% / 17 / IT 39% |
| min_k_sectors=3 | 17.9% / 14 / IT 60% | 17.9% / 14 / IT 60% (arm itself CLEAN) |
| DELTA sector_neutral | +12.6pp | +26.3pp |
| DELTA soft_diversity | **+6.3pp** (ASK-2) | **+3.2pp** |
| DELTA min_k | **+2.1pp** (ASK-1) | **-2.1pp** SIGN FLIP |

The min_k ARM is provably clean under A1 and its reported DELTA is still wrong,
because the corruption is in the shared reference. That is a stronger version of
the cycle-2 finding, not a repeat of it.

### A2 -- soft_sector_diversity w=0.05 into the FLAG_ARMS loop only
RESULT: **SURVIVED**. All six guards green. Every turnover delta reads EXACTLY
as published (+12.6 / +6.3 / +2.1pp) while the baseline's top-sector share moves
**0.72 -> 0.64** and the share deltas move -52.0->-44.0, -32.0->-24.0,
-12.0->-4.0. This is verbatim the "worse than the sign flip" variant Main's own
cycle-3 narrative says was closed -- same w, same 0.72->0.64 -- reproduced one
seam over.

### FALSIFIED CLAIM
`scripts/qa/rank_stability_86_59.py:816-818` (code comment) and
`live_check_86.59.md` s8 / experiment_results cycle-3 both assert:
"An injection anywhere in the replay path -- at the seam, in the kwargs, in a
wrapper -- makes these diverge."
A1 and A2 are injections at the replay seam and they do NOT make them diverge.
The claim's scope exceeds the guard's scope.

### NAMED FIX (small, and the mechanism already exists)
Capture the arms-loop baseline slate per cycle and require it to agree with the
same direct unflagged `rank_candidates` call already computed at :821-828
(compare `arms["baseline"][-1]` against `_direct[:ANALYZE_TOP_N]`), add a cell
that injects a kwarg at :799-802, and narrow the "anywhere in the replay path"
sentence to what the guard actually covers.

### Secondary (NOTE): the "direct" oracle is not fully independent
The direct unflagged call at :821-828 re-uses `build_yf_frame`, `tickers`,
`sectors` and `SCREEN_TOP_N` from the same module, so it cannot detect a
corruption of the frame builder itself. Not exercised further; recorded as a
bound on what M20 licenses.

## E. Scope / plan reconciliation (NOTE-level, non-blocking)
contract_86.59.md's PLAN commits to **P3** ("Standardise cross-sectionally at
screener.py:301-305 using the existing `_zscore`, behind a new default-OFF
flag"), **P4** (criterion-2 DSR/PBO on that change) and **P6** (parity with the
new flag OFF). None was built; the fix is filed as 86.117 BLOCKED-BY 86.116.
The reason given is measured and good (DSR/PBO read a table with 38% duplicate
keys), and 86.116/86.117 were filed rather than absorbed -- the documented
pattern. BUT experiment_results frames this as "That is the deliverable the
criteria describe", a claim about the CRITERIA that steps over the CONTRACT's
own plan; the contract was never reconciled. Disclosure gap, not a falsehood.

## F. Criterion-by-criterion
1. MET -- rho + top-10 turnover measured over 20 consecutive stored sessions
   with the command published; independently reproduced.
2. MET (vacuous, conditional criterion) -- no new/reweighted term shipped, so
   nothing to justify OOS. Disclosed by the author, not exploited. I re-judged
   this freely rather than inheriting cycle 2's ruling and reach the same result:
   criterion 4 mandates measuring the existing flags BEFORE new code, all three
   MOVE the slate, and criterion 2's DSR/PBO demand reads a table this step
   measured at 38% duplicate keys. The obligation is carried into 86.117.
3. MET -- N=20 stated, 12 distinct / 100 slots, IT 72.0%, live "before" 18
   distinct from analysis_results. Reproduced.
4. MET on substance -- all three flags measured and all three MOVE the slate;
   figures reproduced exactly. Its GUARD coverage is the section-D finding.
5. MET -- no .env, no settings.py, no flag promoted in any of the 4 commits;
   4 numbered asks recorded.
6. MET degenerately, with a real oracle -- no new behaviour, and the
   zero-production-file property is checkable (`git show --name-only` on all
   four commits, which I ran). Disclosed rather than claimed.
7. MET literally -- 23 guards, 22 cells + AST coverage gate, control GREEN
   first on all three modes, SHA-256 restore verified, 22/22 killed, reproduced
   by me end to end.

## Log
- [03:52Z] WIP created, qa.md read in full.
- [03:54Z] immutable cmd exit=0; flags run reproduces published table exactly.
- [03:58Z] A1 SURVIVOR established (control green first).
- [04:00Z] A2 SURVIVOR established (0.72->0.64, deltas unchanged).
- [04:02Z] criteria 1+3 and dispersion runs reproduce exactly; ruff clean.
- [04:07Z] mutation matrix reproduces 22/22, control green first, sha restore.
- [04:09Z] HEAD + tree recheck clean; record finalised.

## Verdict returned
CONDITIONAL, ok=false. No criterion is missed and no published number fails to
reproduce, but the criterion-4 baseline reference -- the row all three operator-
ask deltas are computed against -- still has no behavioural guard, and the
artifact claims coverage it does not have.

COMPLETED: 2026-08-18T04:09:31Z
