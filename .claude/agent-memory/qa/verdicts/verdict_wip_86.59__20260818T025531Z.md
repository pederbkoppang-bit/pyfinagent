STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.59
WRITTEN: 2026-08-18T02:55:31Z

# Q/A write-first record for step 86.59

## A. Harness compliance (5 items) -- ALL CLEAN
1. research-gate-before-contract: `research_gate_86.59_rerun_verdict.md` = PASSED.
   sources 8>=5, urls 54>=10 corroborated against the brief, brief_status COMPLETE,
   recency section present, self_report_disagreed=false. Prior gate (2026-08-12) FAILED
   and PLAN was correctly NOT entered -- harness_log Cycle 1226 records
   `phase=86.59 result=GATE-FAILED (research gate, no PLAN entered)`.
2. contract-before-generate (mtimes, local): research_brief 08-14 02:43 <
   gate verdict 08-14 04:28 < contract 08-17 20:56 < mutation_86_59.py 08-18 04:51 <
   rank_stability 04:52 < live_check 04:53 < experiment_results 04:54. ORDER OK.
3. experiment_results_86.59.md present (9,802 B).
4. log-last: masterplan 86.59 status == "pending"; no result=PASS/CONDITIONAL/FAIL row
   for this cycle in harness_log. OK.
5. no-verdict-shopping: qa_wip.py --spawned-at -> source_present=true,
   attempt_number=1, prior_attempts=0, prior_records=[].
   verdict_history_86_21.py --evidence-only -> status=no_rows_for_step, verdicts=(none).
   prior_attempts(0) == ledger rows(0) -> no staleness signal. First spawn.

## B. Deterministic

- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import ast;
  ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'`
  -> stdout `parses`, EXIT=0.
- SCOPE: step files are ALREADY COMMITTED at 15a817cc (`phase-86.59: measure the
  picker rank stability; file 86.116 + 86.117`). HEAD=f9bcd3bf.
  `git show --name-only 15a817cc` = masterplan.json + experiment_results_86.59.md +
  live_check_86.59.md + mutation_86_59.py + rank_stability_86_59.py.
  `git show --name-only 15a817cc | grep -E '^(backend|frontend)/'` -> NONE.
  => the "zero production files modified" claim REPRODUCES at commit level.
- CRITERIA IMMUTABILITY: re-parsed masterplan at 15a817cc^ vs 15a817cc ->
  86.59 verification block IDENTICAL (True), status pending->pending.
  NEW ids added: 86.116, 86.117 (permitted: queue-discovered-defects rule).
- LINT (derived scope = uncommitted *.py U untracked *.py U commit-15a817cc *.py,
  4 files, piped via xargs): `uvx ruff check --select F821,F401,F811` ->
  "All checks passed!" exit=0. Non-empty scope asserted first.
- MUTATION MATRIX re-run BY ME (`python scripts/qa/mutation_86_59.py`, exit 0):
  control --verify/--dispersion/--flags all rc=0 GREEN FIRST;
  KILLED 14 / 14, SURVIVED 0, UNSCORABLE 0; restore sha256 unchanged
  (c2f7982efba00a40...); `git status --short -- scripts/qa/` clean afterwards.
  REPRODUCES the claim exactly.
- MEASUREMENT re-run BY ME (`--cycles 20`, exit 0): rho 0.9622, turnover 15.8%/day,
  12 distinct analysed names / 100 slots, 18 distinct LIVE tickers,
  47,880 of 200,875 (23.8%) duplicate rows dropped, 10 split-shaped bars,
  "OK: all 70 invariants hold". REPRODUCES.
- PANEL PROVENANCE verified independently from the pickle: fetch range
  2025-06-01..2026-08-17, 200,875 rows, 513 tickers, dates 2025-06-02..2026-08-17,
  304 sessions, replay window (126, 20) = 2026-07-21..2026-08-17 = 20 CONSECUTIVE
  sessions ending 2026-08-17 as claimed. Excess rows recomputed by me: 47,880 = 23.8%.
- SECTOR MAP verified: 502/513 carry a GICS sector (claim reproduces).
  Analysed-12 split: 9 IT, 2 Health Care (DVA,HUM), 1 Industrials (DD) ->
  consistent with the reported {IT 72, Industrials 20, Health Care 8} = 100 slots.

## C. FINDING 1 (vacuity, WARN) -- a LITERAL-True guard survives in the shipped file

AST-proven (not grepped): of 19 `_ok(...)` invariants in
`scripts/qa/rank_stability_86_59.py`, exactly ONE has a constant-`True` condition:

    L337  _ok("panel_is_us_only", True, "market='US' filter applied at fetch")

It cannot fail on any input, is counted among the "all 70 invariants hold" total,
and is NOT in the mutation matrix (no cell can kill it -- shape #4, tautology).
This is the same class the step itself reports having found and fixed
(`slate_is_a_prefix_of_the_full_ranking`), left in place one screen above the fix.

MITIGATION MEASURED, so this is WARN not BLOCK: the fetch SQL genuinely carries
`WHERE market = 'US'` (L113) and I verified the cached panel independently --
0 of 513 tickers deviate from a US symbol shape. The tautology is decorative,
not masking an error. But the panel is a CACHED PICKLE: nothing re-verifies that
the cache on disk was produced by that SQL, which is exactly what a real guard
would cover.

## Guards NOT covered by any matrix cell (13 of 19 covered)
panel_is_us_only (uncoverable -- tautology), dedup_actually_fired_on_this_panel,
enough_sessions_for_window, baseline_arm_is_the_unflagged_ranking,
price_only_multidim_arm_ran, displacements_are_tie_explained.
(displacements_are_tie_explained shares `_tie_explained` with M12/M12b, so it is
covered transitively.)

## FINDING 2 (BLOCKING) -- criterion 4's BASELINE ARM IS UNGUARDED
## Two surviving mutants, EXECUTED, with a re-aimed positive control

AST census of guards PER FUNCTION:
    measure()            13 guards   (the --verify / criteria 1+3 path)
    measure_flags()       2 guards   (the CRITERION 4 path)
    measure_dispersion()  4 guards   (finding (a))

Criterion 4 -- the step's own "most consequential result", the source of ASK-1/2/3
which ask the operator to PROMOTE flags on the money path -- runs behind exactly
TWO guards, and one of them cannot fail:

    L707  base_d = out["baseline"]["distinct"]
    L708  _ok("baseline_arm_is_the_unflagged_ranking",
              out["baseline"]["n_distinct"] == len(set(base_d)),
              "baseline distinct list is not a set")

`distinct` is built at L692 as `sorted({t for s in slates for t in s})` -- a
SET comprehension -- and `n_distinct = len(distinct)`. So the predicate is
`len(x) == len(set(x))` on an always-set-derived list: TRUE for every possible
input. Evaluated directly on [], ['A'], ['A','B','C'], sorted({'A','B','A'}) -> all True.
The guard's NAME asserts the load-bearing property of criterion 4 (the baseline arm
carries no flag); its BODY asserts a deduplicated list has no duplicates.
Name-vs-assertion mismatch, vacuity shape #4.

I did not argue this, I executed it (in-process monkeypatch, no repo write):

  CONTROL (unmutated measure_flags, 4 cycles): GREEN, no invariant raised.
    baseline 13.3% / 7 distinct / IT 75%
    sector_neutral 46.7% / 11 / Utilities 20%
    soft_diversity_w0.30 20.0% / 8 / IT 40%
    min_k_sectors=3 26.7% / 9 / IT 60%

  MUTANT Q1 -- poison the baseline: FLAG_ARMS[0] = ("baseline", {"sector_neutral": True})
    RESULT: **SURVIVED**. No invariant raised. Both guards RAN
    (_RAN == ['flag_arms_all_ran','baseline_arm_is_the_unflagged_ranking']).
    baseline turnover 13.3% -> 46.7%; distinct 7 -> 11; top-sector 75% -> 20%,
    i.e. the "baseline" became BYTE-IDENTICAL to the sector_neutral arm, so every
    criterion-4 delta would have been reported as ZERO and nothing caught it.

  MUTANT Q2 -- make an arm INERT: soft_sector_diversity_w 0.30 -> 0.0
    RESULT: **SURVIVED**. arm byte-identical to baseline (13.3% / 7 distinct).
    The headline "all three MOVE the slate" has NO guard behind it.

  POSITIVE CONTROL v1 was MIS-AIMED and I am recording that: I first blanked
  load_sectors, expecting `sector_map_present` to fire -- it did not, because that
  guard lives in measure(), NOT on the criterion-4 path. A GREEN there indicted my
  probe, not the code.
  POSITIVE CONTROL v2 (re-aimed at a guard that IS on the path): measure_flags(1, W)
  -> "INVARIANT FAILED: flag_arms_all_ran -- an arm produced no turnover series".
  KILLED. So the probe method demonstrably reaches and trips guards inside
  measure_flags, and the two SURVIVED verdicts are about the guards, not the probe.

CONSEQUENCE FOR CRITERION 7 ("mutation-test EVERY new guard"): the matrix's
"14/14 KILLED" is a true statement about 13 guards. Two of the 19 new guards
(`panel_is_us_only`, `baseline_arm_is_the_unflagged_ranking`) cannot be killed by
ANY mutation, so they were never mutation-tested and cannot be. Per qa.md 4c a
matrix licenses only "these N mutations were killed", never a global claim.

## Criterion 5 -- verified
settings.py UNTOUCHED (`git status --short -- backend/config/settings.py` empty);
sector_neutral_momentum_enabled=False (L468), multidim_momentum_enabled=False (L478),
paper_soft_sector_diversity_enabled=False (L487), paper_min_k_sectors_analyzed=0 (L489).
No settings/.env file in commit 15a817cc. 4 numbered operator asks are recorded.
BLOCKED CHECK, disclosed: a grep of `backend/.env` was DENIED by the permission
system. I treated the denial as authoritative and did not work around it, so
"no .env override for these flags" is NOT independently verified by me.

## Claim-precision NOTE (non-blocking)
experiment_results L26-27 and live_check L180 quote
`git status --short -- backend/` as the evidence for "no production file modified".
Run verbatim RIGHT NOW that command returns THREE modified files
(backend/api/sovereign_api.py, backend/services/autonomous_loop.py,
backend/services/experiments/perf_results.tsv) -- all from unrelated in-flight work
(the autonomous_loop hunk is a /reports empty-summary UI fix). The SUBSTANTIVE claim
is true and I verified it at commit level; the quoted REPRODUCING COMMAND does not
produce the stated output.

## FINDING 3 (WARN) -- the sigma triple does not reproduce and is self-inconsistent

live_check L156, experiment_results L104-105 and -- durably -- masterplan step
86.117's audit_basis all state "the measured cross-sectional sigmas are ~10.2 (1m),
~19.4 (3m), ~31.0 (6m)". Re-derived from the script's OWN `--dispersion --cycles 20`
table (which I re-ran; the table itself reproduces line for line):

    sigma_1m mean 10.646  (claimed ~10.2, -4.2%)   median 10.375
    sigma_3m mean 19.849  (claimed ~19.4, -2.3%)   median 19.775
    sigma_6m mean 30.442  (claimed ~31.0, +1.8%)   median 29.870

Neither the mean nor the median produces the quoted triple. It is also inconsistent
with the artifact's OWN headline: weight x sigma from the claimed triple gives
effective shares 21.9 / 36.5 / 41.6, whereas the artifact reports 22.6 / 37.0 / 40.4
-- which is exactly what the RE-DERIVED means give (22.6 / 36.9 / 40.4). So the
headline is right and the sigmas quoted to explain it are a different, non-reproducing
set. The load-bearing ratio survives (6m/1m = 2.86x vs the "~3.0x" claimed), so no
conclusion moves -- but the triple has been propagated into masterplan 86.117 where a
future step will read it as the measurement.

## REPRODUCTIONS (all run by me, independently)
- `--cycles 20`      : rho 0.9622 / 0.9319, turnover 15.8% top-10 AND 15.8% top-5,
                       3 of 19 zero-turnover, 12 distinct / IT 72.0%
                       {Industrials 20, IT 72, Health Care 8} = 100 slots,
                       18 distinct LIVE, 80% fidelity, 47,880/200,875 (23.8%) dups,
                       10 split bars, "OK: all 70 invariants hold". EXACT MATCH.
- `--flags --cycles 20` : baseline 15.8/12/IT72, sector_neutral 28.4/22/Industrials20,
                       soft_diversity 22.1/17/IT40, min_k=3 17.9/14/IT60;
                       deltas +12.6 / +6.3 / +2.1pp. EXACT MATCH.
                       Its own footer prints "OK: all 2 invariants hold".
- `--dispersion --cycles 20` : effective 22.6/37.0/40.4, gaps -17.4/+2.0/+15.4pp,
                       50 of 10,139 positions move (0.493%), 5 cycles identical.
                       EXACT MATCH.
- `mutation_86_59.py`: 14/14 KILLED, control GREEN first on 3 modes, sha restore.
                       EXACT MATCH.

## FINAL STATE RECHECK
HEAD f9bcd3bf unchanged from eval start. `git status --short` clean for
scripts/qa/ and both 86.59 handoff artifacts. masterplan 86.59 status = pending.
My mutations were in-process only; md5 of both scripts unchanged after all runs.

## CRITERION MAP
1 MET | 2 MET (vacuous, disclosed) | 3 MET | 4 MET on its face (measured before new
code, reproduced) but its evidence is UNGUARDED per FINDING 2 | 5 MET (.env grep
denied, disclosed) | 6 MET (vacuous, disclosed; corroborated by the null production
diff at commit level) | 7 NOT MET -- "mutation-test EVERY new guard" fails: 2 of 19
new guards cannot be made red by any mutation, and one of them is sole coverage for
criterion 4's baseline, proven by an executed surviving mutant.

VERDICT ISSUED: CONDITIONAL.
COMPLETED: 2026-08-18T03:32:00Z

