STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.59
WRITTEN: 2026-08-18T08:07:14Z

# Q/A write-first record -- step 86.59 (attempt 5)

## Prior-verdict evidence (GATHERED, not applied)
- `qa_wip.py 86.59 --spawned-at 2026-08-18T08:07:14Z` -> source_present TRUE,
  attempt_number 5 (status "ok", is_lower_bound true), prior_attempts 4,
  records_retained 5 (GAUGE, not used as a counter), records_pruned_known null.
- `verdict_history_86_21.py --step 86.59 --evidence-only` -> status "ok",
  "4 verdict(s) from the ledger", sequence
  CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL.
- CROSS-CHECK: prior_attempts (4) == ledger rows (4) => ledger NOT stale.

## A. Harness compliance -- CLEAN (5/5)
1. research-gate-before-contract: research_brief_86.59.md envelope COMPLETE,
   6 read-in-full / 30 URLs / recency true / gate_passed true; contract cites
   the RERUN brief (8 sources, 54 URLs). MET.
2. contract-before-generate (mtime): brief 2026-08-12T09:18 < contract
   2026-08-17T20:56 < rank_stability.py 10:02 < experiment_results 10:03. MET.
3. experiment_results + live_check + critique all present. MET.
4. log-last: masterplan 86.59 status still `pending`; the newest harness_log
   entry ("Cycle 5 ... result=FAIL") records the PRIOR (attempt-4) verdict,
   not this one. MET.
5. no-verdict-shopping: evidence CHANGED -- commit 497ae3ac (2026-08-18 10:05)
   rewrites both scripts + 4 artifacts. MET.

## B. Deterministic -- all reproduced
- IMMUTABLE CMD -> `parses`, exit 0.
- Production scope: ALL 6 commits matching `phase-86.59` -> 16 files, ZERO
  under backend/ or frontend/, ZERO .env/settings. Derived by me, not taken
  from the artifact.
- ruff F821/F401/F811 over the git-DERIVED .py scope (2 files, non-empty
  asserted, xargs-quoted) -> "All checks passed!", exit 0.
- Criteria immutability: worktree == HEAD (7 byte-identical) and all 7 appear
  VERBATIM in contract_86.59.md.
- `--verify` -> "sector map coverage: 502/513 = 97.9%", "OK: all 42 invariants
  hold".
- Mutation matrix re-run BY ME: control GREEN first on --verify/--dispersion/
  --flags, "coverage: 25 guards in target, 25 covered", KILLED 26/26,
  SURVIVED 0, UNSCORABLE 0, restore verified; md5 back to a2312e95 and sha256
  39fc81f531c91cce... unchanged.
- AST guard census re-derived: 22 literal-named `_ok` (19 single-line + 3
  multi-line at :464/:469/:476) + 3 f-string (:525/:535/:547) = 25. Matches.
- Criterion 1/3 headline re-derived at --cycles 20: rho 0.9622 mean /
  0.9319 min, top-10 AND top-5 turnover 15.8%/day, 3 of 19 zero-turnover,
  12 distinct tickers (exact list), IT 72.0% with counts
  {Industrials 20, IT 72, Health Care 8}, 513 tickers, fidelity 80%.
- Criterion 4 table re-derived at --cycles 20 by my own in-process drive:
  15.8 / 28.4 / 22.1 / 17.9, deltas +12.6 / +6.3 / +2.1pp, distinct
  12/22/17/14, top sectors IT 72 / Industrials 20 / IT 40 / IT 60. EXACT.
- 86.116 and 86.117 both exist in the masterplan, status pending.

## C. FINDINGS -- two surviving mutants, executed, control GREEN first

### F1 (WARN) min_k guard records the VARIABLE, not the argument passed
`measure_flags` :893-895 is
  `_k = MIN_K_SECTORS` / `min_k_passed.append(_k)` /
  `picked = _min_k_sector_slice(base, ANALYZE_TOP_N, _k)`.
The guard `min_k_arm_used_the_labelled_k` compares `min_k_passed` to the
integer parsed out of `MIN_K_ARM`. It therefore records what the call site was
ABOUT TO PASS, from a separate statement -- not what `_min_k_sector_slice`
received. The code comment at :831-833 claims the stronger property ("What the
CALL SITE actually received").
MEASURED at the published --cycles 20, control GREEN first:
  CONTROL     min_k row 17.9% / +2.1pp / distinct 14 / IT 60%
  MUTANT (argument forced to k=4, adjacent record line untouched)
              SURVIVED green, guard RAN and PASSED, row still labelled
              'min_k_sectors=3' -> 22.1% / +6.3pp / distinct 15 / IT 49.0%
Those are BYTE-IDENTICAL to the numbers the cycle-4 Q/A published for the
ORIGINAL defect. ASK-1's load-bearing claim ("smallest turnover cost of the
three arms (+2.1pp/day)") ties ASK-2. sha256 unchanged before/after.
Equivalence check: `_min_k_sector_slice(candidates, n, k)` is positional, so
the wrapper is exactly a one-token source edit.
NAMED FIX: record INSIDE the call (one read of k), or assert the returned
slate spans >= labelled_k distinct sectors when available.

### F2 (WARN) coverage guard is not on the path that publishes the table
`sector_map_covers_the_panel_at_the_published_operating_point` and the
`print("sector map coverage: ...")` disclosure both live in `measure()`
(:511/:517). `measure_flags()` (:809-1041) calls `load_sectors()` at :814,
NEVER calls `measure()`, and contains exactly 6 `_ok` guards -- none of them
the coverage guard. Cell M9b runs `["--verify"]`, so its KILL is scored on
`measure()`, not on the criterion-4 path.
MEASURED on the --flags path at --cycles 20, control GREEN first:
  CONTROL 97.9%   15.8 / 28.4(+12.6) / 22.1(+6.3) / 17.9(+2.1)
  DEGRADED 95.5%  SURVIVED (expected, above floor); sector_neutral
                  +12.6 -> +9.5pp and distinct 22 -> 20, with every top-sector
                  share identical to control, so nothing signals it. ASK-1 vs
                  ASK-2 ordering HOLDS -- the 0.95 value is adequate for that
                  ordering, which I verified rather than accepted.
  DEGRADED 78.2%  SURVIVED green -- the EXACT degradation the cycle-4 Q/A used
                  and the one the guard's own message says must block
                  publication. min_k +2.1 -> +6.3pp (tying ASK-2),
                  sector_neutral +12.6 -> +18.9pp, soft_diversity +6.3 ->
                  +8.4pp. 8 guards ran; the coverage guard was not among them.
live_check §5's criterion-4 block carries no coverage line, consistent with
the print being on the other path.
HONEST MITIGATION: a full matrix run does `control --verify` before
`control --flags`, so a degraded cache would be caught there. It is NOT caught
when `--flags` is run directly, which is how the published table is produced.
NAMED FIX: evaluate the coverage guard (and the print) inside `measure_flags`,
and score a cell for it under `--flags`.

### F3 (WARN) experiment_results contradicts itself on the cell count
`experiment_results_86.59.md:24` (present-tense "What this step SHIPS" table):
"criterion 7 -- **23 cells** + an AST coverage gate".
`experiment_results_86.59.md:99-100` (same file): "26 cells, 26 KILLED ...
coverage 25/25".
The claimed sweep DID correctly annotate the cycle-record instances
(:372-378 and escalation:47-49 both carry a current-value pointer); it missed
the un-annotated present-tense one. Same class as cycle-4 finding 3 and
cycle-3 item 10, one line over.

### NOTES (do not degrade the verdict)
- `experiment_results:26` backs "No production file is modified" with
  `git show --name-only 15a817cc` -- ONE commit, while the step has 6. The
  claim is TRUE on the full derived scope (I verified) but the cited command
  is narrower than the claim it supports.
- "86.117 explicitly BLOCKED-BY 86.116" is recorded as PROSE inside
  `audit_basis`. The structured field is `blocked_by: null`, and 0 of 1310
  steps use `blocked_by` at all, while `depends_on` is an existing convention
  in the same file. Nothing mechanical enforces the ordering.
- CONSEQUENCE LEAK, second cycle running: artifacts I was instructed to read
  carry escalation prose (experiment_results:385-389, the escalation file,
  harness_log "one attempt remaining"). Recorded; discounted in BOTH
  directions; ERRJ=0.000 means I cannot certify that from introspection.

## Criterion roll-up
1 MET (reproduced exactly) | 2 MET vacuously, deviation disclosed, reason
measured, filed as 86.117 | 3 MET (N=20 stated, reproduced) | 4 MET on
substance, capped by F1+F2 | 5 MET (zero production/.env across all commits,
4 numbered asks) | 6 MET vacuously, disclosed | 7 MET literally (both new
guards mutation-tested, control GREEN first, byte-identical restore verified
by me).

VERDICT RETURNED: CONDITIONAL.

COMPLETED: 2026-08-18T08:26:24Z
