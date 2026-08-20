STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.59
WRITTEN: 2026-08-18T07:24:36Z

# Q/A write-first record -- step 86.59, EVALUATE

Launch: Workflow structured-output rail (qa-verdict.js), agentType qa.
Attempt evidence (qa_wip.py --spawned-at 2026-08-18T07:24:36Z):
  source_present=true, attempt_number=4 (status ok, lower_bound true),
  prior_attempts=3, records_retained=4 (GAUGE, not a counter).
Ledger (verdict_history_86_21.py --step 86.59 --evidence-only):
  status=ok, "3 verdict(s)", CONDITIONAL -> CONDITIONAL -> CONDITIONAL.
CROSS-CHECK: prior_attempts (3) == ledger rows (3) -> ledger is NOT stale.

## A. HARNESS COMPLIANCE -- CLEAN
- research gate: research_brief_86.59_rerun.md envelope brief_status COMPLETE,
  external_sources_read_in_full 8, urls_collected 54, recency_scan true,
  gate verdict artifact PASSED. Floors cleared.
- contract-before-generate: research 12-14 Aug, contract_86.59.md 17 Aug 20:56,
  experiment/live_check/critique 18 Aug 06:17-06:18. Order holds.
- criteria immutability: all 7 criteria byte-identical between
  .claude/masterplan.json and contract_86.59.md (programmatic compare), and
  identical between HEAD and the working tree. Working-tree masterplan diff is
  a peer session's additive 26-line insert of a NEW step 86.120 -- not 86.59.
- log-last: masterplan 86.59 status = "pending" (NOT flipped).
  harness_log.md:36331 already carries "phase=86.59 result=PARKED
  (3rd-CONDITIONAL rule)" -- a PARK record, not a result claim for this cycle.
- no-verdict-shopping: evidence CHANGED. Commit 099414fe rewrote measure_flags
  (oracle hoisted above the arms loop; new guard
  baseline_ROW_matches_an_unflagged_direct_call) and added cell M22. Diff read.

## B. DETERMINISTIC -- ALL REPRODUCE
1. Immutable command -> "parses", exit=0.
2. Scope DERIVED from git, not typed. `git diff --name-only HEAD -- '*.py'`
   is EMPTY (work is committed). Union over step commits 15a817cc 3e75c2d6
   fb6f8a67 a4a5765c 099414fe (+ this session's guardlib pair 3e532fe4
   ff3ed8c7): 4 .py files, all under scripts/qa/.
   Production files (^backend/|^frontend/) across ALL of them: **0**.
3. ruff F821,F401,F811 over that derived set (xargs, quoted): exit 0,
   "All checks passed!".
4. `rank_stability_86_59.py --verify` -> "OK: all 42 invariants hold", 10.26s.
   SCOPE, from main() at :1157-1197: --verify runs measure() ONLY -- it does
   NOT call measure_flags() or measure_dispersion(), and defaults to 10 cycles.
   So its 42 invariants contain ZERO criterion-4 guards. No artifact overclaims
   this (grep of all 86.59 artifacts for "--verify" returns only the matrix
   control line and an explicit "`--verify` green is a self-check, not a
   verdict"). Main's re-run claim is true as stated.
5. Mutation matrix re-run BY ME, full output captured:
   control --verify/--dispersion/--flags all rc=0 GREEN **first**;
   coverage 24 guards / 24 covered; KILLED 23 / 23, SURVIVED 0, UNSCORABLE 0;
   "restore verified: sha256 unchanged (16164dcb7e04f039...)".
   That sha equals `git show HEAD:scripts/qa/rank_stability_86_59.py |
   shasum -a 256` = 16164dcb7e04f039adfd66afa0349894f3db0243. Block is current.
6. Independent AST guard census on a pristine `git show HEAD:` copy (the live
   file was mid-mutation, md5 8959aed1 with a MUTANT marker -- do NOT import it
   while the matrix runs): **24 distinct guard names**, 21 literal + 3 f-string
   prefixes. Matches the matrix's own 24 and Main's census claim.
7. Confirmed the two guards a prior cycle proved unkillable are GONE:
   `panel_is_us_only` survives only as a comment at :446; the
   `len(x)==len(set(x))` tautology is replaced by an independent-call check.

## C. INDEPENDENT RE-DERIVATION -- every published number reproduces EXACTLY
Driven in-process against the real module (no repo write; sha256 unchanged
before and after). measure(20,126) and measure_dispersion(20,126):
  mean rho 0.9622 / min rho 0.9319 / turnover top10 15.8% / top5 15.8% /
  zero-turnover 3 of 19 / distinct 12 with the exact ticker list /
  sector_concentration IT 0.72 counts {Industrials 20, IT 72, Health Care 8} /
  fidelity 0.7960 (~80%) / 18 live distinct / dedup 47,880 of 200,875 /
  10 split bars / 513 tickers / sigmas 10.646, 19.850, 30.441 at 2.86x /
  effective shares 22.6 / 37.0 / 40.4 / multidim 50 of 10,139 / 5 identical.
measure_flags(20,126) CONTROL, 7 guards ran, green:
  baseline 15.8% / 12 / IT 72.0%
  sector_neutral 28.4% / 22 / Industrials 20.0%   (delta +12.6pp)
  soft_diversity_w0.30 22.1% / 17 / IT 40.0%      (delta +6.3pp)
  min_k_sectors=3 17.9% / 14 / IT 60.0%           (delta +2.1pp)
Zero disagreement with the artifacts on any figure.

## FINDINGS

### V1 [WARN, criterion 4] the min_k arm's k is decoupled from its own label,
### and NO guard can tell -- EXECUTED, two surviving mutants
`arms["min_k_sectors=3"]` (:793) and `_min_k_sector_slice(base, ANALYZE_TOP_N,
3)` (:853) are two independent literals with nothing tying them.
In-process, at the PUBLISHED --cycles 20, control GREEN first:
  k=2 -> KILLED, but by ACCIDENT: the arm degenerated to baseline and tripped
        `flag_arms_are_distinguishable_from_baseline`. That guard detects
        degeneracy-to-baseline, not a label/parameter divergence.
  k=4 -> **SURVIVED**. Row still labelled "min_k_sectors=3", now reporting
        22.1% / delta **+6.3pp** / distinct 15 / IT 49.0%
        (published: 17.9% / +2.1pp / 14 / 60.0%).
  k=5 -> **SURVIVED**. Row reports 23.2% / delta **+7.4pp** / 15 / IT 47.0%.
Consequence: ASK-1 recommends promoting `paper_min_k_sectors_analyzed = 3`
explicitly "at the smallest turnover cost of the three arms (+2.1pp/day)". At
k=4 the row TIES ASK-2 (soft_diversity, +6.3pp); at k=5 it EXCEEDS it. The
ordering the operator ask rests on inverts and the run stays green.
This is vacuity shape #11 (mis-attributed kill mechanism) plus the step's own
recurring family: the reported row not corresponding to the configuration it is
labelled with. Named fix: derive the label from the k variable, or add
`_ok("min_k_arm_used_the_labelled_k", ...)` with a cell.

### V2 [WARN, criterion 4] the sector-map guard's floor is 20pp below the point
### at which the published ordering inverts -- EXECUTED, survives
`sector_map_covers_most_of_the_panel` requires known >= 0.5 * len(tickers).
Operating point is 502/513 = 97.9%. Degrading the cached map to 401/513 =
78.2% (a realistic yfinance-lookup failure mode, well above the floor):
  run stays GREEN, all 7 criterion-4 guards pass, and
  soft_diversity 22.1% -> 17.9% (delta +6.3pp -> **+2.1pp**)
  min_k          17.9% -> 22.1% (delta +2.1pp -> **+6.3pp**)
The two recommended arms SWAP their turnover cost -- the same ASK-1/ASK-2
inversion as V1, by a second independent route. Baseline and the top-sector
shares are unmoved, so nothing in the report signals it.

### V3 [WARN, evidence] live_check section 8 prose contradicts its own adjacent
### verbatim block
live_check_86.59.md:213 (inside the fenced "single verbatim capture"):
  "coverage: 24 guards in target, 24 covered ..."
live_check_86.59.md:222-223 (authored prose, 9 lines below):
  "The cell count (22) is lower than the guard count (23) ..."
Re-derived: 24 guard names; 22 CELLS + 1 NEGATIVE_CONTROL = 23 scored cells.
Guard count is 24, not 23 -- the prose is stale at the pre-M22 value.
experiment_results_86.59.md:99-101 has it right ("23 cells ... coverage 24/24"),
so the two artifacts disagree. Same class as the step's own cycle-3 item 10
("the section 8 evidence block was spliced from two runs"): the block was
regenerated, the authored prose beside it was not.

### N1 [NOTE] a code comment claims a mechanism the code does not have
rank_stability_86_59.py:80-83 says the top-N values are "read from
backend/config/settings.py rather than restated ... a number retyped into a
script is a number that can go stale", immediately above two hardcoded
literals. Values are correct today (settings.py:406 Field(10), :407 Field(5));
the defect is the comment's claim about its own mechanism.

### N2 [NOTE] 5 of 24 guards are covered only transitively
displacements_are_tie_explained, panel_carries_no_non_us_symbols,
enough_sessions_for_window, dedup_actually_fired_on_this_panel and
flag_arms_are_distinguishable_from_baseline are never themselves shown to go
red -- the cell that "covers" them fires a DIFFERENT guard (the predicate
fixture guard). Each is DECLARED in COVERED_TRANSITIVELY with a reason, which
is what the doctrine asks for, so this is a stated limit, not a violation.
Note V1 landed inside exactly this seam.

### N3 [NOTE, harness] the escalation artifact re-delivers the consequence
### payload that phase-86.78 scrubbed from the prompt
The spawn prompt honours the scrub, but the Q/A is INSTRUCTED to read
escalation_86.59_third_conditional.md:6-7 and evaluator_critique_86.59.md:12-13,
which state the trigger and its outcome in the imperative. That is the exact
channel arXiv 2604.15224 measures, arriving through the evidence rather than
the instructions. Recorded and explicitly DISCOUNTED in both directions: I did
not apply the rule and did not compensate for it.

### RETIRED BY EXECUTION (recorded so it is not re-raised)
- "the oracle and the arms can be poisoned together" -- the code comment at
  :845-849 already narrows the guard's scope to exactly this, and defeating it
  needs a coordinated 2-site edit, not a one-site survivor. Honest scoping.
- "k=2 proves the min_k arm is unguarded" -- FALSE, k=2 is killed. It took
  k=4/k=5 to establish V1.

## CRITERIA MAP
1 MET       -- rho + top-10 turnover + commands; reproduced exactly.
2 MET-VACUOUS (disclosed) -- conditional criterion, antecedent false: no new or
              reweighted term. Reason is MEASURED (38% duplicate keys in
              historical_prices would poison any DSR/PBO figure) and filed as
              86.117 BLOCKED-BY 86.116. Contract P3/P4/P6 abandoned and
              disclosed as a deviation.
3 MET       -- N=20 stated, 12 distinct / 100 slots, IT 72.0% {20,72,8};
              live "before" 18 distinct. Reproduced exactly.
4 MET on substance, CAPPED by V1 + V2 -- all three flags measured and the whole
              table reproduces, but two one-input changes invert the ASK-1 vs
              ASK-2 ordering with every guard green.
5 MET       -- 0 production files across ALL step commits (derived from git);
              4 numbered asks; no .env.
6 MET-VACUOUS (disclosed) -- no new behaviour to disable; substitute evidence
              (zero production files + the independent top_n call, cell M3) is
              real and is not claimed to BE flag-OFF parity.
7 MET       -- control GREEN first on 3 modes, 24/24 coverage, 23/23 KILLED,
              byte-identical restore, all re-run by me. N2 is a stated limit.

## DISPOSITION
verdict CONDITIONAL. Product is sound and nothing ships; every published number
survived independent re-derivation. The cap is evidence-layer: two executed,
quantified, one-input mutants that invert the operator-ask ordering while the
run stays green, plus a non-reproducing count in live_check section 8.

COMPLETED: 2026-08-18T07:43:27Z (read from `date -u`, not narrated)
