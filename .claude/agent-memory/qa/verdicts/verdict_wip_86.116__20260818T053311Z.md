STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.116
WRITTEN: 2026-08-18T05:33:11Z

# Q/A write-first record -- step 86.116 (price de-duplication on READ)

## 0. Prior-attempt evidence (gathered, not a trigger)
- `qa_wip.py 86.116 --spawned-at 2026-08-18T05:33:11Z`: `source_present: true`,
  `attempt_number: 2`, `attempt_number_status: "ok"`,
  `attempt_number_is_lower_bound: false`, `prior_attempts: 1`,
  `records_retained: 2` (GAUGE), `records_pruned_known: null`.
- `verdict_history_86_21.py --step 86.116 --evidence-only`: `status: ok`,
  `detail: 1 verdict(s) from the ledger`, `verdicts: CONDITIONAL`.
- CROSS-CHECK: prior_attempts (1) == ledger rows (1) -> ledger NOT stale.
  Sequence as evidence: [CONDITIONAL].

## A. Harness compliance -- CLEAN (5/5)
1. research (06:31:13) < contract (06:34:29). Envelope: brief_status COMPLETE,
   6 sources read in full (>=5), 25 urls_collected (>=10), recency_scan true,
   gate_passed true. My independent URL count of the brief = 25, MATCHES.
2. contract (06:34:29) < commit 539f16eb (07:06:16) and all artifacts.
3. experiment_results present (12,767 B).
4. log-last: masterplan 86.116 `status: "pending"` -- not flipped.
5. no-verdict-shopping: aec5d815 changed 6 evidence files; cache.py NOT among them.
CRITERIA IMMUTABLE: 86.116 `verification` at creation rev 15a817cc vs HEAD ->
IDENTICAL: True (7 criteria). 539f16eb's masterplan diff only ADDS step 86.118.

## B. Deterministic -- ALL GREEN
- IMMUTABLE COMMAND exit=0, stdout `parses`.
- sha256 cache.py = 9f5f1d6798833281c12c4b17387fac31ff62fa167e94cf51c1089f10aa6c8bf6
  = Main's claim; unchanged BEFORE and AFTER my mutation run; 0 MUTANT markers.
- `python scripts/qa/verify_86_116.py` (bare) -> exit=0, "OK: all 31 invariants hold".
  CYCLE-1 FINDING 2 CONFIRMED FIXED.
- `pytest test_phase_86_116_price_dedup.py -q` -> 13 passed, 13 dots (consistent).
- ruff F821,F401,F811 over git-DERIVED scope `539f16eb~1..HEAD -- '*.py'` (5 files,
  non-empty guard asserted first, xargs) -> "All checks passed!" exit=0.
- runtime smoke: `import backend.backtest.cache` OK; dedupes 2->1; identity-inert.
- mutation matrix re-run BY ME: control GREEN at 13 collected, 8/8 KILLED,
  0 SURVIVED, 0 UNSCORABLE, restore sha256 verified.
- FULL SUITE (not a -k selection): 19 failed, 3635 passed, 12 skipped, 5 xfailed,
  1 xpassed in 512.94s. The 19 = exactly the 18 enumerated in 86.118's audit_basis
  PLUS test_phase_86_6_subprocess_channel (the declared ordering artifact).
  test_phase_82_12_string_column_guards is GREEN (the line-pin fix holds) and
  test_phase_86_116_price_dedup is GREEN in the full run.
  ARITHMETIC CHECKS ACROSS THREE RUNS: author 3633 passed/20 failed (pre-pin-fix),
  cycle-1 Q/A 3634/19, mine 3635/19 -- the +1 is exactly the one test cycle 2 added
  (12 -> 13). Zero failures attributable to this step.

## C. Independent re-derivations (not taken from the artifact)
- BigQuery AVB 2026: raw=159, distinct=155, drop_duplicates()=159,
  ~index.duplicated()=155. "THE METHOD IS THE FINDING" is TRUE.
- E1's equivalence justification re-derived from BQ: 394,719 keys whose close
  differs, p50 0.0%, p99 0.0%, max 0.9326%, ZERO above 2%. Exact match.
- Census reproduces exactly (1,859,482 / 1,152,607 / 706,875 = 61.33% keys /
  706,875 = 38.01% rows / 336 of 513 / 394,719); per-year 2017 90.5% .. 2026 0.1%.
- Criterion-2 driven numbers reproduce: 390->250, mom_1m 0.83->-0.52,
  mom_3m -1.60->15.04, rsi 23.7->54.5, vol 0.3343->0.4182, span 12->22.
- Criterion-3 probe run BY ME at both revs: pre-fix 539f16eb~1 gives 0/0/0 for
  drop_duplicates / index\.duplicated / is_unique; POST-fix HEAD gives 2/2/2 with
  the SAME patterns -- the strongest form of the positive control, stronger than
  the shipped set_index control (2 files at the pre-fix rev).
- Criterion-6 WIRED CHAIN verified end to end by me:
  historical_data.py:126 -> backtest_engine.py:1251 -> backtest_trader.py:145-147
  -> backtest_trader.py:89 `vol_scale = min(self.target_vol / stock_vol, 3.0)`.
  Dead key confirmed at rotation_runner.py:64. 1/0.7995 = 1.2508 checks.
- Call-site completeness: `_prices_full[...] =` occurs ONCE (:293, deduped);
  `_prices_cache[...] =` occurs ONCE (:641, deduped); :598/:609 are reads; the
  preload early-return (:261) returns already-deduped frames. No third path.
  All 4 audit_basis consumers route through cache.cached_prices.
- Parity oracle is genuine: `out is df` IDENTITY on unique frames, exact nunique()
  on duplicated, plus an explicit both-branches-exercised assertion.

## D. CAPPING FINDING -- a cycle-2 tripwire cannot fire for what it names
`scripts/qa/verify_86_116.py:319-325`, invariant
`triple_barrier_label_has_no_volatility_term`:

    "vol" not in label_src.split("tp_price")[0].split("def ")[-1].lower()
    or "self.tp_pct" in label_src

MEASURED by execution (in-memory; repo untouched):
  CONTROL unmutated                                 A=True  B=True  GUARD=True
  MUTANT V1 vol term ADDED, self.tp_pct retained    A=False B=True  GUARD=True  SURVIVED
  MUTANT V2 vol term ADDED inline, tp_pct retained  A=False B=True  GUARD=True  SURVIVED
  MUTANT V3 vol term ADDED + self.tp_pct renamed    A=False B=False GUARD=False fires

Clause A (the working half) DOES fire on V1/V2. `or "self.tp_pct" in label_src` is
a pure escape hatch that disarms it: a barrier genuinely re-widened by volatility
(`tp_price = entry*(1 + vol_mult*self.tp_pct...)`) -- the exact condition the
invariant's own failure message names -- leaves the guard GREEN. It fires only when
`self.tp_pct` is ALSO renamed away, i.e. it detects a RENAME, not a volatility term.
qa.md 4c vacuity shape #8 (OR-escape-hatch).

Why it caps: criterion 7 says "mutation-test EVERY NEW GUARD: revert it and show the
check goes red". The two tripwires are NEW guards added in cycle 2 and are not cells
in mutation_86_116.py (which targets cache.py only). When I mutation-tested one, it
SURVIVED. Sole coverage for its proposition -- the sibling tripwire T1 guards a
different claim, and Main's own comment separates them: "even if the key came back,
the barrier is only vol-sensitive if the LABEL reads a volatility. This is the claim
that actually matters." The spawn prompt's claim "If either changes, the section must
be re-derived rather than trusted" is FALSE for T2.
Note the shape: the recorded cycle-2 lesson was that the first tripwire was too
BRITTLE; the remedy for brittleness introduced vacuity.
Secondary: the window is a fixed `eng[i:i+2600]` while the function is 1,788 chars,
overshooting 812 chars into `_compute_sample_weights`; `self.tp_pct` is not in that
spill today, but a neighbour acquiring it would satisfy clause B from outside the
function under test.
T1 checked and is NOT vacuous: `"vol_barrier_multiplier",` (quotes + comma) occurs
exactly ONCE in rotation_runner.py, at :64 inside _DEAD_KEYS; the :22 docstring
mention uses double-backticks and does NOT satisfy it.
NAMED FIX: drop the `or "self.tp_pct" in label_src` clause, bound the slice to the
function rather than 2600 fixed chars, and add matrix cells that mutate BOTH
tripwires and show them go red.

## E. NOTES (non-capping)
- AVB 159/155 is stated in PRODUCTION source (cache.py:222), the test docstring
  (:115), experiment_results, the contract and the commit message, but
  `grep "AVB\|159\|155"` returns NOTHING in verify_86_116.py or live_check_86.116.md.
  It is TRUE (I re-derived it from BQ) -- a reproducibility gap, not a false claim.
- cache.py:224 calls `keep="first"` after `sort_index()` "deterministic".
  sort_index() short-circuits on a monotonic (non-decreasing) index, so it inherits
  the BQ row order, whose tie order among identical dates is not itself guaranteed.
  The load-bearing claim (immaterial: p50/p99 0.0%, max 0.93%) is measured and
  correct, so this is wording, not substance.
- Contract's criterion 6 is not byte-verbatim: masterplan has "the existing gates is
  reported (DSR, PBO)", contract has "the existing gates (DSR, PBO) is reported" --
  a parenthetical transposition, semantically identical, no requirement weakened.
  masterplan.json itself is UNCHANGED. (My spawn prompt carries the contract's form.)
- Unrelated uncommitted .py edits (sovereign_api.py, autonomous_loop.py) contain no
  86.116/dedup content -- not attributable to this step.

## F. Criterion roll-up
1 MET | 2 MET | 3 MET | 4 MET | 5 MET | 6 MET (report correct + independently verified)
7 NOT FULLY MET -- the matrix on the FIX is 8/8 under strict scoring, but a new
  cycle-2 guard was not mutation-tested and survives the condition it names.

VERDICT DIRECTION: CONDITIONAL (one fixable capping finding; product code correct,
untouched, and mutation-proven; harness compliance clean; no unintended change).

COMPLETED: 2026-08-18T05:49:23Z
