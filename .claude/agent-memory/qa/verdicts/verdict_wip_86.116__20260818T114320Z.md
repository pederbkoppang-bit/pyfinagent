STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.116
WRITTEN: 2026-08-18T11:43:20Z

# Q/A cycle 4 (write-first record)

Prior WIP records on disk for 86.116:
- verdict_wip_86.116__20260818T050702Z.md (17246 b)
- verdict_wip_86.116__20260818T053311Z.md (8799 b)
- verdict_wip_86.116__20260818T055647Z.md (16660 b)

Main's disclosure (ADVISORY): 3 prior CONDITIONALs; correction commits
539f16eb, aec5d815, 53fc2106, 69a956ea, f675c0cf.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable cmd, git status/diff scope, ruff, runtime smoke
C. re-derive claims: verify_86_116.py (43 invariants), mutation_86_116.py (14/14)
D. GRADE HARD on the three named questions:
   1. does max(scale_pre, scale_post) < 3.0 genuinely remove the dead clause
      or just hide it?
   2. is _run_tripwires=False driving the real seam or a special-cased path?
   3. do any of the 43 invariants contain another guard that cannot fail?

## Findings (appended as established)

### Counters
- qa_wip: attempt_number 4, prior_attempts 3, source_present true, status ok.
- verdict_history --evidence-only: status ok, CONDITIONAL -> CONDITIONAL -> CONDITIONAL (3 rows).
- prior_attempts (3) == ledger rows (3) -> ledger NOT stale.

### Deterministic (all reproduced by me)
- immutable cmd -> `parses`, EXIT=0.
- verify_86_116.py (full, BQ) -> exit 0, "OK: all 43 invariants hold". Saturation line
  printed verbatim as claimed: `vol_scale pre / post: 0.4487 / 0.3587 (cap 3.0 -- UNSATURATED...)`.
- verify_86_116.py --offline --verify -> "OK: all 29 invariants hold (offline subset)".
- mutation_86_116.py -> control GREEN rc=0 collected=13; tripwire control GREEN;
  KILLED 14/14 SURVIVED 0 UNSCORABLE 0 EQUIVALENT-BY-DESIGN 1. Restore verified,
  `grep -c MUTANT` = 0 on both targets, sha256 cache.py 9f5f1d67... unchanged.
- ruff F821,F401,F811 over the 5 .py files in the 86.116 commits -> "All checks passed!", exit 0.
  (An F401 exists in scripts/qa/rank_stability_86_59.py -- 86.59's file, NOT this step's.)
- runtime smoke: `import backend.backtest.cache` OK, `_dedupe_index` present.
- pytest test_phase_86_116_price_dedup.py + test_phase_82_12_string_column_guards.py -> 43 passed.
- census numbers reproduce EXACTLY: 1,859,482 rows / 1,152,607 keys / 706,875 dup keys
  = 61.33% OF KEYS / 38.01% OF ROWS / max mult 2 / 336 of 513 tickers / 2026 at 0.1%.
- cache.py byte-identical since 539f16eb (`git diff 539f16eb..HEAD` empty; no uncommitted diff).
- V3 restart reproduced: pgrep -> 89340, started 08.26.53 local, fix landed 07:06:16+02:00.

### Harness compliance
1. research gate: brief COMPLETE, gate_passed true, 6 sources read in full, 25 URLs, recency true. OK
2. contract before generate: brief 06:31 < contract 06:34 < fix commit 07:06 (local). OK
3. experiment_results present. OK (but see F4 staleness)
4. log-last: masterplan status=pending; only a PARKED row in harness_log. OK
5. no-verdict-shopping: evidence CHANGED (69a956ea + f675c0cf; verify 33->43 invariants,
   matrix 11->14 cells). OK

### ADVERSARIAL FINDINGS (each DRIVEN, not argued)

F1 [WARN] `size_inflation_is_above_one` CANNOT FAIL on the reachable domain.
  `pre_fix_volatility_is_the_lower_one` asserts vol_ratio < 1.0 three lines above;
  size_inflation = 1.0/vol_ratio, so >1.0 is implied. 256-cell sweep over gate_effect:
  failing invariants were pre_fix_volatility_is_the_lower_one (120), 
  vol_scale_is_unsaturated_at_the_binding_side (69), vol_ratio_is_derivable (16),
  ZeroDivisionError (15) -- size_inflation_is_above_one NEVER. In-memory cell X1
  (forced True) -> GREEN = SURVIVED. Falsifiable only on a NEGATIVE vol_ann, which
  `float(dr.std()*252**0.5)` cannot produce. Same shape as the cycle-3 BLOCKING
  finding and as the dead clause Main removed itself, 3 lines from it. No matrix cell.

F2 [WARN] `_declares_dead_key` does not establish MEMBERSHIP.
  It is `'"{key}",' in src AND "_DEAD_KEYS" in src` -- two independent token searches.
  Driven: a source with `_DEAD_KEYS = ("something_else",)` plus
  `LIVE_PARAMS = ("vol_barrier_multiplier",)` classifies True. rotation_runner._DEAD_KEYS
  holds 8 keys, so a re-wiring that moves this one out while the tuple survives is the
  realistic shape. The fixture has no such case. In-memory cell X3 (delete the
  `_DEAD_KEYS` conjunct) -> GREEN = SURVIVED, because `dead_no_decl` lacks the trailing
  comma and is killed by the FIRST conjunct -- mis-attributed kill mechanism (shape 11).
  The code comment claims it "checks the repo's own authoritative statement"; it does not.

F3 [NOTE] `multiplicity_is_bounded` asserts `max_mult >= 2` -- an EXISTENCE check, not a
  bound. Driven with a stub client: max_mult 2 / 3 / 9 all pass all 5 census invariants,
  while gate_effect returns a HARDCODED `max_inflation_under_full_duplication = 2**0.5`
  valid only at multiplicity 2. Positive control: a repaired table goes RED on
  census_rows_exceed_keys, so the census guards are otherwise live.

F4 [WARN] experiment_results_86.116.md is STALE vs the shipped tree (see below).

### Answers to Main's three GRADE-HARD questions
Q1 max(...) vs the compound: MEASURED -- over 105 inputs that reach the guard, the
   `scale_post` operand changes the outcome 0 times; in-memory cell X2 (delete the
   operand) -> GREEN. So it did NOT make the operand live; `max(a,b)<c` IS `a<c and b<c`.
   What it removed is a separately-MUTABLE clause, and the defensive rationale in the
   comment is legitimate. NOTE, not a violation -- but "no unfalsifiable half" overstates it.
Q2 `_run_tripwires=False`: it gates ONLY the recursive call at the END of gate_effect,
   after every _ok. All four gate invariants execute identically. This IS the real seam.
   No finding.
Q3 other guards that cannot fail: YES -- F1 (and F2's conjunct). The class is not empty.

### F4 detail -- experiment_results is stale vs the shipped tree
- line 20 "verify_86_116.py -- NEW ... **33 invariants**"   (command outputs 43)
- line 21 "mutation_86_116.py -- ... **11 cells across 2 targets**"  (matrix runs 14)
- line 132 "Criterion 7 -- 11 cells, 11 KILLED, 0 SURVIVED..."  (14/14)
- grep for "43 invariants|14 / 14|Cycle-3 Follow-up|max(scale_pre" in experiment_results -> 0
The whole cycle-3 follow-up lives ONLY in evaluator_critique_86.116.md, i.e. the
EVALUATE artifact carries the GENERATE evidence. pre_spawn_gate itself prints TWO
capture blocks (killed 11 and killed 14) and still reports CLEAN -- its class list
does not include "a summary count in a sibling artifact that no longer reproduces".

### Criterion mapping
1 MET (reproduced exactly from the live query, incl. per-year + normalisation rule)
2 MET (real preload_prices + real screener; 390->250, mom sign flips, span 12->22)
3 MET (probe vs 539f16eb~1; positive control set_index=2 files, guarded)
4 MET (read-side only; no DELETE/DDL/write path added in any of the 5 commits; ASK-1..3 numbered)
5 MET (12 shapes, 7 inert `out is df` / 5 active, both-branches guard)
6 MET (mechanism verified independently: backtest_trader.py:54 target_vol=0.15,
       :89 vol_scale = min(target_vol/stock_vol, 3.0); ratio 0.7995 -> 1.2508x) -- F1 WARN
7 NOT FULLY MET (14/14 strong and reproduced, but X1 and X3 SURVIVE and have no cell;
       criterion says "mutation-test EVERY new guard")

### Other observations (NOTE)
- `target_vol = 0.15` in verify_86_116.py is a hardcoded copy of backtest_trader's
  default with no guard pinning it -- if that default moves, the printed vol_scale
  silently stops describing production.
- Tree during EVALUATE also holds an uncommitted backend/api/charts.py (+ untracked
  test_charts_nan_serialisation.py) not named in Main's tree_state disclosure. NOT
  attributable to 86.116: cache.py is byte-identical to 539f16eb (sha 9f5f1d67...).
- live_check's "ZERO failures" heading still sits over the pre-pin-fix 20/3633 block;
  the attribution immediately below it is correct and complete. Carried NOTE.

VERDICT RETURNED: CONDITIONAL (see the structured return; this file is not a verdict).

COMPLETED: 2026-08-18T11:52:16Z
