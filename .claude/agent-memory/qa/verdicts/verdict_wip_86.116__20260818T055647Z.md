STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.116
WRITTEN: 2026-08-18T05:56:47Z
COMPLETED: 2026-08-18T06:09:52Z

# Q/A write-first record -- step 86.116 (price de-duplication on READ)

## 0. Spawn context
- Workflow rail, agentType qa, opus, effort max. Read qa.md IN FULL first.
- qa_wip: source_present=true, attempt_number=3 (status ok), prior_attempts=2,
  records_retained=3 (gauge, not counter).
- verdict_history_86_21 --evidence-only: status=ok, "2 verdict(s)",
  sequence CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (2) == ledger rows (2) -> ledger is NOT stale.

## 1. Harness compliance (5 items) -- ALL CLEAN
1. research-gate-before-contract: research_brief_86.116.md exists, envelope
   brief_status COMPLETE, gate_passed true, external_sources_read_in_full 6 (>=5),
   urls_collected 25 (>=10), recency_scan_performed true. MET
2. contract-before-generate (mtime chain): brief 06:31:13 < contract 06:34:29 <
   cache.py 07:55:00 / experiment_results 07:55:56. MET
3. experiment_results present (15,077 B), live_check present (11,658 B). MET
4. log-last: masterplan 86.116 status="pending" (NOT flipped);
   `grep -F 86.116 handoff/harness_log.md` -> only a phase-86.59 filing mention,
   NO `phase=86.116 result=` row. MET
5. no-verdict-shopping: evidence CHANGED between cycle 2 and 3 -- commit 53fc2106
   modified verify_86_116.py (+172/-x), mutation_86_116.py (+77), experiment_results,
   live_check, evaluator_critique. MET

## 2. Deterministic checks -- ALL GREEN

IMMUTABLE COMMAND:
  bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(
  \"backend/backtest/cache.py\").read()); print(\"parses\")"'
  -> "parses", exit=0

SCOPE (derived, not typed): git diff --name-only 34a56b03..HEAD -- '*.py'
  backend/backtest/cache.py
  backend/tests/test_phase_82_12_string_column_guards.py
  backend/tests/test_phase_86_116_price_dedup.py
  scripts/qa/mutation_86_116.py
  scripts/qa/verify_86_116.py
  (non-empty set asserted BEFORE reading exit code)

RUFF F821,F401,F811 over that derived set (xargs, not unquoted $VAR):
  "All checks passed!" exit=0

RUNTIME SMOKE: `python -c "import backend.backtest.cache"` -> import OK,
  _dedupe_index resolvable. exit=0

PRODUCTION-CHANGE SCOPE: cache.py blob is IDENTICAL at 539f16eb, aec5d815,
  53fc2106 and HEAD (all d6dc0cd8a00d55d37153640b27c54c5074b63739);
  sha256 9f5f1d6798833281...; `git diff --name-only -- backend/backtest/cache.py`
  empty. Main's "no production code changed since the original fix" REPRODUCES.
  Cycles 2 and 3 touched evidence + checkers + tests only. No unintended
  production change in the step's commit range.

PYTEST (step suite): backend/tests/test_phase_86_116_price_dedup.py -> 13 passed.

STEP CHECKERS re-run by me:
  verify_86_116.py --offline -> "OK: all 19 invariants hold (offline subset)"
  verify_86_116.py (full, BigQuery) -> "OK: all 33 invariants hold"
  mutation_86_116.py -> KILLED 11/11, SURVIVED 0, UNSCORABLE 0,
    EQUIVALENT-BY-DESIGN 1; control GREEN first at collected=13; tripwire
    control GREEN; restore verified byte-identical on BOTH targets
    (cache.py 9f5f1d6798833281..., verify_86_116.py 22e302ecb4168f47...);
    I recorded PRE and POST sha256 myself and they match.

## 3. Claim re-derivation (4b) -- EVERY headline number REPRODUCES

Census, re-run against BigQuery by me (not read from the artifact):
  1,859,482 rows / 1,152,607 keys / 706,875 dup keys = 61.33% OF KEYS /
  706,875 excess rows = 38.01% OF ROWS / max multiplicity 2 /
  336 of 513 tickers / 394,719 dup keys whose close differs.
  Per-year 2017 90.5%, 2018 62.3%, 2019 63.1%, 2020 63.5%, 2021 63.9%,
  2022 64.2%, 2023 64.3%, 2024 64.5%, 2025 64.3%, 2026 0.1%.
  ALL EXACT MATCHES to experiment_results.

Driven harm (AKAM 2025), re-run by me:
  rows 390 -> 250; mom_1m 0.83 -> -0.52; mom_3m -1.60 -> 15.04;
  rsi_14 23.7 -> 54.5; vol_ann 0.3343 -> 0.4182; span 12 -> 22 sessions.
  Both momentum terms sign-flip. EXACT MATCH.

AVB 2026 method claim, re-derived by me directly from BQ:
  raw 159 rows / 155 distinct dates; drop_duplicates() -> 159 rows, index STILL
  non-unique; ~index.duplicated() -> 155. EXACT MATCH to the artifact.

Criterion-6 wiring, verified against live source by me:
  historical_data.py:126 features["annualized_volatility"] = std*sqrt(252)
  backtest_engine.py:1251 volatility = fv.get("annualized_volatility", 0.3) or 0.3
  backtest_trader.py:80  def size_position(self, probability, stock_vol, nav)
  backtest_trader.py:89  vol_scale = min(self.target_vol / stock_vol, 3.0)
  settings.py:286 backtest_target_vol = 0.15 ; trader default 0.15
  rotation_runner.py:64 "vol_barrier_multiplier" IS in _DEAD_KEYS (:60-69),
    with :122-128 logging it as inert "reverted in 9fbd9cd6".
  _compute_triple_barrier_label (backtest_engine.py:1066-1104, 1753 chars --
    Main's cycle-3 figure of 1753 is EXACT) uses entry_price, self.tp_pct,
    self.sl_pct, transaction_cost_pct. NO volatility term. CONFIRMED by reading
    the whole body, not by trusting the tripwire.
  vol_ratio 0.7995 -> inflation 1.2508x REPRODUCES.

## 4. My OWN adversarial work (not the author's matrix)

(a) Criterion 3 -- I widened the probe myself. The step's probe is 3 pandas
    tokens. I ran a 13-token census at 539f16eb~1 over backend/ including the
    SQL dedup idioms the step's probe cannot see (DISTINCT, GROUP BY,
    ROW_NUMBER, QUALIFY, groupby, unique()), then inspected EVERY pre-fix file
    referencing historical_prices (15 files). Findings:
      cache.py:758 ROW_NUMBER() -- MACRO series, not prices
      data_ingestion.py:110/214/319 SELECT DISTINCT -- WRITE-side existence
        checks, not a read-path dedup
      no other hit is on a price read path
    => the step's CONCLUSION is correct and I independently corroborate it.
    The probe is NARROWER than the class it names (NOTE, not a cap).

(b) Positive control -- the artifact claims the probe "finds this step's own fix
    in the post-fix tree" but no_dedup_before() never runs against HEAD. I ran it
    myself: at HEAD the same probe returns drop_duplicates 2 files,
    index\.duplicated 2 files, is_unique 2 files. The claim is TRUE; it is just
    asserted in prose rather than executed by the re-runnable script.

(c) Tripwire predicate -- I attacked _volatility_identifiers against the REAL
    subject in memory (no tree mutation). Main's claims reproduce:
      V1 daily_vol/vol_mult      -> ['daily_vol','vol_mult']  FIRES
      V2 self.daily_volatility   -> ['daily_volatility']      FIRES
      comment-only mention       -> []                        correctly silent
    MY mutants, all of which introduce a REAL volatility term:
      A1 sigma = close.pct_change().std(); tp = entry*(1+2*sigma)  -> MISSES
      A2 self.daily_sigma                                          -> MISSES
      A3 inline float(prices['close'].pct_change().std())          -> MISSES
      A4 self._atr_width(prices)                                   -> MISSES
    => the guard rejects one NAMING CONVENTION ("vol" substring), not the class
    "the barrier reads a volatility". It is NOT vacuous (it can fail, T2 and
    V1/V2 prove it) but it is narrower than its own message claims.
    WARN-level; the substantive claim is independently verified by me above.

(d) cap_is_accounted_for asserts the WRONG quantity. The 3.0 cap binds on
    target_vol/stock_vol, not on the inflation ratio 1/vol_ratio. With max
    multiplicity 2 the inflation is bounded by sqrt(2)=1.4142, which the script
    ITSELF prints, so `size_inflation < 3.0` cannot fail on this data --
    near-tautology (vacuity shape #4). I computed the correct quantities:
      AKAM vol_scale pre 0.4487 / post 0.3587 -> real inflation 1.2508,
      cap_binds_pre=False cap_binds_post=False.
    => the REPORTED NUMBER IS CORRECT; only the guard's operationalisation is
    wrong. WARN-level, alongside three genuinely behavioural gate invariants
    (vol_ratio_is_derivable, pre_fix_volatility_is_the_lower_one,
    size_inflation_is_above_one).

(e) MY OWN MUTATION MATRIX on cache.py, via sys.modules injection -- TREE NEVER
    TOUCHED (pre sha 9f5f1d6798833281 == post sha 9f5f1d6798833281):
      CONTROL (unmutated)                          suite GREEN, oracle GREEN
      QA-1 inert branch returns df.copy()          suite RED, oracle RED
           (test_dedupe_is_inert_on_a_unique_index + inert_on_unique_input_n0_dup0)
      QA-2 sort_index() dropped before dedup       SURVIVED  <- see below
      QA-3 cached_prices reverted                  suite RED (test_cached_prices_*)
      QA-4 dedupe becomes identity (df.loc[:])     suite RED (8 tests) + oracle RED
           (exact_on_duplicated_input_n1)
    QA-2 is an EQUIVALENT MUTANT with a named mechanism, NOT a finding: both
    production queries already order (preload SQL `ORDER BY ticker, date ASC`;
    cached_prices SQL `ORDER BY date ASC`) and pandas groupby preserves
    within-group order, so sort_index() is redundant on both read paths.
    sort_index() is also PRE-EXISTING code, not a new guard.
    => criterion 5's oracle and criterion 4's read-path tests are genuinely
    falsifiable on BOTH branches. Independently established.

(f) CAP-GUARD FALSE NEGATIVE -- EXECUTED, NOT ARGUED. I called gate_effect()
    directly with saturating inputs. All 8 gate invariants pass in EVERY case
    while the script prints a materially wrong inflation:
      AKAM real 0.3343/0.4182 -> reports 1.2510x, TRUE 1.2510x, cap unbound  OK
      vol 0.020/0.025         -> reports 1.2500x, TRUE 1.0000x  MISREPORT
      vol 0.040/0.050         -> reports 1.2500x, TRUE 1.0000x  MISREPORT
      vol 0.040/0.060         -> reports 1.5000x, TRUE 1.2000x  MISREPORT
    `cap_is_accounted_for` (verify_86_116.py:446) asserts `size_inflation < 3.0`.
    The cap binds on target_vol/stock_vol, NOT on the inflation ratio, and with
    max multiplicity 2 the inflation is bounded by sqrt(2)=1.4142 -- a bound the
    SAME function prints -- so the guard cannot fail on this table. It names a
    hazard ("at or above the cap the inflation saturates and this arithmetic no
    longer describes the outcome") that it is incapable of detecting, and the
    script takes --ticker/--start/--end so the false negative is REACHABLE.
    NAMED FIX: assert target_vol/a["vol_ann"] < 3.0 AND target_vol/b["vol_ann"]
    < 3.0 (target_vol 0.15, backtest_trader.py:54 / settings.py:286).
    NOTE the reported AKAM figure is CORRECT -- I computed the real vol_scale
    pair (0.4487 / 0.3587) myself and the cap does not bind.

(g) excess_equals_rows_minus_keys is an ALGEBRAIC IDENTITY, not a check. Over
    the grouping k: SUM(n) == total_rows and COUNT(*) of k == keys, so
    SUM(n-1) == total_rows - keys by construction. experiment_results:40-41
    credits it with protection it does not have: "The script asserts
    excess_rows == total_rows - keys so the two cannot drift apart silently."
    It cannot detect a normalisation error. Criterion 1's actual requirement
    (normalisation rule stated beside every share) IS met independently by the
    printed NORMALISATION RULE block and the "% OF KEYS" / "% OF ROWS" labels.

(h) SCOPE-HONESTY: "no restart pending" is FALSE for the in-process API path.
    experiment_results:194-195 states "no restart pending (the change is in the
    backtest read path, not in a running process's hot loop; the backtest loads
    it per run)". MEASURED:
      uvicorn pid 41635 started man. 17 aug. 15.57.16 2026 (etime 16:10:20),
        no --reload; cache.py landed 2026-08-18 07:55.
      backend/backtest/backtest_engine.py:25  `from backend.backtest import cache`
        -- module-level, so the process holds the PRE-FIX module in sys.modules.
      backend/api/backtest.py:1008 `await loop.run_in_executor(_heavy_executor,
        engine.run_backtest)` -- backtests execute INSIDE that uvicorn process.
    So every API-triggered backtest still reads duplicated frames until a
    restart. The parenthetical justification ("the backtest loads it per run")
    is wrong: preload_prices is called per run, but from the already-imported
    module object. CLAUDE.md requires this be recorded as NOT YET IN FORCE with
    the running pid and start time and added to the pending-restart list.
    The fix IS in force for every fresh process (harness, CLI, scripts) -- I
    confirmed a fresh import carries _dedupe_index and preload_prices calls it.

(i) Criterion 4 no-DML check: `git diff 34a56b03..HEAD -- backend/ scripts/`
    grepped for DELETE FROM / TRUNCATE / DROP TABLE / CREATE OR REPLACE TABLE /
    MERGE INTO / load_table_from / insert_rows / UPDATE..SET -> ZERO added lines.
    All four audit_basis consumers (historical_data.py:53,:385,
    candidate_selector.py:52, data_server.py:99) route through
    cache.cached_prices, and the _prices_full slice path inherits the deduped
    frame -- so the two patched read paths cover the whole named surface.

## 5. Criteria roll-up
1 MET  2 MET  3 MET  4 MET  5 MET  6 MET  7 MET-with-gap (see below)

## 6. Findings that cap the verdict
F1 (f) cap_is_accounted_for cannot fail for the condition it names -- executed
   false negative on a money-path evidence script. WARN.
F2 (g) excess_equals_rows_minus_keys is an identity credited in prose with
   drift-protection it does not provide. WARN.
F3 (h) "no restart pending" overclaims -- the running uvicorn (pid 41635,
   started 16h before the fix) still serves backtests from the pre-fix module.
   WARN, scope honesty + CLAUDE.md batched-restart disclosure rule.
F4 (c) _volatility_identifiers rejects one naming convention; sigma/atr-named
   vol terms MISS. NOT vacuous (V1/V2/T2 kill it). NOTE.
F5 (b) the "probe finds this step's own fix in the post-fix tree" positive
   control is prose-only, not executed by no_dedup_before(). I executed it and
   it holds (2/2/2 files at HEAD). NOTE.
F6 live_check:207 full-suite block is the PRE-pin-fix capture (20 failed /
   3633 passed) and is not relabelled as such; cycle 2's evaluator measured
   19/3635. The section is internally coherent. NOTE.

Cycle-2's sole capping finding IS genuinely closed: I reproduced V1/V2 killing
the v3 AST predicate and T1/T2/T3 all KILLED in my own matrix re-run.

## 7. Regression, measured by me independently
FULL suite (no -k selection), backend/tests/, -p no:randomly:
  19 failed, 3635 passed, 12 skipped, 5 xfailed, 1 xpassed in 514.00s
This is EXACTLY cycle 2's evaluator figure (19/3635). Attribution corroborated:
  - NONE of the 19 failing test files references backtest.cache / cached_prices /
    preload_prices / historical_prices / _dedupe_index (grepped all 13 files).
  - test_phase_86_6_subprocess_channel::test_the_optin_IS_honoured... is present
    -- the declared ordering artifact. 18 + 1 = 19.
  - 86.118 IS filed (pending) for the 18 pre-existing failures; 86.117 IS filed
    (pending), so "does not unblock 86.117 by fiat" holds.
  - The pin fix works: test_phase_82_12_string_column_guards -> 30 passed.
The live_check's "20 failed / 3633 passed" block is the PRE-pin-fix capture and
is not relabelled; the three runs reconcile (3633+20=3653; +1 test added in
cycle 2 -> 3635+19=3654).

## 8. Disclosure about my own actions
I ran the author's mutation_86_116.py, which rewrites cache.py and
verify_86_116.py in place and restores them. I recorded sha256 PRE and POST
myself: cache.py 9f5f1d6798833281 -> 9f5f1d6798833281 and verify_86_116.py
22e302ecb4168f47 -> 22e302ecb4168f47, both identical. This DID advance
cache.py's mtime (07:55:00 -> 07:58:32) with byte-identical content.
My own mutation matrix (section 4e) used sys.modules injection and never
touched the tree at all.
No UI claims in this step -> section 1c does not apply; no frontend/** in the
diff -> section 1b does not apply.

## 9. Verdict reasoning (the verdict itself is the structured return, not this)
Product: correct, minimal, complete, byte-identical across all three cycles,
mutation-proven by 11 author cells + 4 independent cells of mine.
All seven criteria substantively MET. Harness compliance clean. No unintended
production change.
Capping: three WARN-level findings (F1 executed false negative in the money-path
evidence script, F2 prose credits a vacuous identity with drift protection, F3 a
false "no restart pending" disclosure measured against a 16h-old uvicorn that
serves backtests in-process). Each has a one-line named fix. None changes a
reported number for the data actually used.
Verdict returned: CONDITIONAL. Not FAIL -- nothing is wrong, no criterion is
unaddressed, no reported number is misstated for the inputs actually used.

