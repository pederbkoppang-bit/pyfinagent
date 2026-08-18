STATUS: INCOMPLETE -- not a verdict
STEP: 86.116
WRITTEN: 2026-08-18T05:56:47Z

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

## 5. Criteria (pending final roll-up)
(continued below)
