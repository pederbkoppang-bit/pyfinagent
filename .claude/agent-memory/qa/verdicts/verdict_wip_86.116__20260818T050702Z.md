STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.116
WRITTEN: 2026-08-18T05:07:02Z

# Q/A write-first record -- step 86.116 (price de-duplication on READ)

## Prior-attempt evidence
- qa_wip.py 86.116 --spawned-at 2026-08-18T05:07:02Z: source_present=true,
  attempt_number=1 (status ok), prior_attempts=0, prior_records=[].
- verdict_history_86_21.py --step 86.116 --evidence-only: status=no_rows_for_step,
  verdicts=(none). Cross-check prior_attempts(0) vs ledger rows(0): CONSISTENT.
  sequence: no prior verdicts recorded for this step.

## A. Harness compliance (5 items) -- ALL CLEAN
1. research-gate-before-contract: research_brief_86.116.md (24,632 B, 06:31:13)
   precedes contract_86.116.md (06:34:29). Envelope: brief_status COMPLETE,
   external_sources_read_in_full=6 (>=5), urls_collected=25 (I counted 25 unique
   URLs independently), recency_scan_performed=true (section at line 110),
   gate_passed=true. Three sources are ADVERSARIAL (duplication IMPROVING models);
   real gate work, not confirmation-seeking.
2. contract-before-generate: contract 06:34:29 < tests 06:36:11 < verify 06:38:23
   < mutation 06:39:26 < cache.py 07:05:12. OK.
3. experiment_results_86.116.md present (8,959 B). OK.
4. log-last: masterplan 86.116 status=pending; harness_log has NO
   `phase=86.116 result=` header (only a "filed by 86.59" mention). OK.
5. no-verdict-shopping: attempt 1. N/A.
Contract cites the brief in References (lines 141-143). OK.

## B. Deterministic checks -- all reproduced by me
- IMMUTABLE COMMAND: stdout `parses`, EXIT=0. REPRODUCED.
- pytest backend/tests/test_phase_86_116_price_dedup.py -q -> 12 passed in 0.81s.
- ruff F821,F401,F811 over DERIVED scope (`git diff --name-only 34a56b03 HEAD --
  '*.py'` = 5 files; non-empty guard asserted first): "All checks passed!" exit=0.
- cache.py sha256 9f5f1d6798833281c12c4b17387fac31ff62fa167e94cf51c1089f10aa6c8bf6
  before AND after every script I ran; `grep -c MUTANT` = 0. Tree not left dirty.
- No frontend files in the diff -> 1b N/A. No UI claims -> 1c N/A.
- Runtime smoke: cache.py imported and both read paths DRIVEN in-process.
- No DML in the commit: grep of added lines for DELETE FROM/DROP TABLE/TRUNCATE/
  MERGE/INSERT INTO/load_table_from/.delete( -> ZERO hits. Criterion 4's "nothing
  deleted" VERIFIED, not taken on trust.
- ASCII-only logger message (security.md rule): compliant.

### FINDING F1 (evidence-side, fixable): the advertised re-runnable evidence
### command CRASHES today and regenerates NONE of the criterion 1/2/5/6 evidence.
`python scripts/qa/verify_86_116.py` (line 1 of the live_check "Re-runnable" block)
aborts immediately:

    AssertionError: INVARIANT FAILED: no_dedup_existed_before_this_step --
    a de-duplication already existed in HEAD:
    {'drop_duplicates': 2, 'index\\.duplicated': 1, 'is_unique': 2}

Root cause: `verify_86_116.py:281` defaults `--base-rev` to `HEAD`, and
`no_dedup_before(a.base_rev)` is the FIRST call in `main()` (:286), before the
census, the driven-harm comparison, the parity-oracle report and the gate section.
HEAD was the PRE-fix tree only while the fix was uncommitted; the fix is now
committed (539f16eb) so HEAD is the POST-fix tree and the criterion-3 invariant is
self-refuting. Consequences: (a) the printed line "pre-fix revision under test:
HEAD" is now FALSE; (b) criteria 1/2/5/6 evidence is unreachable via the advertised
command; (c) `--offline` has the same defect (same first call).
ONE-LINE FIX: default `--base-rev` to the pre-fix sha. PRODUCT CODE UNAFFECTED.

WORKAROUND VERIFIED -- and every number reproduces BYTE-FOR-BYTE:
`python scripts/qa/verify_86_116.py --base-rev 34a56b03` -> "OK: all 27 invariants
hold", EXIT=0, output matching live_check_86.116.md exactly:
  census 1,859,482 rows / 1,152,607 keys / 706,875 dup keys (61.33% OF KEYS) /
  706,875 excess rows (38.01% OF ROWS) / max mult 2 / 336 of 513 tickers /
  394,719 dup keys whose close differs; by-year 2017 90.5% ... 2025 64.3%,
  2026 0.1%; AKAM n 390->250, mom_1m 0.83->-0.52, mom_3m -1.60->15.04,
  rsi 23.7->54.5, vol_ann 0.3343->0.4182, span_21_sessions 12->22;
  barrier scale 0.7995 vs 0.7071.
Hand-checked arithmetic: 706,875/1,152,607 = 61.33%; 706,875/1,859,482 = 38.01%;
1,859,482-1,152,607 = 706,875; the ten year-bucket key counts sum to EXACTLY
1,152,607 and the dup counts to EXACTLY 706,875.

### FINDING F2 (MATERIAL, criterion 6): the quantified gate mechanism credits a
### DOCUMENTED DEAD KEY, and the LIVE amplifier -- one line away in the same file
### -- is missed entirely.
Claimed mechanism (b), in the artifact, the script (verify_86_116.py:253, :377),
the cache.py comment and the commit message: "`quant_optimizer.py` sets
`barriers = daily_vol * vol_barrier_multiplier`, so a depressed daily_vol scales
EVERY triple-barrier proportionally". Traced:
- That string exists in quant_optimizer.py ONLY as a COMMENT at :213 describing a
  search-space bound; the bound is `(0.0, 5.0)` at :214.
- `_apply_params_to_engine` WRITES `engine._strategy_params["vol_barrier_multiplier"]`
  at :715-716. NOTHING READS IT. Repo-wide (excl. .venv) the token appears in 7
  places outside this step's own script: the bound, the write, a cache-key list
  (:738), a docstring + `_DEAD_KEYS` entry in rotation_runner.py, a 0.0 default in
  archetype_library.py.
- `backend/autoresearch/rotation_runner.py:58-69` names it explicitly:
      # Keys written by the optimizer setter but with NO engine reader (reverted
      # in 9fbd9cd6). ... _DEAD_KEYS = (..., "vol_barrier_multiplier", ...)
  and :21-24 "DEAD-KEY HONESTY: ... nothing reads them today."
- The real label fn `backtest_engine._compute_triple_barrier_label` (:1066-1103)
  uses `tp_price = entry_price * (1 + self.tp_pct/100 + round_trip_cost_pct)` --
  FIXED percentage barriers, NO volatility term.
- `features["daily_volatility"]` is WRITTEN once (historical_data.py:132) and read
  NOWHERE.
- Same-file precedent 5 lines below the comment relied on (quant_optimizer.py:215-224):
  phase-82.46 deleted four params for exactly this reason. Also
  `_compute_vol_target_scale`, named in the :718 comment as a reader, DOES NOT
  EXIST anywhere in the repo -- these comments are systematically stale.
Mis-labelling: `gate_effect()` (verify_86_116.py:256-267) computes
`vol_ratio = raw.vol_daily / fixed.vol_daily` and returns THE SAME FLOAT under both
`vol_ratio_pre_over_post` AND `barrier_width_scale`. As a daily-vol ratio 0.7995 is
correct; as a "measured barrier-width scale" it is an inference about dead code.

THE LIVE MECHANISM THE STEP MISSED (I traced every hop):
  historical_data.py:53  get_point_in_time_prices -> cache.cached_prices
  historical_data.py:126-128  annualized_volatility = daily_returns.std()*sqrt(252)
  backtest_engine.py:1251  volatility = fv.get("annualized_volatility", 0.3)
  backtest_engine.py:1253-1258  signals.append({... "volatility": volatility})
  backtest_trader.py:146-147  volatility = sig.get("volatility",0.3);
                              dollar_amount = self.size_position(prob, volatility, nav)
  backtest_trader.py:89   vol_scale = min(self.target_vol / stock_vol, 3.0)
Inverse-volatility sizing. The measured 0.7995x depression of vol therefore
INFLATES position size by ~1/0.7995 = 1.25x on affected tickers (subject to the 3.0
cap) -- a direct NAV/Sharpe/DSR/PBO effect, materially larger than a label-boundary
shift would be. Note `features["daily_volatility"]` (the DEAD feed the step
credits) sits at historical_data.py:132, four lines BELOW the LIVE
`annualized_volatility` at :128. The step picked the dead sibling.
=> criterion 6 NOT MET AS REPORTED. Mechanism (a) (features -> candidate
selection) IS real and I verified it end to end (engine:701 ->
candidate_selector.py:52 -> screener `_pct_change`/`_compute_rsi` ->
`rank_candidates`, whose multipliers screener.py:307-310 the step quotes EXACTLY
right: `if rsi > 80: score *= 0.7 / elif rsi < 20: score *= 0.8`). No threshold was
moved. But the criterion's only quantified claim is attributed to dead code.

### FINDING F3 (WARN, guard vacuity -- EXECUTED, tree untouched)
Independent cells on SCRATCH COPIES of cache.py (repo sha256 unchanged after):
- Q2: `_dedupe_index` mutated to value-keyed `df.drop_duplicates()`. With the
  SHIPPED `_fake_rows` fixture (duplicates are byte-identical `dict(r)` copies)
  BOTH read-path tests are BLIND: preload unique=True len=6, cached unique=True
  len=6 -- identical to control. With `close += 1e-9` (the shape 394,719 of
  706,875 real duplicated keys have) the same mutant leaks: unique=False len=12
  on both paths. => the method choice -- which the step itself calls "THE METHOD
  IS THE FINDING" -- is pinned ONLY by the helper-level
  `test_value_keyed_dedup_is_insufficient`; neither CALL SITE can see a method
  swap. Vacuity shape #5 (fixture cannot represent the failure). Genuine coverage
  exists elsewhere -> WARN. ONE-LINE FIX: vary the duplicate's close in `_fake_rows`.
- Q1: `test_both_read_paths_are_covered` is a source scan and is comment-defeatable
  -- replacing the `cached_prices` call with the raw expression plus a comment
  containing the literal `_dedupe_index` made the scan return True (DEFEATED) while
  the behavioural test still killed (unique=False len=12). Shape #3/#8, NOT sole
  coverage -> NOTE.
- Q3: `keep=False` mutant -> 0 rows on a doubled frame; shipped `len(out)==N`
  assertion kills it. No finding.

### Reproduction of the author's matrix (criterion 7)
`python scripts/qa/mutation_86_116.py` -> control rc=0 collected=12 GREEN;
M1-M6 ALL [KILLED] (rc=1, collected=12, NAMED test among the failures); E1 declared
EQUIVALENT-BY-DESIGN; "KILLED 6 / 6 SURVIVED 0 UNSCORABLE 0"; "restore verified:
sha256 unchanged". I confirmed the sha independently before/after; 0 MUTANT markers.
Harness quality is high: control-first, exit-1-only (pytest exit 5 explicitly
refused), collected-count parity, named-test-must-fail, SIGTERM/SIGINT/SIGHUP
restore, refusal to start from a poisoned baseline, anchor-uniqueness check.

### Independent corroborations (reproduced by me, not taken on trust)
- BQ driven myself: AVB 2026 raw=159 distinct=155 `drop_duplicates()`=**159**
  `~index.duplicated()`=**155**; AKAM 2025 raw=390 distinct=250 BOTH methods=250.
  The method claim AND the step's own counter-example disclosure both hold exactly.
- E1's equivalence evidence, whose derivation is NOT shipped in any script, I
  derived myself: dup keys w/ differing close 394,719; relative gap p50 0.0000%,
  p99 0.0000%, MAX 0.9326%, keys above 2% = 0. REPRODUCES EXACTLY.
- Line pins: re-derived with `schema_oracle._row_key_reads(ast.parse(cache.py))`
  -> report_date line **700**, date line **760**. The modified test_phase_82_12
  pins are correct and source-derived, as claimed.
- Call-site completeness: `_prices_full` has exactly ONE assignment (cache.py:293,
  deduped), `_prices_cache` exactly ONE (:641, from the deduped frame).
  `set_index` exists in 2 files under backend/ (cache.py; yfinance_tool.py is a
  `reset_index` SUBSTRING match on a yfinance path, not a BQ price read). No
  unguarded twin. Both read queries read in full: no DISTINCT/QUALIFY/ROW_NUMBER,
  so no SQL-level dedupe either.
- All four consumers named in the audit_basis (historical_data.py:53 and :385,
  candidate_selector.py:52, data_server.py:99) go through `cache.cached_prices`.
- ASK-1's "bounded and terminal" claim is corroborated by a mechanism the artifact
  does not name: `data_ingestion._get_existing_price_dates` (SELECT DISTINCT
  ticker,date, phase-75.9, fail-closed). Strengthens the claim.

### NOTE N1 (disclosure): "No restart pending" is not established.
The running uvicorn (pid 41635) started 2026-08-17 15:57, ~15h BEFORE the fix
commit. `backend.backtest.cache` is reachable from that process by LAZY import
(api/backtest.py:981, agents/mcp_servers/data_server.py:31), so if any request has
taken those paths the process holds the PRE-FIX module and the fix is not in force
there. The artifact's "the backtest loads it per run" is true for fresh processes
(harness/optimizer/scheduled jobs) but does not cover the long-running API. Correct
disposition under the operator's batch-restarts rule: list it as pending-restart
rather than assert none is pending. NOT a criterion violation.

### NOTE N2: the fix is unconditionally live on the money path -- no flag.
On the 336 affected tickers it fires immediately, so every future backtest/gate
number differs from every historical one as of 539f16eb. This is intended and
ASK-2 gestures at it, but the artifact says "No flag promoted" without stating
that there is no flag at all.

## C. Criterion-by-criterion
C1 census RE-MEASURED + keys/rows/tickers + per-year + normalisation rule: MET.
C2 positional harm DRIVEN through real code: MET (`_pct_change` returns percent,
   so the sign flip is real; reproduced exactly).
C3 no existing layer de-duplicates, POSITIVE CONTROL: MET (reproduced at the
   pre-fix rev; corroborated behaviourally by 390 rows for 250 dates and by
   reading both read queries). NOTE: the control token `set_index` substring-
   matches `reset_index`; harmless -- substring matching can only inflate a
   positive control, never suppress a zero.
C4 de-dupe ON READ in cache.py, no DELETE/table rewrite, repair as numbered ask:
   MET (zero DML verified; ASK-1/2/3 numbered).
C5 fix-absent / inert parity against an ORACLE: MET (12 shapes, 7 inert returning
   the SAME OBJECT / 5 active returning exactly index.nunique() rows, plus
   `oracle_exercised_both_branches` against one-sided coverage).
C6 effect on DSR/PBO reported: **NOT MET as reported** -- F2.
C7 mutation-test every new guard: MET as run, with the F3 read-path fixture caveat.

## Verdict shape
CONDITIONAL. The PRODUCT fix is correct, minimal, complete over both price read
paths and mutation-proven; six of seven criteria are met and every headline number
reproduces. Two evidence defects cap it: F2 (criterion 6's quantified mechanism is
a documented dead key while the live amplifier is missed) and F1 (the advertised
re-runnable evidence command aborts). Both are fixable without touching production
code. F3 is a WARN-level fixture gap with a one-line fix.

## Regression -- I RE-RAN THE FULL SUITE AND IT REPRODUCES EXACTLY
My run: **19 failed, 3634 passed, 12 skipped, 5 xfailed, 1 xpassed in 509.53s**.
Author: 20 failed, 3633 passed, 12 skipped, 5 xfailed, 1 xpassed in 504.54s.
The delta is exactly ONE test moving failed -> passed:
`test_phase_82_12_string_column_guards::test_classified_line_numbers_still_point_at_a_row_read`
-- the pin the author fixed. 20-1=19, 3633+1=3634. Arithmetic agrees exactly.
My observed failing set is EXACTLY the author's enumerated 18
(23_2_6 x1, 40_2 x1, 57_1 x3, 60_3 x1, 62_4 x1, 75_17 x3, 75_19 x1,
75_prompt_contracts x1, 75_sre_ops x2, 82_39 x1, 82_48 x2, portfolio_swap x1 = 18)
PLUS the single ordering artifact
`test_phase_86_6_subprocess_channel::test_the_optin_IS_honoured_...`, which I
confirmed **passes when run alone (1 passed in 6.06s)**.
`test_phase_82_12_string_column_guards.py` alone: **30 passed**.
NONE of the 19 touches cache.py or de-duplication. ZERO failures attributable to
this change -- MEASURED by me, not inherited.
(pytest-randomly is NOT installed, so my `-p no:randomly` was a no-op and the run
is like-for-like with the author's command.)

## NOTE N3: contract transcription of criterion 6 is not verbatim
Masterplan: "the effect on the existing gates is reported (DSR, PBO) rather than
only on the price frames..."; contract_86.116.md:73: "the effect on the existing
gates (DSR, PBO) is reported rather than only on the price frames...". C1-C5 and
C7 are verbatim; C6 transposed the parenthetical. **The masterplan itself is
UNEDITED** -- I read the criteria straight from `.claude/masterplan.json`, so this
is a contract transcription slip, not the forbidden amendment of immutable
criteria. Semantically identical -> NOTE only.

## NOTE N4: unrelated uncommitted production edits in the tree
`backend/api/sovereign_api.py` (mtime 2026-08-17T15:54:50, 6 lines) and
`backend/services/autonomous_loop.py` (2026-08-17T21:42:56, 18 lines) are modified
and uncommitted, but both PREDATE all of 86.116's artifacts (earliest 2026-08-18
06:31) and neither is in commit 539f16eb, whose stat is exactly 10 files /
1,835 insertions / 5 deletions. NOT attributable to this step.

## Code-review heuristics (5 dimensions) -- no BLOCK, no WARN
Security: no secrets, no subprocess/eval on non-literal args in production code
(mutation_86_116.py uses sys.executable + literal args, a dev script), no new deps,
no prompt/LLM surface touched. Trading domain: no kill-switch, stop-loss,
perf_metrics, position-cap or crypto path touched; no LLM output reaches execution.
Quality: no broad except, no print in production, ASCII logger, additive-only
public surface (`_dedupe_index` is private, single-module). Anti-rubber-stamp:
12 behavioural tests accompany 48 changed lines; the only source-scan guard is
supplementary (F3). LLM-evaluator anti-patterns: attempt 1, no prior verdict, no
rebuttal to be swayed by.

STATUS: COMPLETE -- write-first record, still NOT a verdict
COMPLETED: 2026-08-18T05:25:07Z
