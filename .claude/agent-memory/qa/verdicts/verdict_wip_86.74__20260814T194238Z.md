STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-14T19:42:38Z
COMPLETED: 2026-08-14T19:55:09Z

# Q/A write-first record -- step 86.74 FULL-STEP grade

Spawn context: Workflow rail, full-step grade. Prior: Cycle 190 NO-VERDICT (x2 rail
drops), 191 CONDITIONAL, 192 CONDITIONAL, 193 PASS (SCOPED to cycle-4 work only).
Main claims exactly two changes since Cycle 193: (1) C4 now MEASURED (live_check §3
REPLACED), (2) C7 re-derived in new §2g.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable cmd, git status, lint, runtime smoke
C. LLM judgment vs the 10 immutable criteria; mutation matrix M1/M2/M3 independently

## Findings log (append-only)

### Prior-attempt evidence
- qa_wip.py 86.74 --spawned-at 2026-08-14T19:42:38Z: source_present=true,
  attempt_number=8 (status ok, is_lower_bound=true), prior_attempts=7,
  records_retained=8 (gauge, not counter), records_pruned_known=null.
- verdict_history_86_21.py --step 86.74 --evidence-only: status=no_rows_for_step,
  verdicts=(none). LEDGER IS STALE: attempt_number(8) > ledger count(0/no rows).
  sequence: UNKNOWN from the authoritative source. harness_log (secondary) shows
  Cycles 190 NO-VERDICT x2 / 191 CONDITIONAL / 192 CONDITIONAL / 193 PASS(scoped).

### B. Deterministic
- IMMUTABLE CMD `bash -c 'source .venv/bin/activate && python -m pytest
  backend/tests/test_phase_66_2_risk_judge_shape.py -q'` -> **41 passed**, EXIT=0.
- Lint gate over a scope DERIVED from the union of the 10 phase-86.74 commits
  (6 .py files, non-empty guard passed, xargs used -- no zsh word-split trap):
  `uvx ruff check --select F821,F401,F811` -> "All checks passed!", exit 0.
- Runtime import smoke, all 4 changed backend modules import clean in venv:
  risk_debate, autonomous_loop, portfolio_manager, signal_attribution -> OK.
- Working tree: 6 uncommitted production files (backend/api/sovereign_api.py,
  frontend/src/{app/page.tsx,components/*.tsx}). mtimes 11:24Z-11:29Z, i.e. BEFORE
  contract_86.74.md (14:19Z) -> PRE-EXISTING other-workstream work, NOT introduced
  by this step; 9034ddfb touches only handoff/ files. Tree did NOT move during my
  grade (live_check mtime 19:37:01 local = 17:37Z, before my spawn 19:42Z).

### C4 -- INDEPENDENTLY RE-DERIVED FROM BIGQUERY (my own queries)
- BASELINE 2026-07-20..2026-08-13: total=129 dec=0 lvl=0 pct=0  -> EXACT MATCH.
- POST-FIX 2026-08-14:              total=6   dec=6 lvl=6 pct=6 -> EXACT MATCH.
- Per-ticker table reproduces EXACTLY, all six rows incl. timestamps:
  PANW REJECT 0.0 18:35:23Z / WDAY REJECT 0.0 18:36:27Z / HPE REJECT 0.0 18:37:54Z /
  STX APPROVE_REDUCED 2.0 19:02:32Z / MRVL REJECT 0.0 19:04:26Z /
  NTAP APPROVE_REDUCED 2.0 19:32:26Z.
- Enabling facts reproduce: 9d14291e cIso=2026-08-14T16:36:20+02:00 (=14:36:20Z);
  `ps -o pid,lstart -p 85562` = 17.52.08 local (=15:52:08Z), 76 min later;
  cycle_history 68925781 started 18:00:00.145Z completed 19:33:13.393Z, n_trades=0.
- NOTE (mine, not Main's): risk_level is the SINGLE constant 'HIGH' on all 6 rows.
  Main's anti-vacuity argument cites decisions+pcts only, which is accurate -- but
  the risk_level column on its own is indistinguishable from a literal write.

### C7 -- INDEPENDENTLY RE-DERIVED (third-party confirmation, which Main could not do)
- Instrument control reproduced by ME: DELL 2026-08-13 judge.decision='REJECT',
  JSON_VALUE('$.final_synthesis') IS NULL -> TRUE (FALSE POSITIVE);
  JSON_QUERY -> FALSE (correct). Whole table: JSON_VALUE NULL on 573/573 rows
  (Main measured 567/567; table has since grown by today's 6 -> consistent).
- Population: paper_trades all_rows=66, BUY=34, distinct trade_id=66 -> EXACT MATCH.
- My own query, written from the stated enumeration rule (ticker + |analysis_date -
  TIMESTAMP(analysis_id)| < 2s, verdict nested-first then flat, JSON_QUERY only):
    INVERSION                                  1  (DELL, and only DELL)
    PERMITTED                                  0  (bucket empty)
    UNDET_truncated_no_final_synthesis        19
    UNDET_no_row_within_2s                    14
    UNDET_fs_present_but_no_risk_assessment    0  (bucket empty)
    SUM = 34 = full BUY population
  -> EVERY BUCKET REPRODUCES EXACTLY. Main's ask (b) is now satisfied by a
  genuinely independent derivation.
- DISCRIMINATION CONTROL (mine): inverting the INVERSION predicate moves DELL to
  PERMITTED_INVERTED (1 row) -> the predicate is not constant-true; the two zeros
  are measured zeros, not vacuous.

### A. Harness compliance 5/5 CLEAN
1. research_brief_86.74.md: brief_status COMPLETE, gate_passed true,
   external_sources_read_in_full=7 (>=5), urls_collected=27 (>=10),
   recency_scan_performed=true. PASS.
2. contract-before-generate: research 10:24:44Z < contract 14:19:46Z < first code
   commit 9d14291e 14:36:20Z (git, not mtime -- portfolio_manager.py's birth time
   is 15:34:21Z because the mutation harness rewrites the file, so mtime/birth is
   NOT usable for that file). PASS.
3. experiment_results_86.74.md present. PASS (but see contradiction finding below).
4. log-last: masterplan 86.74 status="pending"; harness_log carries only prior
   cycles 190/191/192/193, the in-flight cycle is absent (correct). PASS.
5. no-verdict-shopping: live_check_86.74.md changed (+96/-10 in 9034ddfb) since
   the Cycle-193 grade -- evidence CHANGED. PASS.

### MUTATION MATRIX -- run BY ME, in memory, zero writes to the tree
Control observed GREEN and DISCRIMINATING first (real decide_trades):
  REJECT/0% OFF -> None ; REJECT/0% ON -> None ; APPROVE/3% -> 719.93 (=3% NAV) ;
  ABSENT -> 2399.77 (=10% NAV, the default IS still reachable)
  M1 restore falsy `if raw:` in _coerce_pct  -> OFF 2399.77 / ON 2399.77  KILLED
     (reproduces the DELL harm EXACTLY: 10.00% of NAV; kills in BOTH flag states)
  M2 restore `or 10.0` at the sizing seam    -> OFF 2399.77 / ON 2399.77  KILLED
     differential: legitimate 3% still 719.93 -> kill is ATTRIBUTABLE
  M7 excise the swap $50 floor (real _compute_swap_candidates)
     control 0% -> [] ; 3% -> [SELL OLD, BUY NEW 300.0] ; ABSENT -> [SELL, BUY 1000.0]
     mutant  0% -> [('SELL','OLD',None), ('BUY','NEW',0.0)]  KILLED = the orphan harm
     mutant  3% -> pair INTACT -> attributable, not harness collapse
  M5 drop `or pos_pct is not None` in signal_attribution:
     SURVIVES on the DELL shape (decision='REJECT' already satisfies the guard)
     KILLED on the pct-only shape (no decision text) -> the clause IS load-bearing,
     but for a DIFFERENT shape than the DELL one. Not a finding; mechanism named.
  MUT remove nested-first `risk.get("judge")` -> DELL agents drop to ['Trader'] KILLED
     -> nested-first, not the pct clause, is what fixes the DELL shape.
  FIRST ATTEMPT AT M7 WAS UNSCORABLE (my harness produced [] for every input incl.
  the legitimate 3%); rebuilt against _compute_swap_candidates directly. Recorded
  because an undiscriminating probe scores a false KILL.

### C3 -- derived by me
AST sweep of portfolio_manager.py for BoolOp(Or) with right operand 10.0 /
DEFAULT_POSITION_PCT: **0 sites** (positive control: synthetic `a or 10.0` IS
detected). Exhaustive _sizing_pct grid (6 states x 4 pcts + no-state-key shapes):
the default is yielded ONLY when the resolved state is ABSENT.
FINDING (NOTE): cells (ABSENT, pct=0.0) and (ABSENT, pct=3.0) also return 10.0.
Unreachable from the single write site (:341 forces position_pct=None whenever
kind!=SIZE, verified from source), so the enumeration is CORRECT -- but the
docstring at :1057-1062 claims the enumeration is "TRUE BY CONSTRUCTION rather
than true by a reachability argument that a future caller could silently
invalidate", and for that cell it is still a reachability argument.

### C5 -- MET BEHAVIOURALLY, evidence NOBODY presented
The unit test is a SOURCE SCAN (vacuity shape #1). But the LIVE backend log
(backend.log, running pid 85562) carries six real completion lines from the
18:00Z cycle, all with ticker=, matching the BQ rows to the second AND to the
value:
  20:35:23 ticker=PANW REJECT HIGH 0% | 20:36:27 WDAY REJECT 0% | 20:37:54 HPE REJECT 0%
  21:02:32 STX APPROVE_REDUCED 2%     | 21:04:26 MRVL REJECT 0% | 21:32:26 NTAP APPROVE_REDUCED 2%
(local CEST = UTC+2). Attributable WITHOUT inference. This also cross-corroborates
C4 far better than "two distinct decisions": the persisted columns reproduce the
judge's live per-ticker output line-for-line, so they cannot be a literal write.

### C6 -- seam MET, live gap UNDISCLOSED
Both reference records reproduce EXACTLY: DELL 2026-08-13 = 517 chars, 3 agents
['Quant','SignalStack','Trader'] no RiskJudge; NTAP 2026-07-31 = 1232 chars,
4 agents incl. RiskJudge. Driving the real extract_all_signals: DELL's nested
REJECT/0% now emits RiskJudge; empty risk_assessment still emits nothing
(discrimination control holds). BUT the 18:00Z cycle executed 0 trades, so there
is NO post-fix signals_log row for a gated buy -- C6 rests on the unit seam only,
and experiment_results §6 "What I could NOT verify" does NOT list this.

### C10 -- MET, confirmed live
Every removed line is the defect (4x `or 10.0`, the flag workaround, both `if pct:`
guards). The only threshold-shaped ADDITION is `if buy_amount < 50` at the swap
seam -- a TIGHTENING that mirrors the main path's existing floor at :536.
GET /api/paper-trading/portfolio on the running backend: DELL cost_basis 2392.26,
risk_judge_position_pct=None -- NOT liquidated, NOT resized. NTAP 950.9 / 4.0.

### FLAG IS NOW FULLY VESTIGIAL (stronger than Main claimed)
`grep -rn paper_risk_judge_shape_fix_enabled backend/ | grep -v backend/tests/`
returns ONLY settings.py:350 (the Field), settings_api.py:283 (an env-var name
map) and a docstring at portfolio_manager.py:1104. ZERO production reads. So
criterion 8's "exercised in BOTH flag states" holds BY CONSTRUCTION for every
assertion, not just the 3 parametrised ones. experiment_results §4 says only "the
flag's SIZING half is now vestigial" -- the whole flag is.

### PROSE FINDINGS -- experiment_results_86.74.md does not reproduce
1. CONTRADICTION (blocking-class): §C4 lines 155-160 still say "Post-fix populated
   share is NOT reported as a live number" and "the backend has NOT been
   restarted". live_check §3 says the opposite and I MEASURED the opposite
   (pid 85562 started 15:52:08Z; d6a1500a "session-end backend restart, verified").
   Two live claims across the two artifacts on the ONE criterion this cycle exists
   to close. live_check §3's own parenthetical names this exact failure mode and
   fixes it WITHIN live_check -- the cross-file half was left standing.
2. §6 "What I could NOT verify" item 1 (C4) and item 4 ("the running process still
   holds the pre-fix code") are both FALSE now. Item 4 is dangerous in the
   opposite direction: a reader could restart or force a cycle believing the fix
   is not in force.
3. Header line 4: "Verification command: GREEN, `37 passed`". MEASURED: 41 passed.
   (§9d already corrects a different figure to 41; the header was never updated.)
4. §C8: "9 -> 34" test functions and "17 -> 55" asserts. MEASURED by my own AST:
   38 and 62 (grep -c = 64). Stale by exactly the cycle-4 swap additions. Direction
   is UP so no removal is hidden -- but criterion 8 exists precisely so a net
   removal is VISIBLE, and a stale-low denominator degrades that.
5. §C6 graded OK with no disclosure of the missing post-fix production row.

### ANSWERS TO MAIN'S FOUR ASKS
(a) C4 / n=6: SUFFICIENT AS THE CRITERION IS WORDED. It asks for the post-fix
    share "reported against the measured baseline ... with the query that produced
    each" -- not for stability, and not for a sample size. Reported, reproduced by
    me exactly, limit stated, stability explicitly not claimed. MET.
(b) C7 re-measured BY ME from the stated rule with a from-scratch query: every
    bucket reproduces (1/0/19/14/0, sum 34 = population), instrument control
    reproduces, plus a discrimination control Main did not run. Third-party
    confirmation now EXISTS.
(c) C7 does NOT block. Its literal verbs are "report how many" (1, reported),
    "state the enumeration rule" (stated, and I reproduced it FROM the rule), and
    "report zero as a measured zero with a positive control" (two zeros, positive
    control DELL-detected, completeness check sums to the population). All three
    are satisfied. The criterion never demanded a number the data cannot yield.
    Main's own "PARTIAL" label is MORE conservative than the criterion requires.
(d) The tree did NOT move. HEAD 5f6a8c54 at start and end; experiment_results
    mtime 17:59:51 and live_check 21:37:01, both before my 21:42:38 spawn.

### VERDICT REASONING
All 10 criteria are substantively satisfied on evidence I re-derived independently
(BQ x4 queries, live backend log, live portfolio endpoint, AST sweeps, an
exhaustive grid sweep, and 5 in-memory mutants with green discriminating controls).
No code defect found. The cap is entirely in the PROSE: the GENERATE artifact of
record contradicts the step's own live_check on C4, asserts a false fact about the
running system, and carries three counts that do not reproduce. Docs-only fix, no
code change. -> CONDITIONAL.
