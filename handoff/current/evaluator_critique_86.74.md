# Evaluator critique -- step 86.74

## STATUS: NO VERDICT WAS RETURNED. The step does NOT close.

**Two consecutive Workflow-rail spawns DROPPED**, ~758K subagent tokens, four
agents, all four returning empty:

| cycle | run id | tokens | outcome |
|---|---|---|---|
| 1 | `wf_2e5ddb63-de9` | 385,807 | `completed without calling StructuredOutput (after in-conversation nudge)` |
| 2 | `wf_929b36e7-c8a` | 372,372 | same error |

Per CLAUDE.md an errored/empty rail return is **NO VERDICT, NEVER PASS**, and it
still costs an attempt. **Nothing below is a returned verdict.**

## What IS below, and why it is not a verdict

Write-first preserved the cycle-2 Q/A's working record, and unlike the three
earlier records this one **finished**: it is self-labelled
`STATUS: COMPLETE -- write-first record, still NOT a verdict`, stamped
`COMPLETED: 2026-08-14T15:27:41Z`, and carries a full criterion-by-criterion
analysis. Only the StructuredOutput CALL was lost, not the work.

**Its own computed conclusion is `CONDITIONAL`** (section F: worst-of-N-lenses,
`min = CONDITIONAL`), whose sole blocker is C4's unmeasured post-fix BQ share plus
C7 being 1-of-34 determined.

**I am recording both facts and claiming neither as a verdict**, because the
actionable outcome is identical under either reading: **CONDITIONAL does not
close a step, and NO VERDICT does not close a step.** Main records verdicts; it
never authors them. The next session must obtain a real verdict.

Notably the Q/A did **not** rubber-stamp my self-reported partials -- it ruled
C4 an "uncovered criterion element => CAPS verdict" on its own analysis.

**One correction to its record:** it flags a "DISCLOSURE GAP" -- that my prompt
disclosed one prior cycle with two records while disk showed three. The third
(`150614Z`) was written by **cycle 2's own first agent**, minutes before the
second agent started, so it did not exist when I wrote the prompt. Its
conclusion is unaffected ("all are verdict-less drops, re-spawning is documented
drop-recovery, NOT verdict-shopping").

---

## VERBATIM TRANSCRIPTION -- `.claude/agent-memory/qa/verdicts/verdict_wip_86.74__20260814T151507Z.md`

No editorial edits, no paraphrase, no omissions.

```
STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-14T15:15:07Z

# Q/A write-first record -- step 86.74 (risk-judge falsy-zero inversion)

Spawn: Workflow rail, Opus 5 (1M). Commits under review: 9d14291e (cycle 1) + a541f10c (cycle 2).

## Prior-attempt evidence (GATHERED, not applied as a trigger)
- `qa_wip.py 86.74 --spawned-at 2026-08-14T15:15:07Z`: source_present=True,
  attempt_number=4, prior_attempts=3, attempt_number_status='ok',
  attempt_number_is_lower_bound=True, records_pruned_known=None, is_verdict=False.
  records_retained=4 (a GAUGE, includes my own record -- not used as the counter).
- `verdict_history_86_21.py --step 86.74 --evidence-only`: status=`no_rows_for_step`.
  => **sequence: UNKNOWN.** attempt_number (4) exceeds the ledger's row count for
  this step (none), so the LEDGER IS STALE / never written for 86.74.
- Prior records on disk: 143725Z (4,984 B), 144450Z (8,311 B), 150614Z (4,421 B,
  self-labelled "CYCLE 2/attempt 3"). ALL THREE `INCOMPLETE` => no prior verdict.
- DISCLOSURE GAP (observation, not a trigger): Main's prompt disclosed ONE prior
  cycle with TWO records. Disk shows THREE prior spawns, the newest at 15:06:14Z --
  9 minutes before mine. Since all are verdict-less drops, re-spawning is documented
  drop-recovery, NOT verdict-shopping.

## A. HARNESS COMPLIANCE -- 5/5 CLEAN
1. research-gate-before-contract: `research_brief_86.74.md` 32,922 B; envelope
   parsed: brief_status=COMPLETE, external_sources_read_in_full=7 (>=5),
   urls_collected=27 (>=10), recency_scan_performed=true, gate_passed=true,
   audit_class=false. 35 distinct URLs counted by me.
2. contract-before-generate (birth times, UTC, `date -u -r $(stat -f%B ...)`):
   research 10:24:44Z < contract 14:19:46Z < portfolio_manager.py 14:58:45Z <
   experiment_results 15:03:51Z. ORDER HOLDS.
3. experiment_results_86.74.md present (388 lines added), contract + live_check present.
4. log-last: `grep -c "phase=86\.74" handoff/harness_log.md` = **0**; masterplan
   86.74 status=`pending`. LOG not written, status not flipped. CORRECT.
5. no-verdict-shopping: no prior verdict exists (all 3 prior spawns dropped);
   evidence CHANGED between spawns (commit a541f10c, 6 files). Not shopping.

## B. DETERMINISTIC
- IMMUTABLE COMMAND `source .venv/bin/activate && python -m pytest
  backend/tests/test_phase_66_2_risk_judge_shape.py -q` -> **37 passed, EXIT 0**.
- LINT: scope DERIVED from `git diff --name-only 9d14291e^ HEAD -- '*.py'` U
  `git ls-files --others` (working tree clean of these files, so `HEAD` alone would
  have been the EMPTY-SET trap). 6 files, non-empty guard satisfied, passed via
  `xargs -0` (zsh no-word-split trap avoided). `uvx ruff check --select
  F821,F401,F811` -> **All checks passed!, exit 0**.
- NO UNINTENDED PRODUCTION CHANGE: step diff = 4 backend files + 1 test file +
  1 QA script (+ handoff/CHANGELOG). `.env` NOT touched (no flag promotion).
  paper_trader.py / kill_switch.py NOT touched. No threshold/gate constant changed
  anywhere in the backend diff.
- RUNTIME: drove the real `decide_trades`; queried the live backend (:8000).
  pid 27945 started 13:30:35 CEST (BEFORE both step commits) and has **no
  `--reload`** -- so it holds pre-fix code, and the matrix's transient file writes
  could not reach the live book. Verified before running the matrix.

## C. MUTATION MATRIX -- re-run by me
control GREEN observed first; **6/6 KILLED**; selected counts 7/6/3/9/4/1
(reproduce Main's published figures exactly). Restore byte-identical, verified by
MY OWN sha256 on all 4 subjects AND by `git status --short` returning clean.

### The self-certification hole IS closed (Main's check #1) -- verified independently
Done in-memory via importlib (no writes to the tree):
- `selected('TestHelperDistinguishesZeroFromAbsent')` -> **7**;
  `selected(...+'XX')` -> **0**; `selected('ZZZ_nothing_matches_this')` -> **0**.
- Ran the whole harness with M1's selector TYPO'd: scored **UNSCORABLE**, harness
  returned **rc=1** (0 would have meant self-certification). pm sha unchanged.
- Premise confirmed directly: `pytest -k <bogus> -q` exits **5**. Old rule
  (`killed = rc != 0`) scores that **KILLED**; new rule (`rc == 1`) does **not**.

## D. CRITERION-BY-CRITERION (my own re-derivation)
- **C1 MET.** `_resolve_position_pct` / `_extract_position_pct` take NO settings
  object -- the fix is unconditional, so flag state is not a variable for it.
  Executed both states x reject_binding -> 4/4 no-order on REJECT/0%. M1 KILLED.
- **C2 MET, and by the SIZING path alone.** Discriminated: with
  `paper_risk_judge_reject_binding=False` the block is
  `Skipping BUY TST: buy_amount=0.00 below $50 minimum (... position_pct=0.0)`,
  NOT the binding gate (which logs `BINDING RiskJudge gate: BLOCKED`). Probe
  non-vacuous: APPROVE/3% buys **$719.93 = exactly 3% of NAV** (10% would be
  $2399.77). M2 KILLED.
- **C3 MET.** I re-derived the default-yielding set over a STRICTLY LARGER grid
  than the author's (15 states x 15 pcts incl. "", 'ABSENT ' w/ trailing space, 0,
  False, [], {}, nan, inf, '0.0', 'abc'). Default reached ONLY from
  `state==ABSENT` (any pct) and `(state missing, pct None)` -- both genuinely-absent
  families. The two families cycle 1 found ((SIZE, pct=None); unrecognised state)
  now both return 0.0. **No third family found under my larger grid.**
  `_sizing_pct` called at exactly 4 production sites; `position_pct_state` written
  at exactly 1 (:409); residual `or 10.0` in production = comments/docstrings only.
- **C4 PARTIAL <- THE GAP.** Write path fixed + unit-proven
  (`TestVerdictIsPersistedPerTicker`, with an explicit `assert captured` guard
  because `_persist_analysis` swallows exceptions); M3 KILLED (3 sel). 0-of-129
  baseline stated with its query. But the criterion's **post-fix share is NOT
  measured** -- blocked by the standing session-end-restart instruction, verified
  by me from the process start time. Uncovered criterion element => CAPS verdict.
- **C5 MET (with a WARN-level residual).** `ticker={ticker}` present; M6 KILLED
  (1 sel). Guard is a source-text assert (vacuity shapes #1/#2) but it is the
  strongest available (line is emitted inside a multi-LLM debate), it self-guards
  staleness (`assert marker in src`), and M6 proves it fails. Residual: would
  survive a reword keeping `ticker=` with a wrong value.
- **C6 MET.** Nested-first + `pos_pct is not None`; M5 KILLED (4 sel);
  `test_genuinely_empty_risk_assessment_emits_nothing` is the anti-vacuity negative.
- **C7 PARTIAL, correctly reported as partial.** Rule stated, positive control
  present (DELL detected=True), 1 measured inversion, 33/34 UNDETERMINED and
  explicitly NOT claimed as a measured zero -- which HONOURS the criterion's own
  "any zero reported as a measured zero" clause rather than violating it. But the
  sweep determines only 1 of 34. **I did not re-run the BQ query**, so 34/1/0/33
  are author-reported and unverified by me.
- **C8 MET.** My AST counts: **9 -> 34** test fns, **17 -> 55** asserts, grep
  `assert ` **17 -> 56** (inflated by 1). Reproduces the corrected figures EXACTLY.
  Two rewritten tests read from the diff: old required `b is not None` +
  `abs(amount - NAV*0.10) < 0.5` + `risk_judge_decision == ""`, and
  `_buy(orders) is not None` ("REJECT invisible top-level -> buys"). Both ASSERTED
  THE DEFECT. New assert `_buy(orders) is None` and 3%-not-10%. **Strict
  INVERSION, not a weakening** -- confirmed from the diff, not the summary.
- **C9 MET.** Control-green-first, 6/6, byte-identical restore, UNSCORABLE
  semantics proven live by my bogus-selector injection.
- **C10 MET, verified LIVE.** DELL still held: quantity 4.806437, cost_basis
  2392.26, stop 457.9024 -- unchanged from the incident record. Nothing loosened.
  DIRECTION CHECK on the deliberate flag-OFF change, MEASURED not asserted:
  REJECT/0% 10%->no-buy; nested APPROVE/3% 10%->3%; UNPARSEABLE 10%->0;
  ABSENT 10%->10% (unchanged). **No path became more permissive.**

## E. RESIDUAL NOTES (non-blocking, worth queueing)
1. `_sizing_pct` RAISES ValueError on a non-numeric `position_pct` when the state
   key is absent/SIZE. Unreachable in prod (single writer sets pct from
   `_verdict.pct`, always float-or-None) and NOT a regression (pre-fix
   `cand["position_pct"] or 10.0` with 'abc' raised TypeError one line later).
2. `(ABSENT, pct=<number>)` still yields the default. Unconstructible from the
   single write site (kind==ABSENT => pct is None) and arguably correct (state is
   authoritative), but a latent contradiction a future writer could hit.
3. `pct=nan` passes the $50 floor (`min(nan, cash)` -> nan; `nan < 50` is False).
   IDENTICAL pre- and post-fix (nan was truthy, so `or 10.0` returned nan too).
   Pre-existing, NOT introduced by this step.

## F. WORST-OF-N-LENSES (P0 money path)
- correctness: PASS  - does-it-reproduce: PASS  - scope-honesty: CONDITIONAL
=> min = **CONDITIONAL**. Sole blocker: C4's post-fix BQ share has no covering
evidence (temporally blocked, not a code defect); C7 is 1-of-34 determined.

COMPLETED: 2026-08-14T15:27:41Z
```

---

## Main's response -- what happens next

**NOT fixed in this cycle, deliberately.** The two blockers are the two items I
had already flagged as partial, and neither is a code defect:

- **C4** is *temporally* blocked: the post-fix share needs an autonomous cycle on
  restarted code, and restarts are batched to session end. The Q/A independently
  confirmed the reason from the process start time (pid 27945, 13:30:35 CEST,
  no `--reload`, predating both commits). **I did not restart to unblock it** --
  putting ungraded trading-path code into the live process to make a criterion
  measurable would be exactly backwards.
- **C7** is a coverage limit of the join, reported as 1 confirmed + 33
  undetermined, never as a measured zero.

Three residual notes from section E are queued as their own defects rather than
fixed inline (`nan` passing the $50 floor is explicitly PRE-EXISTING and
identical pre/post, per the Q/A's own check).

**Attempt accounting:** `qa_wip.py` reports `attempt_number=4,
prior_attempts=3, attempt_number_is_lower_bound=True` for step 86.74. Under the
F1b cumulative budget (5 attempts) this step is close to escalation. A third
consecutive drop should go to the **Agent-tool `qa` fallback**, which CLAUDE.md
names for exactly this case, rather than a fourth rail spawn.
