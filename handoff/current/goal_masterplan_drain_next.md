# Goal — Masterplan drain, continuation session (set 2026-08-07 end-of-day)

Supersedes goal_masterplan_drain_2026-08-07.md. That goal's WAVE 1 IS COMPLETE and its
done-definition is substantially met; this goal carries the measured end-of-day state
forward. Autonomy rails unchanged: full harness per step; no AskUserQuestion; a step
needing an operator decision joins the ask list and is SKIPPED.

## Read first
1. This file. 2. CLAUDE.md. 3. .claude/rules/research-gate.md. 4. Auto-memory MEMORY.md.
5. handoff/current/operator_ask_2026-08-07.md — 13 decisions with evidence packets.

## What 2026-08-07 closed (11 cycles, 167-177, commits cb4b3c52..4b1a5449)
DONE+pushed: 82.23 (superseded by 82.27), 83.0, 83.0.3, 83.0.1, 83.1, 83.1.1 — the whole
phase-83 gate set. The go/no-go is MEASURED: CONTINUE on DSR (required 0.6886 at the cap
vs 0.8246 best-ever at measured V=0.008169), PBO is the hard wall (no edge seed clears
0.20), kill rule locked pre-results. 85.3 (auth-latch freshness — 474 consecutive false
alarms ended, latch cleared live, paging drill delivered). 84.1 (memory-link auditor —
PASS cycle 3 after two CONDITIONALs, commit 1625b507; false-breakage class gone;
84.1.1 follow-up queued).
61.2: evidence cycle complete, CONDITIONAL #2, DEFERRED solely on ask #10 (83.5% of
analyses are fabricated neutrals; the fix is one flag from live). 72.0.2: rail-dead
fail-forward BUILT DARK + Q/A PASS, deferred on ask #13 (see wave order #2 —
one Q/A spawn DIED without StructuredOutput; recovery re-spawn worked on a LEAN
budget-capped prompt: cap the evidence block, forbid full-matrix re-runs).
72.0.1: superseded by 78.1+78.16. 85.2: parked at research-done (C7 unreachable — 46 unrelated secretless
failures); 85.2.1/85.2.2 queued; ask #11 sequences it.
Queued new steps: 83.0.4-83.0.8, 83.1.3-83.1.5, 36.28, 36.29 (archive-snapshot
mislabeling, 13 dirs), 36.30 (env-flag test leakage; the operator PROMOTED the 57.1
binding gate — memory updated), 84.1.1, 85.2.1, 85.2.2, 85.3.1-85.3.3.
All Q/A verdicts verbatim in handoff/current/evaluator_critique_*.md (+ raw returns in
qa_returns/).

## Measured scope (2026-08-07 ~16:30 — RE-DERIVE before trusting)
status=="pending": 318 / P0 19, P1 80, P2 146, P3 54, P4 8, unset 11; 53 operator-only.
Genuinely-executor P0s remaining: 61.3, 62.1, 62.2, 63.3, 63.4, 65.2, 68.1, 68.5
(72.0.2 is evidence-complete, ask-#13-gated). Ask-gated P0s: 61.2 (#10), 27.6+27.6.3 (#9/#2), 62.7,
65.4, 68.3 (wall-clock/operator).

## Wave order
1. If ask #10 is ANSWERED: finish 61.2 (promotion live_check Sections C+D → cycle-3 Q/A
   must be PASS — a third CONDITIONAL auto-FAILs; the evidence packet is complete).
2. 72.0.2: CLOSED-DEFERRED cycle 177 — Q/A PASS on the code (37 tests, 14/14
   mutants; the evaluator's own live probe confirmed the Vertex transport), commit
   4b1a5449; status deliberately PENDING, flip HELD on ask #13 (induced metered
   cycle, ~$0.3-1.4, Vertex/GCP-billed). If #13 answered: run the capture per the
   contract §3.8 plan, write live_check_72.0.2.md, flip.
3. 61.3 (money-display/currency, P0), then 62.1/62.2 (away-ops bot), 63.3/63.4, 65.2,
   68.1, 68.5 — re-read each first; the 61.2 pattern (built-dark awaiting evidence) may
   recur on several.
4. Defect-queue tail: 85.3.1 (P1 — probe predicate; the outage REPRODUCES on the next
   real 401 until landed), 85.3.3 (P1 SECURITY — watchdog plist embeds a literal OAuth
   token; git-history check → possible rotation; ask #12), 36.28 (suite/pause-state
   decoupling — scope ALREADY corrected, incl. the stub-signature repair), 83.1.5
   (pre-registration lever-ordering amendment), 83.1.4, 83.0.4-83.0.8, 83.1.3, 85.2.1,
   85.2.2, 85.3.2, 36.29, 84.1.1.

## Non-negotiable (unchanged + today's hard-won additions)
- Full harness per step; qa-verdict Workflow rail; persist every verdict the turn it
  lands; errored/empty return = NO VERDICT (one researcher return dropped today —
  write-first saved it; one Q/A spawn dropped — re-spawned, recovery not shopping).
- A "verbatim" capture must be PIPED from the file/run it claims to quote — NEVER from
  the researcher's prototype (a FAIL was earned this way on 83.1.1); prose summaries of
  recorded data are DERIVED in code; re-derive every fenced measurement after the FINAL
  edit INCLUDING post-verdict fix passes; a Q/A's named instances are a sample — grep
  the whole tree for the claim class.
- Run the ruff lint gate (F821,F401,F811, git-derived scope, tracked UNION untracked,
  non-empty asserted) BEFORE every Q/A spawn — three misses today each cost a cycle.
- A mutation runner must ISOLATE any test that touches real state (-k 'not real_corpora'
  class) — an m5 write-mutant leaked markers into all three live memory corpora today.
- DO-NO-HARM: paper only; historical_macro FROZEN; immutable gates byte-untouched;
  $0 metered; no flag promotions (they are the operator's — batch with evidence).
- git add -An before flips; masterplan edits via python MUST use ensure_ascii=False
  (a json.dumps default mojibake'd the whole file today — caught pre-commit).
- The live kill-switch pause state leaks into uninjected PaperTrader tests (36.28) —
  never read suite redness at face value; classify before counting.

## Done-definition
Every remaining executor P0 above is PASS or deferred-with-recorded-reason; ask list
current; ≥2 of the defect-queue tail closed, prioritizing the two P1s (85.3.1, 85.3.3).
SOFT STOP: 12 cycles or operator-gated-only remaining → regenerate this goal with
measured state.

## Stop conditions
HARD: any live-book move; less-conservative safety-gate change; metered spend; Fable
roster repin. Check git log after every background notification.
