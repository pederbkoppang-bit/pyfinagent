STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-14T15:54:43Z
COMPLETED: 2026-08-14T16:08:00Z

Scope: cycle-4 work. TWO scope corrections, both stated in the return:
- `cba60c0b..HEAD` EXCLUDES cba60c0b and is docs-only; the orphan guard + M7 live
  IN cba60c0b. I graded cba60c0b^..HEAD.
- The tree MOVED during my evaluation. Two commits landed after the range I was
  given (38ba13ad, a33a5117) and they RETRACT the C7 claim I was asked to attack.
  I graded HEAD = 15a6f8f1.

Production code byte-identical to the already-graded 76ac89ee: `git diff --stat
76ac89ee..HEAD -- backend/services/ backend/agents/ backend/api/ backend/config/`
is EMPTY. Restart verified myself: pid 85562 started 17:52:08 CEST (after the
graded commit 17:35:32), pid 27945 GONE.

## Item 1 -- orphan-SELL guard + M7: MET, verified beyond Main's own evidence
Matrix re-run BY ME: control GREEN, M1-M7 all KILLED, restore byte-identical on
all 4 subjects. Target suite 41 passed. Lint clean over a derived 2-file scope.
M7 is a GENUINE orphan, not a floor deletion -- I rebuilt its exact two edits from
`MUTATIONS` in-memory and observed the real function:
  M7 mutant, 0% REJECT -> [('SELL','OLD',None)]        <- SELL alone, no BUY
  M7 mutant, 3% legit  -> [('SELL','OLD',None), ('BUY','NEW',300.0)]
The second line is the part that matters and Main did not report it: M7 ISOLATES
the orphan rather than blanket-breaking the swap path, so the kill is attributable
to the orphan and not to a collapsed harness. The old BUY-subset assertion cannot
see `[('SELL',...)]`; the new whole-list assertion and the SELL-specific test both
fail on it.
Checked the multi-edit normaliser for a fail-open: `if any(o not in original ...)`
scores NOT_APPLIED and `continue`s (mutation_matrix_86_74.py:207-211). Fails
closed. Correct.

## Item 2 -- the comment correction: SUBSTANCE ACCURATE, one residual slip
Verified independently: paper_swap_enabled True (settings.py:368),
paper_swap_min_delta_pct 25.0 (:372), paper_swap_max_per_cycle 2 (:378),
paper_swap_churn_fix_enabled False (:385), paper_atomic_swap_enabled False (:493).
The getattr fallback of 0 is at portfolio_manager.py:719 exactly as the comment now
says. Not overcorrected -- the dangerous-direction framing is right.
WARN: the corrected comment says "These four values" while FIVE kwargs are set, and
says `_settings()` "OMITS them" when `paper_swap_enabled` is PRESENT-but-False at
test:26. All five do match production defaults, so the conclusion holds. But it is
a count that does not reproduce, inside a comment written to fix an inaccurate
comment.

## Item 3 -- C7: the claim was ALREADY WITHDRAWN before I graded it
`live_check_86.74.md` §2c now records the overclaim AND its refutation. The
reasoning is exactly the attack I would have run: "risk_assessment key absent"
supports "NOT PERSISTED", never "never existed" -- and in this very system a real
verdict was already proven droppable from a persisted artifact (the C6
factors_json case). The decisive additional fact is stronger still: final_synthesis
is absent in 19 of 19, so the "reached synthesis, attached nothing" reading is not
even available -- the report is truncated.
Propagation swept: goal_next_2026-08-15.md:24 leads with "C7 IS STILL PARTIAL ...
Do not inherit it"; experiment_results:183 and day_report §8 heading both carry the
retraction in the heading, not merely below it. No surviving forward-looking
"resolved" claim -- the only hits are CHANGELOG rows, which are immutable records
of commit subjects.
STATED LIMIT: I did NOT independently re-derive the BQ decomposition (19 / 14 / 0).
My grade of item 3 rests on internal consistency and the soundness of the
reasoning, not on re-measurement. That is acceptable here only because the claim
now standing is the CONSERVATIVE one -- an unverified "still undetermined" creates
no risk. It would NOT be acceptable for a "resolved".

VERDICT COMPUTED: PASS on the cycle-4 work as scoped. Two WARNs, no blockers.
This does NOT close 86.74: C4 is open and C7 is PARTIAL.
