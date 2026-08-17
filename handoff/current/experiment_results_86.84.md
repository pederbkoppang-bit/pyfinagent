# experiment_results — step 86.84

Filed to close cycle-2 Q/A finding **V-9**: the five-file protocol's GENERATE
artifact was absent for this step, and the rolling
`handoff/current/experiment_results.md` belongs to phase-82.6 (2026-08-06).
Written after the fact and labelled as such rather than back-dated.

## What was built or changed

| File | Change |
|---|---|
| `.claude/agents/qa.md` | **`maxTurns: 30` REMOVED.** Body byte-identical (45,398 == 45,398); only the pin line and its rationale comment changed. |
| `.claude/agents/researcher.md` | **`maxTurns: 40` REMOVED.** Only the pin line and its comment changed. |
| `scripts/qa/rail_turn_cap.py` | **NEW.** Measures whether the drop is turn exhaustion; scores each run against the cap in force when it ran; checks the remediation; four controls; cardinality floors with no opt-out. |
| `scripts/qa/mutate_rail_turn_cap.py` | **NEW.** Executable mutation matrix for the above (V-1). *(15 cells when this row was written; 22 at cycle 4, 29 at cycle 5, 33 at cycle 6 -- the count lives in the matrix output, not in this table)* |
| `scripts/qa/rail_drop_rate.py` | Header superseded; **runtime output** now prints the confound caveat under the by-model table (V-4). |
| `.claude/workflows/qa-verdict.js` | Comment-only. "Mechanism is UNPROVEN" superseded; the `2e-4 … the model split SURVIVES` claim retracted in place (V-4); "REPORTED BELOW" → "REPORTED ABOVE". |
| `.claude/workflows/research-gate.js` | Comment-only. Same supersession, plus the note that two at-40 non-emitters sat inside runs this caller recorded as completed. |
| `handoff/current/live_check_86.84.md` | Regenerated (V-2). |
| `handoff/current/contract_86.84.md` | PLAN, written after the gate passed. |
| `handoff/current/evaluator_critique_86.84.md` | Both verdicts transcribed verbatim. |
| `handoff/harness_log.md` | Cycle 218 + operator review request. |
| `.claude/masterplan.json` | Steps 86.84 and 86.85 filed; 86.84 `audit_basis` corrected (V-3). |

No production code, no threshold, no gate, no `.env`, no flag. Both JS diffs are
comment-only.

## Verification command output, verbatim -- CAPTURED 2026-08-14 under the phase-59.1 pins, SUPERSEDED

*(cycle-5 correction, 2026-08-17: the block below is the 2026-08-14 capture and its
figures no longer reproduce against the grown corpus and the F-E/D1-fixed script --
qa n was 302, now 338; detector 1257/1267, now larger; the 'caps removed at
2026-08-15T00:00:00Z' line describes the retired calendar constant that the
session-birth derivation replaced. The corpus GROWS DAILY, so any frozen figure rots within hours (the cycle-5 evaluator measured n=38/p90=55/past=34 where §Cycle 4 had captured 36/54/32 hours earlier) -- current output comes from RUNNING `python3 scripts/qa/rail_turn_cap.py`, and every block in this file is a dated capture. The 2026-08-14 capture below and
live_check §10. Kept as the historical capture; do not re-run expecting these numbers.)*

```
$ python3 scripts/qa/rail_turn_cap.py --verify   # exit 0

  agentType           cap     n  drop  @cap  >cap  ok p50  ok max  ok@cap
  Explore               -   263     0     0     0       7      56       0
  None                  -   414     0     0     0       9      93       0
  claude-code-guide     -     1     0     0     0       7       7       0
  general-purpose       -   252     0     0     0      12      63       0
  qa                   30   302    39    39     0      18      30       6
  researcher           40    93     9     9     0      24      40       3

Turn counts observed on dropped spawns, per role:
  qa                 cap=30  observed=[30]
  researcher         cap=40  observed=[40]

CONTROLS
  C1 turn counter alive     : 1325 spawns with turns>0; 0 zero-with-assistant-lines (must be 0)
  C2 cap is a real ceiling  : 0 capped spawns exceed their cap (must be 0)
  detector positive control : StructuredOutput emitted by 1257/1267 completed spawns vs 1/48 dropped
  C3 negative control       : 10 spawns in `killed` runs sit at turns [1, 1, 2, 2, 2, 3, 4, 5, 6, 16]; 0 at a cap

CLAIM
  every dropped spawn sits EXACTLY at its cap : True
  drops among UNCAPPED agent types           : 0/930 (uncapped spawns reach 93 turns)
  ... of which AT RISK (>30 turns)        : 0/50  vs a 12.2% drop rate on capped spawns

  spawns sitting AT a cap                    : 57, of which 49 never emitted StructuredOutput
  ... inside runs that COMPLETED anyway      : 2 ['wf_078f4125-57a', 'wf_a6ea31e7-9b9']

REMEDIATION (phase-86.84) -- checked by the same command as the diagnosis
  caps removed at : 2026-08-15T00:00:00Z
  in force before : {'qa': 30, 'researcher': 40}  (phase-59.1 pins)
  live now        : {'qa': None, 'researcher': None}
  all pins removed: True  (must be True)

VERIFY: PASS -- controls green, turn-exhaustion claim holds.
```

```
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify   # exit 0
  control verify_ok=True  live_caps={'qa': None, 'researcher': None}
  ... 22 cells; M4r M5r M9 M8 M7c M7b M7 M15 M16 M17 M18 M19 M20
  ...           M11 M11b M12 M12b M13 M21 KILLED
  ...           M14 (equivalent), M6, M6b (known gap) SURVIVED
BYTE-IDENTICAL RESTORE (md5 before == after, real tree never written): all ok
cells=22  real survivors=0  known/equivalent survivors=3
VERIFY: PASS -- control green, 0 real survivors, tree unchanged.
```

Supporting gates, run and green:

```
$ node --check .claude/workflows/qa-verdict.js         -> 0
$ node --check .claude/workflows/research-gate.js      -> 0
$ node scripts/qa/verify_research_gate_workflow.mjs    -> 0   124 passed, 0 failed
$ node scripts/qa/verify_rail_retry.mjs                -> 0    38 passed, 0 failed
$ uvx ruff check --select F821,F401,F811 scripts/qa/rail_turn_cap.py \
        scripts/qa/rail_drop_rate.py                   -> All checks passed!
```

`verify_rail_retry.mjs` section **[F]** is the executed evidence for criterion 4:
an exhausted retry yields no value, rethrows the original error, `research-gate`
still RECOMPUTES `gate_passed`, and the retry loop assigns no verdict field. It
was green before this step and remains green; it is cited here because the
cycle-2 Q/A found the property proven but unmapped to the criterion (V-10).

## Artifact shape

- `rail_turn_cap.py --verify` → exit 0/1, human table; `--json` → machine summary
  carrying `by_agent_type`, `model_x_agent_type`, `controls`, `claim`,
  `remediation`.
- `mutate_rail_turn_cap.py --verify` → exit 0/1, per-cell matrix plus an md5
  before/after restore proof.

## Honest limits of this GENERATE

- **The change is not in force.** The Agent-tool roster snapshots at session
  start, so the removal takes effect next session. Every number above was
  produced under the OLD caps; none of it is evidence about the uncapped rail.
- **The uncapped qa/researcher turn distribution is unobserved.** "Uncapped
  agents self-terminate" rests on 930 spawns of *other* roles. Re-measuring after
  the next session is the committed follow-up, and it is what would turn this
  from a reasoned fix into a verified one.
- **Ruff was red on this step's own file** (`F401 Counter` unused) from cycle 1
  until it was fixed here; recorded rather than quietly cleaned.

---

## Cycle 4 GENERATE (2026-08-17, operator-attended session, external-audit follow-through)

**Context.** The operator reviewed the 2026-08-14 escalation (three consecutive
CONDITIONALs, loop stopped per F1) and this morning's independent external audit
(claude.ai artifact d5539ae2), then delegated the two escalation decisions to
this session with explicit consent. This cycle discharges them.

**1. The escalation's closure condition has been satisfied by events.** The
escalation recommended keeping 86.84 open until "the post-restart re-measurement
lands". It has landed, and it is now RE-RUNNABLE rather than a one-off note:
`rail_turn_cap.py` prints the realised uncapped turn distribution on every run.
Measured 2026-08-17 (`python3 scripts/qa/rail_turn_cap.py`, REMEDIATION block):

```
REALISED UNCAPPED TURN DISTRIBUTION (the committed re-measurement --
the uncensored sample; percentile rule: sorted[int(frac*(n-1))]):
  qa           n= 36  dropped=0  non-emitters=0  p50=40  p90=54  max=60  >old-cap(30)=32
  researcher   n= 11  dropped=0  non-emitters=0  p50=19  p90=36  max=38  >old-cap(40)=0
```

Read that qa row twice: **32 of 36 post-removal qa evaluations exceeded the old
cap of 30**, and the uncensored p50 is 40 versus the censored p50 of ~20. Under
the old cap, ~89% of the current qa workload would have died at the rail. Zero
drops, zero non-emitters since removal. The right-censoring argument (criterion
3) is no longer an inference; it is lived data.

**2. NEW DEFECT FOUND AND FIXED (external-audit finding D1): the step's own
verification command had gone red for a spurious reason.** Before this cycle:

```
$ python3 scripts/qa/rail_turn_cap.py --verify   # 2026-08-17, pre-fix
VERIFY: FAIL
  - no agent type carries a maxTurns cap; nothing to test
```

Root cause, from `--json`: `analyse()` rows took `cap = group[0]["cap"]` — the
alphabetically-first run record. Once any post-removal session sorted first,
every row read uncapped while `claim.capped_spawns` was 395 and the per-spawn
claim (`every_drop_is_at_its_cap: true`) stayed correct underneath. verify()'s
`capped_types` floor counted zero capped roles and exited 1 with a banner that
invites exactly the wrong fix (restoring a cap). Fix: score each spawn against
ITS OWN era-correct cap (`s["cap"]`), report the group's caps as the observed
SET (`caps_present`, plus `capped_n`/`uncapped_n`). verify() itself is
unchanged — the row inputs were the defect. After:

```
$ python3 scripts/qa/rail_turn_cap.py --verify
VERIFY: PASS -- controls green, turn-exhaustion claim holds.   (exit 0)
$ /usr/bin/python3 scripts/qa/rail_turn_cap.py --verify        # no-PyYAML fallback path
VERIFY: PASS -- controls green, turn-exhaustion claim holds.   (exit 0)
```

Corrected table rows (was: every row `cap=-`): `qa cap=30 n=338 drop=39 @cap=39
>cap=0` · `researcher cap=40 n=104 drop=9 @cap=9 >cap=0`.

**3. Guards re-run after the edit.**

```
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify
cells=22  real survivors=0  known/equivalent survivors=3
VERIFY: PASS -- control green, 0 real survivors, tree unchanged.   (exit 0)
$ node scripts/qa/verify_rail_retry.mjs        # criterion-4 executed evidence
ALL GREEN: 38 passed, 0 failed                                     (exit 0)
$ uvx ruff check --select F821,F401,F811 scripts/qa/rail_turn_cap.py
All checks passed!                                                  (V-6 stays closed)
```

**4. Separation-of-duties review of the agent-file edits (the owed operator
item) — APPROVED.** Reviewer: this session, which authored NEITHER the
qa.md/researcher.md edits (commit 85127353, 2026-08-14 session) NOR any prior
86.84 artifact; its independent audit this morning re-derived the diagnosis from
primary data (580 run records / 1,275 spawns) before reading this step's
artifacts. Basis for approval: (a) the removal diagnosis survived three
adversarial Q/A passes and 22 mutation cells; (b) the post-removal sample above
— 47 uncapped spawns, 0 exhaustion drops, 0 non-emitters — is the verification
the escalation asked for; (c) the residual risk is bounded and now measured on
every run (observed uncapped max 60 turns for qa). Recorded in
`handoff/harness_log.md` per the CLAUDE.md separation-of-duties rule.

**5. Ledger honesty.** `verdict_ledger_write.py --emit-sequence --step 86.84`
returns `[]` — the ledger is stale (86.85's known condition; the drain session
independently confirmed the same on 86.97 today). The verdict sequence passed to
the cycle-4 Q/A is therefore sourced from the verbatim transcriptions in this
step's own `evaluator_critique_86.84.md`, and `qa_wip.py` (3 retained WIP
records for 86.84) is cited as the live attempt counter.

Files changed this cycle: `scripts/qa/rail_turn_cap.py` only (plus these
handoff artifacts). No agent file, no gate, no verdict semantics touched.

---

## Cycle 5 GENERATE (2026-08-17): criterion-8 closure on the cycle-4 findings

Every item below answers a named cycle-4 finding; verdict transcribed verbatim
in `evaluator_critique_86.84.md` §9.

**1. verify() now asserts over the re-measurement (kills QM2/QM7/QM8-class).**
New floors, fail-closed, with the revisit rule in the message: post-removal
non-emitters must be 0 per role, post-removal drops named separately, sample
cardinality >= MIN_POST_REMOVAL_SPAWNS (10), p50 >= 1 when n > 0, and
p50 <= p90 <= max monotone. The immutable command stays green on both
interpreters with the floors live.

**2. The matrix now covers the cycle-4/5 code: 29 cells, 0 real survivors, 0
errors.** Seven SOURCE-level cells load a mutated temp copy (real tree never
written; anchors asserted so a no-match replace fails loudly): S1 reverts the
D1 group[0] scoring (KILLED: "no agent type carries a cap"), S2 floods the
non-emitter counter (KILLED by the new floor), S3 zeroes `_q` (KILLED: p50
floor), S4 breaks the role filter (KILLED: cardinality floor), S5 inverts
past_old_cap (KILLED by the ORACLE: the published numbers must equal the
control's over the same immutable corpus), S6 plants ONE synthetic post-removal
non-emitter with the source unmutated (KILLED: the floor fires -- the paired
positive control), S7 plants the signal AND hardcodes the counter to 0 (KILLED
by the injected-truth assertion: a report that hides a known-planted signal is
caught). The S6/S7 pair exists because the live corpus has zero non-emitters,
so a hardcoded zero is equivalent-on-corpus and only a planted truth can
distinguish it -- the fixture-must-break-the-symmetry lesson as a cell.

**3. M14's annotation now tracks the run, and the summary counts OUTCOMES.**
M14 (boundary moved to 2027) was labelled EQUIVALENT from an era when the whole
corpus predated the boundary; with 59 post-removal spawns on disk it KILLS (C2
fires), so its expectation is now KILL with the history in the cell comment.
The survivor summary counts outcomes, prints ANNOTATION MISMATCH rows when a
label contradicts the run, and `--verify` fails on mismatches and on ERROR
cells (an unscored cell is not evidence).

**4. The four stale-prose sites are corrected AT THE SITE** (cycle-4 violation
4): live_check §4b retitled "AS OF 2026-08-15, SUPERSEDED" with the n=47 state
in the correction note; the "was not run today" sentence carries its discharge
(38/38 + 124/124, fresh today); the two live mentions of the retired
`CAP_REMOVED_AT` constant carry the rename note; experiment_results' undated
2026-08-14 verbatim block is retitled CAPTURED/SUPERSEDED with the
figures-that-moved named.

**5. Consequence framing removed from the evidence** (cycle-4 notes): the
"admissible outcomes" sentences in critique §8 and live_check §10 are replaced
with neutral text plus an edit note. Escalation arithmetic stays caller-side.

**6. Ledger truth restored for this step:** four rows backfilled with true
event dates (`--emit-sequence` now returns the sequence the critique
transcribes); cycle-4's row written at the transcription seam.

Verbatim, post-change:

```
$ python3 scripts/qa/rail_turn_cap.py --verify
VERIFY: PASS -- controls green, turn-exhaustion claim holds.        (exit 0)
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify
cells=29  real survivors=0  known/equivalent survivors (BY OUTCOME)=2  errors=0
VERIFY: PASS -- control green, 0 real survivors, outcomes match annotations, tree unchanged.  (exit 0)
$ uvx ruff check --select F821,F401,F811 scripts/qa/rail_turn_cap.py scripts/qa/mutate_rail_turn_cap.py
All checks passed!
```

Files changed this cycle: `scripts/qa/rail_turn_cap.py`,
`scripts/qa/mutate_rail_turn_cap.py`, the three 86.84 handoff artifacts, and
four `handoff/verdict_ledger.jsonl` data rows. No agent file, no gate weakened.

---

## Cycle 6 GENERATE (2026-08-17): the cycle-5 FAIL's findings closed

Cycle-5 verdict (FAIL) transcribed verbatim in the critique §10. Every finding
answered by the change it named; full evidence in `live_check_86.84.md` §11.

1. **Killed-status conflation removed from the floor**: `killed_n` named
   separately in `post_removal_turns`; `non_emitters` counts only
   completed-without-emitting; cell S11 (killed-run injection, MUST_STAY_GREEN
   negative control) pins that an operator abort never reddens the command.
2. **past_old_cap and the per-role sample guarded by an independent second
   derivation** (`uncapped_past_hist_cap` + `n == uncapped_n` cross-checks in
   verify()) -- the evaluator's five surviving mutants (inversion, qa-only
   break, truncation, emptied set, narrowed counter) now die at a shipped
   assertion or an injected-truth cell, not at a harness oracle.
3. **Monotone-floor cells added** (S8, S9), **third hiding shape covered**
   (S10), **kill modes never pooled** (inline mode labels + per-mode summary).
4. **Stale prose corrected at the site** in all three artifacts: the 22-cell /
   M14-equivalent claims carry their capture cycle and the adjudicated
   correction; the "15-cell" table row and the "Current, reproducing output"
   pointer state the corpus-grows-daily rule; the contract's drifted line
   citation is annotated.

Post-change, verbatim: `--verify` exit 0 both interpreters · matrix
`cells=33 real survivors=0 known BY OUTCOME=2 errors=0`, kills by mode
`{VERIFY 27, ORACLE 1, INJECTED_TRUTH 2, MUST_STAY_GREEN 1}` · ruff clean.
