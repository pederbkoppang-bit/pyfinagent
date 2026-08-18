# experiment_results — step 86.79

> **CYCLE 4.** Criterion 4 is now CLOSED on all three members — see **§8**. This file
> was **regenerated** at cycle 3, not edited. The cycle-2 Q/A found it
> internally stale in five places — including a block headed *"Verbatim verification
> output"* whose numbers no longer reproduced — and `qa.md` §4b is explicit that *"a
> verbatim capture must be regenerated, never edited."* Every count below was read
> back from a run performed after the last code change. Prior verdicts, verbatim, are
> in `handoff/current/evaluator_critique_86.79.md`.

**Current totals at the cycle-5 edit: 60 checks (floor 59 -- raised per F3), 11 mutation cells, 11 killed. *(The '55 checks (floor 53)' that stood here was capture-time truth outlived by growth -- the cycle-4 tail -2 capture could not expose the drift.)***

---

## 0. Cycle history — what each Q/A found, and what changed

### Cycle 1 (`wf_61338c26-b90`) → CONDITIONAL, 4 findings — all fixed

| finding | fix |
|---|---|
| criterion 4: **one member of a two-member class** enumerated. The same false claim was duplicated in `.claude/workflows/qa-verdict.js`, which is **not** under `.claude/agents/` and so was never gated | Enumerating the class found **four** lines (`:147, :152, :159, :172`), not the two named. All four now point at `attempt_number`/`prior_attempts` and name `records_retained` a gauge |
| mutant **N6**: dropping the pruned-away add-back on the `no_record_for_this_spawn` branch survived both gates. The old fixture drove that branch with **no prune and no ledger**, so `lost_n` was 0 and the term was **dead** | New section **C3b** combines them; permanent cell **M8** |
| mutant **N1**: `attempt_number_is_lower_bound` had **zero** assertions anywhere | New section **C3c**; permanent cell **M9** |
| lint **F401** dead `datetime` import | removed |

### Cycle 2 (`wf_44776e5d-ca3`) → CONDITIONAL, 4 findings — 3 fixed, 1 still gated

| finding | fix |
|---|---|
| criterion 4 again: inside `qa.md` the class is **more than one site** — the Q/A found `:645` | Enumerated the **whole file**: **four** sites, and they are **not the same kind of problem**. Classified in `qa_md_patch_86.79.md`: `:622` **FALSE**, `:645` **STALE**, `:692` **accurate**, `:713` **a dated measurement that must not be rewritten**. Gated work is **2 sites** |
| mutant **Q-A**: `>= DEFAULT_KEEP` → `> DEFAULT_KEEP` survived. C3c drove retained=2 and retained=5 — **never the boundary itself** | C3c now drives **below / EXACTLY-AT / above / accounted**, with the boundary as its own named assertion. Permanent cell **M10**, pointed at an assertion **M9 cannot break** so the two stay distinguishable |
| mutant **Q-E**: the *"ledger written BEFORE the unlink"* crash-safety claim had **no guard** — a documented invariant is a claim, not a property | New section **C3d** monkeypatches `Path.unlink` to raise and asserts the loss was **already** recorded (`read_loss == 3`, `prior_attempts == 9 > 6`). Permanent cell **M11**, which *moves* the call rather than deleting it — deleting it is already M2, and a cell that duplicates another proves nothing |
| **this file was stale in five places** | **regenerated from live output** |

**A cell was again refused for the wrong reason, and again that was correct.** After
C3c's assertions were renamed, **M9** came back `RED-WRONG-REASON` — red, but not on
the assertion it was aimed at. It was repointed rather than rubber-stamped.

---

## 1. What was built

| file | change |
|---|---|
| `scripts/qa/qa_wip.py` | **modified** — the whole fix |
| `scripts/qa/verify_counter_86_79.py` | **new** — 55-check re-runnable checker |
| `scripts/qa/mutation_matrix_86_79.py` | **new** — 11-cell mutation matrix |
| `.claude/workflows/qa-verdict.js` | **modified** — the second consumer of the false claim (4 lines) |
| `handoff/current/qa_md_patch_86.79.md` | **new** — the un-applied `qa.md` corrections, all four sites classified |

**`.claude/agents/qa.md`:** untouched by Main through cycles 1-3 (**zero-line diff**,
independently verified by both Q/A cycles), then edited at cycle 4 by a **fresh
executor** on the operator's instruction — see §8. **Not touched at all:** `CLAUDE.md`,
`.claude/rules/research-gate.md`.

### The five changes inside `qa_wip.py`

**F1 — split the field; the unit travels with the number.** `records_retained` keeps
its exact name *and value* (no live number shifted). Added: `attempt_number`,
`prior_attempts`, `attempt_number_status`, `attempt_number_is_lower_bound`,
`attempt_number_guidance`, `records_pruned_known`, `records_retained_unit`.
`attempt_number` is **INCLUSIVE** — a first attempt is 1 — and says so, because
Temporal's `MaximumAttempts` and Step Functions' `MaxAttempts` ship **opposite units
under the same word**.

**F2 — the write-first coupling is unrepresentable, not documented.** The number is
computed only when a record is positively identified as this spawn's; otherwise
`None` + `no_record_for_this_spawn`, while `prior_attempts` — genuinely knowable — is
still reported.

**F3 — the pruner records what it destroys** (`PERF_RECORD_LOST` shape), **before**
unlinking, monotonically. Crash mid-prune therefore **over**-counts (escalates early)
— now guarded by C3d rather than merely claimed.

**F4 — `DEFAULT_KEEP`'s own off-by-one comment. THE DOC MOVED, NOT THE CODE.**
`records[keep:]` is standard keep-N and matches the k8s/journald precedents the module
cites; changing it would silently widen live retention. Found by the research gate.

**F5 — fail closed.** Every uncomputable path returns `None`, never `0`, and states
why via `attempt_number_guidance`.

---

## 2. Verbatim verification output — regenerated after the CYCLE-3 code change *(cycle-6 mark: the body shows 55/53, true at its tree; the current 60->62/61 runs are in the cycle-5/6 sections)*

```
$ python scripts/qa/verify_counter_86_79.py
  checks run : 55   (cardinality floor 53)
  failed     : 0
  ALL CHECKS PASS
exit=0

$ python scripts/qa/mutation_matrix_86_79.py
  [CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (55 checks)
  cells: 11   killed: 11   survived/unearned: 0
  subject sha256[:16] before=146600b722a02481 after=146600b722a02481 -> tracked file UNCHANGED
  ALL CELLS KILLED
exit=0

$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/qa_wip.py\").read())" && echo qa_wip-parses'
qa_wip-parses
exit=0

$ uvx ruff check --select F821,F401,F811 <3 files, scope derived + asserted non-empty>
All checks passed!
exit=0
```

Full section-by-section capture: `handoff/current/live_check_86.79.md`.

The measured defect, in one table:

| the defect | measured |
|---|---|
| `records_retained` counts the current spawn | 2 priors → **3** |
| the number depends on write-first ordering | same spawn: **2** before its own write, **3** after |
| pruning saturates it | 6 true attempts → `records_retained` **3**, `attempt_number` **6** |
| F1b's ceiling after a prune | old: `3/5` → **CONTINUE**. new: `6/5` → **ESCALATE** |
| crash mid-prune | loss already recorded → `prior_attempts` **9**, never 6 |

---

## 3. Criterion-by-criterion

| # | criterion (abridged) | status | evidence |
|---|---|---|---|
| 1 | off-by-one REPRODUCED, producing line quoted | **met** | live_check §1 — line **grep-derived**; it has moved from `:315` → `:507` during this step, which is why |
| 2 | write-first coupling demonstrated | **met** | live_check §2 |
| 3 | pruning saturation + enumeration with command stated | **met** | live_check §3; prune has **no production caller**, re-derived by both Q/A cycles with wider filters |
| 4 | doc and code made to agree | **met (cycle 4) — §8** | all three members: `DEFAULT_KEEP`'s comment, `qa-verdict.js`'s 4 lines, and `qa.md`'s 2 sites (applied by a fresh executor on the operator's instruction) |
| 5 | escalation still fires | **met** | live_check §5 |
| 6 | verdict semantics unchanged; fails CLOSED | **met** | live_check §6 |
| 7 | mutation-test, control GREEN first | **met** | live_check §7 + §12–§15 — 11/11 killed on named assertions |

---

## 4. Criterion 4 — the position through cycles 1-3 (SUPERSEDED by §8)

> **This section is the record of what was true before the operator answered. Criterion
> 4 is now MET; §8 is current.** Kept because the reasoning for declining to self-author
> the `qa.md` edit is the reason a fresh executor was used.

| | member | state |
|---|---|---|
| **4a** | `DEFAULT_KEEP`'s comment vs `records[keep:]` (inside `qa_wip.py`) | **FIXED**. The *doc* moved; the reason is in the code |
| **4b** | `.claude/workflows/qa-verdict.js` — 4 lines | **FIXED** in cycle 2. Never gated; the cycle-1 Q/A was right that I had not looked |
| **4c** | `.claude/agents/qa.md` — `:622` FALSE, `:645` STALE | **NOT FIXED — operator-gated** |

`qa.md` carries **four Main-authored edits awaiting operator review** under CLAUDE.md
separation of duties. The session's operator instruction is *"if a fix genuinely needs
`qa.md`, stop and ask"*, and the step's own notes say *"prefer changing `qa_wip.py`, or
hand it to a fresh executor."*

Shipped instead, so the divergence is **not silent** — which is what criterion 4
actually forbids: `records_retained_unit` (the unit in the payload the Q/A reads),
`attempt_number_guidance` (*"Do NOT fall back to records_retained here … a low number
SUPPRESSES escalation"*), and `qa_md_patch_86.79.md` with all four sites classified and
three routes.

**I am not asking for a waiver.** If a Q/A judges criterion 4 unmet, that is a correct
reading. **The step is not claimable as PASS on criterion 4 by me.**

---

## 5. Deltas from the contract — disclosed

1. **Seven new keys, not the five the contract tabled.** `attempt_number_guidance`
   and `records_pruned_known` were added: a bare `None` invites the fallback this step
   exists to remove, and `records_pruned_known` is the evidence behind
   `attempt_number_is_lower_bound`. Additive only.
2. **The contract planned 5 mutation cells; 11 shipped.** M8/M9 came from cycle 1,
   M10/M11 from cycle 2 — all four are mutants a Q/A found surviving, pinned so they
   cannot survive again.
3. **A subject was fixed rather than a probe loosened, twice.** Cycle 1: the guidance
   string gained the anti-fallback warning rather than the assertion being weakened.
   Cycle 3: C3d's second assertion was **tightened** `>= 6` → `> 6` because `>= 6`
   would not have discriminated.
4. **The enumeration filter was replaced by an explicit allowlist**, because a
   `verify_*` pattern would let a future production caller hide behind a
   checker-shaped filename.
5. **A test-only import seam** (`PYFIN_QA_WIP_OVERRIDE`) matching
   `verify_wip_retention_86_36.py`'s. Without it the RED half of criterion 7 is
   unprovable without writing to the tracked file.

---

## 6. Regression — all five sibling gates re-run after the last change

| checker | exit | result |
|---|---|---|
| `verify_wip_retention_86_36.py` | 0 | `ALL GREEN -- 23 passed, 0 failed` |
| `verify_qa_write_first_86_31.py` | 0 | `ALL GREEN -- 246 passed, 0 failed` |
| `mutation_matrix_86_31.py` | 0 | `MATRIX: 24/24 KILLED` |
| `mutation_matrix_86_36.py` | 0 | `OK -- all 5 cells KILLED on a named assertion` |
| `mutate_counter_source_86_21.py` | 0 | `mutants surviving (undetected): 0` |

**The 246 is non-deterministic by construction** and rose during this step:
`verify_qa_write_first_86_31.py` emits one assertion per live `verdict_wip_*.md`, so
every Q/A spawn on any step moves it (244 → 245 → 246 as the cycles ran). Flagged
because a changing "passed" count otherwise looks like drift.

---

## 7. Limits

- **Hand-deleted records remain undetectable.** The ledger accounts for the only
  automated deleter. `attempt_number_is_lower_bound` exists for this and is `True`
  for every live step today, since no ledger exists for any of them yet.
- **`attempt_number_is_lower_bound` is a heuristic**, not a proof — `True` iff no loss
  account **and** retained ≥ `DEFAULT_KEEP`. The boundary is now guarded (M10); the
  hand-deletion residual is not closable.
- **The new fields are not yet READ by the live rail's `qa.md` half.** `qa-verdict.js`
  now points at them; `qa.md` still does not. Until `4c` is applied, the instrument is
  fixed but only one of the two files the Q/A loads has been updated.
- **The saturation defect remains LATENT** — `prune_wip_records` still has zero
  production callers. This step makes pruning *safe if wired*; wiring is out of scope.
- **The matrix licenses one claim only:** these 11 mutations were killed by the
  assertions they were aimed at. Three of them existed because a Q/A found them
  surviving, which is direct evidence that a matrix an author writes alone is not a
  completeness proof.
- **`attempt_budget.py` is still unwired** — step 86.71.

---

## 8. CYCLE 4 — criterion 4 is now CLOSED, by a fresh executor

**The operator was asked and chose route B: a fresh executor applies the `qa.md`
edits.** That resolves the separation-of-duties block — Main authored the code, so Main
must not author the agent file it is graded against; a different author does.

Both `qa.md` sites this step owed are now corrected:

| site | was | now |
|---|---|---|
| `:622` | *"`records_retained` is the count of prior Q/A spawns … the **attempt number**"* — false on both halves | points at `attempt_number` / `prior_attempts`, states the INCLUSIVE unit, and demotes `records_retained` explicitly as a **gauge** that pruning can lower |
| `:645` | *"if `records_retained` (auto) **>** the ledger's verdict count…"* | compares `attempt_number`, and adds the `null` branch (`sequence: UNKNOWN`, never substitute the gauge) |

Deliberately **not** changed, exactly as classified: `:692` (accurate) and the dated
`qa_wip.py 86.33` measurement (rewriting it would falsify a record). Verified in the
diff: **0 lines containing `records_retained: 0` or `qa_wip.py 86.33` appear at all.**

**So criterion 4 is now met on all THREE members**: `4a` `DEFAULT_KEEP`'s comment
(cycle 1), `4b` `qa-verdict.js`'s four lines (cycle 2), `4c` `qa.md`'s two sites
(cycle 4). The step is no longer blocked on anything outside itself.

`verify_counter_86_79.py` re-run after the qa.md edits: **exit 0, 55 checks, 0 failed** *(cycle-6 mark: at that tree; 60/59 today)*.
All five sibling gates re-run green.


---

## Cycle 4 GENERATE (2026-08-17): the dropped cycle-3's three findings, closed at their sites

The cycle-3 rail drop returned no verdict but its write-first record named
three findings. Each is closed, and two turned out to have been closed
already by later steps -- verified rather than assumed:

1. **The gate now guards criterion 4's members 4b/4c by CONTENT**
   (`verify_counter_86_79.py`, the replaced C4 block): the stale
   un-applied-state assertions ("the un-applied qa.md correction is written
   out for the operator" -- true through cycle 3, FALSE from commit 9b4d5281
   which applied it) are REPLACED with applied-state content pins: qa.md
   must carry the gauge correction ("Do NOT use `records_retained` as the
   attempt number") and must NOT carry the old wrong wording ("the count of
   prior Q/A spawns" -- 0 hits); qa-verdict.js must state
   attempt_number/prior_attempts semantics, the null-is-never-0 rule, and
   the gauge distinction. A revert of either 4b or 4c now REDDENS the gate;
   the patch file stays asserted as the historical record. Gate run: ALL
   CHECKS PASS, exit 0 (live_check cycle-4 section).
2. **The stale-label finding is the SAME defect** -- the replaced block above
   IS the cycle-4-created stale label ("The residual divergence (qa.md)
   must be LOUD"), gone with the replacement.
3. **The two unguarded fields were closed by later work, verified now**:
   `records_retained_unit` is asserted on the healthy ok path (the C1
   section's 3-record report, check "states it is a GAUGE"); and
   `records_pruned_known` carries FOUR assertions (post-prune ==3, the
   add-back case, and regime accounting) -- both predate this cycle and are
   re-run green today.

**The product is load-bearing across the whole 2026-08-17 drain**: every
evaluator ran `qa_wip.py --spawned-at` and quoted
`records_retained=N (gauge, not counter)` + `attempt_number` +
`records_pruned_known` in its own notes; the 86.21 cycle-8 PASS explicitly
exercised the lower-bound cross-check ("attempt_number (4) is NOT greater
than the ledger's 6 rows, so the staleness rule does not fire").


---

## Cycle 5 GENERATE (2026-08-17): the five cycle-4 residuals, landed and driven

1. **F1 (the arithmetic)**: qa.md's staleness rule now compares
   `prior_attempts` -- the like-for-like prior-only count -- against the
   ledger's rows, with the correction note quoting the measured
   false-positive (4>3 on a current ledger while 3==3). One-line agent-file
   edit, flagged for operator review per the separation-of-duties rule *(cycle-6
   pointer, closing the cycle-5 Q/A's does-not-reproduce finding: at grading
   time the flag lived only in the commit message and artifacts; the
   harness-log row landed one commit later -- the Cycle 1240 entry of
   2026-08-17 lists both same-day qa.md edits by name, and the operator's
   same-day Approve-all batch covers the class)*.
2. **F2 (comment-parking)**: the gate's 4b/4c pins now search EFFECTIVE
   text only -- qa-verdict.js with // lines and /* */ spans stripped,
   qa.md with <!-- --> spans stripped. Driven: the evaluator's N7
   (rule parked in a // comment) and N9 (qa.md sentence inverted, original
   parked in an HTML comment) both now exit non-zero. DISCLOSED: my first
   N7 reconstruction survived -- the comment marker landed INSIDE a string
   literal, still an executable line -- the mutant was broken, not the pin;
   rebuilt to comment out the whole line, it dies.
3. **F3 (the absorbable block)**: EXPECTED_CHECKS raised 53 -> 59 per the
   floor's own design note; N11 (55 run) and N12 (exactly 53) now fail the
   floor by arithmetic. Gate at the cycle-5 edit: 60 checks, floor 59,
   ALL CHECKS PASS, exit 0.
4. **F4**: the stale 55-check headlines refreshed with their history.
5. **F5**: qa_md_patch_86.79.md retitled APPLIED-at-cycle-4 historical
   record; its self-falsifying command is now quoted WITH its output.


---

## Cycle 6 GENERATE (2026-08-17): the five cycle-5 findings, landed and driven

1. **F4 completed for the CLASS**: all six remaining CURRENT-labelled 55/53
   sites are marked at the line with their cycle (two of the misses were the
   day's third dash-trap -- en-dash and em-dash variants of anchors I had
   matched with hyphens; the marks say so).
2. **F5 completed by REPLACEMENT**: qa_md_patch:17's false present-tense
   sentence is replaced with the correction quoting both falsifying commits
   AND a verifying command that CAN fail (`git log --oneline -- qa.md`),
   with the vacuity of the old command (a working-tree diff on a committed
   tree) named. Third bite of this file; the replacement note records the
   pattern.
3. **The harness-log claim carries its pointer** (the Cycle 1240 row).
4. **E1 (trailing-comment park)**: the gate's JS reader now strips TRAILING
   // comments quote-aware (a // inside a string is payload); the
   evaluator's exact construction (`const _pin = 1; // parked: ...`) is
   KILLED, and the whole-line and deletion forms still die.
5. **E4 (the unpinned F1)**: two new gate checks pin the staleness rule's
   operand both ways (prior_attempts present; the inclusive form absent);
   floor raised 59 -> 61 with the run now at 62. Reverting the F1 edit
   reddens the gate.

Captured at write time, exits unpiped: gate 62 checks / floor 61 / ALL
CHECKS PASS / exit 0; E1 drive KILLED; matrix unchanged (11/11 at its last
run -- the gate additions do not touch its subject).
