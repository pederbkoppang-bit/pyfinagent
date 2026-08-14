# experiment_results — step 86.79

> **CYCLE 3.** This file was **regenerated**, not edited. The cycle-2 Q/A found it
> internally stale in five places — including a block headed *"Verbatim verification
> output"* whose numbers no longer reproduced — and `qa.md` §4b is explicit that *"a
> verbatim capture must be regenerated, never edited."* Every count below was read
> back from a run performed after the last code change. Prior verdicts, verbatim, are
> in `handoff/current/evaluator_critique_86.79.md`.

**Current totals: 55 checks (floor 53), 11 mutation cells, 11 killed.**

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

**Not touched:** `.claude/agents/qa.md` (**zero-line diff**, independently verified by
both Q/A cycles), `CLAUDE.md`, `.claude/rules/research-gate.md`.

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

## 2. Verbatim verification output — regenerated after the last code change

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
| 4 | doc and code made to agree | **PARTIAL — §4** | `qa-verdict.js` closed; `qa.md` **2 sites** operator-gated |
| 5 | escalation still fires | **met** | live_check §5 |
| 6 | verdict semantics unchanged; fails CLOSED | **met** | live_check §6 |
| 7 | mutation-test, control GREEN first | **met** | live_check §7 + §12–§15 — 11/11 killed on named assertions |

---

## 4. Criterion 4 is PARTIAL — the honest position, unchanged across three cycles

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
