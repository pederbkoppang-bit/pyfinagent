# experiment_results — step 86.79

> **CYCLE 2.** The cycle-1 Q/A (`wf_61338c26-b90`) returned **CONDITIONAL** with four
> findings. All four were remediable in code and **all four are fixed**; the blockers
> and the evidence changed, so a fresh Q/A grades changed evidence, not the same
> evidence twice. See **§0** for the cycle-2 delta and
> `handoff/current/evaluator_critique_86.79.md` for the verbatim cycle-1 verdict.

---

## 0. Cycle-2 remediation — what the cycle-1 Q/A found, and what changed

| # | cycle-1 finding | fix | proof |
|---|---|---|---|
| 1 | **criterion 4: I enumerated ONE member of a TWO-member class.** The same false claim was duplicated in `.claude/workflows/qa-verdict.js` — the launch rail's own prompt — which is **not** under `.claude/agents/`, so separation-of-duties never blocked it | Fixed. Enumerating the whole class found **four** lines (`:147, :152, :159, :172`), not the two named; all four now point at `attempt_number`/`prior_attempts` and name `records_retained` a gauge | `git diff .claude/workflows/qa-verdict.js`; `node --check` passes |
| 2 | **surviving mutant N6** — dropping the pruned-away add-back on the `no_record_for_this_spawn` branch survived both gates. Root cause: C2 exercised that branch with no prune/ledger, so `lost_n` was 0 and the term was **dead**; C3 exercised the ledger only on the matched path | New section **C3b** combines them: a pruned step **with** a non-zero loss account, reported by a spawn that has not yet written. Asserts `prior_attempts == 6` (not 3) and that this is the difference between `ESCALATE` and `CONTINUE`. Added permanently as cell **M8** | C3b: 4 checks; M8 KILLED |
| 3 | **surviving mutant N1** — `attempt_number_is_lower_bound` had **zero** assertions anywhere; flipping it was invisible | New section **C3c** drives all three regimes (below window / at window without account / accounted) and asserts they **differ**. Added permanently as cell **M9** | C3c: 4 checks; M9 KILLED |
| 4 | **lint gate RED**: `F401 'datetime' imported but unused` at `verify_counter_86_79.py:27:8` | Dead import removed | `uvx ruff check --select F821,F401,F811` over the derived union scope (3 files, asserted non-empty) → **`All checks passed!` exit 0** |
| — | note: `EXPECTED_CHECKS = 30` while 42 ran | raised to **48** against a real count of 50 | control prints `50 checks` |

**Totals moved: 42 → 50 checks, 7 → 9 mutation cells, all killed on named
assertions.** Both cycle-1 survivors are now permanent cells, so the guards written
for them are proven able to fail rather than asserted to exist.

**What I did NOT do.** The consequence-framing text in the same `qa-verdict.js` block
(*"return FAIL instead of a third"*, *"at 5+, recommend operator escalation"*) is
sibling step **86.78**'s subject and was left untouched. And
`.claude/agents/qa.md` still has a **zero-line diff** — the remaining member of the
divergence class is still operator-gated, and criterion 4 is therefore still
**PARTIAL**, now at **1 of 2 members fixed** instead of 0 of 2.

---


**Phase:** GENERATE (after RESEARCH → PLAN; contract at
`handoff/current/contract_86.79.md` was written **before** any code change).
**Evidence:** `handoff/current/live_check_86.79.md` — verbatim output, re-runnable.

---

## 1. What was built

| file | change |
|---|---|
| `scripts/qa/qa_wip.py` | **modified** — +220 / −7. The whole fix. |
| `scripts/qa/verify_counter_86_79.py` | **new** — 42-check re-runnable checker, criteria 1–6 |
| `scripts/qa/mutation_matrix_86_79.py` | **new** — 7-cell mutation matrix, criterion 7 |
| `handoff/current/qa_md_patch_86.79.md` | **new** — the un-applied `qa.md` correction, for the operator |
| `handoff/current/{contract,research_brief,live_check}_86.79.md` | **new** — protocol artifacts |

**Not touched:** `.claude/agents/qa.md`, `CLAUDE.md`, `.claude/rules/research-gate.md`
— zero-line diffs, shown in live_check §8.

### The five changes inside `qa_wip.py`

**F1 — the field is split, and the unit travels with the number.**
`report()` keeps `records_retained` at its exact prior name *and value* (no live
number was shifted) and adds: `attempt_number`, `prior_attempts`,
`attempt_number_status`, `attempt_number_is_lower_bound`,
`attempt_number_guidance`, `records_pruned_known`, `records_retained_unit`.
`attempt_number` is **INCLUSIVE** of the current attempt — a first attempt is 1 —
and says so, because the research found Temporal's `MaximumAttempts` and Step
Functions' `MaxAttempts` ship **opposite units under the same word**.

**F2 — the write-first coupling is now unrepresentable rather than documented.**
`report()` tracks `matched_current`: whether a retained record was positively
identified as belonging to *this* spawn. `attempt_number` is computed only when it
was. Otherwise it is `None` with `attempt_number_status =
no_record_for_this_spawn`, while `prior_attempts` — which genuinely *is* knowable —
is still reported. Seemann's remedy for temporal coupling is structural; a sentence
in another file cannot be the fix.

**F3 — the pruner records what it destroys** (`PERF_RECORD_LOST` shape).
`prune_wip_records` writes a monotonic per-step loss account to
`.claude/agent-memory/qa/verdicts/.attempt_lost_<sid>.json` **before** unlinking,
so a crash mid-prune over-counts (escalates early, safe) rather than under-counts.
`report()` adds it back in. Dot-prefixed, so it is invisible to both
`list_wip_records`' globs and `audit_memory.py`'s top-level glob — asserted, not
assumed.

**F4 — `DEFAULT_KEEP`'s own off-by-one comment is fixed. THE DOC MOVED, NOT THE
CODE.** The comment claimed *"Current record + this many prior attempts"* (4
retained) while `records[keep:]` retains 3. `records[keep:]` is standard keep-N
semantics and matches the k8s/journald precedents the module already cites, so
changing the arithmetic would have silently widened live retention for no benefit.
The comment now states the unit: `keep` is the **TOTAL**, **INCLUSIVE** of the
current record. **This off-by-one was found by the research gate, not by me** —
it is a third defect, independent of the two the step was filed for.

**F5 — fail closed.** Every path that cannot compute the number returns `None`,
never `0`, copying `verdict_history_86_21.py`. Zero is a claim about attempts; a
threshold compared against a spurious 0 silently suppresses escalation. Each
refusal also carries `attempt_number_guidance` telling the reader **not** to fall
back to `records_retained` and why.

---

## 2. Verbatim verification output

Full output is in `live_check_86.79.md` §0–§7. Headline:

```
$ python scripts/qa/verify_counter_86_79.py
  checks run : 42   (cardinality floor 30)
  failed     : 0
  ALL CHECKS PASS
exit=0

$ python scripts/qa/mutation_matrix_86_79.py
  [CONTROL] unmutated checker -> exit 0
  cells: 7   killed: 7   survived/unearned: 0
  subject sha256[:16] before=146600b722a02481 after=146600b722a02481 -> tracked file UNCHANGED
  ALL CELLS KILLED
exit=0

$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/qa_wip.py\").read())" && echo qa_wip-parses'
qa_wip-parses
exit=0
```

The three headline numbers, each reproduced in the live_check:

| the defect | measured |
|---|---|
| `records_retained` counts the current spawn | 2 priors → **3** |
| the number depends on write-first ordering | same spawn: **2** before its write, **3** after |
| pruning saturates it | 6 true attempts → `records_retained` **3**, `attempt_number` **6** |
| F1b's ceiling after a prune | old: `3/5` → **CONTINUE**. new: `6/5` → **ESCALATE** |

---

## 3. Criterion-by-criterion

| # | criterion (abridged) | status | where |
|---|---|---|---|
| 1 | off-by-one REPRODUCED, producing line quoted | **met** | live_check §1 (line **grep-derived**, not hardcoded) |
| 2 | write-first coupling demonstrated | **met** | live_check §2 — 2 vs 3 on the same spawn |
| 3 | pruning saturation demonstrated + enumeration with command stated | **met** | live_check §3 — and prune has **no production caller** |
| 4 | doc and code made to agree, and which moved stated | **PARTIAL — see §4** | live_check §4 |
| 5 | escalation still fires after the fix | **met** | live_check §5 — both bounds, incl. through a prune |
| 6 | verdict semantics unchanged; uncomputable fails CLOSED | **met** | live_check §6 |
| 7 | mutation-test, control GREEN first | **met** | live_check §7 — 7/7 killed on named assertions |

---

## 4. Criterion 4 is PARTIAL, and I am not going to dress that up

There are **two** doc/code divergences. One is fixed; the other is deliberately
left, and the step should be judged on that.

| | divergence | state |
|---|---|---|
| **4a** | `DEFAULT_KEEP`'s comment vs `records[keep:]` — both inside `qa_wip.py` | **FIXED** (F4). The **doc** moved; the reason is recorded in the code. |
| **4b** | `.claude/agents/qa.md` — *"`records_retained` is the count of prior Q/A spawns … the attempt number"*, two descriptions of one integer differing by one | **NOT FIXED** |

**Why 4b was not simply done.** `qa.md` already carries **four Main-authored edits
awaiting operator review** under CLAUDE.md's separation-of-duties rule. A fifth
deepens a hold the operator owns. The session's operator instruction is explicit —
*"If a fix genuinely needs `qa.md`, stop and ask"* — and the masterplan step's own
notes say *"prefer changing `qa_wip.py`, or hand it to a fresh executor."*

**What was shipped instead**, so the divergence is **not silent** (which is what
criterion 4 actually forbids):

1. `records_retained_unit` — the unit, in the payload the Q/A reads, at the point
   of use. This is the research's own remedy for E1.
2. `attempt_number_guidance` — on the failure path it says in words: *"Do NOT fall
   back to records_retained here … a low number SUPPRESSES escalation."*
3. `handoff/current/qa_md_patch_86.79.md` — the exact replacement text, three
   routes, and a recommendation (**a fresh executor applies it**).

**Honest consequence:** until route A or B is taken, the new fields are
**available but unread by the live rail**. This step fixes the instrument; it does
not yet change what the rail measures with. If the Q/A judges criterion 4 unmet,
that is a correct reading and I am not asking for it to be waived.

---

## 5. Deltas from the contract — disclosed

1. **Seven new keys, not five.** The contract's table listed five; the
   implementation adds `attempt_number_guidance` and `records_pruned_known` as
   well. Reason: a bare `None` with no explanation invites the reader to fall back
   to `records_retained` — the exact defect being removed — and
   `records_pruned_known` is the direct evidence behind
   `attempt_number_is_lower_bound`. Additive keys only; no behaviour in the
   contract changed.
2. **One assertion was rewritten mid-run, and the SUBJECT was fixed, not the
   probe.** The check "it tells the reader what to do instead of falling back"
   failed on first run because the guidance did not name `records_retained`. The
   guidance was the thing that was wrong — warning against the fallback is the
   whole point — so the guidance gained the warning. Recorded because the opposite
   move (loosening the assertion) would have looked identical in the final output.
3. **The enumeration filter was tightened after first run.** A regex
   (`^\./scripts/qa/(qa_wip|verify_|mutat…)`) was replaced with an **explicit
   allowlist**, because the regex would have let a future production caller hide
   behind a checker-shaped filename.
4. **A test-only import seam was added to the checker**
   (`PYFIN_QA_WIP_OVERRIDE`), identical in shape to the one
   `verify_wip_retention_86_36.py` already carries. Without it the RED half of
   criterion 7 is unprovable without writing to the tracked file.

---

## 6. Regression

All 5 other gates that exercise `qa_wip.py` were run and are green — 23, 244,
24/24, 5/5 and 0-surviving respectively (live_check §9). `mutation_matrix_86_36.py`
independently reports the same subject digest before and after, corroborating that
this step's matrix never wrote to the tracked file.

---

## 7. Limits — repeated here so they are not only in the live_check

- **Hand-deleted records are still undetectable.** The ledger accounts for the only
  automated deleter. `attempt_number_is_lower_bound` is `True` for every live step
  today, because no ledger exists for any of them yet.
- **`attempt_number_is_lower_bound` is a heuristic**, not a proof — see live_check §11.
- **The saturation defect remains LATENT.** `prune_wip_records` still has no
  production caller; this step makes pruning *safe if wired*. Wiring is out of scope.
- **The mutation matrix licenses one claim only:** these 7 mutations were killed by
  the assertions they were aimed at. It is not a "no vacuous guards" proof.
- **`attempt_budget.py` is still unwired** — that is step 86.71.
