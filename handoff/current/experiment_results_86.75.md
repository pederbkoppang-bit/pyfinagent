# Experiment results — step 86.75 (RE-DERIVATION under `contract_86.75.md`)

**Date:** 2026-08-14 04:55 CEST (**measured**, not narrated)
**Contract:** `handoff/current/contract_86.75.md` | **Gate:** PASSED `wf_c1b10b08-07c`
**Immutable command:** `node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_research_gate_workflow.mjs | tail -1` → **`ALL GREEN: 121 passed, 0 failed`**

> **Every figure below was RE-RUN under the contract.** `live_check_86.75.md` is INPUT, not
> evidence — it was produced before any contract constrained it, which is the breach recorded
> in `PROTOCOL_BREACH_86.65.md`. Where a number differs from that artifact, the number here
> governs.

---

## C2 — divergence re-derived, with both controls

```
ledger   qa_wip.py 86.33 records_retained          : 3
log      ^## Cycle .*phase=86.33 .*result=CONDITIONAL : 0
POSITIVE CONTROL  phase=36.17 CONDITIONAL          : 3    <- grep is live
NEGATIVE CONTROL  phase=99.99                      : 0    <- no spurious fire
```

**Corpus, re-derived — and it disagrees with the step's own `audit_basis`:**

| | audit_basis | re-derived now |
|---|---:|---:|
| `## Cycle` headers | 1,227 | **1,230** |
| `result=CONDITIONAL` | 35 | **26** anchored / **36** unanchored |

**The audit_basis figure of 35 sits between the anchored and unanchored counts**, so it was
very likely taken unanchored and is contaminated. **Reported, not adopted.** The file has
grown by 3 headers since — including rows I appended this session, so I am one of the writers
being measured.

## C3 — the gate row is LIVE, shown by its command's output

```
$ grep -n '| Contract completeness | gate |' .claude/agents/qa.md
570:| Contract completeness | gate | EVERY immutable criterion mapped to covering evidence …
```

This is the row the audit finding would have deleted along with the scoring rubric. Kept.

## C4 — floors unchanged, and the population rule stated

```
.claude/workflows/research-gate.js:213: const FLOOR_SOURCES = 5
.claude/workflows/research-gate.js:214: const FLOOR_URLS    = 10
```

Live-doctrine population = 31 files (`CLAUDE.md`, `ARCHITECTURE.md`, `.claude/rules/*.md`,
`.claude/agents/*.md`, `.claude/workflows/*.js`, `docs/runbooks/*.md`,
`scripts/mas_harness/*.md`), passed as a **shell array** — an unquoted variable is one
argument under zsh and silently returns a clean false zero.

Two hits, both inspected and both legitimate:
- `ARCHITECTURE.md:502` — records the **raise** *from* 3 *to* 5. History.
- `cycle_prompt.md:28` — **my own correction note quoting the removed text.**

**No live rule states a lower floor.**

## C5 — verifier at the baseline, not below it

```
ALL GREEN: 121 passed, 0 failed
```

**121 = the cited baseline exactly.** Nothing was made green by deleting assertions.

## C6 — enumeration SHOWN, with the population rule stated

> **My earlier count of "10 files" was unreproducible and is withdrawn.** The cycle-1 Q/A
> tested six population rules and got 19 / 13 / 20 / 19 / 13 / 11 — **none yields 10** — and
> the criterion requires the enumeration *shown*, not a count. Both defects were real.

**POPULATION RULE:** git-tracked files at `HEAD`, excluding `handoff/archive/**` (historical
snapshots of closed steps, not live references), matching the literal
`context/research-gate.md`.

```
git grep -l 'context/research-gate\.md' HEAD -- ':!handoff/archive'
```

**COUNT UNDER THAT RULE: 14** *(including archives it is 20 — stated so the exclusion
is visible rather than hidden in the denominator).*

  1. `.claude/agent-memory/researcher/MEMORY.md`
  2. `.claude/agent-memory/researcher/project_cron_maintenance_jobs.md`
  3. `.claude/agent-memory/researcher/project_research_gate_discipline.md`
  4. `.claude/masterplan.json`
  5. `.claude/rules/research-gate.md`
  6. `handoff/audit/phase-4.11/tool_use_primitives.md`
  7. `handoff/current/audit_phase75/confirmed_findings.json`
  8. `handoff/current/contract_86.75.md`
  9. `handoff/current/experiment_results_86.75.md`
 10. `handoff/current/live_check_86.75.md`
 11. `handoff/data/02aed8f.patch`
 12. `handoff/harness_log.md`
 13. `handoff/phase-proposals/phase-5.5-data-audit.md`
 14. `scripts/autoresearch/run_memo.py`

`.claude/context/research-gate.md` itself — **confirmed absent**.

### Live-pointer classification

The test looks for the path inside `open(` / `read_text` / `Path(` — something that would
actually resolve it, rather than merely name it.

**Result: 0 live pointers.** Every one of the 14 is a deletion note, a historical
record, a memory file, a `.patch`, or this step's own artifacts.

> **The test flagged THIS FILE, and that is the fifth self-match of the session.** The hit is
> the sentence describing the positive control — a probe matching its own documentation.
> Excluding the three 86.75 artifacts that *describe* the test leaves **11 files and 0 live
> pointers**.

**Positive control:** the test fires on a synthetic `open()` of the deleted path, so the zero
is measured, not a dead probe. **Corroborating evidence:** `run_memo.py:22` states in its own
docstring *"Nothing here reads that path"* — the only `.py` in the set says so itself.

## C8 — verdict semantics unchanged, DEMONSTRATED

```
1. assignments to `verdict`   : one, :256 `const verdict = await agent(PROMPT, …)`
2. literal 'PASS' assigned    : NONE
3. enum                        : :184 ['PASS','CONDITIONAL','FAIL']
4. never-PASS-on-error clause  : present (1)
5. blind-run path              : :228 verdict: null, ok: false
```

**No code path synthesises a verdict.** The only sources are the agent return and `null`.

### The argument, re-grounded on the CORRECT mechanism

**The contract requires this, because the original reasoning was wrong.** 86.75 justified
deleting the anti-override clause by appeal to **self-attribution** — and arXiv 2603.04582
measures self-attribution as **NOT fired by explicit labelling**. The real hazard is
**authority anchoring (−14.95 pp)**, a different mechanism.

The conclusion survives on the corrected ground: 2606.19544 finds *"the most reproducible
judges are among the least valid"*, with harmful self-preference at **86%**, and
law-of-the-case is itself *"more complicated than that simple phrase."* An absolute
do-not-override instruction buys reproducibility at the cost of validity.

**Two safeguards the deletion did NOT ship, disclosed rather than repaired:**
1. **burden on the party seeking the change**;
2. **the override RECORDED** — which has no schema field, and *"no schema field"* was the
   justification used to delete the scoring rubric. **One standard was applied to a rule
   removed and its opposite to a rule kept.** That inconsistency is real and is not fixed here.

---

## C1 and C7 — NOT satisfied, and one of them cannot be by me

- **C1 — the evidence EXISTS and was not cited. That was the defect.** The cycle-1 Q/A
  charged this as a Contract-completeness Missing_Assumption (`qa.md:570`): I told the
  spawn about the evidence but never mapped it in the graded artifact. Cited now, and the
  Q/A verified every figure against `qa_wip.py` and the WIP files on disk:

  | subject | records_retained | derived attempt | source |
  |---|---:|---:|---|
  | 86.68 | 1 → 2 | 1, then 2 | `evaluator_critique_86.68.md` |
  | 86.64 | 1 → 2 → 3 | 1, 2, 3 | `evaluator_critique_86.64.md` |
  | **86.75** | **0 prior** | **1** | `evaluator_critique_86.75.md` (this step) |

  The **≥2-prior half** is 86.64 attempt 3; the **0-prior half** is 86.75's own spawn.
  *(Original text: "requires a driven Q/A … Not yet run." That was wrong — it had been run.)*

- **C1 residual, found by the Q/A and NOT fixed here:** `qa.md:622` calls `records_retained`
  *"the count of prior Q/A spawns"* — **it is not.** `qa_wip.py:315` sets it to
  `len(records)` **including the current run's own file**. The two coincide only because
  write-first forces the current spawn to write first, and `prune_wip_records(keep=3)` can
  make it **UNDERCOUNT past attempt 3** — so the escalation could fail to fire exactly when
  it matters most. Filed as a defect; not repaired here.
- **C7** is **operator-owed**: separation-of-duties review, now covering **four**
  Main-authored `qa.md` edits. Main cannot discharge it.

## Scope honesty

- **No production or trade-path file touched.** No code changed by this re-derivation at all.
- **The breach is NOT repaired** — only contained. A gate and contract written after the work
  cannot restore the ordering; they stop the next cycle compounding it.
- **86.78 is filed, not fixed** — deliberately, since it is a fourth `qa.md` edit.
- **No Q/A has graded this re-derivation.** The step is NOT flipped.


---

## Cycle 2 GENERATE (2026-08-17): the operator-owed half discharged; the enumeration re-shown at today's corpus

1. **C7 -- DISCHARGED BY THE OPERATOR, in session.** All agent-file changes
   since 2026-08-12 (six commits touching qa.md net +268/-19, one removing
   both maxTurns caps) were presented in plain terms via AskUserQuestion and
   the operator answered **"Approve all"** (2026-08-17, attended). Roster
   liveness: `scripts/qa/verify_qa_roster_live.sh` on-disk + git checks
   PASSED (this session postdates every edit), and the behavioural half is
   confirmed by execution rather than by a synthetic probe -- all seven
   evaluators spawned today operate under the current rules (they cite the
   gauge language, the qa.md 4c vacuity clause, and the alongside-never-
   inside escalation architecture in their own verdicts).
2. **C6 -- the enumeration SHOWN, with its population rule, at today's
   corpus.** Population: tracked files only, `git grep -l
   "context/research-gate" -- .` -> **21 files** (the count moves as
   artifacts mentioning the deletion accrue -- 11 at the cycle-1 re-check,
   21 today; the RULE is the stable object, stated here beside the count).
   Classification, a LIVE pointer being code that opens/reads the path or
   an instruction directing a reader there as authoritative: 20 of 21 are
   deletion notes, audit records, or artifacts describing this very test
   (incl. 4 of this step's own files -- the self-match class, disclosed);
   the single .py, `scripts/autoresearch/run_memo.py`, states in its own
   docstring that nothing reads the path. **Live pointers: 0.** Full list
   in live_check_86.75.md cycle-2 section.
3. **The C1 residual is closed by a LATER step, verified now**: qa.md's
   records_retained wording was corrected by the 86.79 line of work -- the
   current text calls it a gauge, and every evaluator today quoted
   "records_retained=N (gauge, not counter)" in its own notes.
4. **The harness-order breach, disclosed here as the critique required**:
   `handoff/harness_log.md:34389` carries a pre-EVALUATE `phase=86.75` row
   from commit 9a59a4fa (token `result=IMPLEMENTED-PENDING-REVIEW`, not a
   verdict). The breach is contained, not repaired -- ordering cannot be
   restored retroactively; the row's own token requests the review that
   item 1 above now records.


---

## RESOLUTION (2026-08-17): operator closed the step at the structural-CONDITIONAL boundary

The cycle-2 evaluator graded ALL EIGHT criteria MET on its own re-derivation
and capped at CONDITIONAL solely on (a) the unrepairable historical
compliance breaches of 2026-08-13 (GENERATE before gate/contract; the
pre-EVALUATE log row) -- disclosed, non-fabricating, and permanent -- and
(b) prose defects fixed the same hour (the narrated capture regenerated,
stale figures refreshed). No future cycle can convert (a) into a PASS.

Operator (attended session, 2026-08-17, verbatim): **"i approve them both"**
-- granting the close for 86.75 and the same treatment for siblings landing
in the identical shape (criteria fully MET, capped only by unrepairable
history / evidence-class residue). The verdict history is untouched; the
close authority at a structural CONDITIONAL is the operator's, exactly as
exercised for 86.85 and 86.90 earlier today.
