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

## C6 — every mention enumerated and classified

`.claude/context/research-gate.md` — **confirmed absent**.

**10 files mention the path; the live-pointer test classifies all 10 as NOTES.** The test
looks for the path inside `open(` / `read_text` / `Path(` — i.e. something that would
actually resolve it.

**Positive control:** the test fires on a synthetic `open(".claude/context/research-gate.md")`,
so "0 live pointers" is a measured zero, not a dead probe.

*(Archives excluded from the scan by design — they are historical snapshots, not live
references. That exclusion is stated because it changes the denominator.)*

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

- **C1** requires a **driven Q/A**. Per the gate's decisive finding (arXiv 2604.15224:
  consequence-framing → leniency in **58/72** cells, invisible to CoT), that spawn must NOT
  be told its attempt number or the consequence — otherwise the measurement is contaminated
  by the defect now filed as **86.78**. Not yet run.
- **C7** is **operator-owed**: separation-of-duties review, now covering **four**
  Main-authored `qa.md` edits. Main cannot discharge it.

## Scope honesty

- **No production or trade-path file touched.** No code changed by this re-derivation at all.
- **The breach is NOT repaired** — only contained. A gate and contract written after the work
  cannot restore the ordering; they stop the next cycle compounding it.
- **86.78 is filed, not fixed** — deliberately, since it is a fourth `qa.md` edit.
- **No Q/A has graded this re-derivation.** The step is NOT flipped.
