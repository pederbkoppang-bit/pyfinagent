# Contract -- phase-86.88

**Step:** `86.88` (P1) · **Cycle:** 1 · **Written:** 2026-08-16, AFTER the gate returned.
**Title:** the lite position-pct seam is guarded only at the seam: a caller-side
pre-mangle reintroduces the D6 defect and is caught by NEITHER the suite NOR the
AST checker, and four whole-dict routes are invisible to that checker

---

## 1. Research gate -- PASSED (enforced, not self-reported)

| Field | Value |
|---|---|
| Rail | `.claude/workflows/research-gate.js` by **scriptPath** · run `wf_bb6c3ea8-26a` |
| Brief | `handoff/current/research_brief_86.88.md` (38,584 chars, `COMPLETE`) |
| Sources read in full | **15** (floor 5) · URLs **30** (floor 10) |
| Audit-class | YES -- `coverage.dry` after **2 dry rounds** over **9** rounds |
| `gate_passed` | **true**, RECOMPUTED; self-report agreed |

Read in full: parse-don't-validate (King); CodeQL data-flow; Semgrep taint mode;
Fowler *Mocks Aren't Stubs*; arXiv 2406.09843v3; PIT mutator catalogue;
`copy` module docs; arXiv 2601.19088v2 (PyTation, ICSE'26); Google SWE Book ch13;
**CERT OBJ06-J**; ar5iv 2102.06829; ar5iv 2001.04221; Taoxie ICSM'07;
LangChain-stub practice; Pyre/Pysa basics.

---

## 2. THE STEP'S OWN PREMISE IS WRONG, and the plan is built on the corrected one

The step title and `audit_basis` say the checker's `<whole-dict>` branch is
**DEAD** ("a Call node is not a BoolOp operand, so it can never fire").

**Measured -- by me and, independently, by the research gate -- it is NOT dead.**
Driving the SHIPPED `or_default_sites`:

```
=== the SHIPPED file ===              <whole-dict> fired: False
=== POSITIVE CONTROL `x or _LITE_RISK_DEFAULT` ===
    sites: [(1, '<whole-dict>'), (2, 'recommended_position_pct')]
    <whole-dict> fired on the control: True
=== dict(_LITE_RISK_DEFAULT) ===      sites: []
```

The branch **fires**. It is *unreachable on this file's idioms*, because the four
whole-dict routes are `dict(...)` **Call** nodes and a Call is not a `BoolOp`
operand. The distinction decides criterion 4: the remedy is **widening the
accepted node shapes**, not "resurrecting a corpse" or deleting a branch that
works. A plan built on "dead branch" aims at the wrong thing.

Criterion 4 is immutable and is NOT amended. It offers "made LIVE ... or
DELETED", and this contract chooses **made LIVE on a real match** -- which is
exactly what widening delivers.

---

## 3. The mechanism, restated from measurement

`86.86` guards the **producer**. Four callers replace `risk_dict` with
`dict(_LITE_RISK_DEFAULT)` **before** the producer runs, so the three-state seam
resolves **SIZE(3.0)** instead of **ABSENT**.

**Same number, destroyed provenance** -- and that is precisely why no
number-asserting test sees it. A judge FAILURE (no parseable output at all) is
persisted as though the judge had deliberately specified 3%.

Coverage, measured: the suite, the AST checker and all six 86.86 mutation cells
anchor **at or below** `_build_lite_risk_assessment`. **No test executes
`_run_claude_analysis` or `_run_gemini_analysis`**, so zero routes are exercised
while the checker prints `RESULT: OK`.

---

## 4. Criterion 1 -- ALREADY SATISFIED, before any change

Control GREEN first: `62 passed`. Then, in-process, restore-verified
(`sha256 5b714a9e5f43753c…` identical before and after every cell):

| mutation | 62-test suite | AST checker | verdict |
|---|---|---|---|
| **N1** caller-side pre-mangle before the producer call | **62 passed -- BLIND** | **RESULT: OK -- BLIND** | **SURVIVED** |
| PC1 restore the D6 falsy-or at the seam | 11 failed | RESULT: FAILED (exit 1) | KILLED |
| PC2 neuter `_lite_position_pct` | 11 failed | RESULT: OK | KILLED |

Both controls KILL, so N1's survival is a fact about the guards rather than a
dead probe -- which is what criterion 1 demands.

**PC2 adds something the step's `audit_basis` does not state:** the checker is
blind to a neutered *resolver* too. The two guards have **different** blind
spots, and N1 sits in the **intersection**. Closing only one leaves N1 alive.

---

## 5. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. the N1 mutant is REPRODUCED first, with the control observed GREEN and a discriminating positive control shown (a known-killable cell scored KILLED in the same run), so a SURVIVED result is proven to be about the guard and not about a dead probe
2. an end-to-end test drives at least one real lite risk-judge path with a STUBBED LLM returning {"recommended_position_pct": 0} and asserts no order results -- driving the real _run_*_analysis code, not a copy of its dict construction
3. after the fix the N1 mutant is KILLED, demonstrated by re-running the identical injection, with a byte-identical restore verified by sha256
4. the dead `<whole-dict>` branch is either made LIVE (it must be able to fire and be shown firing on a real match) or DELETED -- a branch retained but unreachable is a zero-assertion guard and fails this criterion; whichever is chosen, the reason is stated
5. all four `dict(_LITE_RISK_DEFAULT)` routes are enumerated from source and each is classified: state whether a judge FAILURE persisting as SIZE 3.0 rather than ABSENT is acceptable, and if it is not, fix it at the seam rather than at the four call sites
6. the reachability of each of the four routes is established by DRIVING it, not by reading the handler that contains it
7. verdict semantics and order outcomes for every input in the 86.86 disclosure table are UNCHANGED unless a change is explicitly justified, demonstrated under both states of paper_risk_judge_reject_binding
8. no gate is loosened, no flag is promoted, and no .env is written to obtain a green result

**Immutable command** (run before planning; currently exit 0):
`bash -c 'source .venv/bin/activate && python scripts/qa/verify_lite_risk_seam_86_86.py'`

---

## 6. Plan

**P1 -- fix at the SEAM, per criterion 5.** The four routes carry 3.0 into
`risk_dict` so a judge FAILURE is indistinguishable from a judge that said 3%.
That is **not acceptable**: it is the same provenance collapse 86.74/86.86 exist
to prevent, one seam over. The fix is **CERT OBJ06-J's copy-then-validate**
applied at the producer: `_build_lite_risk_assessment` must distinguish
*"the caller handed me the default object"* from *"the judge specified a size"*,
so the resolver receives ABSENT rather than SIZE. Criterion 5 explicitly requires
this at the seam and **not** at the four call sites.

**P2 -- widen the checker's node shapes (criterion 4).** `or_default_sites`
accepts a `Subscript` or a bare `Name` as a BoolOp operand. It must also see the
`dict(_LITE_RISK_DEFAULT)` **Call** shape, so the branch fires **on a real
match** in the shipped file. Shown firing, not asserted.

**P3 -- the end-to-end test (criterion 2).** Per Fowler and the Google SWE Book:
**stub the transport, assert the state.** Drive the real `_run_claude_analysis`
with a stubbed LLM returning `{"recommended_position_pct": 0}` and assert **no
order results** -- driving the production path, never a copy of its dict
construction. This is the coverage unit no existing test occupies.

**P4 -- reachability by DRIVING (criterion 6).** Each of the four routes is
reached by forcing its handler (no-JSON, exception) and observing the route
taken -- not by reading the handler.

**P5 -- CALL-SITE mutation, which is the literature's actual name for this.**
Delamaro's interface mutation, second operator group: *mutate invocations*. PIT's
call operators ship **off by default**; PyTation (ICSE'26) mutates call arguments
and reports **69% non-subsumption** by traditional operators -- i.e. unit-level
mutation coverage structurally cannot see this class. The mutation matrix gains
call-site cells, and per **muSE**, a checker that fails to detect a real member
is a **tool flaw**, not an acceptable bound.

**P6** -- criterion 7 under both `paper_risk_judge_reject_binding` states; then
handoff, Q/A, log.

---

## 7. Out of scope (named)

- `86.87` (the lite `risk_assessment` fabricating its own audit trail via the
  RETAINED `or _LITE_RISK_DEFAULT[...]` keys) stays filed and separate.
- The escalated `86.90`/`86.91` remain PARKED; nothing here touches them.

## 8. References

- `handoff/current/research_brief_86.88.md` (run `wf_bb6c3ea8-26a`)
- CERT **OBJ06-J** -- defensive copying of mutable inputs
- Delamaro et al., interface mutation; PIT mutator catalogue (call operators off by default)
- **PyTation**, ICSE'26 (arXiv 2601.19088v2) -- call-argument mutation, 69% non-subsumption
- **Cling** (ar5iv 2102.06829) -- coupled-branch coverage as the integration unit
- **muSE** -- non-detection by a checker is a tool flaw
- Fowler, *Mocks Aren't Stubs*; Google SWE Book ch13 -- stub the transport, assert state
- `handoff/archive/phase-86.86/`, `handoff/current/*_86.86.md`
