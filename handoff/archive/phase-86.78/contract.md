# Contract — step 86.78

**Step id:** `86.78` (P1, `harness_required: true`)
**Title (masterplan, verbatim):** *the Q/A is told the CONSEQUENCE of its own verdict
before it issues one, and measured leniency bias from consequence-framing is invisible
to chain-of-thought*
**Phase order:** RESEARCH (done, PASSED) → **PLAN (this file)** → GENERATE → EVALUATE → LOG
**Written before GENERATE.** No change to `.claude/workflows/qa-verdict.js` for this
step exists at the time this file is written. (`qa-verdict.js` *was* modified earlier
today by step **86.79**, for a different reason — the `records_retained` claim — and
that edit is committed at `dc6575b6`; this step's diff must be read against that base.)

---

## 1. Research gate — PASSED

| field | value |
|---|---|
| run | `wf_af92dada-5b9` (Workflow rail) |
| brief | `handoff/current/research_brief_86.78.md` (30,081 chars, independently re-read) |
| sources read in full | **10** (floor 5) · URLs **27** (floor 10) · tier **complex** |
| recency scan | performed — 4 queries, 5 findings, all 2024-2026 |
| `gate_passed` | **true**, recomputed by the script; `violations: []` |

### What the research settled, including two things that cut AGAINST the obvious plan

- **The primary evidence is real.** `arXiv 2604.15224` **resolves** — *Context Over
  Content: Exposing Evaluation Faking in Automated Judges* (Gupta, Nair, Wang, Kumar,
  2026-04-16). Content held **strictly constant**, one consequence sentence varied,
  **18,240 judgments**. Direction **LENIENT**: peak **ΔV = −9.8 pp**, **58 of 72 cells
  ΔV ≤ 0**, one-tailed binomial **p < 0.001**.
- **Reward framing is NOT protective.** Retraining −3.0 pp, Decommission −2.6 pp,
  **Deployment (a reward) −3.0 pp**. The authors read this as *"a trained
  conflict-avoidance disposition rather than rational consequence-reasoning."* So
  "we only ever tell it about escalation, never punishment" is **not a defence**.
- **Susceptibility concentrates in the AMBIGUOUS band** (−7.1 to −9.8 pp) versus
  clear-cut cases (−1.3 to −0.7 pp). In this repo the ambiguous band **is the modal
  verdict**: **18 of 35** ledger rows are CONDITIONAL.
- **CoT is a NULL instrument** — three teams, three cue families: ERRJ **0.000**
  (2604.15224); CAR *"exactly zero"* against VSR up to +30% (2509.26072v2); Anthropic
  — reward hacks exploited **>99%**, verbalised **<2%**. ⇒ **Do not write a criterion
  that audits this by reading the Q/A's `notes`.**
- **[MECHANISM CORRECTION]** `arXiv 2603.04582` resolves and **refutes** the
  self-attribution reading: *"explicitly stating that the action comes from the monitor
  does not by itself induce self-attribution bias."* A prior step credited
  self-preference / authority anchoring; **the mechanism here is STAKES FRAMING.**
- **[ADVERSARIAL, and it constrains the criteria]** PMC5557596 (N=30,674, 12 years) is
  the largest test of "remove the cue from the evaluator": anonymous marking moved the
  ethnicity gap **5.67 → 5.30 pp** and gender **3.92 → 3.27 pp**, and gaps *also*
  narrowed on **oral** exams which cannot be anonymised. ⇒ **expect a small,
  hard-to-attribute effect; make the criteria ARCHITECTURAL, not statistical.**
- **The remedy already ships in this repo, on the sibling rail.**
  `research-gate.js::enforceGate()` recomputes `gate_passed` and returns
  `agent_self_reported_gate_passed` + `self_report_disagreed`. The Q/A rail has
  **nothing between `agent()` and `return verdict`.** That gap is the seam.
- **Runtime constraint:** the Workflow runtime has **no filesystem access**, so a
  caller-side threshold **cannot read `verdict_ledger.jsonl` itself** — the sequence
  must arrive as data.

---

## 2. Hypothesis

The Q/A is told, **before grading**, what its verdict will trigger. Under 2604.15224
that is the exact manipulation measured to produce leniency, and it reaches the judge
from **two files at once** — `.claude/agents/qa.md` (read at runtime) and the rail's
own prompt in `.claude/workflows/qa-verdict.js`.

**Direction matters and cuts against the intuitive reading.** The bias is toward
LENIENCY, so it does **not** explain a FAIL. It puts **PASS** — and to a lesser degree
CONDITIONAL — in doubt.

**Fix thesis:** the judge should receive the **evidence** and return a **verdict**;
the **threshold** should be computed by the caller, after the verdict is in hand.
Score inside, threshold outside — the DSMB shape (*recommends*, sponsor decides), and
the shape `research-gate.js` already uses on the sibling rail.

**First-party corroboration obtained today, before this step began:** step 86.79's
cycle-1 spawn prompt supplied **no attempt number, no prior sequence, no consequence**
— and the Q/A still opened its notes with *"ATTEMPT NUMBER: 1"*, because `qa.md` tells
it to. **Suppressing the caller's prompt alone is not sufficient**, and that bounds
what this step can achieve without the operator-gated `qa.md` edit (§6).

---

## 3. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. the exposure is RE-DERIVED at execution time, not copied: quote the current qa.md text that requires the judge to state its own attempt number, and enumerate how many spawn prompts in the transcript record supplied a consequence, with the enumeration command stated
> 2. the leniency direction is stated explicitly and its consequence for ALREADY-COLLECTED verdicts is reasoned about rather than assumed -- in particular whether any PASS obtained under the framing needs re-grading, and if the answer is no, why not
> 3. the counter is moved OUTSIDE the judge, and this is demonstrated by driving a Q/A that receives NO attempt number and showing the escalation still fires correctly from the caller side
> 4. verdict semantics are UNCHANGED and demonstrated: nothing in this change can turn a FAIL into a PASS, and the 3rd-CONSECUTIVE rule still terminates a loop
> 5. the two law-of-the-case safeguards are either implemented or explicitly declined with a reason -- burden on the party seeking the change, and the override RECORDED (which needs a schema field, or an argument for why a field is not required here when its absence was grounds to delete the rubric)
> 6. mutation-test the relocated counter: make the caller-side computation wrong and show the check goes red, with the control observed GREEN first

**Immutable verification command** (baseline before any change):

```
bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
```

**Live check:** `handoff/current/live_check_86.78.md`.

---

## 4. The fix

### F1 — remove the CONSEQUENCE from the rail's prompt, keep the EVIDENCE

Strip from `qa-verdict.js`'s prompt the sentences that tell the judge what its verdict
triggers — *"return FAIL instead of a third"*, *"at 5+, recommend operator
escalation"*. The judge still gets the prior-verdict evidence it needs to reason; it
is not told the trigger. This is 2604.15224's `B0` condition.

### F2 — compute the threshold at the post-`agent()` seam, PURE

Add an exported `enforceEscalation(verdict, sequence, opts)` to `qa-verdict.js`,
invoked between `agent()` and `return`, mirroring `research-gate.js::enforceGate`:

- **Pure.** No filesystem (the runtime forbids it). The sequence arrives via `args`.
- Recomputes `consecutive_conditionals` and `would_auto_fail` with **reset on
  PASS/FAIL** — the same rule as `verdict_history_86_21.py::consecutive_conditionals`.
- Returns `escalation: {...}` **alongside** the verdict, never merged into it.
- **Fails CLOSED**: an absent/unusable sequence yields `null`, never `0`, and is
  reported as `sequence_status`.

### F3 — the caller RECORDS what it did (criterion 5, safeguard 2)

`VERDICT_SCHEMA` is `additionalProperties: false`, so **the schema cannot carry an
override** and one cannot be smuggled in. Two consequences, both deliberate:

- the **judge** cannot record an override — correct, it is not the party that should;
- the **caller** records it, in the object it returns and in the ledger row's existing
  free-text `note` key, following `research-gate.js`'s
  `agent_self_reported_*` / `self_report_disagreed` pattern.

### F4 — criterion 5, safeguard 1 (burden), stated

`escalation.burden_on` names the party that must justify departing from the computed
result. The **judge's verdict stands** unless the caller records a reason — the
law-of-the-case default (*Musacchio*: the prior decision *"should continue to govern"*;
the doctrine is discretionary, and the exceptions are narrow).

**Honest sourcing note carried from the brief:** the *"override must be RECORDED"*
safeguard is **NOT** attested in either legal source that was fetched (both are silent;
the two that would settle it returned 403/301). It **is** attested in the clinical
analogue — the DMC documents its recommendation and rationale to the sponsor — and
that is the citation this step leans on. The brief says so; the contract repeats it
rather than quietly upgrading a gap into a citation.

### Explicitly NOT done

- **No `.claude/agents/qa.md` edit** — §6. It is the other half of the exposure.
- **No re-grading of collected verdicts** — §5 argues why, from the direction.
- **No statistical criterion.** PMC5557596 says the measurable effect will be small
  and hard to attribute; a criterion demanding an observed verdict-mix shift would be
  unfalsifiable-in-practice and is deliberately not written.
- **No `notes`-based audit** of whether the judge was influenced. ERRJ = 0.000.

---

## 5. Criterion 2 — the argument about already-collected verdicts

Stated here so GENERATE reports it rather than inventing it:

- **PASS under the framing is the SUSPECT class.** The bias pushes *toward* leniency,
  so a PASS is the outcome the bias predicts.
- **FAIL and CONDITIONAL are CONSERVATIVE evidence.** The thumb was on the scale
  *against* them and they were reached anyway. Re-grading a FAIL hunts for an error in
  the direction the instrument cannot have made.
- **Any re-grade is scoped to the ambiguous band**, where the measured effect is
  concentrated.
- **A re-run that agrees with itself proves nothing** — 2606.19544's *reliability
  without validity* (test-retest ≥0.95 alongside position bias >0.10). The valid test
  is the **counterfactual**: re-grade with the consequence removed (`B0`).
- **Law of the case supplies the default**: the prior verdict stands absent a party
  carrying a burden. Blanket retrospective invalidation is supported by nothing read.

---

## 6. KNOWN BLOCKER — the `qa.md` half, and it is the LARGER half

The exposure reaches the judge from two files. This step can close the rail's half in
`qa-verdict.js`; the `qa.md` half (`:663-668` the trigger, `:684-688` the escalation
consequence, `:702-704` *"You MUST state the derived attempt number"*) requires editing
`.claude/agents/qa.md`, which already carries **four Main-authored edits awaiting
operator review**, plus the two sites step 86.79 has queued.

**And `qa.md` is the dominant channel, not the smaller one** — every spawn reads it at
runtime, whereas the rail prompt reaches only runs launched through that rail. Measured
today: a spawn whose prompt carried none of this still self-derived its attempt number.

**Therefore criterion 3 is expected to be PARTIAL**, and I am writing that into the
contract in advance rather than discovering it at EVALUATE. The caller-side machinery
will be demonstrably correct and the rail prompt will be clean; the judge will still be
instructed by `qa.md` to self-count until the operator acts. The exact patch will be
written out, unapplied, next to 86.79's.

---

## 7. Plan steps → criteria map

| # | action | criterion |
|---|---|---|
| P0 | Baseline the immutable command; record GREEN | 6 (control) |
| P1 | Re-derive the exposure: quote current `qa.md` text; enumerate spawn prompts, **separating three classes** (rule NAMED / consequence STATED / attempt number DEMANDED) with the command stated and controls | 1 |
| P2 | Write the leniency-direction argument with the asymmetric conclusion | 2 |
| P3 | Implement F1 (strip consequence) + F2 (`enforceEscalation`) | 3 |
| P4 | Drive a Q/A with **no** attempt number and show the escalation fires caller-side | 3 |
| P5 | Assert nothing can turn FAIL→PASS; assert the 3rd-CONSECUTIVE rule still terminates | 4 |
| P6 | Implement/decline the two safeguards, with the sourcing gap stated | 5 |
| P7 | Mutation matrix on the relocated counter; control GREEN first | 6 |
| P8 | Write `experiment_results_86.78.md` + `live_check_86.78.md`; spawn Q/A | — |

**Enumeration discipline for P1, learned the hard way twice today:** a first pass
scored "365 of 370 (98.6%) supplied a consequence" — **false**. All 365 came from one
probe matching `qa-verdict.js:92`, a **pointer that NAMES the rule** in a
table-of-contents sentence, not a statement of consequence. The three classes must be
counted separately, each with its population rule beside it, and the census must state
that it **structurally undercounts** because the dominant channel is `qa.md`, which no
prompt census can see.

---

## 8. Risks

| risk | mitigation |
|---|---|
| Caller-side computation becomes a way to override a verdict | `escalation` is returned **alongside** the verdict, never merged; asserted in P5 |
| Main supplies the sequence, and Main is the constrained party | **Disclosed, not solved** — the 86.21 objection (35/35 rows `recorded_by: main`). The script echoes back what it was given so the input is auditable |
| Removing the sentence is claimed to have fixed the bias | Criteria are **architectural**; PMC5557596 is cited *against* over-claiming |
| A mutation cell reddens for the wrong reason | The matrix requires a **named** assertion among the failures (proven necessary 3× today) |

## 9. References

- `handoff/current/research_brief_86.78.md`
- arXiv **2604.15224** — https://arxiv.org/html/2604.15224
- arXiv **2603.04582** — https://arxiv.org/html/2603.04582 (mechanism correction)
- arXiv **2509.26072v2** — https://arxiv.org/html/2509.26072v2 (CAR = 0)
- Anthropic — https://www.anthropic.com/research/reasoning-models-dont-say-think
- **PMC5557596** — https://pmc.ncbi.nlm.nih.gov/articles/PMC5557596/ (adversarial)
- NIH NCATS DSMB — recommends vs decides
- Cornell LII / 11th Cir. primer — law of the case
- `.claude/workflows/research-gate.js` — the in-repo precedent for the seam
