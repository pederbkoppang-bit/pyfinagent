# experiment_results — step 86.78

**Phase:** GENERATE (after RESEARCH → PLAN). Contract at
`handoff/current/contract_86.78.md`, written **before** any change.
**Evidence:** `handoff/current/live_check_86.78.md` — verbatim, re-runnable.
**Totals (cycle 2): 43 checks (floor 41), 13 mutation cells, 13 killed.**

> **CYCLE 2.** The cycle-1 Q/A returned **CONDITIONAL**, 5 of 6 criteria met, and its
> own 7-cell battery left **3 survivors**. All three are now closed and pinned as cells
> **M11/M12/M13** — see `live_check_86.78.md` "CYCLE 2" and
> `evaluator_critique_86.78.md` for the verbatim verdict. Criterion 3 is **unchanged at
> PARTIAL**: it is operator-gated, and no amount of further work here moves it.

---

## 1. What was built

| file | change |
|---|---|
| `.claude/workflows/qa-verdict.js` | **modified** — consequence stripped from the prompt; `enforceEscalation()` added at the post-`agent()` seam |
| `scripts/qa/verify_escalation_86_78.mjs` | **new** — **43**-check checker driving the REAL function |
| `scripts/qa/mutation_matrix_86_78.mjs` | **new** — **13**-cell mutation matrix |

**Not touched:** `.claude/agents/qa.md` (**zero-line diff**), `CLAUDE.md`,
`.claude/rules/research-gate.md`, `scripts/qa/qa_wip.py`.

### F1 — the consequence is out of the prompt; the evidence stays

Removed from the rail's prompt: *"return FAIL instead of a third"*, *"at 5+ …
recommend operator escalation"*, *"State the derived attempt number"*. Replaced with an
explicit statement that the consequence is withheld **on purpose**, and why. The judge
still gets the pointer to `verdict_history_86_21.py` — it needs the **evidence**, it
does not need the **trigger**. This is 2604.15224's `B0` condition.

### F2 — `enforceEscalation()` at the post-`agent()` seam

Pure (the Workflow runtime has no filesystem, so the sequence arrives as
`args.verdict_sequence`). Recomputes `consecutive_conditionals` with **reset on
PASS/FAIL**, treats `NO_VERDICT` as a dropped **attempt** that neither extends nor
resets the run, and returns `escalation` **alongside** the verdict — never merged into
it. Fails closed: an absent or unusable sequence yields `null`, **never `0`**.

Not `export`ed — the shipped workflow exports only `meta`, as `research-gate.js` does.
The checker drives the **real** function by extracting its exact source span and
importing it, so it can never drift from a hand-copy.

### F3/F4 — the two law-of-the-case safeguards

- **Burden** (`escalation.burden_on`): named, and on *"the party departing from the
  computed escalation"*. The judge's verdict stands by default.
- **Recording** (`override`, `override_reason`, defaulting to `null`): `VERDICT_SCHEMA`
  is `additionalProperties: false`, so **the judge structurally cannot record an
  override** — which is correct, it is not the party that should. The **caller** records
  it, following `research-gate.js`'s `self_report_disagreed` pattern.
- **Sourcing gap, stated not upgraded:** the *"override must be RECORDED"* safeguard is
  **not** attested in either legal source fetched (both silent; the two that would
  settle it returned 403/301). It rests on the clinical analogue only.

---

## 2. Criterion-by-criterion

| # | criterion (abridged) | status | evidence |
|---|---|---|---|
| 1 | exposure RE-DERIVED; enumerate consequence-supplying prompts, command stated | **met, with the correction recorded** | live_check §1 — **three classes**: A 365/370 named, B **25/370 (6.8%)** stated, C 16/370 demanded. §1c records that a first pass reported a **false 98.6%** by pooling a table-of-contents pointer with a stakes sentence |
| 2 | leniency direction stated; already-collected verdicts reasoned about | **met** | §3 below |
| 3 | counter moved OUTSIDE the judge; a Q/A with NO attempt number still escalates caller-side | **PARTIAL — §4** | live_check §2 — driven; but `qa.md` still instructs self-counting |
| 4 | verdict semantics UNCHANGED; 3rd-CONSECUTIVE still terminates | **met** | live_check §2 C4 — no input verdict is mutated under any sequence; `would_auto_fail` can arm only on a CONDITIONAL |
| 5 | the two safeguards implemented or declined with a reason | **met** | live_check §2 C5; sourcing gap stated |
| 6 | mutation-test the relocated counter, control GREEN first | **met** | live_check §3 + "CYCLE 2" — **13/13** killed on named assertions |

---

## 3. Criterion 2 — the argument, in full

**The measured direction is LENIENCY**: arXiv 2604.15224, content held strictly
constant, one consequence sentence varied, 18,240 judgments — **58 of 72 cells ΔV ≤ 0,
p < 0.001, peak −9.8 pp**. Reward framing was **as lenient as** punishment framing, so
"we only mention escalation" is not a defence.

**This makes re-grading ASYMMETRIC, and the asymmetry runs opposite to intuition:**

- **PASS collected under the framing is the SUSPECT class** — the thumb was on the
  scale *toward* leniency, so a PASS is the outcome the bias predicts. This includes
  **86.68's closing PASS**.
- **FAIL and CONDITIONAL are CONSERVATIVE evidence** — the bias pushed *against* them
  and they were reached anyway. Re-grading a FAIL hunts for an error in the direction
  the instrument cannot have made.
- **Scope any re-grade to the AMBIGUOUS band**, where the effect concentrates
  (−7.1 to −9.8 pp) versus clear-cut cases (−1.3 to −0.7 pp). Here the ambiguous band
  **is the modal verdict**: **18 of 35** ledger rows are CONDITIONAL.
- **A re-run that agrees with itself proves nothing** — 2606.19544 measures *reliability
  without validity* (test–retest ≥0.95 alongside position bias >0.10). The valid test is
  the **counterfactual**: re-grade with the consequence removed, i.e. 2604.15224's `B0`.
- **Law of the case supplies the default**: the prior verdict **stands** unless the
  party seeking the change carries a burden. **Blanket retrospective invalidation is
  supported by nothing read.**

**Conclusion: no automatic re-grade is triggered by this step.** What *is* warranted is
a targeted, stakes-free re-grade of **ambiguous PASSes**, 86.68 among them — filed as a
follow-up rather than performed here, because performing it would mean re-opening a
closed step mid-cycle.

**Mechanism correction, on the record:** a prior step credited **self-preference /
authority anchoring**. That is wrong. arXiv 2603.04582: *"explicitly stating that the
action comes from the monitor does not by itself induce self-attribution bias"*, and
2410.21819 shows self-preference fires **without** labelling (perplexity-driven). The
mechanism here is **STAKES FRAMING**.

---

## 4. Criterion 3 is PARTIAL — and the untouched half is the LARGER one

The exposure reaches the judge from **two** files. This step closes the rail's half:

- the prompt no longer states any consequence (4 probes, all 0 — live_check §2 C1);
- the threshold is computed caller-side, after the verdict, and cannot alter it.

**The `qa.md` half is untouched and operator-gated** — `:663-668`, `:684-688`,
`:702-704`. `qa.md` already carries four Main-authored edits awaiting operator review,
plus the two sites step 86.79 queued. The session instruction is *"if a fix genuinely
needs `qa.md`, stop and ask."*

**And `qa.md` is the dominant channel, not the smaller one.** Every spawn reads it at
runtime; the rail prompt reaches only runs launched through that rail. **Measured
today, first-party:** step 86.79's cycle-1 spawn prompt carried none of this, and the
Q/A still opened with *"ATTEMPT NUMBER: 1"*. The census in §1 therefore reports
**Class B at 25 of 370 (6.8%)** while the true exposure is **370 of 370** — and the
live_check says so rather than letting the smaller number stand.

**So criterion 3 is demonstrably met on the caller side and NOT met end-to-end.** I am
not asking for a waiver. The `qa.md` patch will be written out unapplied alongside
86.79's.

---

## 5. Deltas from the contract — disclosed

1. **The checker extracts the function's source span rather than slicing a prefix.**
   The contract assumed the `research-gate.js` pattern would transfer; it does not —
   `qa-verdict.js` has top-level `return`s both before and after the function (legal
   under the Workflow runtime, a SyntaxError under ESM). Extraction is by brace
   matching, and the first attempt was defeated by the `opts = {}` **default
   parameter**, which the naive scan mistook for the body.
2. **Two defects in my own checker, both caught by the matrix** (live_check §4): `SRC`
   read the tracked file instead of the mutant — the same defect fixed in 86.79 hours
   earlier — and a `[FAIL]`-regex that truncated labels containing ` -- `. Both would
   have produced a false green.
3. **13 cells, not the contract's implied minimum** (10 authored, then 3 added from the
   cycle-1 Q/A's own survivors). Every cell targets a distinct
   failure the relocation could have introduced.

---

## 6. Limits

- **The `qa.md` half is not fixed** and is the dominant channel (§4).
- **No claim that verdicts changed.** PMC5557596 predicts a small, hard-to-attribute
  effect from cue removal; criteria are architectural by design.
- **Single-lab evidence.** 2604.15224 is under review with no replication of the stakes
  result.
- **The caller supplies the sequence and the caller is the constrained party** — the
  86.21 objection, disclosed and only partly mitigated by echoing the input back.
- **Nothing downstream consumes `escalation` yet.** Wiring a consumer is out of scope
  (and overlaps 86.71).
- **The recording safeguard is not legally sourced** — clinical analogue only.

---

## 7. Cycle-2 remediation, and one thing the Q/A framed better than I did

**"The counter is DUPLICATED, not MOVED."** That is the cycle-1 Q/A's phrasing for
criterion 3 and it is sharper than mine. The rail no longer supplies the consequence,
but `qa.md` — which the spawn prompt's own **binding STEP 0** orders the judge to read
in full — still does. So the composite instruction set still delivers it. The Q/A
confirmed this **first-party**: it was itself instructed to self-count in that very run.

It also confirmed the two halves I could verify only indirectly:

- **The live test holds.** `THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED
  HERE` occurs in **zero of ~100 persisted `qa-verdict-wf_*.js` snapshots** and in the
  on-disk file — independent proof the graded run used the disk script, and independent
  corroboration of the session-snapshot finding in live_check §7.
- **The census reproduces exactly** under an independently written population rule:
  365 / 25 / 16 after subtracting the two transcripts created since I measured. It adds
  two reproduction caveats I had not stated: matching **must be case-insensitive** (19
  otherwise), and the literal probes are `return FAIL instead of a` / `recommend
  operator`, **not** the fuller labels printed in the §1b table — the old prompt
  line-wrapped mid-phrase. **So the number is right and my printed recipe was not
  exact.** Corrected here rather than left standing.
- **Both self-reported checker defects are load-bearing**, proven by its own mutants
  rather than taken on trust.

### Three notes it raised that I am recording rather than quietly fixing

1. **The census self-contaminates on re-run.** Re-derived today it counts *this
   evaluation's own prompt* as Class B, because my `extra` quotes the two removed
   strings as probe targets — the same mention-vs-use error I caught in my own first
   pass. The census needs an as-of timestamp. **Queued, not patched**, because editing
   the census mid-evaluation is tree movement during EVALUATE.
2. **`live_check_86.78.md` was modified 22 seconds after that spawn began** (adding §7,
   the snapshot finding). The Q/A flagged it as tree movement during EVALUATE and
   verified the subject sha was unchanged. It was correct to flag it, and I should have
   held the append.
3. **The C3 source-order check is a source scan, not a behavioural observation.** It
   happens to be backed by a JS temporal-dead-zone guarantee, so it is not vacuous in
   effect — recorded so nobody mistakes it for a behavioural guard.

### Still NOT met, and not fixable here

Criterion 3 end-to-end. Its stated route to PASS is *"the operator-gated `qa.md` edit
removing `:665`/`:685`/`:702`, or an explicit operator decision to accept the
duplication with the patch written out unapplied."* Both routes are the operator's.
