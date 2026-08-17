# experiment_results — step 86.78

**Phase:** GENERATE (after RESEARCH → PLAN). Contract at
`handoff/current/contract_86.78.md`, written **before** any change.
**Evidence:** `handoff/current/live_check_86.78.md` — verbatim, re-runnable.
**Totals (cycle 2): 43 checks (floor 41), 13 mutation cells, 13 killed.**

> **CYCLE 3.** Criterion 3 is now CLOSED end-to-end — see **§8**.
> **Cycle 2:** the cycle-1 Q/A returned **CONDITIONAL**, 5 of 6 criteria met, and its
> own 7-cell battery left **3 survivors**. All three are now closed and pinned as cells
> **M11/M12/M13** — see `live_check_86.78.md` "CYCLE 2" and
> `evaluator_critique_86.78.md` for the verbatim verdict. Criterion 3 was PARTIAL at
> that point and operator-gated; the operator has since answered and **§8 closes it**.

---

## 1. What was built

| file | change |
|---|---|
| `.claude/workflows/qa-verdict.js` | **modified** — consequence stripped from the prompt; `enforceEscalation()` added at the post-`agent()` seam |
| `scripts/qa/verify_escalation_86_78.mjs` | **new** — **43**-check checker driving the REAL function |
| `scripts/qa/mutation_matrix_86_78.mjs` | **new** — **13**-cell mutation matrix |

**`.claude/agents/qa.md`:** untouched by Main (**zero-line diff** through cycles 1-2),
then edited at cycle 3 by a **fresh executor** on the operator's instruction — §8.
**Also changed at cycle 3:** `scripts/qa/verdict_history_86_21.py` (the `--evidence-only`
mode) and a renamed ADR. **Not touched at all:** `CLAUDE.md`,
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
| 3 | counter moved OUTSIDE the judge; a Q/A with NO attempt number still escalates caller-side | **met (cycle 3) — §8** | rail prompt, `qa.md`, AND the tool's stdout all closed; all five consequence probes return 0 |
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

## 4. Criterion 3 through cycles 1-2 (SUPERSEDED by §8)

> **This section is the record of what was true before the operator answered. Criterion
> 3 is now MET end-to-end; §8 is current.** Kept because its argument — that `qa.md` is
> the LARGER half — is exactly what the cycle-3 work had to close.

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

---

## 8. CYCLE 3 — criterion 3 is now CLOSED end-to-end, and the executor found more than I asked

**The operator chose route B: a fresh executor applies the `qa.md` edits.** All three
sites this step owed are corrected — the 3-consecutive trigger, the F1b escalation
consequence, and the self-count demand. **All five consequence probes against `qa.md`
now return 0.**

But the executor did not stop at the brief, and three of its findings were things I had
missed entirely:

1. **The consequence reached the judge through a TOOL, not the prose.** `qa.md` tells
   the judge to run `verdict_history_86_21.py`, whose stdout announced
   `auto-FAIL armed : True (a further CONDITIONAL would be the 3rd)`. **A prose scrub
   cannot close a channel that runs through a tool.** Fixed: `--evidence-only`.
2. **After that, the mode still printed `consecutive : 2`.** An **aggregate** hands the
   judge the boundary's unit *and its own position within it* — "you are near a boundary
   of shape X", the same class of information. It found this by **re-running the mode
   rather than trusting its own earlier reading of it**. Every aggregate is now withheld;
   the sequence, which is the evidence, is not.
3. **The ADR filename named both units**, and `qa.md` cited the path. A path is text the
   judge reads. Renamed to `docs/adr/0003-verdict-bound-provenance.md`.

It also caught **its own** first draft quoting the suppressed line inside the warning
that forbids it — *a warning that reproduces its payload delivers the thing to every
reader* — and rewrote it to describe the line's shape without stating the rule.

**And it pushed back on my own directive, correctly.** My change-C brief told it to
"state plainly that the 3-consecutive-CONDITIONAL threshold is computed by the CALLER",
which names the threshold's **shape** while withholding only its outcome. Under
2604.15224 the operative variable is the presence of consequence information at grading
time, so that was still a leak — *my* wording, not its execution. Now the judge is told
only that a threshold exists, is the caller's to apply, and is deliberately undescribed
— with the explicit rider **not** to read that silence as evidence that nothing follows.

**The historical CORRECTION block moved VERBATIM** to the ADR rather than being
rewritten: 13 lines, 980 bytes, sha256 identical, `diff` empty — and the losslessness
was proven by **re-extracting from the finished artifact**, not by comparing against the
buffer that wrote it.

### The residual count is 5, NOT 0 — and it is reported that way on purpose

Asked for the count rather than a silent zero, the executor returned **five** and ranked
them. Its top two (the tool aggregate, the ADR filename) are now **closed**. Three
weak ones remain and are recorded rather than zeroed: `consecutive_conditionals` as a
caller-side **field name**, a mention that an aggregate **exists**, and the
`"do NOT let the zero suppress the escalation"` line inside a paragraph deliberately
left alone. None names a threshold value, unit or outcome.

**The new checks are DRIVEN** — they execute the tool and assert on its real stdout,
because a source scan cannot prove a tool's output. **51 checks (floor 49), 13/13
cells.**


---

## Cycle 3 GENERATE (2026-08-17): the recorder computed, the census two-level, the leaks owned

Cycle 2's rail drop established (write-first) that criterion 3's
architecture was complete but NOT IN FORCE -- its own evaluator ran on the
pre-scrub qa.md snapshot. THIS session is post-restart: every evaluator
today demonstrably runs the post-fix architecture (each verdict's notes
carry the "gathered, not applied; the caller computes anything that
follows" discipline, and each escalation envelope computes
n/would_auto_fail from args.verdict_sequence alone). The cycle-3 work:

1. **Criterion 1 -- the exposure census, two-level, commands stated**
   (live_check §10): 413/413 qa spawn prompts contain the RULE text
   (qa.md embedded -- design, not exposure; the first census version that
   counted it is disclosed as an overcount); slicing to CALLER-controlled
   sections, **8/413 carry POSITIONAL claims** -- four pre-fix
   (2026-08-11, "Attempt N of 5" era) and four on 2026-08-17, TWO OF THEM
   MAIN'S OWN (wf_4e01adc8 "FIFTH counted attempt...", wf_6821f477). The
   party that relocated the counter still leaked position in its
   extra.note. Current qa.md text quoted: the post-scrub architecture
   note at :808 ("alongside -- never inside").
2. **Criterion 3 -- driven on live spawns**: the 86.84 cycle-10 and
   cycle-11 evaluations received NO attempt number and no positional
   framing; their envelopes computed consecutive_conditionals from the
   supplied sequence alone (cycle-11's notes explicitly gather-not-apply).
3. **Criterion 4 -- the rule TERMINATED a loop today**: 86.85 cycle 11 was
   bound PASS-or-FAIL by three consecutive CONDITIONALs and returned FAIL;
   the F1 arc completed end-to-end on a live step.
4. **Criterion 5 -- the recorder is now a MEASUREMENT** (the fix):
   `judge_was_told_consequence` was hardcoded `false` at qa-verdict.js
   with a comment delegating to caller self-declaration -- and no caller
   ever declared it (all 8 leaky spawns recorded nothing). It is now
   COMPUTED at the seam from the same caller-controlled strings the
   prompt renders (`caller_text` = evidence + JSON(extra)), via
   `POSITIONAL_CLAIM_RE` -- the census's own pattern -- with the matched
   substring recorded as `judge_was_told_consequence_evidence`. qa.md's
   rule text never passes through the function, preserving the
   rule-vs-position distinction (pinned by a dedicated check).
5. **Criterion 2 -- leniency direction + already-collected verdicts**:
   direction per arXiv 2604.15224 (leniency, invisible in CoT). Of the 8
   framed spawns: 2 PASS, 1 FAIL, 4 CONDITIONAL, 1 no-result. The FAIL and
   CONDITIONALs argue against a captured judge (framing pushed toward
   leniency and they graded hard anyway); both PASSes (wf_20a27baa
   2026-08-11; wf_4e01adc8 = 86.71's close, framed by Main the same day
   this step's census ran) are QUEUED FOR A RE-GRADE DECISION in step
   86.93 rather than quietly kept -- with the mitigation stated: the 86.71
   PASS's substance was independently executed by its evaluator
   (12-process drive, 4,368-cell sweep, its own mutants).
6. **Criterion 6 -- mutation-tested with control green first**:
   checker section [8]: 8-recorder-neutered (regex -> never-match) and
   8-recorder-hardcoded-false (revert to the hardcode) both KILLED; the
   positive/negative/rule-text/absent drives are the control set. Family
   green at write time (126/124/96; the prompt-render count moves as the
   86.72 cycle-2 work adds cells to the same section -- 136 at the last
   pre-eval run; the two recorder cells and the four control drives are
   unchanged).

*(cycle-4 REPLACEMENT of a falsified forward claim -- the original said
"Caller discipline forward: sequence-as-data only; no positional prose in
evidence/extra", and the cycle-3 evaluator measured Main's VERY NEXT spawn
-- the cycle-3 evaluation itself -- tripping the shipped detector on the
phrase naming the termination rule's two outcomes, used in HISTORICAL
narration. The honest discipline is stronger and is now stated as such:
spawn prompts avoid the detector's literal phrases ENTIRELY, narration
included, because the recorder cannot distinguish narration from framing
-- and per the underlying study the bias channel does not require intent.
The recorder is a fixed-literal detector: the evaluator's 7-case probe
missed 6 of 6 rewordings, so every census figure and the boolean are LOWER
BOUNDS, stated wherever they appear.)*


### Cycle-3 addendum: the recorder's first LIVE measurement

The 86.72 cycle-1 evaluation (wf_9bc5bc53-351, launched after this fix
shipped) returned `escalation.judge_was_told_consequence: false` computed
from its caller text -- the first live spawn measured by the recorder, on a
prompt kept clean under the new discipline. The census, the computed
recorder, and a live measurement now exist end to end.


---

## Cycle 4 GENERATE (2026-08-17): the blocker repaired at the named line; everything else queued as graded

1. **B1/B2 (the blocker the evaluator refused to queue)**: this step's two
   shipped instruments were broken by a SIBLING step's legitimate edit --
   86.72 added `research_routing` to the merge line, and
   `verify_escalation_86_78.mjs:159` asserted the WHOLE-LINE literal.
   Repaired exactly as the verdict named: the assertion is now the
   PROPERTY (the merge line carries `escalation` as a bare key -- regex
   `const merged = \{ \.\.\.verdict, [^}]*\bescalation\b` -- keeping the
   anti-spread conjunct), and matrix cell M11's anchor tracks the current
   line with the flatten mutant still dying. Post-repair, exits unpiped:
   verify 51 checks ALL CHECKS PASS exit 0; matrix 13/13 ALL CELLS KILLED
   exit 0; the :6-7 recipe cardinalities refreshed (37->51, 10->13) with
   the history stated. The un-disclosure is owned: the artifact claiming
   family-green was written 23 minutes after the breaking commit and
   quoted the three FAMILY checkers while this step's own two were red --
   the scope-must-be-derived lesson, recorded here.
2. **The falsified forward claim is REPLACED in place** (above): my own
   cycle-3 spawn tripped the detector in historical narration;
   the discipline now bans the literal phrases entirely and states the
   recorder's lower-bound nature.
3. **The stale ledger is BACKFILLED** (the evaluator measured attempt 4 vs
   0 rows): cycles 1/2 CONDITIONAL + the cycle-2 rail drop as NO_VERDICT,
   each labelled BACKFILL/reconstruction with sources named, keyed by
   cycle per the 86.85 precedent; today's cycle-3 row recorded at the
   seam. The sequence source for this step is no longer stale.
4. **Queued as graded (evidence-quality, per the verdict's own triage)**:
   F4 -- the override field has no writer (structurally unsettable; 0/76
   rows carry it); F5 -- the recorder's false-negative surface; both ride
   the 86.107 residual queue via the transcribed verdict.


---

## Cycle 5 GENERATE (2026-08-17): the rail's own leak closed; the comment trap killed after killing me once

The cycle-4 verdict (returned CONDITIONAL, recorded FAIL by the mechanical
rail rule) found the leak in the half this step owns. Each finding closed:

1. **The 420/420 STEP-0 leak**: qa-verdict.js:345's enumeration named the
   rule's value, unit and outcome in EVERY spawn prompt, 60 lines above the
   deliberately-withheld block -- and my census had attributed the
   100%-prevalence hit to qa.md-embedding, which the evaluator falsified
   with three qa.md-body markers scoring 0/421. Fixed both ways: the
   enumeration now reads "the loop-termination rule" (value/unit/outcome
   dropped), and the census's causal clause is CORRECTED at the site in
   live_check section 10. A fifth GONE probe pins the leak's exact phrase.
2. **The comment-token vacuity (QX2/QX6)**: the nesting assertion now
   locates the merge statement among EXECUTABLE LINES ONLY and
   comment-strips it before requiring the escalation token. DISCLOSED:
   my FIRST version of this fix survived QX2 -- the naive statement regex
   matched 'const merged' INSIDE the '// was:' comment -- caught by driving
   both mutants before shipping (the discipline the cycle-1 evaluator of
   86.72 taught this session). Post-fix drive: QX1 flatten, QX2
   comment-was-line, QX6 inline-comment ALL exit non-zero (KILLED);
   an anti-vacuity check pins that the found statement is real.
3. **The 86.93 landing place**: extended in the masterplan audit_basis with
   both framed run ids and the mitigation -- the previously-false queue
   claim is now true, with its falsification recorded in the critique.
4. **The four stale live_check blocks** carry SUPERSEDED marks; the
   nonexistent scratchpad citation is replaced by the inline derivation;
   section 11 is regenerated with matching commands and outputs (verified
   reproducing before commit).

Captured at write time, exits unpiped: verify_escalation 52 checks ALL
CHECKS PASS exit 0; mutation_matrix 13/13 ALL CELLS KILLED exit 0;
prompt-render 136 and research-gate-workflow 124 both green (the STEP-0
wording change altered no check counts).


---

## Cycle 6 GENERATE (2026-08-17): the block-comment decoy, the reworded restoration, and the sibling sweep

1. **MN**: /* */ SPANS are stripped from the whole source BEFORE
   line-splitting (the decoy's unprefixed interior line vanishes with its
   span), then the per-line filter handles // and stragglers. The
   evaluator's exact decoy construction is KILLED; MO/MD/ME-equivalents
   (QX1/QX2) re-verified dying.
2. **MB**: the STEP-0 enumeration line is content-pinned -- it must exist
   with the neutral name and be free of the rule's value/unit/outcome
   tokens in any spelling the pin can see (/3rd|third|CONDITIONAL|
   auto-?FAIL|straight|consecutive/i). The evaluator's reworded
   restoration is KILLED. Stated bound: no string pin enumerates all
   phrasings -- the same inherent limit the cycle-1 Q/A ruled WARN; this
   pin kills the measured families.
3. **F4/F5 have a REAL queue entry**: 86.107's audit_basis now carries both
   with their identifying phrases, mechanisms, and named fixes -- the
   86.93-class sweep completed for the sibling instance the cycle-5 Q/A
   caught.
4. The :187 anti-vacuity check's non-independence is acknowledged as the
   evaluator noted it (MJ reddens it only jointly); its property is
   documentation of the found statement, and it rides as-is.

Captured at write time, exits unpiped: verify_escalation ALL CHECKS PASS
exit 0 (55 checks after the MB pin); matrix 13/13 ALL CELLS KILLED exit 0;
MN/MB/QX1/QX2 all KILLED in the pre-ship drive.
