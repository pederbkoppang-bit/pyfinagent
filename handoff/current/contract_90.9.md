# Contract -- step 90.9

**Step:** 90.9 -- "the criteria are the loop's fuel -- classify criterion SHAPE by
execution at filing time"
**Phase:** phase-90. **Contract written:** 2026-08-20.

---

## 1. Research gate -- PASSED (enforced), on the SECOND attempt

First attempt `wf_722b01b9-67d` returned enforced **`gate_passed: false`**:
`self_report_disagreed: true`, violation *"over-claim: urls_collected=42 but only 9
distinct URLs appear in the brief"*. The agent self-reported `true`; **the enforced value
governed and the gate held.** Recorded because it is the gate working, not a nuisance.

Re-run `wf_a963cdc4-7d4` PASSED: 8 sources read in full, 53 URLs collected, 57 distinct in
the brief, `self_report_disagreed: false`, `violations: []`, 12/12 checks.
Brief: `handoff/current/research_brief_90.9.md` (37,850 chars).

### The finding that reshapes criterion 1

**The filing's figures reproduce to the digit -- but only at a PINNED corpus.** At the
filing commit `252090a3` the rule "walk nodes carrying an `id` + a dict `verification` +
non-empty `success_criteria`, then EXCLUDE step 90.9 itself" reproduces **every** figure:
155 steps / 980 criteria / 403 apparatus / **41.1%** / 78 terminal / 1026 of 4670 =
**22.0%** project-wide, and the filed 1.6x-1.9x range collapses to **1.87x**.

At HEAD it gives 174 / 1045 / 414 / 39.6% / 1.81x, because commit `085c74e8` added 19
steps **49 minutes after filing**.

So the missing variable was the **corpus timestamp, not the regex** -- and my own
pre-research first pass (35.8%, 156 steps, 80 terminal, 40 unbounded) was wrong for
exactly that reason plus a looser keyword list. This matters directly:

> criterion 1: "*reproduces the filing figures by execution on the live tree; where a
> figure does not reproduce, the RULE is corrected and the new figure printed*"

**"On the live tree" is already unsatisfiable**, and taking the escape hatch would
*corrupt a correct rule to chase a moved corpus* -- the precise inverse of the criterion's
intent. The classifier therefore takes a **git-rev corpus pin**, house precedent
`scripts/qa/replay_changelog_rule_86_68.py:34` and
`scripts/qa/sweep_absent_verification_paths.py:421`, and prints both: the pinned figures
(which reproduce) **and** the live-tree figures (which do not), with the delta attributed
to `085c74e8`. The numbers are never edited to match.

### The one figure that genuinely does not reproduce

**The unbounded count 44 reproduces only as a keyword PROXY.** The *literal*
self-reference test returns **0 of 155** -- the self-reference is implied, never written.
So 44 is the right number reached from the wrong property, and the detector must test the
quantified **noun class** against the artifact class the step grows. Criterion 1's
"correct the RULE" clause applies here and only here.

### Three citation corrections carried out of the research, adopted

1. **`arXiv 2501.04810` must not be cited for detector accuracy.** Its 0.98/0.83 figures
   measure *LLM traceability performance on requirements containing* each smell class; the
   paper does not address keyword-detector limitations. The number is real; the claim it
   was previously used for is not what was measured.
2. **`qa-verdict.js:336` does NOT pass `verdict_sequence`/`attempt_number` to the judge.**
   Line 336 is the unknown-key detector; the prompt at :340-437 never renders them. Prior
   researcher memory said otherwise. The live consequence channel is a **SELF-read** at
   :430-435 (`qa_wip.py --spawned-at`) -- so criterion 7's binding verb is **"never
   reads"**, not "never given".
3. **`arXiv 2509.06770`'s "3-4 iterations" is a paraphrase, not a verbatim quote.** The
   filing quotes it as if verbatim; the full text confirms the claim but not the wording.

---

## 2. Hypothesis

The loop is fuelled by what the criteria ask for. If ~41% of phase-86..90 criteria grade
the **verification apparatus** rather than product behaviour, and ~44 steps demand an
unbounded "every new guard", then each remediation adds guards the same criterion then
demands be mutation-tested -- a fixed point, which `arXiv 2607.24300` supplies the
impossibility bound for (alpha + beta >= 1 - TV for any endogenous gate; 35/35 self-scores
above 0.70, 15 below random; *"tests evolve to accommodate degraded behavior"*).

Classifying criterion SHAPE **at filing time** attacks the fuel rather than the fire. It
grades nothing and changes no verdict; it exits non-zero on ONE thing only -- unbounded
scope.

---

## 3. Immutable success criteria (VERBATIM)

1. a re-runnable classifier labels every criterion of a named step PRODUCT_BEHAVIOUR or EVIDENCE_APPARATUS by a rule PRINTED beside its output, and reproduces the filing figures by execution on the live tree; where a figure does not reproduce, the RULE is corrected and the new figure printed with its step-inclusion rule stated -- the number is never edited to match, and the 1.6x-1.9x ratio range is collapsed to a single value only by fixing the inclusion rule

2. red-first and discriminating, control observed GREEN first: a hand-built control step whose criteria are all product-behaviour scores 0% evidence-class and exits 0; a hand-built step whose terminal criterion reads 'mutation-test every new guard this step adds' is flagged unbounded-scope and exits non-zero; a mutant classifying every criterion as evidence-class must be KILLED by the all-product control, and a mutant that fails to run scores ERROR, never a kill

3. it runs at FILING time on a proposed NEW step object only, and exits non-zero SOLELY on unbounded scope -- proven by executing it against all 155 phase-86..90 steps and capturing the exit code of every one, showing it exits 0 on each step carrying no unbounded criterion

4. it cannot mutate the plan: sha256 of .claude/masterplan.json is byte-identical before and after a full classification run over every step in the file, captured both times, AND the source contains no write path to that file (no open(...,'w'), no json.dump targeting it) -- both checks required, neither sufficient alone

5. grading is untouched: no existing criterion is edited, weakened or removed (criterion 4's sha256 proves it), and sha256 of handoff/verdict_ledger.jsonl is byte-identical before and after the run

6. any bound proposed for the unbounded-guard shape is JUSTIFIED against the record rather than asserted: print how many of the 44 unbounded criteria the bound would have flagged, name the most serious real historical finding it would have DEFERRED, and file that finding's disposition as its own queued step rather than dropping it

7. the classifier is never given, and never reads, a step's verdict history, round index or remaining attempt budget -- asserted by a test over its inputs; phase-86.78 closed that consequence channel on the measured basis of arXiv 2604.15224 (18,240 judgments, LENIENT in 58 of 72 cells, p<0.001, peak -9.8pp, ERRJ 0.000 so the leniency is invisible in chain-of-thought and unauditable from notes)

**Immutable command:**

```
bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'
```

---

## 4. Plan

### 4.1 `scripts/qa/criteria_shape_90_9.py` (criteria 1, 3)

- `--masterplan <path>` and `--rev <git-rev>` (house precedent above). Default prints
  **both** the pinned-`252090a3` census and the live-tree census, with the delta
  attributed to `085c74e8`.
- The classification rule is **printed beside every output**, per criterion 1. Since no
  external standard exists for a product-vs-apparatus axis (the research is explicit that
  citing an RE taxonomy as authority would be false), **the printed rule is the entire
  warrant** and is written to be read and disputed.
- Step-inclusion rule stated on every ratio: node carries an `id`, a dict `verification`,
  and non-empty `success_criteria`; step 90.9 itself excluded (its own 7 criteria are the
  whole delta that made the figures look irreproducible).
- Exit code: **non-zero SOLELY on unbounded scope.** Classification alone never fails.

### 4.2 The unbounded-scope detector (criteria 1, 6)

Tests the quantified **noun class** against the artifact class the step grows, not a
keyword. The keyword proxy reaches 44 by luck and the literal self-reference test reaches
0, so neither is the property. Criterion 6 requires the bound be justified against the
record: print how many of the 44 it flags, **name the most serious real historical finding
it would have DEFERRED**, and file that finding's disposition as its own queued step.

### 4.3 Criterion 7 -- the input surface, and a correction

Asserted over the classifier's **inputs**: it is never given and never reads verdict
history, round index, or remaining budget. The research correction matters here --
`qa-verdict.js` does *not* hand these to the judge; the live residual channel on the
sibling rail is a **self-read** (`qa_wip.py --spawned-at` at :430-435). So the binding
verb is **"never reads"**, and the test must cover a self-read path, not only the
caller's hand-off. The divergence from the sibling rail is stated, not hidden.

### 4.4 Criterion 4 -- write-capability at AST level, not by two literal patterns

The criterion names `open(...,'w')` and `json.dump`. **The house's own idiom is
`Path.write_text` (148 sites, and both masterplan writers use it)**, which those two
patterns miss entirely. Resolving write-capable calls at AST level satisfies the criterion
strictly *more* than its literal wording; the gap is stated so the strengthening is
visible rather than silent. Both required checks are kept: the sha256 pair AND the source
scan.

### 4.5 Controls and matrix (criterion 2)

Two hand-built control steps, fixed and held **outside the classifier's own generation
path** -- the exogenous audit set SEAL's conditions require, since an endogenous gate
cannot hold both error rates. Control observed GREEN first; a mutant classifying
everything as evidence-class must be KILLED by the all-product control; a mutant that
fails to run scores ERROR. **The 90.1 lesson applies: "fails to run" means fails to
IMPORT, not merely fails to parse** -- the matrix will smoke-import.

## 5. Doctrinal basis

Anthropic harness-design's hard-threshold-or-fail loop and its "negotiate the contract
before any code" -- 90.9 is the filing-time analogue. `docs/runbooks/per-step-protocol.md`
§4.

## 6. References

- `handoff/current/research_brief_90.9.md` (gate PASSED on re-run, enforced).
- https://arxiv.org/html/2607.24300v1 -- the impossibility bound and the measured fixed point.
- https://arxiv.org/html/2604.15224v1 -- ERR_J 0.000; the leniency is unauditable from notes.
- https://arxiv.org/html/2509.06770v1 -- scoped feedback (**paraphrase, not verbatim**).
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- **NOT cited for detector accuracy:** `arXiv 2501.04810`. See §1 correction 1.
