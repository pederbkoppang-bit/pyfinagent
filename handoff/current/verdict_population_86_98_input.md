# Verdict population measurement — the input to 86.98

**Written 2026-08-17 after the R2 circuit breaker tripped**, per the night goal's
§4: *"Extend today's 15-verdict measurement across ALL sessions … A population
figure would move 86.98 from one session's evidence to the repo's. Do NOT
implement 86.98. Measure for it."*

**86.98 was not implemented. This is measurement only.**

---

## Corpus

```
run records found:      2034      (find <project>/*/workflows/wf_*.json)
Q/A verdicts recovered:  377      (records whose result.verdict ∈ {PASS, CONDITIONAL, FAIL})
sessions:                 44
```

Recovered by parsing the run records directly, not by reading any artifact's
summary of them.

| verdict | n | share |
|---|---:|---:|
| CONDITIONAL | 186 | 49.3% |
| PASS | 138 | 36.6% |
| FAIL | 53 | 14.1% |

---

## The headline the day session measured, now at population scale

The 2026-08-16 day session reported **8 of 15** verdicts (53%) saying every
criterion was MET and returning CONDITIONAL anyway. Across the whole repo:

| verdict | n | reason asserts **ALL** criteria MET |
|---|---:|---:|
| CONDITIONAL | 186 | **106 (57.0%)** |
| FAIL | 53 | 11 (20.8%) |
| PASS | 138 | 120 (87.0%) |

**The day session's figure generalises: 53% → 57.0% over 12× the sample.**

### The classifier, and why it should be believed

It was built **from the harvested sentences**, not from my guesses — 231 real
`MET`-bearing sentences were extracted from CONDITIONAL reasons first, and the
quantifier preceding `MET` was tabulated before any pattern was written. It
detects the all-quantified form and **excludes partials** (`5 of 6`,
`criteria 1/2/3 are MET`).

Two controls:

- **Positive:** 87.0% of PASS verdicts classify as all-met. That is what a
  working classifier must do — a PASS essentially always says every criterion was
  met. A broken pattern would not score PASS that high.
- **Manual:** sampled classifications in both directions read correctly. The
  TRUE class is *"All 6 immutable criteria are MET on substance…"*; the FALSE
  class correctly excludes *"criteria 1/2/3 are MET"* (partial) and *"all three
  cycle-1 BLOCK findings are genuinely closed"* (findings, not criteria).

---

## The finding that actually decides 86.98

Of the **106** CONDITIONAL verdicts asserting every criterion was MET:

```
      EMPTY violated_criteria             :   0
      cite a NUMBERED criterion           :  13
      cite ONLY quality tags              :  93
```

**Not one of the 106 returned an empty finding list.** Every single "all criteria
MET → CONDITIONAL" verdict named a concrete defect.

The tags they cite:

| n | tag |
|---:|---|
| 20 | `illusory-guard` |
| 19 | `Contradiction` |
| 13 | `Missing_Assumption` |
| 12 | `Overgeneralization` |
| 9 | `scope-honesty` / `scope_honesty` |
| 3 | `Unjustified_Inference` |
| 3 | `evidence_integrity` |
| 3 | `claim-auditing (qa.md 4b)` |
| 3 | `consumer-contract-break` |

So: **93 of 186 CONDITIONALs (50.0%) met every immutable criterion and were
capped purely on findings no criterion names.**

---

## What this does and does not license

**It is NOT evidence of rubber-stamping.** The zero-empty result is decisive:
there is no population of verdicts withholding PASS without saying why. And the
single most common tag — `illusory-guard`, 20 of 93 — is a correctness concern,
not a style nit: a guard that cannot fail when its subject is broken is a test
that certifies nothing.

**Tonight is the counter-sample, and it is unambiguous.** All three steps capped
on exactly this pattern, and in **7 of 7** cycles the finding was a real defect in
my own work, each one reproduced by me before I accepted it — twice my
re-measurement found the evaluator had *understated* the problem. Concretely:
a positive control that could not fail; a buildability oracle blind to the only
failures it gated; a measured figure invalidated by my own edit in the commit
that stated it; a fail-**open** `continue` inside a module whose thesis is
fail-closed; a correction that *accompanied* instead of replacing, inside the
step whose criterion says exactly that; and an effectiveness claim that measures
zero.

**So the question 86.98 has to answer is a POLICY question, not a bug report:**

> When all N immutable criteria are met and the only findings are quality tags,
> should the step close?

The measurement says this is the *normal* outcome (50% of all CONDITIONALs), not
an anomaly. Three defensible answers, none of which this document chooses:

1. **Keep it.** Quality findings are the harness's main yield; the criteria are a
   floor, not a ceiling. Cost: steps park while correct product sits finished —
   tonight, 86.94 parked with all 7 criteria met.
2. **Split the verdict.** Distinguish PRODUCT-correct from EVIDENCE-complete and
   let a step close with residual quality findings queued as their own steps.
   This is already the shape `attempt_budget.py` was designed around (still
   unwired — step 86.71).
3. **Promote or demote the tags.** If `illusory-guard` should block, make it a
   criterion so it is named in the contract up front rather than discovered at
   EVALUATE. 20 occurrences say it is load-bearing enough to deserve that.

**Whichever is chosen, note what must not change:** every one of these 106
verdicts named a real finding. Any rule that lets a step close must still make
those findings land somewhere an operator will read — otherwise the harness's
highest-yield output is discarded to buy throughput.

---

## Reproducing this

The population is derived at run time from the run records; no figure here is
transcribed. Re-run the extraction over
`find <project-dir> -path '*/workflows/wf_*.json'`, parse `result.verdict`,
`result.reason` and `result.violated_criteria`, and apply the two regexes
(all-met, partial-exclusion) documented above. **The counts will drift as more
verdicts land** — which is the point of phase-86.94, and why the corpus size
(2034 records / 377 verdicts / 44 sessions) is stated alongside every figure.
