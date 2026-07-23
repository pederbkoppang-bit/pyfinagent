# Root-cause research -- the unmeasured-scope-claim defect (phase-75.5)

**Workflow**: `wf_b550e771-aa7` (2026-07-20). 12 of 13 agents completed; the completeness
critic errored on a weekly usage limit (noted; its job was partly done by the forensic
agent, which found the 11th instance itself). Forensics + 5 research lenses + adversarial
verify + 4 design candidates + judge all completed.

**This brief asserts no count it did not measure.** Where a number appears, the command
that produced it is named. Two figures quoted from the research agents (`~68 unresolved
claims`, `101 numeric-claim prose lines`, `5 hits / 2 TP / 3 FP`) are **single
derivations under rules the agents wrote**, i.e. the exact shape of the defect under study,
and are marked as such -- do not build on them without re-deriving.

---

## 1. The corrected root cause

Main's working hypothesis -- "every instance is a claim about a SET whose membership rule
was never written down" -- was graded **PARTIAL / loose post-hoc fit** by both the forensic
agent and the adversarial verifier. By the forensics' own per-instance notes the membership
rule was **stated-and-correct** in instances #1, #3, #4 and #5, and the defect shipped
anyway. Stating the rule changed the prose; it did not change the outcome.

The two explanations that fit all the evidence:

1. **STRONGEST -- the harness had no instrument pointed at the CLAIMS.** Verification
   effort targeted the *code*, which was correct on all seven Q/A cycles. The *claims about
   the code* were never a verification target. This explains the two facts the
   set-membership story cannot: the engineering was right every time, and every defect was
   in the surrounding prose.
2. **Why self-audit could not catch it: the claim and its check are authored in the same
   pass by the same author, so the check inherits the claim's blind spot.** The decisive
   evidence is instance #9 -- a completeness scan Main wrote *to catch* instance #8, which
   shared the same blind spot and could not find its own members. This is why all 11 were
   caught by the *independent* Q/A and none by Main.

A causally-distinct third strand covers several instances: **staleness under a moving
referent** (#2, #4, #5, and the still-live "three F401s") -- the claim was true when written
and the world moved underneath it. The remedy there is re-execution / non-hand-editing of
measured numbers, not rule-statement.

**The generator observation** (verifier, upheld): instances #8, #9, #10 are each a defect
committed *inside the fix for the previous one*, and they got more emphatic as the
rule-statement got more explicit -- culminating in an unmeasured number ("~36 functions")
inside a section titled "WHY THERE IS NO NUMBER IN THIS DOCSTRING". More prose discipline
made it worse, not better. This is the case against "just be more careful".

---

## 2. What the evidence supports building (and what it does NOT)

The judge re-derived every candidate's catch-rate against the 11 instances and was blunt:
**mechanism reliably catches only the four CHEAP instances** (#2 lint-scope omission, #4
carried-forward count, #5 spliced capture, #7 phantom step-ids) -- each a single
dereference against an external authority (`git diff`, re-run pytest, count dots, walk the
masterplan). **The six expensive instances** (#1, #3, #6, #8, #9, #10) are caught only by
Q/A JUDGMENT -- mutation testing, witness testing, reading a truncated composition, an
independent re-derivation -- and **no proposed mechanism reliably catches any of them.**

Instance #9 is the proof and it is decisive: it IS "write a checked-in completeness scan"
(the headline mechanism) *applied and failing*, so that mechanism cannot be banked as a
catch for the class it most wants to solve.

**External sources, fetched and read in full, verbatim-verified by the adversarial pass:**
- Beer/Ben-David/Eisner/Rodeh, *Efficient Detection of Vacuity in ACTL Formulas* (FMSD
  18:141-163, 2001): "typically 20% of formulas are found to be trivially valid, and
  trivial validity always points to a real problem" -- the industrial precedent for
  vacuity/witness testing, and the reason a passing assertion is not evidence until a
  mutant kills it.
- Goodenough & Gerhart 1975: a validity proof for a test criterion "exists if and only if
  the program is already correct" -- **the hard ceiling.** No completeness mechanism can
  prove a census right; it can only raise the cost of a wrong one.
- Anthropic harness-design: "agents tend to respond by confidently praising the work -- even
  when the quality is obviously mediocre" -- the independent-evaluator justification.
- arXiv 2606.09863 (*From Confident Closing to Silent Failure*) and 2607.05391
  (*LLM-as-a-Verifier*): both real, numbers quoted accurately; the latter's own thesis is
  that criteria-in-a-rubric an LLM grades are weaker than criteria-as-a-command.

**The binding conclusion, in the research's own words:** this class of fix "makes four of
ten instances structurally impossible and hands Q/A a bounded, machine-denominated list for
the rest. It does not reduce the need for a skeptical independent evaluator by one iota, and
if adopting it is used to justify lighter Q/A scrutiny it will make the system strictly
worse than doing nothing."

---

## 3. What was DONE this cycle (STAGE 0 -- mechanical, no new machinery)

Each verified by measurement, not assertion:

- **`.claude/agents/qa.md` lint gate**: the hand-typed `<changed .py files>` replaced with
  `$(git diff --name-only HEAD -- '*.py')` plus a mandatory empty-set guard (ruff prints
  "All checks passed!" and exits 0 on an empty/nonexistent path -- verified). This is the
  one-line fix the judge said to adopt immediately and independently; it permanently kills
  the instance-#2 shape.
- **`.claude/agents/qa.md` new section 4b (claim auditing)**: points the Q/A explicitly at
  the PROSE -- derive scopes, reproduce every count, require a known-member recall test for
  completeness claims, compare independent derivations by symmetric difference, reject
  edited "verbatim" captures. Codifies what the Q/A already did ad-hoc into a named duty.
- **The live 11th instance**, fixed by measurement: the F401 count was asserted "3" for six
  cycles but the derived scope showed **4** (the fourth: `import pytest` unused in the
  lock-roster test, a file 75.5 edited, never in the typed list). Removed the two
  `[*]`-fixable dead imports in files 75.5 already touched (`pytest`, `typing.Any`),
  re-measured to **2** (both `find_spec` probes in `autonomous_loop.py`), and corrected
  `experiment_results.md` + masterplan `75.5.6` to the measured 2. The stale "3 duplicated
  parse helpers" summary bullet was corrected to the six named sites.
- **CLAUDE.md live_check paragraph**: corrected -- a gate hold skips the commit+changelog
  too (holds `exit 0` before `git add -A`), not just the push. Verified against the hook.

## 4. What was QUEUED (larger builds -- research-gated, per operator rule)

- **75.5.9** -- Claims Ledger, advisory-first for two cycles, four measured resolvers
  (changed-files, lint-findings, masterplan-ids, collected-tests), every resolver asserting
  a non-empty population before comparing. Do NOT build the claimlock lock-file layer or the
  claimc `.md.in` compiler -- the judge rejected both as unaffordable/opt-in-defeatable for
  one local dev.
- **75.5.10** -- `live_check_gate.py` content-reading: today it only `.exists()`s the
  artifact, so an empty file passes the operator's audit gate. Minimum bar: non-empty +
  >=1 fenced block + not solely prose.
- **75.5.8 amended** -- the census must produce TWO independent derivations of the
  truncation-blind population and report their **symmetric difference**, not counts
  (cardinality agreement is not member agreement -- instance #10 returned 17 and 20 for
  "the same rule").

## 5. Residual risk (verbatim from the judge -- the honest ceiling)

The semantic half stays open (#1, #8, #10 caught by nothing mechanical); paraphrase evasion
is free ("the touched surface", "the affected modules" carry no numeral); a machine-emitted
worklist can be sampled rather than worked (instance #7 was certified by three Q/As who
checked whether the defects were real, never whether the steps existed); fail-open is an
attack surface that must stay; and moving denominators produce confident false holds. **None
of this removes the independent skeptical evaluator; it makes its list bounded.**
