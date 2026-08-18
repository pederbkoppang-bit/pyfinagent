# Contract — step 86.75 (written AFTER the gate, as a REPAIR)

**Step:** 86.75 | **Priority:** P1 | **Status:** pending
**Research gate:** PASSED — `wf_c1b10b08-07c`, 26 sources read in full, 66 URLs,
audit-class 18 rounds / 2 dry. Brief: `handoff/current/research_brief_86.75.md`.
Verdict: `handoff/current/research_gate_86.75_verdict.md`.

> **THIS CONTRACT CANNOT UNDO THE BREACH, AND DOES NOT PRETEND TO.**
> `PROTOCOL_BREACH_86.65.md` records that 86.75's GENERATE ran with no gate and no
> contract. Writing a contract now, after the work, would normally be the exact
> "contract at the end" failure the protocol forbids — a document shaped to match
> work already done. **So this contract binds the RE-DERIVATION, not the original
> work.** Everything in `live_check_86.75.md` is demoted to INPUT. Any criterion
> below must be satisfied by evidence produced under this contract, and where the
> prior artifact already contains a measurement, it must be RE-RUN, not cited.

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. the counter repoint is proven by DRIVING it, not by reading the prompt: run a Q/A on a step with >=2 prior WIP records and show the returned notes state the correct derived attempt number, and show a step with 0 prior records is unaffected

2. the qa_wip.py-versus-harness_log divergence is INDEPENDENTLY re-derived with a positive control on the grep, and any disagreement with the figures in this audit_basis is reported rather than silently adopted

3. the Contract-completeness gate row still exists in qa.md as a LIVE table row, demonstrated by a command whose output is shown -- this is the line the audit finding would have deleted

4. no research floor was weakened by deleting .claude/context/research-gate.md: show FLOOR_SOURCES and FLOOR_URLS are unchanged and that no remaining file states a lower number

5. scripts/qa/verify_research_gate_workflow.mjs is green at NO FEWER than 121 assertions, with the count reported, so the verifier cannot have been made green by deleting assertions

6. every remaining mention of the deleted .claude/context/research-gate.md is confirmed to be a deletion NOTE and not a live pointer, with the enumeration shown

7. the agent-file changes are reviewed by the operator per the separation-of-duties rule before any step depends on them, and the roster is confirmed live after a session restart

8. verdict semantics are UNCHANGED and this is demonstrated rather than asserted: nothing in this change can turn a FAIL into a PASS


**Verification command (immutable):**

```
bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_research_gate_workflow.mjs | tail -1'
```

**Required live_check:** live_check_86.75.md with the driven Q/A showing the derived attempt number in notes, the re-derived qa_wip-vs-harness_log divergence with its positive control, and the verifier assertion count against the 121 baseline

---

## Hypothesis

The five shipped changes are individually defensible, but the gate found that **two
of the arguments used to justify them were wrong**, and that **one of them created a
new defect**. So the step's own claim — "5 subtraction-first changes landed" — is
true about the changes and false about the reasoning.

## What the gate changed about the plan

| finding | effect on this step |
|---|---|
| Consequence-framing → **leniency in 58/72 cells**, invisible to CoT | change (1) created a NEW exposure; filed as **86.78**, NOT fixed here |
| Anti-override deletion justified by the WRONG mechanism (self-attribution, not authority anchoring −14.95 pp) | criterion 8's "verdict semantics unchanged" must be re-argued on the correct mechanism |
| Law-of-the-case pairs override with **burden** + **recorded override**; neither shipped | a gap this step must disclose, not repair silently |
| Change (1) already superseded by **86.21** | criterion 2's divergence re-derivation must say so |
| **35/35** ledger rows `recorded_by: main`; **5/35** `NO_VERDICT` | criterion 1's counter evidence is Main-authored — state it |

## Plan

1. **Re-run** criteria 2–6 and 8 under this contract, treating `live_check_86.75.md`
   as input. Every number re-derived at execution time.
2. **Criterion 1 needs a driven Q/A** — and per the gate, the spawn must NOT supply
   the attempt number or its consequence, or the measurement is contaminated by the
   very bias 86.78 records.
3. **Criterion 7 is operator-owed** and cannot be discharged by Main: four `qa.md`
   edits now await separation-of-duties review.
4. **Do not fix 86.78 here.** It is a fourth Main-authored `qa.md` edit and belongs
   to a fresh executor.

## Traps this step has already hit

- A probe matching its own documentation (three times today).
- `zsh` not word-splitting an unquoted var → a clean false zero.
- `exit=$?` after a pipe reporting `tail`'s status.
- Wall-clock times **narrated rather than measured** — 14 artifacts corrected.

## References

- `handoff/current/research_brief_86.75.md` (79,363 chars)
- `handoff/current/research_gate_86.75_verdict.md`
- `handoff/current/PROTOCOL_BREACH_86.65.md`
- arXiv 2604.15224 (consequence-framing), 2603.04582 (self-attribution vs anchoring),
  2606.19544 (reproducible ≠ valid)
