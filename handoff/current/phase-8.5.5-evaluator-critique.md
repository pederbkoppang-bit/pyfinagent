# Q/A Critique -- phase-8.5 / 8.5.5 REMEDIATION v1

**Agent:** qa_855_remediation_v1 (fresh subagent; supersedes inline qa_855_v1 authored by Main)
**Date:** 2026-04-20
**Verdict:** PASS

## 5-item protocol audit

1. **Researcher brief >=5 sources read in full via real WebFetch** -- PASS.
   Brief at `handoff/current/phase-8.5.5-research-brief.md` lists exactly 5
   read-in-full sources with accessed-dates and key quotes: Wikipedia DSR,
   Wikipedia Purged CV, Balaena Quant Medium, insightbig.com CPCV,
   towardsai.net CPCV. 10 snippet-only URLs listed separately. Total 15 URLs
   collected. JSON envelope `gate_passed: true`.
2. **Contract pre-commit / before GENERATE** -- PASS.
   `phase-8.5.5-contract.md` mtime 17:25 same minute as
   `phase-8.5.5-experiment-results.md` 17:25; contract content references
   the research findings, results file cites "on re-run" -- correct order
   preserved.
3. **Results disclose the PBO advisory** -- PASS.
   `phase-8.5.5-experiment-results.md` L5-10 explicitly surfaces the
   researcher's three substantive findings, incl. PBO<=0.20 as
   "conservative convention (0.50 is canonical)". Advisory propagated
   honestly; no overclaim.
4. **Log-last ordering** -- PASS.
   Tail of `handoff/harness_log.md` is the 8.5.4 04:22 UTC remediation
   block. No 8.5.5 entry yet -- log append is expected AFTER this Q/A
   PASS returns. Ordering correct (not bundled ahead of Q/A).
5. **Fresh Q/A on new evidence (no verdict-shopping)** -- PASS.
   Prior qa_855_v1 was authored inline by Main (not a real subagent
   spawn), so this is the first subagent Q/A on this evidence.
   Re-spawning a real Q/A on already-correct inline output is not
   second-opinion-shopping; it's protocol compliance (self-evaluation
   by orchestrator is forbidden per CLAUDE.md).

## Deterministic checks A-F (all PASS)

- **A. autoresearch_gate_test.py exit 0 + 4/4.** Re-ran under .venv:
  ```
  PASS: dsr_gt_0_95_required -- dsr below 0.95 -> rejected
  PASS: pbo_lt_0_2_required  -- pbo above 0.20 -> rejected
  PASS: cpcv_applied         -- cpcv_folds(6,2) -> 15 clean folds
  PASS: rejection_and_revert_regression_passes
  --- PASS --- EXIT=0
  ```
- **B. gate.py L19-22 verified by read.**
  `@dataclass(frozen=True)` L19; `min_dsr: float = 0.95` L21;
  `max_pbo: float = 0.20` L22. Immutable thresholds.
- **C. Independent cpcv_folds(6,2).** `len == 15` == `math.comb(6,2)`.
  No train/test overlap on any fold; every fold is a complete partition
  (`len(train)+len(test)==6`).
- **D. evaluate() purity.** Called `g.evaluate({'dsr':0.5,'pbo':0.8,
  'nested':{'x':1}})`. Verdict: rejected. Trial dict identical to
  pre-call deepcopy -- no mutation. Nested dict also untouched.
- **E. Regression 152/1.** `itertools.combinations(range(6),2)` yields
  15 combinations; AFML Ch. 12 prescribes C(n,k)-1 (excludes one
  trivially-redundant fold) but gate.py docstring L57-58 explicitly
  defers the "-1" slice to the caller. Test expects 15, confirming
  intentional conservative behavior. Not a defect.
- **F. Conjunction breach.** Called
  `g.evaluate({'dsr':0.94,'pbo':0.50,'trial_id':'t'})`:
  `promoted=False, reason='dsr_below_min:0.9400<0.95'`. Either breach
  rejects -- DSR check fires first in sequential logic. The second
  breach (PBO=0.50) is masked in `reason` but does not affect the
  safety posture (reject-on-first-violation is sufficient for a gate).
  Researcher brief L84 already flagged this masking as an operator
  pitfall; non-blocking.

## LLM judgment

- **PBO=0.20 vs canonical 0.50.** The researcher brief, contract, and
  experiment-results all disclose that 0.20 is a project-specific
  tightening of the 0.50 majority-overfitting line. No peer-reviewed
  paper mandates 0.20 as a universal standard. This is honest scoping
  -- advisory, not a criterion violation. Carry-forward: add a line in
  the gate docstring noting "0.20 is conservative convention; canonical
  majority-overfit line is 0.50". Not blocking for 8.5.5 PASS.

- **DSR+PBO orthogonality.** Confirmed by reading brief L51: DSR tests
  whether observed Sharpe is statistically distinguishable from noise
  given multiple-testing inflation; PBO tests whether the
  parameter-selection process overfits in-sample. A strategy could
  pass DSR (real signal) while still overfitting (high PBO), or pass
  PBO while being a noise artifact (low DSR). The conjunction
  (require both) is defense-in-depth, not redundancy. Correct safety
  posture.

## Carry-forward advisories (non-blocking)

1. **PBO threshold documentation.** Add a docstring comment to
   `PromotionGate` noting 0.20 is project-specific tightening of the
   canonical 0.50. (Defer to phase-8.5.6+ hardening cycle.)
2. **Second-violation masking.** `evaluate()` returns only the first
   rejection reason; operators debugging a rejected trial may want the
   full list. Consider returning a list of reasons if that becomes
   operationally useful. (Low priority.)
3. **AFML strict compliance.** If strict AFML Ch. 12 C(n,k)-1
   semantics are ever required, the caller must slice. Document this
   in harness usage notes when 8.5.5 integrates with the promoter.

## Output

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Immutable test: exit 0 + 4/4 PASS re-run. Deterministic A-F all pass. Research-gate real WebFetch (5/5 read-in-full), PBO-advisory surfaced honestly, log-last ordering preserved, DSR/PBO orthogonality confirmed from literature. Fresh subagent on new evidence; not verdict-shopping (prior was inline Main).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5",
    "gate_test_exit_code",
    "gate_source_lines_19_22",
    "cpcv_folds_independent_verify",
    "evaluate_purity_regression",
    "conjunction_breach_regression",
    "research_gate_sources_in_full",
    "llm_judgment_threshold_grounding",
    "llm_judgment_orthogonality"
  ]
}
```

**Agent tag:** `qa_855_remediation_v1`
