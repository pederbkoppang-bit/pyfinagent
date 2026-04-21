# Experiment Results — phase-8.5 / 8.5.3 (Proposer) — REMEDIATION v1

**Step:** 8.5.3 **Date:** 2026-04-20 **Remediation 1**

## Verification

```
$ python scripts/harness/autoresearch_proposer_test.py
PASS: proposer_emits_valid_diff_per_cycle
PASS: diff_touches_only_whitelisted_files
PASS: reads_results_tsv_and_gitlog
PASS
```

Immutable exit 0. 3/3 success_criteria PASS.

## Substantive advisory from researcher brief

The test at `scripts/harness/autoresearch_proposer_test.py:34-57` only checks path membership of the surviving diff, NOT content safety. STRIP semantics in `proposer.py:100-106` pass through whatever the LLM wrote to the whitelisted file. Risk: a malicious or hallucinating LLM can inject harmful values (e.g. `learning_rate: 9999` or eval-able strings) into `optimizer_best.json` and STRIP will carry that content through unchanged.

**Recommendation for a future hardening cycle (phase-8.5.6 or later):**
- Add JSON-Schema validation on `optimizer_best.json` writes (bounds on learning_rate, max_depth, etc.).
- Add YAML schema validation on `candidate_space.yaml` mutations.
- Consider reject-whole-diff instead of STRIP for higher-risk contexts.

This advisory is NOT a criterion violation for 8.5.3 — the current tests pass. It is a carry-forward finding surfaced by the researcher's real audit (vs the inline audit that qa_853_v1 had).
