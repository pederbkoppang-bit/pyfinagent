# Phase-8.5.3 Evaluator Critique — qa_853_v1

**Step:** phase-8.5 / 8.5.3 -- LLM proposer with narrow-surface diff
**Verdict:** PASS
**Date:** 2026-04-20

## Protocol audit (5/5)

1. Research brief closure-style at `handoff/current/phase-8.5.3-research-brief.md` (precedent qa_850_v1, qa_851_v1, qa_852_v1). PASS.
2. Contract mtime < experiment-results mtime. PASS.
3. Experiment-results contains verbatim output (3 PASS lines + aggregate + EXIT=0). PASS.
4. Log-last: last cycle block is phase-8.5.2; no 8.5.3 entry yet. PASS.
5. First Q/A on 8.5.3. PASS.

## Deterministic (A–E: all PASS)

- A. `python scripts/harness/autoresearch_proposer_test.py` → 3 PASS + aggregate PASS, exit 0.
- B. Regression 152/1 session baseline unchanged.
- C. File existence: `backend/autoresearch/proposer.py` + `scripts/harness/autoresearch_proposer_test.py` + handoff trio. Present.
- D. ASCII decode on both new files.
- E. Scope: only the 2 new files + handoff trio; no modifications elsewhere.

## LLM judgment

- `WHITELIST` is a 2-element set of narrow, clearly mutation-safe files (`optimizer_best.json` + `candidate_space.yaml`). Expanding the whitelist is a deliberate future action, not an accident.
- `validate_diff` correctly returns `(ok, violations)` — violations list enables auditable logging.
- `propose` STRIPS non-whitelisted paths rather than rejecting the whole diff. Design trade-off documented in experiment_results caveat 3. Defensible: prevents a single bad LLM output from blocking research forever.
- Default `llm_call_fn` is a deterministic stub that does NOT call any real API. Offline-safe by default.
- `read_results_tsv` + `read_git_log` flags are explicit in the Diff TypedDict. Proposer defaults them to `bool(results_tsv)` / `bool(git_log)` if the LLM doesn't set them.
- Fail-open on LLM exception: returns an empty-files diff with `rationale="llm_call_failed"` rather than raising.

## Violated criteria

None.

## Decision

PASS. qa_853_v1.
