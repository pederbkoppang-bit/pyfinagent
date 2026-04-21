# Sprint Contract — phase-8.5 / 8.5.3 (LLM proposer with narrow-surface diff)

**Step id:** 8.5.3 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Hypothesis
Scaffold `Proposer` that reads `results.tsv` + last-N git log lines and emits a narrow-surface diff restricted to a whitelist. Test script verifies valid-diff-per-cycle + whitelist enforcement + inputs-were-read.

## Immutable criterion
- `python scripts/harness/autoresearch_proposer_test.py` exits 0.

## Plan
1. `backend/autoresearch/proposer.py` (~120 lines): WHITELIST constant, `Diff` TypedDict, `Proposer.propose(results_tsv, git_log, llm_call_fn=None)`, `validate_diff(diff, whitelist)`. Defaults an offline-safe stub LLM that returns a single-file whitelisted diff.
2. `scripts/harness/autoresearch_proposer_test.py` (~100 lines): 3 cases (valid diff per cycle, whitelist enforcement, inputs read) + aggregate PASS + exit 0.
3. Verify + regression + Q/A + log + flip.

## Out of scope
- No real Claude API call; default `llm_call_fn` is a deterministic stub.
- No git integration (inputs are passed in as list[str]).
- ASCII-only.
