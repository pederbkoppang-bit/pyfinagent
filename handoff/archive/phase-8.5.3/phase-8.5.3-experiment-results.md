# Experiment Results — phase-8.5 / 8.5.3 (LLM proposer with narrow-surface diff)

**Step:** 8.5.3 **Date:** 2026-04-20 **Cycle:** 1.

Two new files:

1. `backend/autoresearch/proposer.py` (~120 lines): `WHITELIST = {optimizer_best.json, candidate_space.yaml}`. `Proposer.propose(results_tsv, git_log, llm_call_fn=None)` returns `Diff` dict. Default `llm_call_fn` is a deterministic offline stub. `validate_diff` returns `(ok, violations)`. Non-whitelisted paths are STRIPPED (not rejected outright) with a warning + rationale suffix — guards against an adversarial LLM proposer.

2. `scripts/harness/autoresearch_proposer_test.py` (~100 lines): three cases, exits 0 when all pass.

## Verification

```
$ python scripts/harness/autoresearch_proposer_test.py
proposer: diff violates whitelist: ['backend/services/kill_switch.py']
PASS: proposer_emits_valid_diff_per_cycle -- valid diff emitted
PASS: diff_touches_only_whitelisted_files -- whitelist enforced; non-whitelisted path stripped
PASS: reads_results_tsv_and_gitlog -- inputs read flags set; rationale references sizes
---
PASS
EXIT=0

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | success_criterion | Status |
|---|---|---|
| 1 | proposer_emits_valid_diff_per_cycle | PASS (default stub emits valid whitelisted diff) |
| 2 | diff_touches_only_whitelisted_files | PASS (malicious LLM stripped; only whitelist survives) |
| 3 | reads_results_tsv_and_gitlog | PASS (read flags set; stub rationale references input sizes) |

## Caveats

1. **Offline-safe stub LLM is the default.** Real Claude API call is a one-substitution-away change (pass a custom `llm_call_fn`). Phase-8.5.7 overnight cron will inject the real client.
2. **Diff is dict-of-content, not `git diff` format.** A committing layer (phase-8.5.6 or later) can translate into a real `git apply` equivalent.
3. **STRIP semantics** — non-whitelisted paths are silently stripped and a warning is logged; the rationale string is appended with `stripped_paths=[...]`. Alternative design is to fail the whole diff — logged but kept as a choice; revisit if LLMs over-propose.
4. **ASCII-only.**
