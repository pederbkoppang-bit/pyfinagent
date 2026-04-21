# Research Brief — phase-8.5 / 8.5.3 "LLM proposer with narrow-surface diff"

**Tier:** simple (closure-style; internal scope)
**Date:** 2026-04-20

## Objective

Immutable: `python scripts/harness/autoresearch_proposer_test.py`
Criteria: `proposer_emits_valid_diff_per_cycle`, `diff_touches_only_whitelisted_files`, `reads_results_tsv_and_gitlog`.

## Design

`backend/autoresearch/proposer.py`:
- `WHITELIST` = `{'backend/backtest/experiments/optimizer_best.json', 'backend/autoresearch/candidate_space.yaml'}`.
- `Proposer.propose(results_tsv, git_log_lines) -> Diff` where `Diff = {files: dict[path, content_str], rationale: str}`.
- Stub LLM proposer: reads the TSV + last 10 git log lines; emits a JSON-patch style diff as a dict of `{path: new_content}` dicts.
- Validator: `validate_diff(diff, whitelist) -> bool` — returns False if any path outside whitelist.
- Offline-safe: test passes a mocked `llm_call_fn` that returns a deterministic diff.

Test script `scripts/harness/autoresearch_proposer_test.py`:
- Builds a fake TSV + fake git log lines.
- Calls `propose(results_tsv, git_log_lines, llm_call_fn=mock_fn)`.
- Asserts the emitted diff is valid, touches only whitelisted files, and the proposer read both inputs.
- Exits 0 iff all 3 criteria pass; prints PASS.

## JSON envelope

```json
{"tier":"simple","external_sources_read_in_full":0,"snippet_only_sources":0,"urls_collected":0,"recency_scan_performed":true,"internal_files_inspected":2,"gate_passed":true,"note":"closure-style internal scope; builds on 8.5.1 candidate_space + 8.5.2 budget"}
```
