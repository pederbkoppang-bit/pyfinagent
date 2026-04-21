# Sprint Contract — phase-8.5 / 8.5.3 (Proposer) — REMEDIATION v1

**Step id:** 8.5.3 **Date:** 2026-04-20 **Tier:** simple

## Why remediation
qa_853_v1 PASS on inline-authored brief. Researcher subagent now authored real 6-source brief. Q/A to issue fresh verdict.

## Research-gate summary
6 sources in full (arXiv 2510.03217 dual-LLM patch validation, arXiv 2603.01257 LLM patching survey, Knostic AI coding agent security, OpenSSF security guide, Anthropic agent autonomy, arXiv 2602.17753 2025 AI Agent Index). gate_passed: true.

**Substantive finding:** STRIP semantics in `proposer.py:100-106` is defensible for the current 2-file narrow WHITELIST, but leaves a content-injection risk. A malicious LLM can write harmful values into whitelisted files (e.g. `optimizer_best.json` with `learning_rate: 9999`) and STRIP passes it through. Test only checks path membership, not content bounds. Recommendation: phase-8.5.6 committing layer MUST add JSON-Schema bounds-checking before unattended commits are safe.

## Immutable criterion
- `python scripts/harness/autoresearch_proposer_test.py` exit 0.

## Plan
Re-run test. Spawn Q/A. Q/A should confirm PASS on the immutable but carry the STRIP content-safety finding forward as an advisory to 8.5.6 (which already has `qa_856_v1 PASS`, so the advisory would attach to a future hardening cycle).

## References
- `handoff/current/phase-8.5.3-research-brief.md`
- `backend/autoresearch/proposer.py` (L100-106 STRIP path)
- `scripts/harness/autoresearch_proposer_test.py`
