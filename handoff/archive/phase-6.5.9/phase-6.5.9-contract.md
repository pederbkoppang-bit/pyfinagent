# Sprint Contract — phase-6.5 / step 6.5.9 (E2E smoketest with fixtures)

**Step id:** 6.5.9 — final step of phase-6.5 Path D.
**Cycle:** 1. **Date:** 2026-04-19. **Tier:** moderate.

Parallel-safe: phase-scoped handoff files.

## Research-gate summary

7 sources read in full (Bunnyshell e2e best-practices 2026 + microservices 2026, Dojo Five exit-codes, Sonar audit-logging, Last9 JSON log format, CircleCI smoke-tests, Microsoft engineering playbook smoke-testing), 17 URLs, recency scan present, three-variant queries visible, 8 internal files inspected including the phase-12.4 `rainbow_rehearsal.py` exemplar. Brief at `handoff/current/phase-6.5.9-research-brief.md`. `gate_passed: true`.

## Hypothesis

A 5-stage smoketest script `scripts/smoketest/intel_e2e.py` composes the shipped phase-6.5 modules end-to-end under `--fixtures` (YAML fixture + stub embedder). Each stage returns a `{ok, ...}` dict. S5 assembles the JSON summary (the Path-D "digest" per `superseded_by: 6.5.9`) and appends a row to `handoff/audit/intel_e2e.jsonl`. Script exits 0 when `overall_ok` is True.

## Path-D interpretation of immutable criteria

Criteria authored 20:30 UTC were drafted against the 9-step pre-Path-D scope. Path D (decided 20:52 UTC; on-disk `.claude/masterplan.json::phase-6.5.path_decision`) kept 4 steps, dropped 5, and explicitly wired `phase-6.5.8.superseded_by = 6.5.9`. The interpretation below is grounded on those on-disk facts — NOT an amendment to the criteria:

- **`overall_ok_true`** ← all 5 stages `ok: True` → `summary["overall_ok"] = True`.
- **`at_least_one_record_per_extractor_family`** ← each distinct `source_type` in the active (`kill_switch=False`) fixture sources emits ≥1 `DocumentCandidate` from `BaseScanner(dry_run=True)`. Path D collapsed all type-specific extractors into `BaseScanner`; the "family" construct survives as `source_type`. Active fixture sources include `http` and `rss` (2 families), each producing 1 stub candidate → 2 ≥ 1 per family.
- **`novelty_and_digest_stages_pass`** ← S3 (`score_novelty`) `ok: True` AND S5 (`digest_and_audit`) `ok: True`. "Digest" = the JSON-summary stage, which replaces Slack digest per the `superseded_by: 6.5.9` masterplan pointer baked in by the 20:52 UTC path decision.
- **`exit_0`** ← `raise SystemExit(0)` on overall-ok; non-zero only on truly fatal exception (matches `rainbow_rehearsal.py` pattern at `scripts/smoketest/rainbow_rehearsal.py:275-290`).

## Immutable success criteria (verbatim from .claude/masterplan.json)

- `overall_ok_true`
- `at_least_one_record_per_extractor_family`
- `novelty_and_digest_stages_pass`
- `exit_0`

Not edited. Interpreted above.

## Plan steps

1. Create `scripts/smoketest/intel_e2e.py` (~200 lines):
   - `S1 load_registry()` → returns `{ok, active_count, active_sources}` from `load_from_yaml(fixture)` with kill-switch filtering.
   - `S2 scan_sources(active)` → returns `{ok, candidates_by_family, total_candidates}`; `ok` requires each distinct `source_type` to have ≥1 candidate.
   - `S3 score_novelty(candidates)` → returns `{ok, scores, scorer}`; uses `novelty_client` with injected `_stub_embed` (no live API keys).
   - `S4 enqueue_patches(candidates, scores)` → returns `{ok, patch_ids}`; monkeypatches `prompt_patch_queue._insert` with a captive store so no BQ needed.
   - `S5 digest_and_audit(summary)` → builds a dict, appends to `handoff/audit/intel_e2e.jsonl`, prints to stdout; `ok: True` if the JSONL write succeeded.
   - `main(argv)` wires stages with fail-open per stage (same shape as rainbow_rehearsal). Final `raise SystemExit(0 if overall_ok else 1)`.
2. Adopt the `rainbow_rehearsal.py:275-290` top-level pattern verbatim so Q/A can trace exact precedent.
3. Run the immutable command; capture verbatim stdout + exit code.
4. Confirm `handoff/audit/intel_e2e.jsonl` was appended to (row count went up by 1).
5. Run full regression to confirm no test-surface changes broke anything.
6. Write `phase-6.5.9-experiment-results.md`, spawn Q/A, log-last, flip 6.5.9 status, then flip phase-6.5 phase-level status to `done` (4/4 kept steps done).

## Out of scope

- No pytest target for the smoketest (same design as `rainbow_rehearsal.py` — the smoketest IS the e2e validation).
- No live BQ, no live HTTP, no live Voyage/Gemini (`--fixtures` enforces stubs).
- No source-specific extractor reintroduction.
- No Slack wiring.
- ASCII-only logger + print strings.

## References

- `handoff/current/phase-6.5.9-research-brief.md`
- `handoff/current/phase-6.5-decision-contract.md` (Path D + `path_decision.open_issue`)
- `scripts/smoketest/rainbow_rehearsal.py:275-290` (top-level orchestrator precedent)
- `backend/intel/source_registry.py`, `backend/intel/scanner.py`, `backend/intel/novelty_client.py`, `backend/intel/prompt_patch_queue.py`
- `backend/tests/fixtures/intel_sources.yaml`
- `.claude/masterplan.json` → phase-6.5 (path_decision + 6.5.9 immutable verification + 6.5.8 `superseded_by: 6.5.9`)
