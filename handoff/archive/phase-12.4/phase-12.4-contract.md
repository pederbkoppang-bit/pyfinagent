# Sprint Contract -- phase-12.4 First Rainbow Migration (dummy rehearsal)

**Written:** 2026-04-19 PRE-commit.
**Step id:** `12.4` in phase-12. **Final phase-12 step.**
**Immutable verification:** `grep -q 'rainbow' docs/VERTEX_AI_GENAI_MIGRATION.md && echo ok` — ALREADY satisfied (verified in research gate). Contract adds a second, functionally meaningful verify below.

## Research-gate summary

Researcher envelope `{tier: simple, external_sources_read_in_full: 5, snippet_only_sources: 7, urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 8, gate_passed: true}`. 3-query compliance confirmed.

**Scope-reassignment context** (per phase-12.0 cycle-2): masterplan step 12.4 was originally "first real migration using Rainbow: Vertex → google-genai cutover". Phase-11 shipped the Vertex migration **without** Rainbow (direct migration; 79p/1s regression; zero incidents; 2026-04-19). That candidate is gone. Replacement picked: **dummy color-flip-only rehearsal** — exercise the full Rainbow machinery (manifests + promote.py + rollback.py + canary SLO diff) end-to-end with synthetic data, no real cluster, no production code change.

Staked rec adopted: smoketest script `scripts/smoketest/rainbow_rehearsal.py`, mirroring the proven `scripts/smoketest/phase6_e2e.py` shape (serial stages, fail-open, audit JSONL, JSON summary, exit 0/1).

## Hypothesis

One ~250-line smoketest that composes `promote.py --dry-run` + `canary_snapshot_from_buffer` + `rollback.py --dry-run` into a 5-stage serial rehearsal proves the phase-12 machinery works end-to-end, and gives operators a single-command drill before a real migration.

## Success criteria

**Functional:**
1. `scripts/smoketest/rainbow_rehearsal.py` (new) with 5 stages:
   - S1 **promote dry-run**: load `promote.py` via importlib, call `promote.main(["--dry-run", "--to", "green"])`, assert rc==0 + "DRY-RUN" in captured stdout.
   - S2 **canary inject equal latency**: reset api_call_log buffer, inject 20 blue + 20 green rows at 100ms each, run `canary_snapshot_from_buffer(...)`, assert `reason=="ok"` and `regression is False`.
   - S3 **canary inject regression**: inject green at 250ms, run snapshot, assert `regression is True`, `ratio > 2.0`.
   - S4 **rollback dry-run**: load `rollback.py`, call `rollback.main(["--dry-run"])`, assert rc==0 + "blue" in stdout.
   - S5 **emit audit row** to `handoff/audit/rainbow_rehearsal.jsonl` (append-only) + print JSON summary to stdout.
2. CLI: `--dry-run` flag (defaults to true, since the whole rehearsal is inherently dry-run); `--verbose` optional.
3. Exit 0 on all stages PASS; exit 1 on any uncaught exception escaping stage fail-open.
4. No changes to production modules (`backend/` + `scripts/deploy/rainbow/`). Rehearsal is read-only on those.
5. Audit JSONL row shape matches phase-6.8 pattern (`ts`, `stages[] with {name, ok, detail}`, `overall_ok`).

**Correctness verification commands:**
- Immutable (masterplan): `grep -q 'rainbow' docs/VERTEX_AI_GENAI_MIGRATION.md && echo ok` → `ok` (already satisfied).
- Syntax: `python -c "import ast; ast.parse(open('scripts/smoketest/rainbow_rehearsal.py').read()); print('ok')"`.
- **Live rehearsal**: `source .venv/bin/activate && python scripts/smoketest/rainbow_rehearsal.py` → exit 0, prints JSON summary with all 5 stages `ok: true`.
- Audit row: `test -f handoff/audit/rainbow_rehearsal.jsonl && python -c "import json; print(json.loads(open('handoff/audit/rainbow_rehearsal.jsonl').readlines()[-1]))"` — parses.
- Regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` → 103p/1s unchanged (no test changes).

**Non-goals:**
- NOT running against a real Kubernetes cluster.
- NOT performing a real migration.
- NOT adding pytest tests (the smoketest IS the e2e validation; phase-12.2/12.3 already have unit-test coverage).
- NOT touching any production code.

## Plan

1. Write `scripts/smoketest/rainbow_rehearsal.py`.
2. Run it live + confirm stages.
3. Verify JSONL + regression.

## Researcher agent id

`a5b2dd1f2a2abdef9`
