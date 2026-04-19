# Sprint Contract -- phase-12.2 promote.py + rollback.py CLI

**Written:** 2026-04-19 PRE-commit.
**Step id:** `12.2` in phase-12.
**Immutable verification:** `source .venv/bin/activate && python scripts/deploy/rainbow/promote.py --dry-run --to green && python scripts/deploy/rainbow/rollback.py --dry-run` — both exit 0, each prints the exact kubectl/service-selector patch it would apply.

## Research-gate summary

Researcher envelope `{tier: simple, external_sources_read_in_full: 6, snippet_only_sources: 4, urls_collected: 10, recency_scan_performed: true, internal_files_inspected: 5, gate_passed: true}`. 3-query compliance confirmed.

Staked recs adopted:
- **argparse** (stdlib) over click — scripts/harness/run_harness.py precedent, no new dep.
- **Explicit `--dry-run` flag, default live** — matches `run_harness.py` convention and the masterplan verify command passes `--dry-run` explicitly.
- **`kubectl get service ... -o jsonpath='{.spec.selector.color}'`** for current-color detection — cluster is single source of truth; no local `.rainbow-state` file that can drift.

## Hypothesis

Two ~80-line CLI scripts (promote.py + rollback.py) with `--dry-run` printing the exact kubectl patch JSON satisfy the immutable verify + give operators a one-liner for the rollout/rollback recipes in `deploy/rainbow/README.md`.

## Success criteria

**Functional:**
1. `scripts/deploy/rainbow/__init__.py` (empty, makes it importable for tests).
2. `scripts/deploy/rainbow/promote.py`:
   - `argparse` CLI: `--to {blue,green,...}` required; `--service NAME` default `pyfinagent-backend`; `--dry-run` flag.
   - Behavior: in `--dry-run`, print the exact `kubectl patch service <svc> -p '<json>'` command + the selector-patch JSON; exit 0 without running.
   - Live mode: call `subprocess.run(["kubectl", "patch", "service", svc, "-p", json])`; exit 0 on success, 1 on failure.
   - Fail-open: subprocess exception → exit 1 with printed error; never raises unhandled.
3. `scripts/deploy/rainbow/rollback.py`:
   - `argparse` CLI: `--to COLOR` optional (if omitted, uses a file fallback); `--service NAME` default `pyfinagent-backend`; `--dry-run` flag.
   - Reads current color via `kubectl get service -o jsonpath='{.spec.selector.color}'`.
   - Default rollback target: "blue" if current is "green", "green" if current is "blue" (simple toggle). For 7-color future: operator explicitly passes `--to <previous-color>`.
   - Dry-run prints the kubectl patch command + JSON; live runs it.
4. Both scripts print the patch JSON to stdout in dry-run so the caller can copy-paste if automation fails.
5. Both scripts use argparse with `--help` producing usable output.
6. `backend/tests/test_rainbow_cli.py` with ≥3 tests (monkeypatch subprocess.run; assert dry-run prints right JSON; assert live mode calls kubectl with right args; rollback toggle logic).
7. Immutable verify passes (`--dry-run --to green` for promote; `--dry-run` for rollback both exit 0).

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('scripts/deploy/rainbow/promote.py').read()); ast.parse(open('scripts/deploy/rainbow/rollback.py').read()); print('ok')"`.
- Immutable: as above.
- Tests: `pytest backend/tests/test_rainbow_cli.py -x -q` → all pass.
- Help output: `python scripts/deploy/rainbow/promote.py --help` and `rollback.py --help` both exit 0.
- Regression: full test suite passes (79p/1s + 3 new = 82p/1s).

**Non-goals:**
- NOT calling kubectl in live mode during this cycle (no cluster; deferred to 12.3/12.4 smoke).
- NOT reading yaml manifests (scripts flip Service selector; yaml is operator-applied).
- NOT wiring into CI.
- NOT supporting ≥2 Services simultaneously (single-service MVP).

## Plan

1. Create `scripts/deploy/rainbow/` directory.
2. Write `promote.py` + `rollback.py` + `__init__.py`.
3. Write `backend/tests/test_rainbow_cli.py`.
4. Run verification commands.

## Researcher agent id

`a63b35b9ccf3d4efe`
