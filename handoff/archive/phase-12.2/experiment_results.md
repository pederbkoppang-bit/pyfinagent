# Experiment Results -- phase-12.2 promote.py + rollback.py

**Step:** 12.2 (3rd of phase-12).
**Date:** 2026-04-19.

## What was built

Two thin Python CLI scripts + 11 unit tests. No new deps (stdlib argparse + subprocess).

- `scripts/deploy/rainbow/__init__.py` (empty, package marker).
- `scripts/deploy/rainbow/promote.py` (~95 lines) — `--to <color>` + `--service` + `--dry-run`. Builds `{"spec":{"selector":{"color":...}}}` patch JSON, shells `kubectl patch service ...`. Dry-run prints the exact kubectl command + JSON without running. Exit 0/1/2 on success/kubectl-fail/input-invalid.
- `scripts/deploy/rainbow/rollback.py` (~140 lines) — reads current color via `kubectl get service ... -o jsonpath='{.spec.selector.color}'`, toggles blue↔green via `_TOGGLE = {"blue": "green", "green": "blue"}`. `--to COLOR` overrides. Dry-run works without a cluster (falls back to "assume current=green, rollback to blue"). Exit 0/1/2/3.
- `backend/tests/test_rainbow_cli.py` — 11 tests covering `build_patch_json` JSON shape (promote + rollback), `build_kubectl_cmd` contents, dry-run exits 0, dry-run does NOT call subprocess, live mode calls kubectl with correct args, invalid-color returns 2, rollback dry-run without cluster defaults to blue, explicit `--to` overrides detection, live mode reads current then patches with the toggle, `_TOGGLE` symmetry. All 11 pass.

## File list

Created:
- `scripts/deploy/rainbow/__init__.py`
- `scripts/deploy/rainbow/promote.py`
- `scripts/deploy/rainbow/rollback.py`
- `backend/tests/test_rainbow_cli.py`

No code modifications to any existing file.

## Verification command output

### Immutable (from masterplan)

```
$ python scripts/deploy/rainbow/promote.py --dry-run --to green
[promote] DRY-RUN: would patch service=pyfinagent-backend to color=green
[promote] kubectl command:
  kubectl patch service pyfinagent-backend -p '{"spec": {"selector": {"color": "green"}}}'
[promote] patch JSON: {"spec": {"selector": {"color": "green"}}}

$ python scripts/deploy/rainbow/rollback.py --dry-run
[rollback] DRY-RUN: no cluster reachable; assuming current=green and rolling back to blue
[rollback] DRY-RUN: would patch service=pyfinagent-backend to color=blue
[rollback] kubectl command:
  kubectl patch service pyfinagent-backend -p '{"spec": {"selector": {"color": "blue"}}}'
[rollback] patch JSON: {"spec": {"selector": {"color": "blue"}}}
```

Both exit 0. Each prints the exact kubectl/service-selector patch it would apply.

### Unit tests

```
$ pytest backend/tests/test_rainbow_cli.py -x -q
...........
11 passed in 0.02s
```

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
90 passed, 1 skipped, 1 warning in 5.33s
```

+11 new tests. Zero regressions (prior 79p/1s preserved).

### Help output sanity

Both scripts accept `--help`:
- `python scripts/deploy/rainbow/promote.py --help` → usable argparse help, exit 0.
- `python scripts/deploy/rainbow/rollback.py --help` → same.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `scripts/deploy/rainbow/__init__.py` | PASS |
| 2 | `promote.py` with argparse `--to` + `--service` + `--dry-run`, prints patch JSON | PASS |
| 3 | `rollback.py` reads current color + toggles + `--to` override | PASS |
| 4 | Dry-run prints patch JSON to stdout | PASS |
| 5 | `--help` works | PASS |
| 6 | `test_rainbow_cli.py` ≥3 tests | PASS (11) |
| 7 | Immutable verify: both CLIs exit 0 in dry-run | PASS |

## Known caveats

1. **Dry-run rollback without a reachable cluster falls back to "blue"** as the default rollback target (assumes current=green). This is a sensible default for demonstrating the CLI without infra but means an operator demo-ing rollback from a live "blue" cluster would see a green→blue recipe that's already correct. Acceptable for phase-12.2 demo semantics; live rollback always reads the cluster.
2. **No live kubectl invocation this cycle** — no cluster available. Correctness of live path verified via monkeypatch in `test_rollback_live_reads_current_then_patches`.
3. **`_TOGGLE` covers only the 2-color MVP palette**. For the full 7-color Dimcheff rollout, operators pass `--to <previous-color>` explicitly. The helpful error message in the rollback "current not in toggle" branch guides them.
4. **Pre-Q/A self-check** re-ran the immutable command + the tests + a manual `--help` check before submission. No issues surfaced.
5. **Phase-12.1 flagged a slack-bot sentinel writer as follow-up** — that's NOT scope-overlap with 12.2; these CLIs operate on Service selectors, not app code.
