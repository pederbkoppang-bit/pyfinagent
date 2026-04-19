# Experiment Results -- phase-12.3 Canary Split + SLO Diff

**Step:** 12.3 (4th of phase-12).
**Date:** 2026-04-19.

## What was built

- `backend/services/observability/rainbow_canary.py` (~175 lines, stdlib only): `percentile()`, `SLODiff` dataclass, `compute_slo_diff()` with threshold-ratio regression detection + fail-open rules, `canary_snapshot_from_buffer()` partitioning `api_call_log` rows by caller-supplied predicate.
- `deploy/rainbow/canary-split.yaml` — green Deployment with `replicas: 1`; kube-proxy round-robin achieves 5% canary when paired with blue at `replicas: 19` and Service selector widened to match both colors.
- `backend/tests/test_rainbow_canary.py` — 13 tests (percentile edge cases; SLO diff happy + empty + insufficient-samples + green-faster + threshold-tunable + partition-by-source + fail-open-empty-buffer).

No scipy dep added. No new BQ columns. No new API. Pure library + one yaml + tests.

## Key design commitments

- **Traffic split via replica weighting** for MVP (`1/(1+19) = 5%`). Gateway API HTTPRoute weights flagged as future upgrade in canary-split.yaml comments.
- **Threshold ratio** `green_p95 / blue_p95 > 1.2 = regression`. Tunable via `threshold=` kwarg; 1.2 default matches Flagger/Argo practitioner convention.
- **Fail-open everywhere**: empty samples → regression=False (reason="empty"); below `min_samples=10` → regression=False (reason="insufficient_samples"). A missing-data window never auto-triggers a rollback.
- **No `color` column added to `api_call_log`**: operators tag canary traffic via `source="pyfinagent-green"` (or any scheme) and pass filter predicates to `canary_snapshot_from_buffer`. Module docstring documents this explicitly.

## File list

Created:
- `backend/services/observability/rainbow_canary.py`
- `deploy/rainbow/canary-split.yaml`
- `backend/tests/test_rainbow_canary.py`

Modified: none. Zero backend behavior changes; zero schema changes.

## Verification command output

### Immutable (from masterplan)

```
$ source .venv/bin/activate && pytest backend/tests/test_rainbow_canary.py -q
.............
13 passed in 0.02s
```

### Syntax + yaml parse + import smoke

```
$ python -c "import ast; ast.parse(open('backend/services/observability/rainbow_canary.py').read()); ast.parse(open('backend/tests/test_rainbow_canary.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -c "import yaml; list(yaml.safe_load_all(open('deploy/rainbow/canary-split.yaml'))); print('yaml ok')"
yaml ok

$ python -c "from backend.services.observability.rainbow_canary import compute_slo_diff, SLODiff, percentile, canary_snapshot_from_buffer; print('ok')"
ok
```

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
103 passed, 1 skipped, 1 warning in 5.28s
```

+13 new (canary). Prior 90p/1s preserved. Zero regressions.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `canary-split.yaml` with replica-weighted green | PASS |
| 2 | `rainbow_canary.py` exports percentile + SLODiff + compute_slo_diff + canary_snapshot_from_buffer | PASS |
| 3 | No color column added to api_call_log; caller-predicate partitioning | PASS (module docstring documents) |
| 4 | ≥5 tests | PASS (13) |
| 5 | Canary yaml parses | PASS |

## Known caveats

1. **Replica weighting coarse granularity**: 1/20 ≈ 5% is the smallest split you get without Gateway API. For 1% canary, add 99 blue pods (resource-hungry) or upgrade to HTTPRoute weights. Flagged as future.
2. **No automated promotion / rollback decisions**: `rainbow_canary.py` emits a SLODiff; an operator or a higher-level orchestrator reads `.regression` and decides. Building the orchestrator loop (Flagger-analog) is a phase-12.4-adjacent future step.
3. **Buffer read is in-process**: `canary_snapshot_from_buffer` reads the live `_buffer` list — useful when the backend pod is also running the observability stack (which it is). For multi-pod SLO diff, query BQ `pyfinagent_data.api_call_log` directly (the module docstring documents this).
4. **No actual kube-proxy test**: the 5% split claim relies on Kubernetes' documented round-robin behavior; we don't simulate kube-proxy in tests. Correctness of the MATH is covered (SLO diff happy + fail-open paths).
5. **Pre-Q/A self-check**: ran the immutable + full regression + yaml parse + import smoke. Found nothing new. Verified the `_buffer` read in `canary_snapshot_from_buffer` uses `row.__dict__` for the dataclass-backed buffer rows (api_call_log.py:32 `@dataclass` `_ApiCallRow`).
