# Sprint Contract -- phase-12.3 Canary Split + SLO Diff Tooling

**Written:** 2026-04-19 PRE-commit.
**Step id:** `12.3` in phase-12.
**Immutable verification:** `source .venv/bin/activate && pytest backend/tests/test_rainbow_canary.py -q` — all canary tests pass.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 6, snippet_only_sources: 8, urls_collected: 14, recency_scan_performed: true, internal_files_inspected: 9, gate_passed: true}`. 3-query compliance confirmed.

Staked recs adopted:
- **Traffic split primitive**: weighted Service REPLICAS for MVP (1 green + 19 blue = ~5% canary via kube-proxy round-robin); Gateway API HTTPRoute weights flagged as future upgrade path in the plan doc.
- **SLO diff**: simple threshold ratio `green_p95 / blue_p95 > 1.2 = regression`. Matches Flagger/Argo Rollouts practitioner convention; deterministic + readable; Mann-Whitney U flagged as future upgrade path.
- **Metric source**: synthetic injection into `api_call_log` buffer for MVP tests; production path reads `pyfinagent_data.api_call_log` via BQ MCP.

## Hypothesis

A `backend/services/observability/rainbow_canary.py` module + `deploy/rainbow/canary-split.yaml` (replica-weighted canary example) + `backend/tests/test_rainbow_canary.py` (≥5 tests) lands the MVP canary-split + SLO-diff primitives without requiring a live cluster or Gateway API install.

## Success criteria

**Functional:**
1. `deploy/rainbow/canary-split.yaml` — variant of `backend-green.yaml` with `replicas: 1` (vs blue at `replicas: 19`) demonstrating the 5% split via kube-proxy round-robin. README section explains the approach.
2. `backend/services/observability/rainbow_canary.py` (new) exporting:
   - `percentile(values: list[float], p: float) -> float` — stdlib-only p50/p95/p99 helper (no scipy).
   - `SLODiff` dataclass with `{blue_p95: float, green_p95: float, ratio: float, regression: bool, threshold: float = 1.2}`.
   - `compute_slo_diff(blue_latencies: list[float], green_latencies: list[float], *, threshold: float = 1.2, min_samples: int = 10) -> SLODiff` — returns the diff + regression flag. Fail-open: empty/too-few samples → regression=False + ratio=0.0.
   - `canary_snapshot_from_buffer(blue_color_filter: Callable, green_color_filter: Callable) -> SLODiff` — pull latencies from the `api_call_log` in-process buffer, partition by color, compute diff. Color partition is caller-supplied (via source/endpoint/metadata filter) since api_call_log has no `color` column today.
3. **Design decision**: do NOT add a `color` column to `api_call_log` schema (would require BQ migration; overkill for MVP). Instead, operators tag canary traffic via `source="pyfinagent-green"` or similar and filter on that. Document in the module docstring.
4. `backend/tests/test_rainbow_canary.py` with ≥5 tests:
   - `test_percentile_basic` — p50/p95/p99 of a known series.
   - `test_compute_slo_diff_no_regression` — identical latency distributions → regression=False.
   - `test_compute_slo_diff_regression` — green 2x slower → regression=True.
   - `test_compute_slo_diff_empty_samples_no_regression` — empty → fail-open regression=False.
   - `test_compute_slo_diff_below_min_samples` — <min_samples → fail-open.
   - `test_canary_snapshot_from_buffer` — inject synthetic rows via `log_api_call`, partition by source, assert ratio.

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('backend/services/observability/rainbow_canary.py').read()); ast.parse(open('backend/tests/test_rainbow_canary.py').read()); print('ok')"`.
- Immutable: `pytest backend/tests/test_rainbow_canary.py -q` → all pass.
- Import smoke: `python -c "from backend.services.observability.rainbow_canary import compute_slo_diff, SLODiff, percentile, canary_snapshot_from_buffer; print('ok')"`.
- Regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` → 90p + 6 new = 96p/1s expected.
- Canary yaml parses: `python -c "import yaml; list(yaml.safe_load_all(open('deploy/rainbow/canary-split.yaml'))); print('ok')"`.

**Non-goals:**
- NOT installing Gateway API / Nginx Gateway Fabric (research-backed deferral).
- NOT adding scipy Mann-Whitney (simpler threshold ratio for MVP).
- NOT adding a `color` column to `api_call_log` BQ schema.
- NOT wiring to Prometheus / OpenTelemetry.
- NOT running against a real cluster.

## Plan

1. Write `backend/services/observability/rainbow_canary.py`.
2. Write `deploy/rainbow/canary-split.yaml`.
3. Update `deploy/rainbow/README.md` with a "Canary split (phase-12.3)" section.
4. Write `backend/tests/test_rainbow_canary.py`.
5. Run verification.

## Researcher agent id

`a54f8f685607c586a`
