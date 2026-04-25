---
step: phase-10.7.1
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-10.7.1

## What was done

Shipped Alpha Velocity metric (Sharpe-slope-per-day) + BQ table migration + 6 unit tests. First step of phase-10.7 meta-evolution series. NO existing code modified.

### Files created (5 new files, ~330 LOC)

| Path | Lines | Purpose |
|------|-------|---------|
| `backend/meta_evolution/__init__.py` | 9 | Package marker + scope note (distinct from deprecated meta_coordinator.py) |
| `backend/meta_evolution/alpha_velocity.py` | 152 | `AlphaVelocitySample` dataclass + `compute_alpha_velocity()` + `persist_sample()` |
| `scripts/migrations/create_alpha_velocity_table.py` | 110 | BQ migration (--apply, --verify, --dry-run) mirroring 10.5.1 pattern |
| `tests/meta_evolution/__init__.py` | 0 | Package marker |
| `tests/meta_evolution/test_alpha_velocity.py` | 144 | 6 unit tests (positive, negative, insufficient_obs, zero_window, BQ insert, dry-run) |
| `handoff/current/contract.md` | rewrite | rolling |
| `handoff/current/experiment_results.md` | rewrite | this |
| `handoff/current/phase-10.7.1-research-brief.md` | created | researcher |

NO files modified. NO behavior changes to backend/frontend/scripts that exist.

## Verification (verbatim, immutable command)

```
$ python -m pytest tests/meta_evolution/test_alpha_velocity.py -v

============================= test session starts ==============================
platform darwin -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/ford/.openclaw/workspace/pyfinagent
plugins: langsmith-0.7.31, anyio-4.13.0
collected 6 items

tests/meta_evolution/test_alpha_velocity.py::test_positive_velocity_basic PASSED [ 16%]
tests/meta_evolution/test_alpha_velocity.py::test_negative_velocity_decay PASSED [ 33%]
tests/meta_evolution/test_alpha_velocity.py::test_insufficient_observations_returns_null PASSED [ 50%]
tests/meta_evolution/test_alpha_velocity.py::test_zero_window_days_raises PASSED [ 66%]
tests/meta_evolution/test_alpha_velocity.py::test_compute_and_insert_mocked_bq PASSED [ 83%]
tests/meta_evolution/test_alpha_velocity.py::test_migration_script_dry_run PASSED [100%]

============================== 6 passed in 0.04s ===============================
```

**Result: PASS** — 6/6 tests, 0 fails, 0 skips, 0.04s. Verification command exit 0.

## Implementation summary

### Formula (Candidate B from research brief)

```
alpha_velocity_score = (sharpe_end - sharpe_start) / window_days
```

Guards:
- `MIN_OBSERVATIONS = 20`: below floor → `score = None` (avoids spurious slopes from thin samples)
- `window_days <= 0`: `ValueError` on score property access (caller must pick non-degenerate window)
- DSR gate (Bailey & Lopez de Prado) is documented in the research brief as a downstream filter — NOT applied in this cycle (separate consumer responsibility)

### BQ schema (`pyfinagent_pms.alpha_velocity_samples`)

11 columns: strategy_id (REQUIRED), window_start (REQUIRED + partition key), window_end (REQUIRED), n_obs, sharpe_start, sharpe_end, alpha_velocity_score, window_days, macro_regime, components_json, computed_at.

Partitioned by `DATE(window_start)`. Clustered on `(strategy_id, macro_regime)`.

### Test coverage (6 cases)

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_positive_velocity_basic` | SR 1.0→1.5 over 30d = +0.0167; BQ row shape; macro_regime="EASING" passes through |
| 2 | `test_negative_velocity_decay` | SR 1.5→0.8 → negative score; "HIKING" regime |
| 3 | `test_insufficient_observations_returns_null` | n_obs=15 < 20 → score=None; row.alpha_velocity_score == None |
| 4 | `test_zero_window_days_raises` | window_start == window_end → ValueError on `.alpha_velocity_score` access |
| 5 | `test_compute_and_insert_mocked_bq` | End-to-end: compute → persist → FakeBQ.insert_rows_json called with correct table_fqn + row shape; components_json round-trips |
| 6 | `test_migration_script_dry_run` | `--dry-run` exits 0; emits `CREATE TABLE IF NOT EXISTS`, `alpha_velocity_samples`, `PARTITION BY DATE(window_start)`, `CLUSTER BY strategy_id, macro_regime` |

## Honest disclosures

1. **Migration NOT applied this cycle.** Tests use FakeBQ stub. The `--dry-run` is verified; `--apply` is NOT executed (BQ mutation requires user approval per CLAUDE.md). Operator runs `python scripts/migrations/create_alpha_velocity_table.py --apply` when ready.

2. **Computer is pure compute.** No call to `autonomous_loop.run_daily_cycle()` integration — that's deliberately deferred to phase-10.7.4 (Cron Budget Allocator wiring) per the research brief's scope note.

3. **`alpha_velocity` is a NEW term.** Not a standard metric in 2024-2026 literature. Inspired by QuantaAlpha's IC-slope (arXiv 2602.07085) and AgentEvolver's "convergence velocity" (arXiv 2511.10395). Pyfinagent has latitude to define it; we're chosen Sharpe-slope per the research brief's recommendation.

4. **No regression on existing pytest suite.** The new tests live under `tests/meta_evolution/` (root-level) and don't touch `backend/tests/`. No imports of existing modules (only `backend.meta_evolution.alpha_velocity`).

5. **Distinct from deprecated `backend/agents/meta_coordinator.py`.** The research brief flagged that file as DEPRECATED. The new package `backend/meta_evolution/` is the canonical home for phase-10.7 work.

6. **DSR not applied.** Bailey & Lopez de Prado deflation should be applied downstream (e.g., when alpha velocity is consumed by champion/challenger gate or recursive prompt optimizer), not at compute time. Documented in module docstring.

7. **`tests/meta_evolution/` is a NEW test directory.** Matches the masterplan 10.7.2-10.7.7 verification commands' expected layout.

## No-regressions

```
$ git diff --stat | head -10
 backend/meta_evolution/__init__.py                    |   9 +
 backend/meta_evolution/alpha_velocity.py              | 152 +++++++
 scripts/migrations/create_alpha_velocity_table.py     | 110 ++++
 tests/meta_evolution/__init__.py                      |   0
 tests/meta_evolution/test_alpha_velocity.py           | 144 +++++
 (handoff/ files)
```

Pure additions. No `-` removals. No imports of existing modules outside the new test file's stdlib + `backend.meta_evolution.alpha_velocity`.

## Next

Spawn Q/A. If PASS → log + flip → end of cycle.
