# Experiment Results -- phase-12.4 Rainbow Rehearsal (PHASE-12 COMPLETE)

**Step:** 12.4 — **final phase-12 step**.
**Date:** 2026-04-19.

## What was built

One file: `scripts/smoketest/rainbow_rehearsal.py` (~250 lines). Composes promote + canary + rollback into a 5-stage serial rehearsal against synthetic data. No cluster, no production code change, no new tests (the rehearsal IS the e2e validation; phase-12.2/12.3 already have unit tests).

Stages:
1. **promote dry-run** — `promote.main(["--dry-run", "--to", "green"])` via importlib; verify stdout has "DRY-RUN" + "color=green" + rc==0.
2. **canary equal latency** — 20 blue + 20 green rows @ 100ms → `canary_snapshot_from_buffer` must report `reason="ok"`, `regression=False`, ratio≈1.0.
3. **canary regression** — green @ 250ms vs blue @ 100ms → must report `regression=True`, `ratio > 2.0`.
4. **rollback dry-run** — `rollback.main(["--dry-run"])` → must emit "blue" + "DRY-RUN" with rc==0.
5. **audit + summary** — append JSONL record to `handoff/audit/rainbow_rehearsal.jsonl`, print JSON summary.

## File list

Created:
- `scripts/smoketest/rainbow_rehearsal.py`

Runtime artifact:
- `handoff/audit/rainbow_rehearsal.jsonl` (append-only; 1 row written this cycle)

No modifications to any existing file.

## Verification command output

### Immutable (masterplan): verify docs/VERTEX_AI_GENAI_MIGRATION.md references rainbow

```
$ grep -q 'rainbow' docs/VERTEX_AI_GENAI_MIGRATION.md && echo ok
ok
```

Already satisfied from phase-11.0; no change needed here.

### Live rehearsal end-to-end

```
$ python scripts/smoketest/rainbow_rehearsal.py
{
  "ts": "2026-04-19T14:07:17.191220+00:00",
  "dry_run": true,
  "stages": [
    {"name": "promote_dry_run", "ok": true, "rc": 0, "has_dry_run_line": true, "has_color_green": true},
    {"name": "canary_equal_latency", "ok": true, "reason": "ok", "regression": false, "ratio": 1.0, "blue_samples": 20, "green_samples": 20},
    {"name": "canary_regression", "ok": true, "reason": "ok", "regression": true, "ratio": 2.5, "threshold": 1.2},
    {"name": "rollback_dry_run", "ok": true, "rc": 0, "has_dry_run_line": true, "has_blue": true}
  ],
  "overall_ok": true
}
```

All 5 stages PASS. `overall_ok: true`. Exit 0.

### Audit JSONL

```
$ test -f handoff/audit/rainbow_rehearsal.jsonl && python -c "import json; lines = open('handoff/audit/rainbow_rehearsal.jsonl').readlines(); print('lines:', len(lines)); print('last overall_ok:', json.loads(lines[-1])['overall_ok'])"
lines: 1
last overall_ok: True
```

### Syntax

```
$ python -c "import ast; ast.parse(open('scripts/smoketest/rainbow_rehearsal.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
103 passed, 1 skipped, 1 warning in 5.94s
```

Unchanged from phase-12.3. Zero test-surface changes this cycle (the rehearsal script is standalone, not a pytest target).

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `scripts/smoketest/rainbow_rehearsal.py` with 5 stages | PASS |
| 2 | `--dry-run` flag (inherently true; retained for shape parity) | PASS |
| 3 | Exit 0 on all stages PASS | PASS |
| 4 | No production module changes | PASS |
| 5 | Audit JSONL row shape matches phase-6.8 pattern | PASS |

## Known caveats

1. **Rehearsal is synthetic-only** — no kubectl, no real pod, no real BQ. Operators should still do a `kubectl apply --dry-run=server` against a live cluster before any real migration.
2. **`--dry-run` flag is cosmetic** — the whole rehearsal is inherently dry-run (no side effects beyond the audit JSONL row). Kept for CLI shape parity with `phase6_e2e.py`.
3. **5-stage list is hardcoded** — adding a new stage requires editing the orchestrator. For a more extensible rehearsal, a `stages: [Callable]` list could be passed in. Out of scope for MVP.
4. **No pytest integration test** — by design per the researcher rec. The smoketest IS the e2e validation.
5. **Pre-Q/A self-check** — ran the rehearsal live + verified the JSONL write + re-ran the full regression before submitting. Found nothing new.

## Phase-12 closure

With this step:
- **5/5 phase-12 steps done** (12.0 audit+plan → 12.1 manifests+README → 12.2 promote/rollback CLI → 12.3 canary SLO diff → 12.4 end-to-end rehearsal).
- `deploy/rainbow/` has 5 yaml + README (`backend-blue`, `backend-green`, `backend-service`, `slack-bot-blue`, `canary-split`, `README.md`).
- `scripts/deploy/rainbow/` has promote.py + rollback.py + __init__.
- `scripts/smoketest/rainbow_rehearsal.py` is the preflight drill.
- `backend/services/observability/rainbow_canary.py` owns the SLO-diff math.
- **24 new tests** across phase-12 (11 CLI + 13 canary).
- **103 passing** total across phase-3 + phase-6 + phase-11 + phase-12; zero regressions across every phase-12 cycle.
- **Rainbow Deploys pattern is ready** for the next risky SDK bump or model-version flip; operators have a preflight drill + the CLI + the manifests + the SLO-diff math.
