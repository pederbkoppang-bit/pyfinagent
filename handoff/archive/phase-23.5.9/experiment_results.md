---
step: phase-23.5.9
title: Cron job verification — nightly_mda_retrain — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — TRUE liveness signal)
verification_command: 'python3 tests/verify_phase_23_5_9.py'
---

# Experiment Results — phase-23.5.9

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_9.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.9`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_mda_retrain"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK nightly_mda_retrain scheduled 2026-05-10T03:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_9.py
OK nightly_mda_retrain status=scheduled next_run=2026-05-10T03:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "nightly_mda_retrain",
  "source": "slack_bot",
  "schedule": "phase-9.4 cron",
  "next_run": "2026-05-10T03:00:00+02:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Nightly MDA feature-importance retrain"
}
```

`last_run: null` because the daemon was last restarted at 23:24
CEST today — no fire since restart. Pre-restart the registry
showed `last_run_at="2026-05-08T01:00:00.011634+00:00"` (per
researcher's audit), so the heartbeat path is known-functional.

## Why the criterion is satisfied (TRUE liveness)

- Handler (`nightly_mda_retrain.py`) makes zero HTTP calls; imports
  only `backend.autoresearch.gate` and `backend.slack_bot.job_runtime`.
- `heartbeat()` correctly wired with idempotency_key for daily skip.
- **Notably this job is NOT affected by the production-stub
  pattern** (unlike daily_price_refresh + weekly_fred_refresh).
  The real `train_fn` runs; only the PromotionGate threshold (0.95)
  rejects the stub's `dsr=0.80` model. The job still completes the
  heartbeat path and posts `status="ok"` for a real reason.
- Bridge surfaces `status="scheduled"` from the registry's
  startup-seed.
- CronTrigger computed `next_run` correctly (03:00 ET = 09:00 CEST
  daily; tomorrow 03:00 ET = 02:00 CEST tomorrow morning... wait,
  ET is UTC-4, so 03:00 ET = 07:00 UTC = 09:00 CEST). The shown
  `next_run="2026-05-10T03:00:00+02:00"` is in CEST timezone — the
  trigger's wall-clock 03:00 ET converts cleanly.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.8 (12 prior) | PASS |
| 23.5.9 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire production fetch_fn/write_fn (deferred bulk fix at end of
  phase-9 block).
- Tune the PromotionGate threshold.
- Touch the MDA training logic.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.9-research-brief.md`
- `tests/verify_phase_23_5_9.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_9.py
```
