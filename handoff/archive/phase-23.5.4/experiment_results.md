---
step: phase-23.5.4
title: Cron job verification — evening_digest — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — no false-positive caveat)
verification_command: 'python3 tests/verify_phase_23_5_4.py'
---

# Experiment Results — phase-23.5.4

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_4.py` — replayable verifier mirroring
   23.5.3 / 23.5.2.5 pattern.

## Verification command — verbatim from `.claude/masterplan.json::23.5.4`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="evening_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result (run 2026-05-09)

```
$ <verbatim immutable command>
OK evening_digest scheduled 2026-05-09T17:00:00-04:00
EXIT=0

$ python tests/verify_phase_23_5_4.py
OK evening_digest status=scheduled next_run=2026-05-09T17:00:00-04:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "evening_digest",
  "source": "slack_bot",
  "schedule": "cron daily evening_digest_hour:00 ET",
  "next_run": "2026-05-09T17:00:00-04:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Slack evening digest (P&L + closed trades)"
}
```

`last_run` is null because the slack-bot daemon was last restarted
at 09:49 today (PID 63639 from 23.5.3.1's deploy). Evening digest
fires at 5 PM ET (= 23:00 CET) so the next fire is later today.

## Why the criterion is satisfied (and is a TRUE liveness signal)

Researcher confirmed the four-link chain is closed:

1. **httpx call** uses `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"`
   (`scheduler.py:247, 250`) — Docker-alias bug fixed in 23.5.3.1.
2. **`/api/paper-trading/trades?limit=10`** returns trades data
   from a healthy backend (port 8000 confirmed up).
3. **`format_evening_digest`** produces 5-6 blocks max
   (formatters.py:354-400). Section text < 3000 chars per block;
   no Slack size-limit risk.
4. **`chat.postMessage`** accepts the payload normally.
5. **`EVENT_JOB_EXECUTED`** fires on genuine completion → heartbeat
   listener records `status="ok"` for a real reason (no fail-open
   exception swallowing).

Unlike phase-23.5.3, this verification is NOT a structural false
positive.

## Adjacent finding (NOT a regression, NOT in scope)

Researcher's brief surfaced one residual concern: `chat.postMessage`
has no native idempotency key. In a daemon-restart-near-fire-time
race, theoretical double-send is possible. This is a known
architectural limitation of the single-instance local deployment
(not introduced by any recent phase). Researcher cited the canonical
Redis SET NX dedup pattern as the mitigation. Out of scope for
verification.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| `tests/verify_phase_23_5_1.py` | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2.py` | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2_5.py` | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2_6.py` | PASS (4/4), EXIT=0 |
| `tests/verify_phase_23_5_3.py` | PASS, EXIT=0 |
| `tests/verify_phase_23_5_3_1.py` | PASS (4/4), EXIT=0 |
| `tests/verify_phase_23_5_4.py` | PASS, EXIT=0 (this step) |

## What this step does NOT do

- Add idempotency for `chat.postMessage` (out of scope).
- Touch the digest handlers or formatters.
- Investigate the 14 sibling jobs.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.4 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.4-research-brief.md` — researcher.
- `tests/verify_phase_23_5_4.py` — replayable verifier.

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_4.py
```
