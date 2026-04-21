# Contract — phase-9.1 heartbeat + idempotency primitives.

Immutable: ast.parse(backend/slack_bot/job_runtime.py) && pytest tests/slack_bot/test_job_runtime.py -q.

Plan: job_runtime.py with IdempotencyStore + IdempotencyKey helpers (daily/weekly/hourly) + `heartbeat(job_name, idempotency_key, sink)` context manager. Failed runs do NOT mark idempotent (retry-safe).
