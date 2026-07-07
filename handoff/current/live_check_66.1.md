# live_check 66.1 -- Restore the decision path (2026-07-06/07)

Required shape (masterplan): "live_check_66.1.md with pytest output, drill P1
permalink, and the post-deploy scheduled-cycle BQ rows."

## 1. Pytest output (immutable command, verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q
........                                                                 [100%]
8 passed in 0.25s
```
Adjacent regression: test_phase_60_4_observability.py + test_phase_62_4_sentinel.py
-> `25 passed, 1 warning in 17.76s`.

## 2. Drill P1 (criterion 2) -- REAL paging chain, exactly once

```
resolved binary: .../scratchpad/stubbin/claude   (exit-1 stub, first on PATH)
drill_id=drill-66.1-1783377034
25 calls in 2.49s
breaker_tripped=True consecutive_failures=20 skipped_calls=5
alert bot-token fallback delivered=True source=claude_code_rail
  title='Claude Code rail breaker OPEN -- 20 consecutive failures; remaining rail calls skipped this cycle'
```
Server-side read-back (conversations.history via bot token, 62.8 doctrine):
```
ts: 1783377037.178179
text: [P1] Claude Code rail breaker OPEN -- 20 consecutive failures; remaining rail
calls skipped this cycle -- claude_code_rail: cycle_id=drill-66.1-1783377034 |
consecutive_failures=20 | threshold=20 | last_error=claude CLI exited with code 1:
drill 66.1: simulated auth failure (401) ...
permalink: https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783377037178179
```
Exactly ONE message: 25 calls produced 20 attempted failures (threshold=20 trip) + 5
skips + 1 page. The probe-gate no-double-page path is unit-covered
(test_rail_guard_no_page_when_probe_already_paged).

## 3. Deploy evidence (criterion 3 precondition)

- Import smoke test: `IMPORT_OK` (backend.main + all three changed modules).
- Code commit `27d40df5` 2026-07-07 00:31:37 +0200, pushed to origin/main.
- `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` -> single uvicorn
  instance (pgrep: 1 uvicorn + its caffeinate wrapper), lstart `Tue Jul  7 00:31:39
  2026` (+0200).
- Timing note (62.1 lesson, commit-time vs file-mtime): `git log -1` shows the
  auto-changelog trailer a854640a at 00:31:45 (hook commit, docs-only) -- the CODE
  commit 27d40df5 (00:31:37) and the changed files' mtimes (claude_code_client.py
  00:22:16, autonomous_loop.py 00:23:16) all PREDATE the 00:31:39 lstart, so the
  running process holds the deployed guard code.
- `curl http://localhost:8000/docs` -> HTTP 200.

## 4. Post-deploy scheduled-cycle BQ rows (criterion 3) -- SATISFIED 2026-07-07

Scheduled cycle `0725d2aa` fired by the scheduler at exactly 18:00:00.125 UTC
(cycle_history row below; not a manual run). BQ `pyfinagent_data.llm_call_log`,
queried 18:12 UTC (cycle still running):

```
2026-07-07 18:02:45+00 | cc_rail | claude-sonnet-4-6 | ok=True  | in=3863 out=3274 | cycle=0725d2aa
2026-07-07 18:04:49+00 | cc_rail | claude-sonnet-4-6 | ok=True  | in=3370 out=1692 | cycle=0725d2aa
2026-07-07 18:06:21+00 | cc_rail | claude-sonnet-4-6 | ok=True  | in=5218 out=2069 | cycle=0725d2aa
2026-07-07 18:06:32+00 | cc_rail | claude-sonnet-4-6 | ok=True  | in=5195 out=2897 | cycle=0725d2aa
... (12 of 37 shown)
TOTAL at 18:12 UTC: 37 rows, ok=26, fail=11
```

```
{"cycle_id": "0725d2aa", "started_at": "2026-07-07T18:00:00.125788+00:00", ..., "status": "started", ...}
```

First ok=true rail rows since 2026-06-14. NOTE (register, non-blocking): 11
ok=false rows (0 tokens) interleave with successes -- transient CLI failures at
~30% rate; the guard's success-reset design correctly kept the breaker closed
(never >=20 consecutive); transient-failure cause (concurrency/timeout) is a
follow-up diagnosis item, distinct from the credential outage class.
The launchd-context prediction (research_brief_66.2.md section 8) did NOT fire:
the credential works in the backend process context.

## 4b. Original PENDING framing (superseded by section 4 above)

The next SCHEDULED trading cycle is 2026-07-07 18:00 UTC (cycles run mon-fri 18:00;
manual runs do not count per the 39.1 scheduled-evidence lesson). The credential is
verified live (claude_code_health_probe -> (True, 'ok') post-/login), so ok=true
`agent LIKE 'cc_rail%'` rows are expected. TO BE PASTED by the closing session:

```sql
SELECT ts, agent, model, ok, input_tok, output_tok
FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
WHERE agent LIKE 'cc_rail%' AND ok = true AND DATE(ts) = '2026-07-07'
ORDER BY ts LIMIT 5
```

Until then this section is honestly PENDING -- criteria 1/2/4 closed, criterion 3
open; the step stays `pending` in the masterplan.

## 5. Criterion 4 (policy)

docs/runbooks/claude-rail-degraded-mode.md: rail-down => HOLD (fail-safe, current
behavior); Gemini fallback NOT implemented; any future implementation config-gated
DEFAULT OFF + operator token. No gate/threshold/risk-cap touched (git diff scope:
claude_code_client, autonomous_loop imports+wiring, cycle_health kwargs, settings new
field, tests, runbook).
