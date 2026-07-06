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

## 4. Post-deploy scheduled-cycle BQ rows (criterion 3) -- PENDING, wall-clock-gated

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
