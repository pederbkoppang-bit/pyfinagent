---
name: auth-latch-85-3
description: Step 85.3 away-ops auth alarm — a latch clear path that EXISTS but is unreachable via if/elif ordering; a recovery probe that gates on exit code instead of the failure signal; health.jsonl streak claims need re-derivation across rotation gaps
metadata:
  type: project
---

Away-ops auth alarm stuck ok:false for 28 days. Three failure CLASSES worth
carrying forward — each one refuted or extended a stated step premise.

**1. A latch's clear path can EXIST and be unreachable through branch ordering.**
The step asserted the only `cleared_at` writer was `run_away_session.sh:151`.
False — `healthcheck.sh:164` writes it too, inside an `elif`. The preceding
`if [ "$auth_ok" = "false" ]` always won, so the clear arm never ran, so
`cleared_at` stayed None, so `auth_ok` stayed false. Self-sustaining inside ONE
file.
**Why:** when a step says "there is no exit path", grep for the writer anyway —
the difference between *absent* and *unreachable* changes the whole fix (make an
existing path reachable vs. build a new mechanism), and it changes which
criterion is already half-satisfied.
**How to apply:** for any latch/incident/circuit-breaker audit, enumerate ALL
writers of the clear field FIRST, then ask separately whether each is reachable
under the stuck state. Trace the if/elif chain, not just the predicate.

**2. A recovery probe that gates on process exit code is a false-negative test.**
`run_away_session.sh:150` required `probe_rc -eq 0` AND no 401. A HEALTHY
credential returned rc=1 (`subtype: error_max_turns` — `printf 'ping'` with
`--max-turns 1` makes the model reach for a tool, `stop_reason: tool_use`) or
rc=124 (gtimeout 20s vs a measured 50,373-token cache-creation prompt, because
`claude -p` runs cwd=repo and reloads CLAUDE.md every call). The 401 leg was
clean the whole time. Never once rc=0 in 28 days.
**Why:** exit code conflates "the tool ran badly" with "the thing being tested is
broken". The probe had the RIGHT signal available (401 absence) and threw it away
by AND-ing an unrelated one.
**How to apply:** whenever a health probe wraps a CLI, read a real captured
output artifact before trusting the rc gate. Gate on the domain signal, and size
any timeout against the measured prompt-cache cost, not a guess.

**3. Never assert a health-log streak without re-deriving across rotation.**
Step said "every record in health.jsonl carries ok:false". Measured: 34 of 509
are ok:true; the streak is 473 records over 9.85 days, not 28 — the watchdog was
itself dead 07-06..07-28 and `scripts/ops/rotate_logs.sh` truncates the file.
Also found one structurally INVALID JSON record (`"api_health":000000`, from
curl printing `000` AND `|| echo 000` firing) which silently kills the whole
Slack digest health section via a bare `except: pass`.
**How to apply:** parse the log and count; check for a rotation authority and a
gap in timestamps before quoting any "N days of X" figure.

Related: [[project_away_watchdog_p1_path]] (same watchdog, P1 deduper threshold),
[[feedback_measure_dont_assert_claims]], [[reference_vacuous_type_guards_on_bq_string_columns]].
