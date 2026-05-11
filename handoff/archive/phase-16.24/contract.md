---
step: phase-16.24
title: Operational hardening (cron TZ + autoresearch ENOENT diag + Anthropic key reminder)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.24

## Research-gate summary

`handoff/current/phase-16.24-research-brief.md`. tier=simple, 7 in-full, 17 URLs, recency scan, gate_passed=true.

## Research findings load-bearing for plan

1. **4 broken cron sites confirmed** with exact before/after patches:
   - `backend/slack_bot/scheduler.py:34-42` morning_digest
   - `backend/slack_bot/scheduler.py:45-53` evening_digest
   - `backend/slack_bot/scheduler.py:68-75` prompt_leak_redteam
   - `backend/autoresearch/cron.py:28-34` autoresearch_overnight
   - Both files need `from zoneinfo import ZoneInfo` import added (zoneinfo is stdlib in 3.9+).

2. **autoresearch ENOENT exit=127 root cause IDENTIFIED** (not venv corruption!):
   - **`.env` line 25 has unquoted value** that bash interprets as a command. Specific log entry: `/Users/ford/.openclaw/workspace/pyfinagent/backend/.env: line 25: [REDACTED-phase-23.3.7]: command not found`.
   - When `run_nightly.sh:16` does `set -a; . backend/.env; set +a`, it bombs at line 25 and **all env vars at/after that line are NOT exported**.
   - Cascades to gpt_researcher's `parse_llm()` raising `ValueError: Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>'`.
   - Fix is a user-runnable `.env` quote fix — NOT a code change. I document the fix command; user runs it.

3. **mcp_health_cron.py:200 is already correct** (uses `CronTrigger(timezone="UTC")` — different API, deliberate UTC choice). Do NOT touch.

4. **Anthropic key still oat-***. Reminder stays in 16.24 notes; user action separate.

## Hypothesis

Patching 4 cron sites with `timezone=ZoneInfo("America/New_York")` and adding the import to both files makes all 5 (4 + autoresearch's own cron) explicit-TZ. Documenting the autoresearch ENOENT diagnosis in handoff_log gives the user a runnable fix path. Backend bounce so APScheduler picks up the changes. Verification command grep counts >=4.

## Success Criteria (verbatim, immutable)

```
source .venv/bin/activate && python3 -c "import ast; ast.parse(open('backend/slack_bot/scheduler.py').read()); ast.parse(open('backend/autoresearch/cron.py').read()); print('syntax_ok')" && grep -c 'timezone=ZoneInfo' backend/slack_bot/scheduler.py backend/autoresearch/cron.py
```

- cron_tz_added_5_sites
- autoresearch_diagnosis_documented
- key_swap_reminder_logged
- no_regressions

## Plan steps

1. Edit `backend/slack_bot/scheduler.py`: add `from zoneinfo import ZoneInfo` import + `timezone=ZoneInfo("America/New_York"),` arg in 3 cron `add_job(...)` calls
2. Edit `backend/autoresearch/cron.py`: add `from zoneinfo import ZoneInfo` import + same arg in 1 cron `add_job(...)` call
3. Syntax-check both files (verification command stage 1)
4. Grep-count `timezone=ZoneInfo` should return at least 4 hits (3 in scheduler + 1 in cron) (verification stage 2)
5. Bounce backend so APScheduler picks up changes (slack_bot scheduler may need separate restart)
6. Document autoresearch fix command in experiment_results
7. Document Anthropic key reminder in experiment_results
8. Spawn Q/A

## What Q/A must audit

1. 4 cron sites patched correctly (re-grep + read patch context)
2. Imports added cleanly, no shadowing
3. Patches don't break ASTparser (re-run syntax check)
4. autoresearch diagnosis is honest + actionable (specific quote-fix recommendation)
5. Key swap reminder explicit
6. No production-code-path regressions (the cron sites are scheduling, not trading-decision logic)
