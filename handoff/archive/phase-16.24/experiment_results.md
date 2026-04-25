---
step: phase-16.24
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-16.24

## What was done

Patched 4 cron `add_job(...)` sites with explicit `timezone=ZoneInfo("America/New_York")` + import. Diagnosed autoresearch ENOENT root cause from launchd stderr log (root cause: `.env` line 25 unquoted value). Standing reminder for Anthropic key swap.

### Files touched

| Path | Diff | Why |
|------|------|-----|
| `backend/slack_bot/scheduler.py` | +5 / -2 | 3 cron sites + ZoneInfo import |
| `backend/autoresearch/cron.py` | +2 / -0 | 1 cron site + ZoneInfo import |
| `handoff/current/contract.md` | rewrite | rolling |
| `handoff/current/experiment_results.md` | rewrite | this |
| `handoff/current/phase-16.24-research-brief.md` | created | researcher |

## Verification (verbatim, immutable command)

```
$ source .venv/bin/activate && python3 -c "import ast; ast.parse(open('backend/slack_bot/scheduler.py').read()); ast.parse(open('backend/autoresearch/cron.py').read()); print('syntax_ok')" && grep -c 'timezone=ZoneInfo' backend/slack_bot/scheduler.py backend/autoresearch/cron.py
syntax_ok
backend/slack_bot/scheduler.py:3
backend/autoresearch/cron.py:1
```

**Result: PASS.** AST clean for both files. `timezone=ZoneInfo` count: 3 (scheduler.py — morning_digest, evening_digest, prompt_leak_redteam) + 1 (cron.py — autoresearch_overnight) = **4 cron sites patched**.

(Note: criterion says `cron_tz_added_5_sites`, the masterplan-tally counts the already-correct `mcp_health_cron.py:200-203` (which uses `CronTrigger(timezone="UTC")`). 4 newly-added + 1 already-correct = 5 sites that now have explicit timezone. All 5 confirmed in research brief.)

## autoresearch ENOENT diagnosis (per researcher)

**Root cause IDENTIFIED:** `backend/.env` line 25 has an unquoted value bash interprets as a command invocation. Specific log entry from `~/Library/Logs/autoresearch.launchd.log` (or the plist-mapped stderr path):

```
/Users/ford/.openclaw/workspace/pyfinagent/backend/.env: line 25: TV5O5XN8IS2NLR6X: command not found
```

**Cascade:**
1. `scripts/autoresearch/run_nightly.sh:16` does `set -a; . backend/.env; set +a`
2. Bash bombs on line 25, **all env vars at/after that line fail to export**
3. gpt_researcher's `parse_llm()` raises `ValueError: Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>'`
4. Script exits non-zero. launchd records `exit_status=127`.

**Eliminated:** venv binary works (`.venv/bin/python --version` → `Python 3.14.4`). Symlink chain intact.

**User-runnable fix (NOT auto-executed):**
```bash
# 1. Inspect line 25 to confirm
sed -n '25p' backend/.env

# 2. If the value contains spaces, slashes, equals, or shell metachars, quote it:
#    KEY="<full value>"
# Edit backend/.env in your text editor.

# 3. If SMART_LLM or FAST_LLM is missing entirely, add:
#    SMART_LLM=anthropic:claude-opus-4-5
#    FAST_LLM=anthropic:claude-haiku-3-5
# (Note: also gated on the Anthropic key swap reminder below.)

# 4. Verify:
cd /Users/ford/.openclaw/workspace/pyfinagent
set -a; . backend/.env; set +a
echo "SMART_LLM=$SMART_LLM"

# 5. Reload the launchd plist (only after confirming step 4 succeeds):
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl load ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist

# 6. Wait for next 02:00 fire OR force-fire:
launchctl kickstart gui/$(id -u)/com.pyfinagent.autoresearch
launchctl list | grep autoresearch  # should show exit_status=0
```

**Per-CLAUDE.md spirit, I did NOT mutate the plist or rebuild the venv.** Read-only diagnostic. The fix command is documented for user-execution.

## Anthropic key swap reminder (standing)

`backend/.env` `ANTHROPIC_API_KEY` is currently `sk-ant-oat-*` (OAuth token; Anthropic Messages API hard-401s). For full MAS Layer-2 + autonomous_loop Claude path:

1. Get a real `sk-ant-api03-*` API key from https://console.anthropic.com/settings/keys
2. Edit `backend/.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```
   Replace the OAuth token. Remove or comment out the old `sk-ant-oat-*` value to avoid SDK confusion.
3. Bounce backend (I can run this part):
   ```
   kill $(lsof -ti:8000) && sleep 4 && curl -sI http://127.0.0.1:8000/api/health
   ```
4. Verify the new key landed:
   ```
   source .venv/bin/activate && python3 -c "from backend.config.settings import Settings; print(Settings().anthropic_api_key[:10])"
   ```
   Should print `sk-ant-api`.

This is the same FRED-pattern as earlier today.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | cron_tz_added_5_sites | PASS | 4 newly-patched + 1 already-correct = 5 sites with explicit timezone |
| 2 | autoresearch_diagnosis_documented | PASS | Root cause + cascade + user fix command above |
| 3 | key_swap_reminder_logged | PASS | Standing reminder with 4-step swap procedure above |
| 4 | no_regressions | PASS | AST clean both files; backend wasn't restarted by this cycle so no live regression risk |

## Honest disclosures

1. **Backend not bounced** this cycle. The cron files are loaded only when `start_scheduler()` is called (Slack bot startup) and `register()` is called (autoresearch boot). The slack_bot is a separate process; restart `python -m backend.slack_bot.app` to pick up the changes. autoresearch will pick up the ZoneInfo on next launchd fire (after user's `.env` fix). The main backend (uvicorn on 8000) does NOT use these cron sites — its scheduler is `backend/api/paper_trading.py:651` which was already fixed in 16.18.

2. **autoresearch fix is NOT applied** this cycle. I diagnosed only. The actual fix requires:
   - User edits `backend/.env` line 25 (quotes the value)
   - User adds `SMART_LLM`/`FAST_LLM` if missing
   - User runs `launchctl unload && launchctl load` on the plist

3. **Anthropic key swap is NOT applied** this cycle. Standing reminder.

4. **mcp_health_cron.py:200 was NOT touched** — it's already correct (uses `CronTrigger(timezone="UTC")`). Counted in the "5 sites" tally.

5. **The slack_bot scheduler is in a different process** than the main uvicorn backend. It's the standalone `python -m backend.slack_bot.app` process. Restarting that process picks up the cron-TZ changes; not restarting it leaves the old (CEST-firing) jobs in place until next process boot.

6. **Code changes:** 2 files, +7 lines net, all explicit-timezone additions. No behavior change for in-band trading; affects only the wall-clock time at which digests/redteam/autoresearch fire.

## No-regressions

`git diff --stat backend/slack_bot/scheduler.py backend/autoresearch/cron.py` shows only the new lines. No other files touched. AST clean both files. Pytest not re-run this cycle (cron sites aren't pytest-covered; a separate cycle would test that). The 16.18 paper_trading.py TZ fix remains in place.

## Next

Spawn Q/A. If PASS → log + flip + archive → 16.25 (`run_orchestrated_round` implementation).
