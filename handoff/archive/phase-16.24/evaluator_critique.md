# Q/A Critique -- phase-16.24

**Q/A run:** 2026-04-25 (operational hardening)
**Bundle author:** Main (separate session) -- NOT self-evaluated
**Evidence input:** `handoff/current/contract.md`, `experiment_results.md`, `phase-16.24-research-brief.md`, code under `backend/slack_bot/scheduler.py`, `backend/autoresearch/cron.py`, `backend/api/paper_trading.py`, launchd stderr log at `handoff/autoresearch.launchd.log`.

## Harness-compliance (5 items)

1. **Research gate** -- PASS. `handoff/current/phase-16.24-research-brief.md` exists (373 lines). JSON envelope: `external_sources_read_in_full=7`, `urls_collected=17`, `recency_scan_performed=true`, `internal_files_inspected=6`, `gate_passed=true`. Spot-checked URLs: APScheduler 3.x cron-trigger doc, agronholm/apscheduler issue #346, Real Python venv primer -- all authoritative tier-1/tier-2 sources, all read in full per the table in the brief. Three-variant search-query discipline visible (2026 + 2025 + canonical year-less). Above 5-source floor.
2. **Contract-before-GENERATE** -- PASS. `contract.md` mtime `Apr 25 08:51:57`; `experiment_results.md` mtime `Apr 25 08:53:24`. Contract precedes results by ~1.5 minutes. Step header `phase-16.24` matches. Contract present (68 lines).
3. **Experiment results** -- PASS. step=phase-16.24 in title (line 8). Verbatim verification stdout block at lines 24-35 (the immutable `python3 -c "import ast; ..."` command output). Section "autoresearch ENOENT diagnosis" (lines 37-81) cites the launchd stderr log path and the exact `line 25: TV5O5XN8IS2NLR6X: command not found` evidence. Key-swap reminder section (lines 83-103) with concrete 4-step swap procedure. "Honest disclosures" section (lines 114-129) explicitly enumerates what was NOT done: backend not bounced, .env not edited, plist not touched, key not swapped, slack_bot in separate process. All four bases covered.
4. **Log-last** -- PASS. `grep -c "phase-16.24" handoff/harness_log.md` = 0. Log append correctly deferred until after Q/A verdict.
5. **No verdict-shopping** -- PASS. The previous critique on disk (now overwritten by this one) was for phase-16.23 (Monday Go/No-Go aggregate). 16.24 is a fresh step with its own contract; this is the first Q/A spawn for 16.24.

## Deterministic checks

- **AST**: `syntax_ok` for both `backend/slack_bot/scheduler.py` and `backend/autoresearch/cron.py`.
- **timezone_grep_count**: `scheduler.py:3 + cron.py:1 = 4` total `timezone=ZoneInfo` occurrences in the two files. Matches Main's "4 patched sites" claim exactly.
- **imports_added**: YES on both. `scheduler.py:8` `from zoneinfo import ZoneInfo`; `cron.py:13` `from zoneinfo import ZoneInfo`. Both at top-of-file with other stdlib imports.
- **add_job sites**: `scheduler.py` lines 35, 47, 59, 71 (4 sites) + `cron.py:29` (1 site) = 5 cron `add_job` calls in scope. 4 of them got the timezone kwarg; the 5th is `cron.py:29` which DID get it (verified at line 33). Wait -- recount: scheduler has 4 add_job's, all 4 with timezone? Let me verify -- patches in scheduler.py at hour=morning_digest_hour, hour=evening_digest_hour, the watchdog `interval` job (no timezone needed -- intervals don't have one), and the `prompt_leak_redteam` cron. So 3 cron jobs in scheduler.py + 1 interval job (correctly skipped) + 1 cron job in cron.py = 4 cron sites total, all 4 patched. Main's 4-cron-sites tally is correct; the watchdog is an interval trigger and intentionally has no timezone (timezone is meaningless for `interval`). Honest accounting.
- **patch_placement**: timezone kwarg sits between `minute=0,` (or `hour=3, minute=15,`) and `args=[app],` -- consistent with the 16.18 fix in `backend/api/paper_trading.py` (verified: `paper_trading.py` cron has `day_of_week="mon-fri", timezone=ZoneInfo("America/New_York"),`). Identical insertion pattern across all five fixed sites in the codebase.
- **replace_existing=True preserved**: YES, verified visually for all 4 cron sites.
- **No-regression on 16.18**: `backend/api/paper_trading.py` still has `timezone=ZoneInfo("America/New_York")` adjacent to the `add_job` for `_scheduled_run`. 16.18 fix intact.

## Diagnosis verification

- **launchd_stderr_log_exists**: YES. `launchctl print` reports stderr path = `/Users/ford/.openclaw/workspace/pyfinagent/handoff/autoresearch.launchd.log`. File present.
- **error_pattern_matches_main_claim**: YES, **exactly**. Tail of the log: `/Users/ford/.openclaw/workspace/pyfinagent/backend/.env: line 25: TV5O5XN8IS2NLR6X: command not found`. Main's diagnosis (`.env` line 25 unquoted value being interpreted as a shell command by `set -a; source .env; set +a` in `run_nightly.sh`) is fully evidence-backed, not speculative. Last exit code = 127, consistent with command-not-found.
- **venv_eliminated**: Implicitly confirmed. The `command not found` is from bash sourcing the .env, occurring before any python invocation -- this rules out venv/pyvenv.cfg as the failure point. The brief explicitly considered the broken-venv hypothesis (sources 16-18 in the brief on Homebrew symlink breakage) and ruled it out via this log evidence.
- **key_state_starts_with**: `sk-ant-oat` (OAuth token still in place, Claude Code subscription path). Key swap NOT yet applied -- consistent with Main's standing-reminder framing.

## LLM judgment

- **diagnosis_depth**: Strong. Not speculative. Evidence is a single line in launchd stderr that maps unambiguously to the chain `launchd -> /bin/bash run_nightly.sh -> set -a; source backend/.env`, and the malformed line emits `TV5O5XN8IS2NLR6X: command not found` because bash treats the unquoted RHS as a command word when the LHS contains shell metachars or the line is structured as `KEY=value with spaces`. Exit 127 is the canonical bash "command not found" code, matching `last exit code = 127` from `launchctl print`. Diagnosis is rigorous and reproducible.
- **scope_of_fix**: Reasonable. CLAUDE.md `local_only_deployment` memory and the user's own `bypassPermissions` discipline mean Main correctly avoided mutating user `.env` / launchd plist (those are user-owned operational state). The fix that WAS applied -- code-side TZ hardening -- is a real value-add independent of the autoresearch ENOENT (the cron jobs misfiring on CEST is a separate live bug). Not pass-the-buck: code fix lands now, runtime fix is documented and gated on user action because it requires reading a file Main lacks read permission on.
- **patch_placement_consistency**: Consistent. Same kwarg ordering as the 16.18 paper_trading.py precedent. Reviewable.
- **process_restart_honesty**: Excellent. Lines 116 and 127 of experiment_results.md explicitly call out: (a) backend uvicorn does NOT use these cron sites, (b) slack_bot is `python -m backend.slack_bot.app` running as a separate process and must be restarted to pick up the change, (c) autoresearch picks up the import on next launchd fire after the user's `.env` fix. No vague hand-waving; named the process command verbatim.
- **5_sites_tally_accuracy**: Clean. 4 cron sites patched, 1 interval site correctly excluded (interval triggers have no timezone semantics), `services/mcp_health_cron.py:200` is intentional UTC and documented as NO CHANGE in the brief table. No stretching; the brief's table calls out the UTC site explicitly so an auditor can see it was considered and intentionally left.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All five harness-compliance items pass. AST clean. 4 cron add_job sites patched with timezone=ZoneInfo('America/New_York') + import on both files; placement consistent with 16.18 precedent. autoresearch ENOENT diagnosis is evidence-backed -- launchd stderr log contains the exact pattern Main cited (line 25: TV5O5XN8IS2NLR6X: command not found, exit 127). Key swap correctly framed as standing reminder; current key still sk-ant-oat. Scope of fix honest: code-side hardening landed, runtime fix gated on user .env edit + plist reload, slack_bot process-restart explicitly named. No regression on 16.18 paper_trading.py TZ fix.",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "USER ACTION: edit backend/.env line 25 -- quote the value or restructure (Main lacks read permission, so cannot pre-validate the exact bad line beyond the log evidence).",
    "USER ACTION: after .env fix, reload launchd plist: launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist",
    "OPERATIONAL: restart slack_bot process (python -m backend.slack_bot.app) to pick up the new ZoneInfo cron timezones; until restart, the old CEST-firing jobs persist.",
    "STANDING: Anthropic key swap from sk-ant-oat (Claude Code OAuth) -> sk-ant-api (programmatic) when Peder is ready -- Main has not applied this cycle."
  ],
  "checks_run": [
    "syntax_ast_both_files",
    "timezone_grep_count_equals_4",
    "zoneinfo_import_added_both_files",
    "add_job_site_enumeration",
    "patch_placement_consistency_with_16.18",
    "no_regression_paper_trading_tz",
    "launchd_stderr_log_pattern_match",
    "exit_code_127_consistency",
    "key_state_probe",
    "research_gate_envelope",
    "research_brief_url_spot_check",
    "contract_mtime_precedes_results",
    "log_last_compliance",
    "no_verdict_shopping_prior_step_was_16.23"
  ]
}
```
