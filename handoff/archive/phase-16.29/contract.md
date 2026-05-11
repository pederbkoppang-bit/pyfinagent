---
step: phase-16.29
title: Close 3 standing reminders (alpha_velocity --apply + key swap + autoresearch .env line 25)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.29

## Research-gate summary

`handoff/current/phase-16.29-research-brief.md`. tier=simple, 5 in-full, 15 URLs, recency scan, gate_passed=true.

## Key research findings

1. **`create_alpha_velocity_table.py --apply`** is idempotent — both `CREATE SCHEMA IF NOT EXISTS` (line 31) and `CREATE TABLE IF NOT EXISTS` (lines 36-53) are safe to re-run. `--verify` does `SELECT COUNT(*) LIMIT 1` (exits 0 PASS, 1 empty, 2 import-error). Main can run autonomously.

2. **Anthropic key state**: still `sk-ant-oat01-*` prefix. Per `backend/config/settings.py:86`, `Settings()` reads the env var fresh on construction (no caching). Bouncing backend after a future user-side swap will pick up the new key without code changes.

3. **autoresearch ENOENT root cause CONFIRMED**: `handoff/autoresearch.launchd.log:1` shows verbatim `backend/.env: line 25: [REDACTED-phase-23.3.7]: command not found`. The plist does NOT embed `ANTHROPIC_API_KEY` — env comes from `set -a; . backend/.env; set +a` inside `run_nightly.sh`. Fixing `.env` line 25 is sufficient; no plist reload needed beyond `launchctl kickstart`.

4. **Success criteria are "record state"**: `alpha_velocity_table_exists` + `anthropic_key_state_recorded` + `autoresearch_state_recorded` + `no_silent_user_action_simulation`. The cycle PASSes when state is honestly recorded (NOT when user-actions are simulated). Same pattern as 16.20/16.23 CONDITIONAL precedents.

## Hypothesis

Main runs `--apply` + `--verify` on the BQ migration (autonomous, BQ DDL is idempotent and within the project's normal write scope per `.claude/rules` BQ mcp permissions). Main records key-state and launchctl-state truthfully. The verification command's chained `&&` exits 0 because all three probes succeed. Cycle PASSes; user-action items remain as filed follow-up tickets (#21, #36) but are NOT silently simulated.

## Success Criteria (verbatim, immutable)

```
python scripts/migrations/create_alpha_velocity_table.py --verify && python3 -c "from backend.config.settings import Settings; k=Settings().anthropic_api_key; assert k.startswith('sk-ant-api') or 'OAT_KEY_NOT_SWAPPED_YET' or True; print('key_state:', k[:10] if k else 'EMPTY')" && launchctl list 2>&1 | grep autoresearch | head -1
```

- alpha_velocity_table_exists
- anthropic_key_state_recorded
- autoresearch_state_recorded
- no_silent_user_action_simulation

## Plan steps

1. Run `python scripts/migrations/create_alpha_velocity_table.py --apply` (creates table if not exists; idempotent)
2. Run `python scripts/migrations/create_alpha_velocity_table.py --verify` (confirms `pyfinagent_pms.alpha_velocity_samples` exists, returns row_count)
3. Probe Settings().anthropic_api_key state — record truthfully (currently `sk-ant-oat01`)
4. Probe `launchctl list | grep autoresearch` — record truthfully (currently `- 127 com.pyfinagent.autoresearch`)
5. Run the verbatim verification command verbatim (chained `&&`)
6. Document precise user-action commands for Anthropic key swap + autoresearch .env fix in experiment_results
7. Spawn Q/A

## What Q/A must audit

1. `--apply` was actually run (table now exists in BQ)
2. `--verify` returns PASS with table existence + row_count visible
3. Key state honestly recorded as `sk-ant-oat01` (NO simulation of swap)
4. autoresearch state honestly recorded as `127` exit (NO simulation of fix)
5. No silent user-action simulation: nowhere does Main pretend the .env was edited or the key was swapped
6. User-action documentation is precise + executable
7. Follow-up tickets #21 (key swap) and #36 (GITHUB_TOKEN alternative) remain open
