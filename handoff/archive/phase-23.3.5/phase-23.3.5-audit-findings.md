# Phase-23.3.5 audit findings — Log file inventory

**Cycle date:** 2026-05-07
**Scope:** every `*.log` file in `handoff/` and `handoff/logs/`, plus
the `/api/logs/tail` allowlist + frontend dropdown.

## Verdict: PASS WITH FIX + 3 OPERATOR-FIX-REQUIRED

Two structural bugs found and fixed:
1. **/cron Logs tab was showing 18-21-day stale duplicates** for
   3 of 6 keys (mas-harness, autoresearch, mas-harness.launchd). The
   live launchd-managed services write to `handoff/<x>.log` (repo
   root) but the allowlist pointed at `handoff/logs/<x>.log`.
2. **3 useful logs were not allowlisted at all**: ablation.log,
   ablation.launchd.log, autoresearch.launchd.log.

Plus three operator-fix-required findings surfaced via the now-correct
launchd logs.

## What was changed

```diff
 # backend/api/cron_dashboard_api.py:_log_paths()
 return {
     "backend":               _REPO_ROOT / "backend.log",
     "watchdog":              _REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log",
     "restart":               _REPO_ROOT / "handoff" / "logs" / "backend-restart.log",
-    "harness":               _REPO_ROOT / "handoff" / "logs" / "mas-harness.log",
-    "autoresearch":          _REPO_ROOT / "handoff" / "logs" / "autoresearch.log",
-    "mas_harness_launchd":   _REPO_ROOT / "handoff" / "logs" / "mas-harness.launchd.log",
+    "harness":               _REPO_ROOT / "handoff" / "mas-harness.log",
+    "autoresearch":          _REPO_ROOT / "handoff" / "autoresearch.log",
+    "mas_harness_launchd":   _REPO_ROOT / "handoff" / "mas-harness.launchd.log",
+    "autoresearch_launchd":  _REPO_ROOT / "handoff" / "autoresearch.launchd.log",
+    "ablation":              _REPO_ROOT / "handoff" / "ablation.log",
+    "ablation_launchd":      _REPO_ROOT / "handoff" / "ablation.launchd.log",
 }

 # frontend/src/app/cron/page.tsx:LOG_KEYS  --  6 entries -> 9 entries (mirrored)
```

Live verification post-restart:
- `curl /api/logs/tail?log=harness` -> 38 MB live file (was 2.9 MB stale).
- `curl /api/logs/tail?log=autoresearch_launchd` -> exit-127 errors visible.
- `curl /api/logs/tail?log=ablation_launchd` -> exit-127 errors visible.

## Three OPERATOR-FIX-REQUIRED findings (`backend/.env`)

The now-correct launchd logs surface THREE distinct `.env` parsing
bugs that have been failing the nightly autoresearch + ablation jobs
silently:

```
backend/.env: line 24: [REDACTED-phase-23.3.7]: command not found
backend/.env: line 25: [REDACTED-phase-23.3.7]: command not found
backend/.env: line 56: sk-ant-api03-fx4D_..._RQAA: command not found
```

Pattern: leading space after `=` causes bash (`set -e`; `set -a;
. backend/.env; set +a`) to tokenise the value as a standalone
command. Phase-23.3.4 already documented line 24; lines 25 and 56
are NEW findings from this phase.

### Operator fix sequence (cannot run from this Claude Code session — sandbox blocks .env)

```bash
# Inspect the broken lines first:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Surgical fix: collapse the leading space after `=` on broken lines.
# The pattern is `KEY= value` -> `KEY=value`. sed handles all 3 in one go:
sed -i '' '24s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '25s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '56s/^\([A-Z_]*\)= /\1=/' backend/.env

# Verify:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Recovery (force the next nightly to pick up the fix immediately):
launchctl bootout gui/501/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl bootout gui/501/com.pyfinagent.ablation 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.ablation.plist
launchctl kickstart gui/501/com.pyfinagent.autoresearch
launchctl kickstart gui/501/com.pyfinagent.ablation
sleep 5
launchctl list | grep -E "(autoresearch|ablation)"   # both should be exit 0
```

## Sibling concerns deferred

- Log rotation for `backend.log` (164 MB, growing). Apple
  `newsyslog.d` rules or a daily rotation script. P2 follow-up.
- `slack_bot.log` allowlist key — depends on phase-23.3.2 operator
  restart actually writing the file. Add when the file exists.
- `seed_stability_output.log` (1.15 MB, 2026-04-16) — audit-only
  artifact, not surfaced to operator.

## Verification

- `python tests/verify_phase_23_3_5.py` -> 6/6 OK including 2 live
  HTTP probes.
- `pytest tests/services/test_log_path_allowlist.py -q` -> 6 passed.
- `tsc --noEmit` clean (frontend LOG_KEYS in sync with backend
  allowlist, both at 9 keys).

## Q/A

Per same-session pragmatism: deterministic verifier is canonical
gate. The three operator-fix .env line numbers are surfaced as
explicit bash recovery commands.
