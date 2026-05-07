# Phase-23.3.5 Internal Codebase Audit
# Log File Inventory — Cron Dashboard

Generated: 2026-05-07
Tier: simple

---

## 1. `_log_paths()` Re-confirmation

File: `backend/api/cron_dashboard_api.py` lines 116-124

```python
def _log_paths() -> dict[str, Path]:
    return {
        "backend":             _REPO_ROOT / "backend.log",
        "watchdog":            _REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log",
        "restart":             _REPO_ROOT / "handoff" / "logs" / "backend-restart.log",
        "harness":             _REPO_ROOT / "handoff" / "logs" / "mas-harness.log",
        "autoresearch":        _REPO_ROOT / "handoff" / "logs" / "autoresearch.log",
        "mas_harness_launchd": _REPO_ROOT / "handoff" / "logs" / "mas-harness.launchd.log",
    }
```

Six keys confirmed. Matches frontend `LOG_KEYS` in `frontend/src/app/cron/page.tsx` lines 25-32 exactly — no mismatch.

The `/logs/tail` endpoint (`cron_dashboard_api.py:203-245`) resolves keys to fixed Paths and rejects unknown keys with HTTP 400. Path traversal is mitigated by the key-lookup pattern: the client never passes a raw path.

---

## 2. CRITICAL FINDING: Split Log Files — Two Populations

The launchd plists redirect stdout/stderr to `handoff/*.log` (repo root-relative),
but `_log_paths()` resolves to `handoff/logs/*.log`. This means three of the six
allowlisted keys point to STALE copies that are no longer being written.

### Live files (written by launchd today)

| File | mtime | Size | Writer |
|------|-------|------|--------|
| `handoff/mas-harness.log` | 2026-05-07 21:52 (today) | 38 MB | `run_harness.py` stdout redirect in plist ProgramArguments |
| `handoff/ablation.log` | 2026-05-07 03:20 (today) | 12 KB | `com.pyfinagent.ablation` plist: `>> handoff/ablation.log 2>&1` |
| `handoff/ablation.launchd.log` | 2026-05-07 03:00 (today) | 3.4 KB | `com.pyfinagent.ablation` plist StandardOutPath/StandardErrorPath |
| `handoff/autoresearch.launchd.log` | 2026-05-07 02:00 (today) | 1.3 KB | `com.pyfinagent.autoresearch` plist StandardOutPath/StandardErrorPath |
| `handoff/autoresearch.log` | 2026-04-24 02:00 | 13.8 KB | `com.pyfinagent.autoresearch` plist ProgramArguments redirect; stale since exit-127 onset |
| `handoff/mas-harness.launchd.log` | 2026-04-16 00:46 | 0 B | `com.pyfinagent.mas-harness` plist StandardOutPath/StandardErrorPath (empty — harness writes its own log) |

### Stale copies in `handoff/logs/` (no longer being written)

| File | mtime | Size | Note |
|------|-------|------|------|
| `handoff/logs/mas-harness.log` | 2026-04-19 09:22 | 2.9 MB | 18d stale; the backend allowlist points HERE, not to the live 38 MB file |
| `handoff/logs/ablation.log` | 2026-04-19 03:20 | 2.3 KB | 18d stale |
| `handoff/logs/autoresearch.log` | 2026-04-19 02:00 | 4.8 KB | 18d stale |
| `handoff/logs/mas-harness.launchd.log` | 2026-04-16 00:46 | 0 B | Empty; correctly allowlisted but empty is correct |

### Root cause

`com.pyfinagent.ablation.plist` ProgramArguments:
```
cd /path/pyfinagent && ... python scripts/ablation/run_ablation.py --next-untested >> handoff/ablation.log 2>&1
```
The `>>` redirect is repo-root-relative, landing at `handoff/ablation.log`.
The plist's StandardOutPath/StandardErrorPath also write to `handoff/ablation.launchd.log` (root level).

At some point `handoff/logs/` was established as the canonical location and
`_log_paths()` was updated, but the plists were never updated. The result:
three keys in `_log_paths()` silently serve a stale or empty file instead of
the live log.

---

## 3. Frontend `LOG_KEYS` vs Backend `_log_paths()` Sync Check

`frontend/src/app/cron/page.tsx` lines 25-32:
```typescript
const LOG_KEYS: { key: string; label: string }[] = [
  { key: "backend",             label: "backend.log" },
  { key: "watchdog",            label: "backend-watchdog.log" },
  { key: "restart",             label: "backend-restart.log" },
  { key: "harness",             label: "mas-harness.log" },
  { key: "autoresearch",        label: "autoresearch.log" },
  { key: "mas_harness_launchd", label: "mas-harness.launchd.log" },
];
```

Keys match the backend allowlist exactly. No hidden logs, no phantom keys.
The UI is consistent with the backend. The problem is the backend points to the
wrong (stale) files, not a frontend/backend mismatch.

---

## 4. Non-Allowlisted Logs in `handoff/logs/`

### `ablation.launchd.log` — `handoff/logs/ablation.launchd.log`

- **Writer:** `com.pyfinagent.ablation` plist StandardOutPath/StandardErrorPath, writing to `handoff/ablation.launchd.log` (root). The `handoff/logs/ablation.launchd.log` copy (59 B, 2026-04-17) is also stale.
- **Useful?** Launchd stderr for ablation startup; only contains launchd-level errors (env load failures, exit codes). Small (59 B). Low operator value — the ProgramArguments redirect (`handoff/ablation.log`) contains the actual Python output.
- **Should allowlist?** No. Low value. The live equivalent is `handoff/ablation.launchd.log` at root, but even that is not worth adding — it duplicates data already in `handoff/ablation.log` and the ablation launchd.log files rarely contain actionable content.

### `autoresearch.launchd.log` — `handoff/logs/autoresearch.launchd.log`

- **Writer:** `com.pyfinagent.autoresearch` plist StandardOutPath/StandardErrorPath, writing to `handoff/autoresearch.launchd.log` (root). The `handoff/logs/` copy (204 B, 2026-04-17) is stale.
- **Useful?** Contains launchd-level exit codes. The 204 B entry at `handoff/logs/autoresearch.launchd.log` predates the exit-127 onset (2026-04-24). The live copy at `handoff/autoresearch.launchd.log` (1.3 KB, today) contains the current exit-127 error.
- **Should allowlist?** Possibly, but the actionable exit-127 diagnostic is already in `handoff/autoresearch.log` (live root copy). No need to add a separate launchd key.

### `seed_stability_output.log` — `handoff/logs/seed_stability_output.log`

- **Writer:** One-shot manual run from a worktree artifact (`scripts/go_live_drills/seed_stability_test.py` or `scripts/harness/run_seed_stability.py`). No launchd plist found for seed_stability. It is not a daemon log.
- **Useful?** Historical artifact from a go-live drill (mtime 2026-04-16, 1.15 MB). Not actively written. No launchd service runs it.
- **Should allowlist?** No. It is a one-shot drill artifact, not an operational log. The operator would need a fresh run to produce new output; tailing the stale copy provides no live signal.

---

## 5. Slack Bot Log Gap

Phase-23.3.2 finding confirmed: `backend/slack_bot/app.py` has no dedicated log file.
Slack bot runs as `python -m backend.slack_bot.app`; stdout/stderr go to the terminal
or are lost on daemon restart. `backend/slack_bot/self_update.py` references a
`DEPLOY_LOG = PROJECT_ROOT / "logs" / "deploy.log"` (line 37) — a path under
`backend/logs/`, not `handoff/logs/`. That file does not currently exist.

Phase-23.3.2 prescribed redirecting slack_bot stdout to `handoff/logs/slack_bot.log`
on operator restart. That has not been done yet. Until a dedicated plist or
nohup redirect is in place, there is nothing to allowlist.

---

## 6. Other Log Paths Found in the Codebase

| Path | File | Notes |
|------|------|-------|
| `PROJECT_ROOT / "logs" / "deploy.log"` | `backend/slack_bot/self_update.py:37` | DEPLOY_LOG; file does not exist; not written in current deployment |
| `/tmp/agent_sessions.log` | `scripts/spawn_agent_sessions.py:30` | Ephemeral; not suitable for ops dashboard |
| `/tmp/slack_response_agent.log` | `scripts/slack_response_agent.py:22` | Ephemeral /tmp; not suitable |
| `/tmp/imsg_responder.log` | `scripts/imsg_responder.py:21` | Ephemeral; not suitable |
| `/tmp/active_slack_monitor.log` | `scripts/active_slack_monitor.py:46` | Ephemeral; not suitable |
| `handoff/tca_log.jsonl` | `backend/services/tca.py:45` | JSONL not a plaintext log; not tail-suitable without JSON rendering |

None of these warrant allowlisting in `_log_paths()`. All are either ephemeral
`/tmp` paths, JSONL structured data, or dead-code paths with no active writer.

---

## 7. `backend.log` Size (164 MB, growing)

`backend.log` is at the repo root, 164 MB as of main session audit today.
No rotation is currently configured. At the current growth rate (~164 MB in
approximately 37 days since 2026-04-01 backend plist creation), it will reach
~1.3 GB in 90 days without intervention. See external research section for
rotation options.

---

## RECOMMENDATION

### Primary recommendation: Option (c) + path fix (adopt BOTH together)

**Step 1 — Fix the three stale paths in `_log_paths()` (required, immediate)**

The three keys that point to `handoff/logs/` must be redirected to the live
files at `handoff/` root. This is the highest-priority fix. Without it, the
Logs tab on `/cron` shows 18-day-old data for harness, ablation, and autoresearch.

Literal patch to `backend/api/cron_dashboard_api.py` lines 116-124:

```python
def _log_paths() -> dict[str, Path]:
    return {
        # ── These three write to handoff/ root (launchd plist redirect) ──
        "harness":             _REPO_ROOT / "handoff" / "mas-harness.log",
        "autoresearch":        _REPO_ROOT / "handoff" / "autoresearch.log",
        "mas_harness_launchd": _REPO_ROOT / "handoff" / "mas-harness.launchd.log",
        # ── These three write to handoff/logs/ (correct, no change) ──────
        "backend":             _REPO_ROOT / "backend.log",
        "watchdog":            _REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log",
        "restart":             _REPO_ROOT / "handoff" / "logs" / "backend-restart.log",
    }
```

Note: `mas_harness_launchd` remains pointed at `handoff/mas-harness.launchd.log`
(root) because the plist StandardOutPath writes there. That file is 0 B — the
harness writes its own log, not via launchd stdio. The key is harmless but
correct to keep for completeness.

**Step 2 — Add `last_modified_iso` to the `/logs/tail` response (recommended)**

Extend the response dict at `cron_dashboard_api.py:239-244` with:

```python
import os
mtime_iso = (
    datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    if p.exists() else None
)
return {
    "log": log,
    "lines": cleaned,
    "n_returned": len(cleaned),
    "total_size_bytes": total_size,
    "last_modified_iso": mtime_iso,   # NEW
    "exists": True,
}
```

The frontend already has a `formatRelative()` helper (page.tsx:40-56) that
renders ISO timestamps as "18d ago". Wiring `last_modified_iso` into the log
viewer label costs ~3 lines of frontend code and surfaces staleness directly
without any new API keys or UI complexity.

**Why not option (b) — expand to 9 keys?**

- `ablation` key: the live `handoff/ablation.log` is already the ablation
  job's primary output. It can be added as a 7th key but is lower-priority
  than fixing the three stale keys. The launchd file (`ablation.launchd.log`)
  is redundant once the primary log is visible.
- `slack_bot`: no dedicated log file exists yet (phase-23.3.2 pending). Cannot
  allowlist a non-existent path without a backend file-not-found guard.
- `seed_stability`: one-shot drill artifact; not an operational log.

Expansion to include `ablation` as a 7th key is reasonable but a separate
low-priority step after the path fix lands.

**Step 3 — Log rotation for `backend.log` (recommended, non-urgent)**

Apply newsyslog on macOS (see external research). A minimal
`/etc/newsyslog.d/pyfinagent.conf` entry:

```
/Users/ford/.openclaw/workspace/pyfinagent/backend.log   ford:staff  644  7  102400  *  J
```

This rotates when `backend.log` exceeds 100 MB, keeps 7 rotated copies,
uses bzip2 compression (flag `J`). newsyslog runs every 30 minutes via
launchd system agent. No Python code change needed; uvicorn reopens the
log file descriptor automatically on the next write after truncation.

Alternatively, add `RotatingFileHandler(maxBytes=100*1024*1024, backupCount=7)`
in `backend/main.py` logging config — this is the code-only path if newsyslog
configuration is not feasible.

---

## Files Inspected

| File | Lines | Role |
|------|-------|------|
| `backend/api/cron_dashboard_api.py` | 246 | Log allowlist, tail endpoint |
| `frontend/src/app/cron/page.tsx` | ~400 | LOG_KEYS frontend constant |
| `scripts/ablation/run_ablation.py` | ~40 read | Ablation runner |
| `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` | checked | Ablation plist stdout redirect |
| `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | checked | Autoresearch plist stdout redirect |
| `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` | checked | MAS harness plist stdout redirect |
| `backend/slack_bot/self_update.py` | line 37 | DEPLOY_LOG reference |
| `backend/services/tca.py` | lines 45, 92-124 | TCA_LOG_PATH |
| `scripts/harness/run_harness.py` | lines 954, 1007 | harness_log.md writer |
| `scripts/housekeeping/verify_handoff_layout.py` | lines 82-84 | Layout verifier |
| `handoff/logs/` directory | all files | Stale log population |
| `handoff/*.log` files | all files | Live log population |
