# Phase-23.3.5 External Research Brief
# Log Rotation, Allowlist Patterns, Stale Log Surfacing

Generated: 2026-05-07
Tier: simple (assumption stated; caller specified simple)

---

## Search Queries Run (3-variant discipline)

1. **Current-year frontier:** "logrotate TimedRotatingFileHandler log rotation best practices ops dashboard 2026"
2. **Last-2-year window:** "Python logrotate newsyslog macOS log rotation 164MB backend log daemon 2025" + "stale log detection last-modified metadata freshness indicator dashboard UI pattern 2025"
3. **Year-less canonical:** "log allowlist path traversal safe tail endpoint security web application" + "logging.handlers Python RotatingFileHandler TimedRotatingFileHandler"

---

## Read in Full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://oneuptime.com/blog/post/2026-01-25-log-rotation-strategies/view | 2026-05-07 | Blog (ops platform) | WebFetch full | Tiered size thresholds: access logs 100 MB, app logs 50 MB, debug 10 MB; hybrid size+time recommended; Python TimedRotatingFileHandler preferred for app-level control |
| https://www.dash0.com/guides/log-rotation-linux-logrotate | 2026-05-07 | Official guide | WebFetch full | logrotate `maxsize` forces rotation before schedule; `copytruncate` has data-loss window; `delaycompress` prevents immediate compression of newest backup; `postrotate` SIGHUP for log reopen |
| https://docs.python.org/3/library/logging.handlers.html | 2026-05-07 | Official docs (Python) | WebFetch full | RotatingFileHandler: `maxBytes` + `backupCount`; TimedRotatingFileHandler `when`/`interval`/`atTime`; rollover is lazy (only on emit); `WatchedFileHandler` for Unix external-rotation compatibility |
| https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/ | 2026-05-07 | Blog (macOS admin) | WebFetch full | newsyslog config in `/etc/newsyslog.d/`; must specify `owner:group` matching launchd service user or post-rotation writes fail with "Permission denied"; size threshold in KB; flag `J` = bzip2 |
| https://portswigger.net/web-security/file-path-traversal | 2026-05-07 | Official docs (PortSwigger Web Security Academy) | WebFetch full | Allowlist-first defense: validate input against permitted values, then canonicalize and verify path starts with expected base directory; key-lookup pattern (never echo raw path to client) is the recommended mitigation |
| https://tacnode.io/post/what-is-stale-data | 2026-05-07 | Industry blog | WebFetch full | Surface `last_modified` timestamp; compute freshness SLA per asset type; color-coded indicators preferred over manual audits; automated alerts when freshness exceeds threshold |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://choudharycodes.hashnode.dev/python-log-rotation-a-comprehensive-guide-to-better-log-management | Blog | Covered by oneuptime.com and Python docs |
| https://tutorialedge.net/python/python-logging-best-practices/ | Blog | Generic Python logging; not specific to rotation or macOS |
| https://runebook.dev/en/docs/python/library/logging.handlers/rotatingfilehandler | Docs mirror | Python official docs fetched directly |
| https://betterstack.com/community/guides/logging/how-to-manage-log-files-with-logrotate-on-ubuntu-20-04/ | Blog | Linux-focused; macOS newsyslog more relevant here |
| https://owasp.org/www-community/attacks/Path_Traversal | OWASP | PortSwigger covers same material with implementation detail |
| https://www.apisec.ai/blog/path-traversal-in-apis-detection-and-prevention | Blog | Covered by PortSwigger |
| https://systemweakness.com/file-path-traversal-vulnerable-vs-secure-implementation-in-python-b618ddcf597d | Blog | Python-specific but PortSwigger + current code already correct |
| https://atlan.com/stale-data/ | Blog | Overlaps with tacnode.io |
| https://anomalyarmor.ai/data-freshness-monitoring/ | Blog | Focused on data pipelines, not log files |
| https://medium.com/@allclonescript/20-best-dashboard-ui-ux-design-principles-you-need-in-2025-30b661f2f795 | Blog | Too generic for this finding |

---

## Recency Scan (2024-2026)

Searched explicitly for 2026 and 2025 literature on log rotation strategies and freshness indicators. Results:

- **2026:** oneuptime.com (Jan 2026) published an updated rotation strategies guide confirming 100 MB threshold for high-volume logs and hybrid size+time approach. No superseding papers found; confirms existing practice.
- **2025:** newsyslog macOS pattern is unchanged; PortSwigger path traversal guidance updated 2025 to note container-escape extensions of path traversal — not directly relevant to pyfinagent's local-only deployment but confirms the allowlist pattern remains the correct defense.
- **No new findings supersede** the canonical Python logging docs or logrotate guidance. The 2024-2026 window confirms best practices, not replaces them.

---

## Key Findings

1. **Log rotation threshold for `backend.log`:** At 164 MB and growing, `backend.log` is already past the 100 MB threshold recommended for high-volume access logs (Source: oneuptime.com 2026, https://oneuptime.com/blog/post/2026-01-25-log-rotation-strategies/view). The recommended trigger is 100 MB with 7 rotated copies.

2. **macOS-native rotation via newsyslog:** newsyslog is the correct tool on macOS (`/etc/newsyslog.d/`). Critical: the config entry must specify `ford:staff` as owner/group to prevent "Permission denied" when uvicorn tries to write to a newly rotated file owned by root (Source: patelhiren.com, https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/). newsyslog checks size every 30 minutes via system launchd.

3. **Python-level alternative — RotatingFileHandler:** If newsyslog configuration is not available, switching `backend/main.py` to `RotatingFileHandler(maxBytes=100*1024*1024, backupCount=7)` achieves the same result without system config. Rollover is lazy (occurs on next emit after threshold, not at an exact size) (Source: Python docs, https://docs.python.org/3/library/logging.handlers.html). For external rotation compatibility on Unix, use `WatchedFileHandler` instead — it detects inode changes after external rotation and reopens the file automatically.

4. **Path traversal defense in `_log_paths()`:** The current key-lookup pattern is the correct mitigation. PortSwigger confirms: "validate the user input against a whitelist of permitted values" then "canonicalize the path" and verify it starts with the expected base directory (Source: https://portswigger.net/web-security/file-path-traversal). The current implementation at `cron_dashboard_api.py:209-214` satisfies both requirements. Phase-23.2.23 researched this; the pattern is sound and should not be changed when adding new keys.

5. **Stale log surfacing:** The canonical approach is to expose `last_modified` as a metadata field and compute age client-side, then apply color-coded or textual freshness indicators (Source: tacnode.io, https://tacnode.io/post/what-is-stale-data). The `/logs/tail` response currently returns `total_size_bytes` but not `last_modified_iso`. Adding it is a 2-line backend change; the frontend `formatRelative()` helper at `page.tsx:40-56` already handles ISO timestamps.

6. **`delaycompress` pattern:** When using logrotate/newsyslog compression, `delaycompress` is recommended to prevent compressing the most recent backup immediately — processes that haven't finished writing will still reference the original inode (Source: Dash0 logrotate guide, https://www.dash0.com/guides/log-rotation-linux-logrotate). For Python daemons using `WatchedFileHandler`, this is less critical since the handler reopens on inode change.

---

## Consensus vs Debate

**Consensus:** Key-lookup allowlist is the settled pattern for path-traversal-safe tail endpoints. No debate in 2024-2026 literature. Python's RotatingFileHandler and macOS newsyslog are both valid solutions; the choice depends on operational preference (code-only vs system config). Both are widely used.

**Debate:** newsyslog vs RotatingFileHandler for `backend.log`. newsyslog is zero-code-change but requires system config and correct owner/group. RotatingFileHandler requires a code change to `backend/main.py` logging setup but is fully self-contained. For a local-only deployment on one Mac, newsyslog is simpler operationally.

---

## Pitfalls (from Literature)

- **newsyslog owner mismatch:** If the `/etc/newsyslog.d/` entry does not specify `ford:staff`, the rotated file is created as `root:root` and uvicorn (running as `ford`) cannot write to it. Silent log loss results (Source: patelhiren.com).
- **`copytruncate` data loss window:** logrotate's `copytruncate` option has a brief window between copy and truncate where log entries can be lost. Prefer `create` + postrotate SIGHUP instead (Source: Dash0 guide).
- **Lazy rollover in RotatingFileHandler:** Rollover only triggers on the next `emit()` call after the size threshold is crossed. A burst of large writes can push the file well past `maxBytes` before rollover fires (Source: Python docs). Use `maxBytes` conservatively (e.g., 50 MB) to account for burst headroom.
- **`backupCount=0` means no deletion:** In both RotatingFileHandler and TimedRotatingFileHandler, `backupCount=0` disables deletion of old backups entirely — rotated files accumulate indefinitely. Always set `backupCount >= 1` (Source: Python docs).

---

## Application to pyfinagent (Mapped to file:line Anchors)

| Finding | File:line | Action |
|---------|-----------|--------|
| Three allowlisted keys point to stale `handoff/logs/*.log` files | `backend/api/cron_dashboard_api.py:121-123` | Redirect to `handoff/*.log` (live root files) — see internal audit patch |
| `backend.log` at 164 MB, no rotation configured | `backend/main.py` logging setup (root logger) | Add newsyslog entry or RotatingFileHandler |
| No `last_modified_iso` in tail response | `cron_dashboard_api.py:239-244` | Add `p.stat().st_mtime` to response dict |
| Frontend `formatRelative()` already handles ISO timestamps | `frontend/src/app/cron/page.tsx:40-56` | Wire `last_modified_iso` into log viewer label |
| Slack bot has no log file | `backend/slack_bot/app.py` (no FileHandler) | Phase-23.3.2 pending: redirect stdout to `handoff/logs/slack_bot.log` on next restart |
| Path traversal mitigation is correct | `cron_dashboard_api.py:203-214` | No change needed; confirm when adding new keys |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see internal audit + application table)

Soft checks:
- [x] Internal exploration covered every relevant module (cron_dashboard_api, frontend LOG_KEYS, all launchd plists, slack_bot, scripts)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim with URL

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/phase-23.3.5-external-research.md + phase-23.3.5-internal-codebase-audit.md",
  "gate_passed": true
}
```
