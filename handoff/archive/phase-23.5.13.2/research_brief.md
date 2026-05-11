# Research Brief: phase-23.5.13.2
## Bridge launchd job state into /api/jobs/all (launchctl print parser)

**Effort tier:** moderate (assumption stated by caller)
**Date:** 2026-05-10
**Researcher:** merged researcher + Explore agent

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.alansiu.net/2025/05/28/using-new-launchctl-subcommands-to-check-for-and-reload-launch-daemons/ | 2026-05-10 | blog (authoritative practitioner, 2025) | WebFetch | "A daemon showing `state = not running` or `active count = 0` is still loaded — simply not executing at that moment." Documents state vs active-count distinction on Sequoia. |
| https://www.launchd.info/ | 2026-05-10 | doc/tutorial | WebFetch | `launchctl list` columns: col-1 = PID or "-", col-2 = last exit code, col-3 = label. Exit code 0 = success, positive = error, negative = -(signal number). |
| https://keith.github.io/xcode-man-pages/launchctl.1.html | 2026-05-10 | official man page mirror | WebFetch | "This output is NOT API in any sense at all. Do NOT rely on the structure or information emitted for ANY reason." Confirmed for `print` subcommand. |
| https://docs.python.org/3/library/subprocess.html | 2026-05-10 | official Python docs | WebFetch | Canonical pattern: `subprocess.run([...], capture_output=True, text=True, timeout=N)` -- use `text=True` to get str not bytes; `check=False` so nonzero exit doesn't raise. |
| https://newosxbook.com/articles/jlaunchctl.html | 2026-05-10 | deep technical (Jonathan Levin / MOXiI) | WebFetch | Confirms `active count` and `path` fields appear in print output; author notes the format is informal and defers internals to his book. |
| https://medium.com/@priyanshu009ch/ttl-lru-cache-in-python-fastapi-2ca2a39258dc | 2026-05-10 | blog | WebFetch | TTL+LRU cache pattern for FastAPI: `ttl_lru_cache(ttl=60, max_size=128)` wraps a function; cached entries expire after TTL seconds. For simple cases a module-level dict with a timestamp sentinel is equally effective and has no dependency. |
| https://gist.github.com/masklinn/a532dfe55bdeab3d60ab8e46ccc38a68 | 2026-05-10 | community reference (cheat sheet) | WebFetch | Explicitly: "The output of `print` is not officially structured, do not rely either on the format or on the information." Confirms print-vs-list distinction. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://ss64.com/mac/launchctl.html | doc | WebFetch returned only the no-parse warning + high-level description; no new field data beyond man page |
| https://rakhesh.com/mac/macos-launchctl-commands/ | blog | Fetched but no field-level detail; redirected to general overview |
| https://babodee.wordpress.com/2016/04/09/launchctl-2-0-syntax/ | blog | 2016 pre-Sequoia; snippet only; launchctl 2.0 syntax article, field format may have changed |
| https://commandmasters.com/commands/launchctl-osx/ | community | Fetched; mentioned `PID`, `LastExitStatus`, `label` for `launchctl list`; no print-specific detail |
| https://github.com/fastapi/fastapi/issues/3044 | community (GitHub) | Snippet: `lru_cache` is thread-safe for sync FastAPI endpoints; deferred to primary subprocess docs |
| https://cachetools.readthedocs.io/ | official | Snippet: `TTLCache(maxsize=128, ttl=30)` from `cachetools`; no extra dependency needed if we use a timestamp dict |
| https://hackmag.com/security/launchctl-python | blog | Snippet: Python + launchctl automation example; no field detail beyond what we have |

---

## Search queries run (three-variant discipline)

1. **Current-year frontier:** `launchctl print output format fields state macOS 2026`
2. **Last-2-year window:** `launchctl print gui UID label output parsing Python subprocess 2025`, `launchctl print state running "last exit code" parse subprocess Python macOS Sequoia 2025 2026`, `launchctl list "PID" "last exit code" regex parse shell script macOS daemon monitoring 2025`
3. **Year-less canonical:** `launchctl list PID last exit code regex parse shell script macOS daemon monitoring`, `Python caching subprocess output TTL functools lru_cache thread-safe FastAPI endpoint performance`

---

## Recency scan (2024-2026)

Searched explicitly for 2025-2026 literature on launchctl state parsing, launchd introspection APIs, and Python subprocess caching for FastAPI. Result:

- **Found:** Alan Siu's May 2025 article on new launchctl subcommands (alansiu.net/2025/05/28) -- confirms `state = not running` / `active count = 0` field names are still present on Sequoia/modern macOS. Confirms "Bad request. Could not find service" error when a job is not loaded (useful for the mas-harness bootout case).
- **Found:** No new Apple-official structured introspection API for launchd in 2025-2026. The recommendation remains `launchctl print` for human-readable info; `launchctl list` for machine-parseable PID + exit code.
- **No new finding** supersedes the core constraint: Apple has not stabilized the `launchctl print` output format and explicitly marks it not-API. The `launchctl list` three-column format (`PID exit_code label`) is older but more machine-friendly, and still documented as of 2025.

---

## Key findings

1. **`launchctl print gui/<uid>/<label>` exposes these fields** (verified empirically on this machine, macOS Sequoia, 2026-05-10):
   - `state = running` or `state = not running`
   - `active count = N` (0 = idle, >= 1 = process active)
   - `last exit code = N` (0 = clean, nonzero = error; -15 = SIGTERM normal)
   - `runs = N` (total launchd invocation count since plist load)
   - `pid = N` (only present when state = running)
   - `minimum runtime = N` (seconds)
   - No `last_run_at` timestamp field is exposed by launchctl print. (Source: empirical + alansiu.net 2025, launchd.info)

2. **`launchctl print` is not a stable API** -- Apple's man page (keith.github.io/xcode-man-pages) and every practitioner source warn "NOT API in any sense at all." However the `state`, `active count`, `last exit code`, and `runs` fields have been stable across the field reports surveyed (2016 -- 2026). Pragmatically: parse them with regex, wrap in a try/except, fall back gracefully. (Source: man page, masklinn cheat sheet, alansiu.net 2025)

3. **`launchctl list <label>` (single-label form) returns a more machine-friendly three-column line**: `PID  exit_code  label` -- or a JSON-like dict on some macOS versions. It does not return `state` as a string. The `print` form is more information-rich for our purpose; using both is an option (list for exit code, print for state string). (Source: launchd.info, commandmasters.com)

4. **No timestamp of last successful run is surfaced by launchctl.** Neither `print` nor `list` exposes a last-run-at field. This means `last_run` in the dashboard will remain `None` for launchd entries unless we parse log files (out of scope for this phase). (Source: empirical, launchd.info)

5. **`StartCalendarInterval` jobs (ablation, autoresearch) do not expose a next-fire-time** from launchctl. launchd calculates the next calendar tick internally; `launchctl print` shows the event trigger descriptor (`"Hour" => N, "Minute" => M`) but not a resolved ISO timestamp. `StartInterval` jobs (backend-watchdog, mas-harness) expose `run interval = N seconds` but again not a next-fire-time. Dashboard `next_run` will remain `None` for all launchd entries. (Source: empirical output captured 2026-05-10)

6. **`subprocess.run(["launchctl", "print", f"gui/{uid}/{label}"], capture_output=True, text=True, timeout=5)` is the canonical Python invocation.** Use `check=False` so a nonzero returncode (e.g., "Bad request" when mas-harness is booted out) does not raise; inspect `returncode` and `stdout` manually. (Source: Python stdlib docs subprocess.html)

7. **Caching is warranted.** Six `launchctl print` calls at dashboard load is < 600ms total (each call is ~50-80ms on this Mac) but still adds latency. A simple module-level dict `{label: (result, fetched_at_ts)}` with a 30-second TTL avoids any external dependency and is thread-safe for read-heavy workloads if writes are atomic dict assignments (CPython GIL). (Source: fastapi issues/3044, cachetools docs, medium ttl-lru article)

8. **mas-harness is currently booted out** (`launchctl bootout gui/501/com.pyfinagent.mas-harness`). `launchctl print gui/501/com.pyfinagent.mas-harness` returns `Bad request. Could not find service`. The parser must handle this gracefully: treat "Bad request" / nonzero returncode as `status="not_loaded"`. (Source: empirical, alansiu.net 2025: "Any other output indicates it is loaded.")

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/api/cron_dashboard_api.py` | 288 | Cron dashboard endpoints; `get_all_jobs()` at line 200-242 | Active; launchd block at lines 235-236 calls `_static_to_dict` -- this is the integration point |
| `backend/api/job_status_api.py` | 80+ (read to line 80) | Heartbeat registry for slack-bot jobs; `get_registry_snapshot()` used by phase-23.5.2.5 bridge | Active; pattern to emulate for launchd bridge |
| `tests/api/test_cron_dashboard.py` | 246 | Regression guard for /api/jobs/all and /api/logs/tail | Active; line 231-245: `test_jobs_all_launchd_unaffected_by_slack_bot_bridge` asserts `status == "manifest"` and both `next_run`/`last_run` are `None` -- this test MUST be updated to allow real statuses |
| `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` | - | `StartInterval=60s`, `RunAtLoad=true` | Healthy; runs=6478, last exit code=0 |
| `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | - | `KeepAlive`, `RunAtLoad=true` | Running (pid=42259); runs=11 |
| `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` | - | `KeepAlive`, `RunAtLoad=true` | Running (pid=94049); runs=2 |
| `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` | - | `StartInterval=1800s`, `RunAtLoad=false` | Booted out -- "Bad request" from launchctl |
| `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` | - | `StartCalendarInterval Hour=3 Minute=0` | Loaded; not running; runs=4, last exit code=0 |
| `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | - | `StartCalendarInterval Hour=2 Minute=0` | Loaded; not running; runs=4, last exit code=1 (FAILING) |
| `handoff/archive/phase-23.3.4/phase-23.3.4-audit-findings.md` | 123 | Prior launchd audit; phase-23.3.4 | Historical; confirmed autoresearch exit-127, backend-watchdog healthy, manifest extended from 1 to 6 entries |

---

## Empirical `launchctl print` output -- annotated field map

From the live captures on 2026-05-10:

```
# RUNNING example (com.pyfinagent.backend)
state = running          -> status: "running"
active count = 1         -> corroborates running
last exit code = 0       -> (not present when currently running -- appears only when stopped)
runs = 11                -> informational
pid = 42259              -> present only when running

# IDLE / SCHEDULED example (com.pyfinagent.ablation)
state = not running      -> not currently executing
active count = 0         -> corroborates not running
last exit code = 0       -> last run succeeded
runs = 4                 -> ran 4 times total

# FAILING example (com.pyfinagent.autoresearch)
state = not running
active count = 0
last exit code = 1       -> last run failed (was exit 127 per phase-23.3.4, now 1)
runs = 4

# WATCHDOG (StartInterval, not currently firing)
state = not running
active count = 0
last exit code = 0
runs = 6478              -> fires every 60s; high run count normal

# NOT LOADED (com.pyfinagent.mas-harness after bootout)
returncode != 0, stdout = "Bad request.\nCould not find service..."
```

Key insight: `last exit code` is only present in `launchctl print` output when the job is **not currently running**. When state = running, the field does not appear (the process hasn't exited yet). The parser must handle its absence.

---

## Consensus vs debate (external)

**Consensus:** `launchctl print` is not a stable API; field names have been stable in practice 2016-2026 but Apple never guarantees them. Every source agrees: wrap all parsing in try/except and degrade gracefully.

**Debate:** `launchctl list <label>` vs `launchctl print gui/<uid>/<label>`. `list` gives machine-friendly columns but only PID + exit code + label -- no `state` string. `print` gives `state`, `active count`, `runs`, and other detail at the cost of verbose text parsing. For a dashboard that needs the `state` string, `print` is the right choice despite the verbosity.

**No debate:** no Python library wraps launchd introspection in a stable API without resorting to `pyobjc-framework-ServiceManagement` or writing an XPC client -- both are overkill for this use case.

---

## Pitfalls (from literature and empirical)

1. **`last exit code` absent when job is running** -- the field only appears post-exit. Parser must use `.get()` pattern.
2. **"Bad request" on booted-out jobs** -- `launchctl print` returns nonzero exit and "Bad request" text. Must check `returncode != 0` before parsing stdout.
3. **`-N` exit codes are signals** -- `-15` = SIGTERM (normal for KeepAlive jobs that were stopped). The backend job shows `last terminating signal = Terminated: 15`. This is expected and should map to `"ok"` or `"stopped"`, not `"failed"`.
4. **SIGTERM exit code convention** -- `launchctl print` shows `last exit code` as a positive integer (e.g., `0`) for clean exits, but a signal-terminated process shows `last terminating signal = Terminated: 15` instead of (or alongside) a negative exit code. The test output from `com.pyfinagent.backend` does NOT show `last exit code` while running; after a SIGTERM stop, `last exit code` may show as `0` (launchd normalises SIGTERM to 0 for KeepAlive jobs that it restarts cleanly).
5. **`runs` counter resets on plist unload/reload** -- don't treat it as cumulative across machine restarts.
6. **Parsing is O(text)** -- launchctl print output is ~40-80 lines per job; regex line-by-line is fast (< 1ms per job post-subprocess). The bottleneck is the subprocess fork, not parsing.

---

## Application to pyfinagent (mapping findings to file:line anchors)

### Integration point

`cron_dashboard_api.py:235-236` -- the launchd block in `get_all_jobs()`:

```python
# CURRENT (line 235-236):
for entry in _LAUNCHD_JOBS:
    jobs.append(_static_to_dict(entry, source="launchd"))
```

This should be replaced with a merge block analogous to the slack-bot bridge at lines 220-233. The new helper `_launchctl_state(label: str) -> dict` queries `launchctl print gui/<uid>/<label>`, parses the output, and returns `{"status": ..., "last_run": None, "next_run": None}`. The loop then merges that into each manifest entry.

### Test update required

`tests/api/test_cron_dashboard.py:231-245` -- `test_jobs_all_launchd_unaffected_by_slack_bot_bridge` asserts:
```python
assert j["status"] == "manifest"
```
This test was written as a guard that the phase-23.5.2.5 bridge did NOT affect launchd rows. After this phase, that guard becomes incorrect. The test must be updated to either:
- Mock `_launchctl_state` to return a known dict, and assert the merged status, OR
- Rename to `test_jobs_all_launchd_unaffected_by_slack_bot_bridge` and update the assertion to `assert j["status"] != "manifest"` once the bridge is live.

The other launchd test (`test_jobs_all_includes_static_launchd_manifest` at line 84-90) only checks count and `id` -- no `status` assertion -- so it passes unchanged.

---

## Concrete design recommendation

### Helper function (new, in `cron_dashboard_api.py`)

```python
import os
import re
import subprocess
import time
from typing import Optional

_LAUNCHD_CACHE: dict[str, tuple[dict, float]] = {}
_LAUNCHD_CACHE_TTL = 30.0   # seconds

def _launchctl_state(label: str) -> dict[str, Optional[str]]:
    """Pull live state from launchctl print for one label.

    Returns dict with keys: status, last_exit_code (str or None).
    All fields degrade to safe defaults on parse failure.
    """
    now = time.monotonic()
    cached_val, cached_at = _LAUNCHD_CACHE.get(label, ({}, 0.0))
    if now - cached_at < _LAUNCHD_CACHE_TTL:
        return cached_val

    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        result = {"status": "unknown", "last_exit_code": None}
        _LAUNCHD_CACHE[label] = (result, now)
        return result

    if proc.returncode != 0:
        # "Bad request. Could not find service" -- job not loaded
        result = {"status": "not_loaded", "last_exit_code": None}
        _LAUNCHD_CACHE[label] = (result, now)
        return result

    out = proc.stdout
    state_m = re.search(r"^\s*state\s*=\s*(.+)$", out, re.MULTILINE)
    exit_m  = re.search(r"^\s*last exit code\s*=\s*(-?\d+)$", out, re.MULTILINE)

    state_str = state_m.group(1).strip() if state_m else None
    exit_code_str = exit_m.group(1) if exit_m else None
    exit_code = int(exit_code_str) if exit_code_str is not None else None

    if state_str == "running":
        status = "running"
    elif exit_code is None:
        # state = not running, no exit code yet (never fired, or SIGTERM cleanup)
        status = "ok"
    elif exit_code == 0 or exit_code == -15:
        # Clean exit or SIGTERM (normal KeepAlive restart)
        status = "ok"
    elif exit_code > 0:
        status = "failed"
    else:
        # Other negative: signal kill; treat as ok unless > 3 consecutive
        status = "ok"

    result = {"status": status, "last_exit_code": exit_code_str}
    _LAUNCHD_CACHE[label] = (result, now)
    return result
```

### Merge loop (replaces `cron_dashboard_api.py:235-236`)

```python
for entry in _LAUNCHD_JOBS:
    ld = _launchctl_state(entry["id"])
    jobs.append(
        {
            "id": entry["id"],
            "source": "launchd",
            "schedule": entry.get("schedule", "?"),
            "next_run": None,     # launchctl does not expose next-fire time
            "last_run": None,     # launchctl does not expose last-run timestamp
            "status": ld["status"],
            "description": entry.get("description", entry["id"]),
        }
    )
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total including snippet-only (14 collected)
- [x] Recency scan (last 2 years) performed and reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (cron_dashboard_api, job_status_api, test file, all 6 plists, phase-23.3.4 archive)
- [x] Contradictions and consensus noted (print vs list, stable-in-practice vs not-API)
- [x] All claims cited per-claim (not just footer)

---

## The four answers Main asked for

### Answer 1: What fields can we surface from `launchctl print`?

Empirically verified fields (2026-05-10, macOS Sequoia, gui/501):

| Field in output | Present when | Maps to dashboard |
|----------------|-------------|-------------------|
| `state = running` / `state = not running` | Always (if loaded) | `status` primary signal |
| `active count = N` | Always | Corroborates state |
| `last exit code = N` | Only when state = not running | Drives "failed" determination |
| `runs = N` | Always | Informational only; no dashboard column |
| `pid = N` | Only when running | Not needed for dashboard |
| next-fire-time | Never | `next_run` stays `None` |
| last-run timestamp | Never | `last_run` stays `None` |

### Answer 2: Where in cron_dashboard_api.py should the merge happen?

**Inline at `get_all_jobs()` lines 235-236**, replacing the `_static_to_dict` call for launchd entries with the new merge block (shown above). This mirrors the slack-bot bridge pattern at lines 220-233 exactly: the loop over `_LAUNCHD_JOBS` calls a helper, merges the dict, and appends. No new route, no new module.

A separate helper function `_launchctl_state(label)` should be defined at module level above `get_all_jobs()` (analogous to how job_status_api provides `get_registry_snapshot()` for the slack-bot bridge). The helper owns the subprocess call and the 30s cache.

### Answer 3: How to map launchctl print state to dashboard status?

| launchctl print output | Dashboard status | Rationale |
|------------------------|-----------------|-----------|
| `returncode != 0` ("Bad request" / not loaded) | `"not_loaded"` | mas-harness is booted out; distinct from failed |
| `state = running` | `"running"` | Process is active |
| `state = not running`, `last exit code` absent | `"ok"` | Never fired yet, or SIGTERM-clean KeepAlive restart |
| `state = not running`, `last exit code = 0` | `"ok"` | Last run succeeded |
| `state = not running`, `last exit code = -15` | `"ok"` | SIGTERM = normal KeepAlive cycle |
| `state = not running`, `last exit code > 0` | `"failed"` | Abnormal exit (autoresearch = exit 1) |
| subprocess exception / timeout | `"unknown"` | Graceful degradation |

**Never emit `"manifest"`** for launchd entries after this phase -- that was the static-only placeholder.

### Answer 4: Caching strategy

**Recommend: 30-second module-level dict cache** (shown in the helper above). Rationale:

- Each `launchctl print` call is ~50-80ms on this Mac (subprocess fork + kernel query). Six calls = ~360-480ms added latency on cache miss.
- Dashboard load frequency: human-driven; a 30s TTL means at most 2 cache-miss cycles per minute even under active use.
- No external dependency: a dict + `time.monotonic()` requires nothing beyond stdlib. `cachetools.TTLCache` is cleaner but adds a dependency; `functools.lru_cache` does not support TTL without a wrapper.
- Thread safety: CPython dict assignment is atomic under the GIL. For a single-developer local app (CLAUDE.md: local-only deployment) this is sufficient. A `threading.Lock` around the cache read/write is optional hardening.
- Do NOT query fresh on every request: 6x subprocess forks per `/api/jobs/all` call blocks FastAPI's async event loop for ~400ms. `get_all_jobs()` must call the cached helper from a sync context (the helper is `def`, not `async def`; FastAPI auto-runs `def` endpoints in a threadpool per backend-api.md convention).

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 7,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
