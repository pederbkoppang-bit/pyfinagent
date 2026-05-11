---
step_id: phase-23.5.14
step_name: "Cron job verification: com.pyfinagent.backend-watchdog (launchd)"
tier: simple
date: 2026-05-10
researcher: merged researcher+Explore
---

## Research: launchd backend-watchdog — schedule type, bridge correctness, next_run criterion

### Queries run (three-variant discipline)

1. Current-year frontier: `launchd StartInterval KeepAlive semantics watchdog macOS 2026`
2. Last-2-year window: `macOS launchd StartInterval next fire time launchctl 2025`
3. Year-less canonical: `launchd watchdog process supervision SIGUSR1 restart on failure`
4. Supplemental: `launchctl print next scheduled run time StartInterval programmatic API 2024 2025`
5. Supplemental: `process supervision watchdog SIGUSR1 N consecutive failures restart pattern 2025 2026`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | 2026-05-10 | official doc (man page) | WebFetch | "The manual does not document any mechanism for launchctl to display next scheduled run times" — next fire time exposure is absent from the documented interface |
| https://www.launchd.info/ | 2026-05-10 | authoritative tutorial doc | WebFetch | "None of these mechanisms—nor the launchctl command-line tool—appear to surface upcoming execution timestamps" confirming StartInterval, KeepAlive, and RunAtLoad all have no next_run exposure |
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | 2026-05-10 | official Apple doc | WebFetch | "The documentation does not mention any mechanism to query next scheduled run time or next fire time" — StartCalendarInterval is offered as an alternative but also lacks introspection |
| https://gist.github.com/dabrahams/4092951 | 2026-05-10 | authoritative notes (well-cited gist) | WebFetch | "StartInterval timer starts over when the job exits, so a 20s job with a 10s StartInterval will run every 30s" — confirms timer-restart semantics; no next_run introspection documented |
| https://en.wikipedia.org/wiki/Watchdog_timer | 2026-05-10 | reference (canonical definition) | WebFetch | "during normal operation, the computer regularly restarts the timer to prevent timeout; multistage watchdogs cascade two or more timers, each triggering corrective actions sequentially" — two-stage pattern (diagnose then kill) is the established pattern |
| https://www.embeddedrelated.com/showarticle/1276.php | 2026-05-10 | industry practitioner blog | WebFetch | "The watchdog should be triggered when the device's essential functionality is lost" — consecutive-failure threshold is the correct triggering criterion; SIGUSR1+kill-after-N matches this doctrine |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://alvinalexander.com/mac-os-x/launchd-plist-examples-startinterval-startcalendarinterval/ | tutorial | Snippet confirmed StartInterval/StartCalendarInterval difference; full read not needed — covered by launchd.info |
| https://alvinalexander.com/mac-os-x/launchd-examples-launchd-plist-file-examples-mac/ | tutorial | Snippet only; duplicate coverage of plist format |
| https://github.com/tjluoma/launchd-keepalive | code/example | Snippet confirmed KeepAlive semantics; no next_run info needed beyond official man page |
| https://killtheyak.com/schedule-jobs-launchd/ | tutorial | Snippet; duplicate StartInterval coverage |
| https://developer.apple.com/forums/thread/23361 | forum | Snippet confirmed StartInterval behavior change history |
| https://blog.darnell.io/automation-on-macos-with-launchctl/ | blog | WebFetch attempted but yielded no new info beyond what launchd.info already provided |
| https://gist.github.com/johndturn/09a5c055e6a56ab61212204607940fa0 | gist/tutorial | WebFetch returned no next_run info; confirms launchctl list/start are the documented commands |
| https://github.com/diffstorm/processWatchdog | code | Snippet only; Linux-focused; confirms N-consecutive-failure pattern exists broadly |
| https://discussions.apple.com/thread/3011039 | forum | Snippet only; confirms StartInterval has no effect on already-running jobs |
| https://maketecheasier.com/use-launchd-run-scripts-on-schedule-macos/ | tutorial | Snippet only; duplicate coverage |

---

### Recency scan (2024-2026)

Searched `launchd StartInterval KeepAlive semantics watchdog macOS 2026` and `macOS launchd StartInterval next fire time launchctl 2025`.

Result: no new findings in the 2024-2026 window that supersede the canonical sources. The launchd interface for StartInterval-based jobs has not changed: `launchctl print` still does not surface next fire time for interval jobs as of macOS 25 (Darwin 25.4.0, confirmed from live `launchctl print` output below). The man page definition remains unchanged. One 2026 source (sshmac.com blog for OpenClaw scheduled tasks) was found in search snippets but was not authoritative enough for full-fetch; it did not indicate any new next_run field in launchctl output.

---

### Key findings

1. **Schedule type is StartInterval=60** -- The plist at `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` uses `<key>StartInterval</key><integer>60</integer>` plus `<key>RunAtLoad</key><true/>`. There is NO KeepAlive key. (Source: plist file read in full, line 13.)

2. **launchctl does not expose next_run for StartInterval jobs** -- Confirmed by the official man page, launchd.info, Apple developer docs, and the dabrahams gist. `launchctl print` surfaces `runs`, `last exit code`, `state`, and `run interval = 60 seconds`, but NO next fire timestamp. This is a documented limitation of the launchd interface. (Sources: keith.github.io man page; launchd.info; Apple dev docs — all read in full.)

3. **Live launchctl print output confirms** -- `runs = 6497`, `last exit code = 0`, `state = not running` (between 60s intervals), `run interval = 60 seconds`. The job is active and healthy.

4. **Bridge correctly surfaces status = "ok"** -- Live `/api/jobs/all` returns `{"id":"com.pyfinagent.backend-watchdog","source":"launchd","schedule":"launchd interval 60s","next_run":null,"last_run":null,"status":"ok"}`. The `_classify_launchctl_state()` function correctly interprets `state="not running"` + `exit_code=0` as `status="ok"`. (`cron_dashboard_api.py` line 289, 293-294.)

5. **CRITICAL — criterion is structurally unmeetable as written** -- The immutable verification criterion asserts `j.get("next_run") is not None`. This is impossible for any launchd StartInterval job because launchctl does not expose next fire time. The code already documents this at line 293: `"next_run": None,  # launchctl doesn't expose this`. This is the same category of structural mismatch as phase-23.5.3's morning_digest criterion. The criterion was written when the masterplan assumed launchd would be able to surface next_run (like APScheduler does). It cannot.

6. **Watchdog logic is correct** -- The script (`scripts/launchd/backend_watchdog.sh`) implements the canonical two-stage watchdog pattern: N consecutive health failures -> SIGUSR1 (diagnostic stack dump) -> sleep 2s -> `launchctl kickstart -k`. The threshold is 3, persisted via a counter file. This matches the Wikipedia/EmbeddedRelated watchdog doctrine of staging corrective actions before a hard reset.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` | 25 | launchd plist — defines StartInterval=60, RunAtLoad, script path | Healthy; StartInterval (not KeepAlive) |
| `scripts/launchd/backend_watchdog.sh` | 80 | Watchdog shell script — curl health check, SIGUSR1 + kickstart -k after 3 fails, Slack alert | Healthy; correct two-stage pattern |
| `backend/api/cron_dashboard_api.py` | ~370 (inspected lines 32-360) | Bridge API — `_probe_launchctl`, `_launchctl_state`, `_LAUNCHD_JOBS`, `/api/jobs/all` | Healthy; `next_run: None` is intentional and documented at line 293 |
| `handoff/archive/phase-23.3.4/phase-23.3.4-audit-findings.md` | 123 | Prior watchdog audit — inventory of all 6 launchd services, autoresearch exit-127 bug, watchdog confirmed healthy | Historical context; backend-watchdog was already healthy at audit time |

---

### Consensus vs debate (external)

Consensus: all five authoritative sources (man page, launchd.info, Apple dev docs, dabrahams gist, community discussion) agree that launchctl does not expose next fire time for StartInterval jobs. There is no debate on this point. The information simply does not exist in the launchd public interface.

---

### Pitfalls (from literature)

- Do NOT add KeepAlive=true alongside StartInterval: the man page (via launchd.info) explicitly warns this causes interference — launchd will try to keep the process alive continuously while also trying to schedule it, leading to timing corruption. The current plist correctly uses StartInterval alone.
- ThrottleInterval only activates on failure, not on successful completions (dabrahams). The watchdog's counter-file reset-on-success is the correct pattern to avoid throttle accumulation.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| External finding | File:line anchor | Implication |
|-----------------|-----------------|-------------|
| StartInterval timer resets on job exit (dabrahams) | `plist line 13: StartInterval=60` | The 60s interval is wall-clock from job exit, not job start — actual firing is ~60s + job runtime (minimal for a curl check) |
| launchctl does not expose next fire time (man page, launchd.info, Apple docs) | `cron_dashboard_api.py:293 "next_run": None` | The bridge is correct; next_run cannot be populated without parsing the plist's StartCalendarInterval key (which this job doesn't use) |
| Two-stage watchdog pattern (Wikipedia) | `backend_watchdog.sh:51-76: SIGUSR1 + sleep 2 + kickstart -k` | Implementation matches canonical doctrine |
| Criterion structural mismatch | Immutable criterion in masterplan.json phase-23.5.14 | `next_run is not None` is permanently false for all launchd StartInterval entries; criterion requires amendment or a structural-false-positive disclosure in the evaluator verdict |

---

### Anthropic immutable-criteria doctrine — criterion mismatch analysis

Anthropic's harness design mandates that immutable verification criteria are never silently waived. The doctrine for structural false positives (criteria that are impossible by construction, not by implementation failure) is:

- If the criterion cannot be satisfied regardless of what the generator produces — because the underlying platform API does not expose the required information — this is a **specification defect**, not an implementation defect.
- The correct harness response is a CONDITIONAL verdict with explicit disclosure: "criterion X cannot be met because [platform limitation]; the implementation correctly handles this by [Y]; recommend criterion amendment."
- A PASS verdict with undisclosed criterion deviation is worse — it hides the defect in the audit trail.
- A FAIL verdict (treating a platform limitation as an implementation failure) is also wrong — it would trigger a retry loop that cannot converge.

**Recommendation for Main:** issue a CONDITIONAL verdict for phase-23.5.14 with criterion-mismatch disclosure. Document the disclosure in `evaluator_critique.md` and flag the masterplan criterion for amendment. This is the same pattern used for phase-23.5.3's morning_digest. Do not attempt to "fix" next_run for launchd StartInterval jobs as part of this substep — that would require either (a) parsing StartCalendarInterval from the plist (not applicable here) or (b) computing `now + remaining_interval_seconds` from launchctl output (which does not expose remaining interval). Both are out of scope per the instructions.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (plist, script, bridge API, prior audit)
- [x] Contradictions / consensus noted (unanimous: no next_run in launchd interface)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "gate_passed": true
}
```
