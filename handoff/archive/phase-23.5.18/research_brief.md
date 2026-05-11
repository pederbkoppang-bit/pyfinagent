---
step: phase-23.5.18
name: Cron job verification: com.pyfinagent.ablation (launchd)
tier: simple
date: 2026-05-10
---

## Research: launchd StartCalendarInterval + nightly feature ablation runner

### Queries run (three-variant discipline)
1. Current-year frontier: `launchd StartCalendarInterval sleep wake behavior macOS 2026`
2. Last-2-year window: `launchd StartCalendarInterval missed jobs sleep wake coalesce macOS` (2025/2024 hits)
3. Year-less canonical: `launchd LaunchAgent plist best practices macOS Sequoia 2025` + `feature ablation study automated ML experiment scheduling reproducibility 2025`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.launchd.info/ | 2026-05-10 | official doc / tutorial | WebFetch | "If multiple intervals transpire before the computer is woken, those events will be coalesced into one event upon wake from sleep." |
| https://developer.apple.com/forums/thread/52369 | 2026-05-10 | Apple Developer Forum | WebFetch | Explicit sleep vs. shutdown distinction: asleep -> fires on wake; powered-off -> skips, fires at next scheduled occurrence only. |
| https://deniapps.com/blog/scheduling-a-cron-job-on-macos-with-wake-support | 2026-05-10 | practitioner blog | WebFetch | `WakeSystem: true` plist key is the mechanism that lets launchd wake a sleeping Mac; without it the job relies on the system already being awake. |
| https://www.josephspurrier.com/macos-sleep-cron | 2026-05-10 | practitioner blog | WebFetch | Cron skips missed jobs entirely; launchd StartCalendarInterval defers and fires on next wake -- "closer to what most people actually want." |
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | 2026-05-10 | Apple official doc | WebFetch | `StartCalendarInterval` wildcard semantics (absent key = wildcard); minimum runtime 10s; `StandardOutPath`/`StandardErrorPath` preferred over stdio redirect in code. |
| https://pykeen.readthedocs.io/en/stable/tutorial/running_ablation.html | 2026-05-10 | official library doc | WebFetch | Ablation framework pattern: one script invocation = one experiment; TSV log for results; systematic single-component removal to measure contribution. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://alvinalexander.com/mac-os-x/launchd-plist-examples-startinterval-startcalendarinterval/ | blog | Covered by deeper Apple docs already read |
| https://launchd-dev.macosforge.narkive.com/ZF2IQriC/launchd-startinterval-and-sleep | mailing list | Search snippet sufficient; StartInterval (not StartCalendarInterval) topic |
| https://launchd-dev.macosforge.narkive.com/MWudXExk/launchd-and-system-sleep | mailing list | Older thread; key findings confirmed by fuller sources |
| https://discussions.apple.com/thread/5137946 | community forum | Search snippet sufficient; same coalescing point |
| https://www.manpagez.com/man/5/launchd.plist/ | man page | Authoritative but coverage already captured by Apple docs |
| https://gist.github.com/Matejkob/f8b1f6a7606f30777552372bab36c338 | GitHub gist | Practitioner guide; key points covered |
| https://ml.recipes/notebooks/6-ablation-study.html | ML recipe | Snippet captured ablation pattern adequately |
| https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002 | journal | Reproducibility focus -- tangential to scheduling question |
| https://euromlsys.eu/pdf/euromlsys25-33.pdf | conference paper | LLM-assisted ablation; out of scope for scheduling research |
| https://dcatkth.github.io/papers/autoablation.pdf | paper | Parallel ablation; this project is sequential single-host |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on launchd StartCalendarInterval and nightly ML ablation scheduling. Result:

- **launchd (2025-2026):** No substantive changes to StartCalendarInterval semantics found. macOS Sequoia (15.x) did not alter the sleep/wake coalescing behavior that has been stable since macOS 10.x. The `WakeSystem` key remains the only plist mechanism to auto-wake the host. No regression reports for the calendar-interval trigger in 2025-2026 forums.
- **Feature ablation runners (2025-2026):** EuroMLSys 2025 paper on LLM-assisted ablation planning (ABGEN/AblationMage) is the most recent work. The dominant pattern remains unchanged: one-script-one-backtest with TSV result appending and a baseline comparison gate. pyfinagent's `run_ablation.py --next-untested` matches the idiomatic pattern.

---

### Key findings

1. **StartCalendarInterval fires on next wake after sleep** -- "If the system is asleep, the job will be started the next time the computer wakes up." (launchd.info, 2026-05-10)
2. **Missed intervals coalesce to one fire** -- Multiple skipped intervals consolidate into a single execution on wake, not a backlog of runs. (launchd.info, Apple Dev Forum thread/52369, 2026-05-10)
3. **Powered-off is NOT sleep** -- If the Mac is shut down at 03:00 the job does NOT run; it waits for the next 03:00 occurrence. (Apple Dev Forum thread/52369, 2026-05-10)
4. **`WakeSystem: true` absent in ablation plist** -- The plist does NOT include `WakeSystem`, so the ablation job relies on the host being awake or asleep (not off) at 03:00. This is acceptable for a nightly experiment on a Mac that typically sleeps rather than shuts down. If strict nightly execution is needed, `WakeSystem: true` could be added.
5. **plist `ExitTimeOut: 1200`** -- 20-minute hard kill ceiling. Ablation runs (one backtest) take well under 20 minutes based on log evidence.
6. **Bridge correctly classifies `"ok"`** -- `_classify_launchctl_state("not running", 0)` returns `"ok"` (clean exit). Confirmed live: `state="not running"`, `last exit code=0`, `runs=4`.
7. **Ablation runner pattern is idiomatic** -- `--next-untested` with TSV appending and a DSR/Sharpe gate matches the AutoAblation and PyKEEN patterns from literature.

---

### Internal code inventory

| File | Lines read | Role | Status |
|------|-----------|------|--------|
| `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` | full | launchd schedule definition | Active; `StartCalendarInterval Hour=3 Minute=0` confirmed |
| `backend/api/cron_dashboard_api.py` | lines 100-230 | `/api/jobs/all` endpoint + launchctl bridge | Active; `_LAUNCHD_JOBS` row at line 110 matches plist label |
| `scripts/ablation/run_ablation.py` | lines 1-60 | Ablation experiment runner | Active; `--next-untested` invoked by plist |
| `handoff/ablation.launchd.log` | full (0 bytes) | launchd stderr capture for ablation | Empty -- no stderr emitted on successful runs (expected) |
| `handoff/ablation.log` | last 30 lines | ablation script stdout | Active; last entry at 2026-05-10 03:21 shows `total_revenue delta=-0.5350 dsr=1.0000 verdict=keep` |

---

### Consensus vs debate (external)

**Consensus:** StartCalendarInterval is the correct macOS replacement for cron; sleep coalescing is well-documented and reliable. No debate.

**Debate:** Whether to add `WakeSystem: true` -- some practitioner sources recommend it for reliability, others note it is unnecessary if the host is a desktop or always-on server. For a Mac that sleeps rather than shuts down, the current plist is sufficient.

---

### Pitfalls (from literature)

- **Shutdown vs sleep:** If the Mac is powered off at 03:00, the job skips entirely. Current plist has no `WakeSystem` guard. (Apple Dev Forum, 2026-05-10)
- **Minimum runtime:** launchd requires a job to run >= 10 seconds or it may throttle respawn. The ablation script satisfies this (one full backtest takes minutes).
- **ExitTimeOut:** Set to 1200s (20 min). If a future ablation backtest stalls, launchd will SIGKILL at 20 min. Log will show exit code -15.
- **`ablation.launchd.log` empty:** stderr for ablation goes to `handoff/ablation.launchd.log`, which is 0 bytes -- meaning no errors have reached stderr. All stdout goes to `handoff/ablation.log` via the ProgramArguments redirect (`>> handoff/ablation.log 2>&1`). The redirect in the script conflicts with the plist's `StandardOutPath`/`StandardErrorPath` -- in practice the script redirect wins for stdout; launchd's stderr path captures only pre-exec errors. This is a known quirk; no action needed.

---

### Application to pyfinagent (file:line anchors)

| Finding | File:line |
|---------|-----------|
| StartCalendarInterval Hour=3 Minute=0 | `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` (plist `StartCalendarInterval` dict) |
| `launchctl print` confirms `runs=4`, `last exit code=0`, `state=not running` | live launchctl output |
| `/api/jobs/all` `_LAUNCHD_JOBS` row for ablation | `backend/api/cron_dashboard_api.py:110` |
| `_classify_launchctl_state` maps `(not running, 0)` to `"ok"` | `backend/api/cron_dashboard_api.py:223-229` |
| Ablation runner script, `--next-untested` mode | `scripts/ablation/run_ablation.py:1-60` |
| Last ablation run at 03:21 on 2026-05-10, exit 0 | `handoff/ablation.log` (last 30 lines) |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (plist, bridge API, runner script, both logs)
- [x] Contradictions / consensus noted (WakeSystem debate)
- [x] All claims cited per-claim

---

## Three answers

**Answer 1 -- Plist trigger type?**
Confirmed `StartCalendarInterval` with `Hour=3`, `Minute=0`. Fires daily at 03:00. Verified verbatim from plist and cross-confirmed by `launchctl print` event trigger descriptor showing `"Minute" => 0 / "Hour" => 3`.

**Answer 2 -- Bridge surfaces correct status?**
Yes. Live `curl /api/jobs/all` returns `"status": "ok"` for `com.pyfinagent.ablation`. Immutable verification command output: `OK com.pyfinagent.ablation ok`. launchctl confirms `state=not running`, `last exit code=0`, `runs=4`.

**Answer 3 -- Amended criterion meetable?**
Yes. The verification command asserts `status in ("running","ok","failed","not_loaded","unknown")` -- current status is `"ok"`, which is in the allowed set. The command passes as demonstrated.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
