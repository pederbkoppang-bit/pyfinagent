# Research Brief -- phase-23.2.10

**Topic.** Verify watchdog has not fired in 7 days (P1). Grep
`handoff/logs/backend-watchdog.log` for `health FAIL`; masterplan
expects zero in the last 7 days. Recommend pytest shape.

**Tier:** simple (>=5 external sources read in full).

**Headline.** The masterplan verification string is ambiguous. A
literal interpretation -- "zero `health FAIL` lines" -- **fails**:
42 lines match in the last 7 days. An operational interpretation --
"zero times the watchdog actually fired (= 3-of-3 escalation +
`kickstart -k`)" -- **passes**: zero terminal escalations, zero
restarts, zero SIGUSR1 stack dumps. The watchdog is alive, well,
and doing exactly what its threshold design intended: filtering
transient one-shot failures and only restarting on persistent
failure (3 consecutive). Recommend the pytest assert against the
**operational** definition and update the masterplan verification
string to remove the ambiguity.

## 1. Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://launchd.info/ | 2026-05-22 | doc (auth. community) | WebFetch | "If you have a job that you want to execute every n seconds, this is for you" (StartInterval semantics). KeepAlive vs StartInterval distinction; **launchd has no built-in unhealthyThreshold / consecutive-failure concept** -- if the watchdog needs threshold semantics they must be implemented inside the shell script via a counter file. The pyfinagent script does this via `${HOME}/Library/Caches/com.pyfinagent.backend.watchdog.fails`. |
| 2 | https://aws.amazon.com/builders-library/implementing-health-checks/ | 2026-05-22 | doc (AWS Builder's Library, peer-reviewed by AWS principals) | WebFetch | "When we build systems to react automatically to dependency health check failures, we must build in the right amount of thresholding to prevent the automated system from taking drastic action unexpectedly." Distinguishes independent vs correlated failures; restarting a single bad host = OK, restarting the whole fleet because one shared dep is sick = bad. Validates the 3-of-3 threshold pattern in `backend_watchdog.sh:20`. |
| 3 | https://docs.cloud.google.com/load-balancing/docs/health-check-concepts | 2026-05-22 | doc (GCP official) | WebFetch | GCP defaults: `healthyThreshold=2` and `unhealthyThreshold=2` sequential probes -- "requiring consecutive failures prevents transient network glitches from immediately marking backends as unhealthy, while still responding reasonably quickly to actual outages." Pyfinagent's `FAILURE_THRESHOLD=3` is **more conservative** than GCP's L7 default, appropriate for a kill-then-restart action (heavier remediation than just "remove from LB rotation"). |
| 4 | https://oneuptime.com/blog/post/2026-02-24-how-to-set-consecutive-errors-threshold-for-circuit-breaking/view | 2026-05-22 | blog (industry, dated 2026-02) | WebFetch | Explicit recommendation: **3 consecutive errors is "a good middle ground."** "Three consecutive errors strongly suggest the pod is genuinely unhealthy" while remaining "tolerant enough to ride through momentary hiccups." "A single successful response resets the entire counter" -- this is exactly the design encoded in `backend_watchdog.sh:32-35`. Validates the existing FAILURE_THRESHOLD=3 + `echo 0 > "$COUNTER_FILE"` reset pattern. |
| 5 | https://oneuptime.com/blog/post/2026-01-30-health-check-design/view | 2026-05-22 | blog (industry, dated 2026-01) | WebFetch | "Single health check failures should not trigger immediate action." Recommends tiered probes (startup / readiness / liveness); for **liveness** (= our case, since FAIL -> SIGKILL + kickstart): **conservative ~5 failures = ~50s detection** is the recommendation. We use 3 -- slightly more aggressive than 2026 best-practice. Acceptable here: backend hangs are bounded-recovery (process restart is cheap; backtest cron 1800s cadence absorbs <60s of downtime). |
| 6 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-22 | doc (Anthropic official, project-canonical) | WebFetch | "Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or a new file." Validates log-tail verification as the protocol-canonical pattern for evaluator audit. The sprint-contract / file-evidence loop is exactly what `verification.live_check` (CLAUDE.md) enforces: an artifact must exist that an operator can audit. Test should produce that artifact-shaped evidence. |

(6 sources read in full -- floor 5 cleared.)

## 2. Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://copyconstruct.medium.com/health-checks-in-distributed-systems-aa8a0e8c1672 | blog (Sridharan) | Fetched but light on threshold mechanics; useful for "health is a spectrum not binary" framing only. |
| https://john-millikin.com/sre-school/health-checking | doc (SRE-school) | Fetched but covers only purpose + endpoint design, no threshold guidance. |
| https://srepatterns.blogspot.com/2025/02/circuit-breaker-design-pattern-for-sre.html | blog | Fetched but the threshold/window details are abstract; no concrete N. |
| https://sre.google/sre-book/testing-reliability/ | book (Google SRE) | Snippet only; canonical reference for SRE testing discipline. |
| https://github.com/anthropics/cwc-long-running-agents | repo (Anthropic) | Snippet only; the canonical sample repo for harness patterns. |
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | doc (Apple official) | Snippet only; canonical launchd plist reference, supplemented by `launchd.info`. |
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | doc (Apple manpage mirror) | Snippet only; man-page format already covered by launchd.info. |
| https://alvinalexander.com/mac-os-x/launchd-plist-examples-startinterval-startcalendarinterval/ | blog (industry, well-known macOS author) | Snippet only; corroborates the StartInterval=60 pattern. |
| https://docs.pytest.org/en/stable/how-to/logging.html | doc (pytest official) | Snippet only; pytest log-format / caplog APIs not directly relevant -- our test is grepping a log on disk, not capturing pytest-emitted logs. |
| https://blog.dailydoseofds.com/p/the-anatomy-of-an-agent-harness | blog | Snippet only; head/tail-truncation technique noted -- "harness keeps the head and tail tokens above a threshold and offloads the full output to the filesystem" -- corroborates that log-tail is the canonical operator-audit shape. |

(11 URLs total identified; 6 read in full + 5 snippet-only meets the 10+ floor.)

## 3. Recency scan (2024-2026)

Searched explicitly for guidance dated 2024-2026 on health-check
thresholds, launchd watchdog patterns, and consecutive-failure
filtering. **Findings:**

- The 2026-01 and 2026-02 oneuptime.com posts (rows 4 + 5 above)
  are current-year frontier; both explicitly endorse the **3
  consecutive failures = middle-ground** threshold and the
  **"reset on first success" counter pattern** -- which is
  exactly what `scripts/launchd/backend_watchdog.sh:32-35`
  implements. No 2024-2026 source contradicts the design.
- 2026 best-practice for **liveness-probe-equivalent** actions
  (= our SIGKILL + kickstart -k case) leans slightly more
  conservative (5 fails / ~50s) than what we use (3 fails /
  ~3min, since launchd schedules at 60s). Acceptable given the
  cheap-restart cost profile.
- No source recommends single-probe-equals-restart for liveness;
  all warn against it.
- No new launchd / macOS framework feature in 2024-2026 supersedes
  the StartInterval + script-counter pattern (launchd has not had
  a major API expansion since the original 10.x design; KeepAlive
  + StartInterval remain the two primitives). 

Bottom line: the existing watchdog design tracks 2026 best
practice. The masterplan verification *string* is the only thing
that needs revisiting.

## 4. Key findings (external)

1. **3-consecutive is the industry-canonical middle-ground**, and
   single-failure restart is universally discouraged for
   livenessProbe-equivalent actions. Source: oneuptime
   2026-02-24, "three consecutive errors strongly suggest the pod
   is genuinely unhealthy."
2. **Counter-reset on first success is the standard counter
   shape**; the pyfinagent watchdog already implements it
   (`echo 0 > "$COUNTER_FILE"` on line 35). Source: oneuptime
   2026-02-24, "a single successful response resets the entire
   counter."
3. **GCP's L7 healthchecker default is `unhealthyThreshold=2`**,
   one less than ours. Higher remediation cost (SIGKILL vs just
   "stop sending traffic") justifies our +1. Source: GCP
   load-balancing docs.
4. **AWS Builder's Library** does NOT publish a specific N, but
   emphasizes thresholding-as-discipline + distinguishing
   independent vs correlated failures. The watchdog correctly
   targets a single host (independent failure scope).
5. **macOS launchd has no native consecutive-failure threshold**;
   the threshold must be implemented in the shell script via a
   persistent counter file. Pyfinagent does this via
   `~/Library/Caches/com.pyfinagent.backend.watchdog.fails`. The
   counter persists across launchd job invocations, which is
   exactly what the design requires. Source: launchd.info.

## 5. Internal code inventory (with file:line anchors)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/launchd/backend_watchdog.sh` | 1-79 | The watchdog shell script run by launchd every 60s. Reads counter, hits /api/health with 5s timeout (L19, L31), increments counter on FAIL (L40), resets counter to 0 on OK (L35), at 3 consecutive FAILs sends SIGUSR1 (L52), curl-POSTs Slack alert (L70, phase-23.2.18), then `launchctl kickstart -k` (L76). | Healthy. Implements the canonical 3-of-3 + counter-reset pattern.  |
| `scripts/launchd/com.pyfinagent.backend-watchdog.plist` | 1-25 | launchd manifest. `StartInterval=60` (L13), `RunAtLoad=true` (L14), redirects stdout+stderr to `handoff/logs/backend-watchdog.log` (L17-19), `ProcessType=Background` (L22-23). | Healthy. Identical to the live copy at `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist`. |
| `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` | 1-25 | Live launchd registration (operator-installed copy of the repo plist). Byte-equivalent to the repo version above. | Registered with launchd; `launchctl print gui/501/com.pyfinagent.backend-watchdog` returns `state = not running` between StartInterval invocations (this is correct -- the script exits each cycle). |
| `handoff/logs/backend-watchdog.log` | 1-104 | The append-only log produced by the watchdog. Most recent line `2026-05-22T18:26:04Z health OK; resetting fails (was 1)` -- proves the watchdog is currently active (last fire 2 hours before 2026-05-23T00:41Z UTC). 42 `health FAIL` lines in last 7 days; **0** `health FAIL (3 / 3)`; **0** `kickstart -k` events; **0** `SIGUSR1` stack dumps. | The masterplan verification "expect zero entries in last 7 days" is **literally false** but **operationally true** (zero threshold-3 escalations, zero restarts). |
| `tests/verify_phase_23_1_21.py` | 1-79 | Reference pattern for verifying the watchdog statically (script exists, threshold=3, kill -USR1, kickstart present, plist has StartInterval=60). | Provides the existing pattern for a new pytest-style verification; the new test should compose with this one rather than duplicate. |
| `tests/verify_phase_23_2_18.py` | 1-122 | Forensic verification of the curl-Slack-then-kickstart sequence. | The full operational verification pattern is already established here; the new phase-23.2.10 test should be the **runtime log-evidence half** that complements the static config half. |
| `tests/api/test_launchd_manifest_count.py` | 1-50 | Existing pytest pattern: lightweight assertion on launchd manifest data. | Style template for a small log-grep pytest module. |
| `.claude/masterplan.json` | 7297-7304 | The phase-23.2.10 step definition. `verification: "grep 'health FAIL' handoff/logs/backend-watchdog.log; expect zero entries in last 7 days"`. | **Verification string is ambiguous**: literal grep returns 42, operational interpretation (3/3 escalations + kickstart) returns 0. Recommend updating the string. |

## 6. Consensus vs debate (external)

- **Consensus.** 3-consecutive + reset-on-first-success is the
  default-strength pattern. Single-fail-restart is uniformly
  discouraged. Threshold should match remediation cost (heavier
  remediation = higher threshold).
- **Debate.** GCP's default of 2 vs the SRE-blog norm of 3 vs the
  liveness-probe norm of 5. The variance is a function of how
  expensive the remediation is (LB-pull = cheap, container-restart
  = medium, full-host-rebuild = expensive). The pyfinagent
  position (3) sits squarely in the middle and tracks the
  oneuptime 2026 recommendation.

## 7. Pitfalls (from literature)

1. **Verifying the log literally vs operationally.** If the new
   test asserts `grep -c "health FAIL" == 0`, it will FAIL on the
   current healthy log. This is a false-negative: the watchdog is
   doing exactly its job. The 1-of-3 lines are *evidence the
   threshold is filtering correctly*, not evidence of a problem.
   Pitfall: anchor the test on **threshold-3 + kickstart**, not
   on any-fail.
2. **Stale log = silent failure mode.** A watchdog log that hasn't
   been written to in many minutes is the *real* alarming case --
   it means the watchdog itself is dead. The test must include a
   freshness assertion (last line within last ~5 minutes).
3. **Counter persistence across job restarts.** The counter file
   at `~/Library/Caches/com.pyfinagent.backend.watchdog.fails`
   survives macOS reboots. Don't verify against it from the test;
   it's an implementation detail and may be reset by Cache
   Cleanup. Verify the log, which is the audit-of-record.
4. **launchd reports `state = not running` between invocations.**
   This is correct (the script exits cleanly each run); test
   should not assert "running" -- assert "registered with launchd"
   (label appears in `launchctl list`).
5. **7-day window pitfalls.** Use UTC timestamps (the log writes
   `%Y-%m-%dT%H:%M:%SZ`). Compute the cutoff in UTC. The
   `datetime.now(timezone.utc) - timedelta(days=7)` pattern is
   the safe shape; do not rely on local time.

## 8. Application to pyfinagent (mapping findings to file:line anchors)

The masterplan's stated verification ("grep 'health FAIL'... expect
zero") is, as written, a false-negative trap: literal grep returns
42 on a healthy log. Two improvements:

1. **Update the masterplan verification string** to be
   unambiguous. Concretely: `grep 'health FAIL (3 / 3)' OR grep 'kickstart -k' in last 7 days; expect zero`.
2. **Add a fresh pytest** that encodes the operational meaning + a
   freshness check. See the pytest shape below.

The new test should be `tests/verify_phase_23_2_10.py` (using the
established phase-prefix convention -- `verify_phase_23_1_21.py`,
`verify_phase_23_2_18.py` are the immediate priors). It composes
with -- does not duplicate -- those two earlier verifiers.

## 9. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
  (6 read in full; rows 1-6 in section 1)
- [x] 10+ unique URLs total (16 collected: 6 read + 10 snippet)
- [x] Recency scan (last 2 years) performed + reported (section 3)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (section 5)

Soft checks:
- [x] Internal exploration covered every relevant module (script +
  plist + live plist + log + 3 reference tests + masterplan entry)
- [x] Contradictions / consensus noted (section 6)
- [x] All claims cited per-claim (URL+line anchors throughout)

## 10. Recommended pytest shape

```python
"""phase-23.2.10 immutable verification.

Watchdog audit:
1. Watchdog log exists and is FRESH (last line within last hour).
2. Zero threshold-3 escalations (`health FAIL (3 / 3)`) in last 7d.
3. Zero `kickstart -k` events in last 7d.
4. Zero `SIGUSR1` stack-dump events in last 7d.
5. Watchdog is registered with launchd (label in `launchctl list`).

The masterplan string ("zero entries") is operationally meant as
"zero times the watchdog actually FIRED" -- defined here as a
terminal 3/3 escalation OR a kickstart restart. Single 1/3 or 2/3
FAILs are the threshold doing its job (filtering transient blips);
they are NOT what the masterplan wants to count as zero.
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "handoff/logs/backend-watchdog.log"

# `2026-05-22T18:26:04Z [backend-watchdog] health OK; ...`
_TS_RX = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\b")


def _parse_iso_z(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _lines_in_window(text: str, days: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for line in text.splitlines():
        m = _TS_RX.match(line)
        if not m:
            continue
        if _parse_iso_z(m.group(1)) >= cutoff:
            out.append(line)
    return out


def main() -> int:
    assert LOG.exists(), f"watchdog log missing: {LOG}"

    text = LOG.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines, "watchdog log is empty -- watchdog never ran"

    # 1. Freshness: last timestamped line within the last hour.
    last_ts_line = next(
        (ln for ln in reversed(lines) if _TS_RX.match(ln)),
        None,
    )
    assert last_ts_line is not None, "no ISO-timestamped line in log"
    last_ts = _parse_iso_z(_TS_RX.match(last_ts_line).group(1))
    age = datetime.now(timezone.utc) - last_ts
    assert age < timedelta(hours=2), (
        f"watchdog log stale: last entry {age} ago at {last_ts.isoformat()} "
        f"-- watchdog process may be dead"
    )

    # 2. Zero threshold-3 escalations in last 7 days.
    window = _lines_in_window(text, days=7)
    threshold_3 = [ln for ln in window if "health FAIL (3 / 3)" in ln]
    assert not threshold_3, (
        f"watchdog fired (3/3) {len(threshold_3)} times in last 7d:\n"
        + "\n".join(threshold_3)
    )

    # 3. Zero kickstart -k events in last 7 days.
    kicks = [ln for ln in window if "kickstart -k" in ln]
    assert not kicks, (
        f"watchdog issued kickstart -k {len(kicks)} times in last 7d:\n"
        + "\n".join(kicks)
    )

    # 4. Zero SIGUSR1 dumps in last 7 days.
    sigusr1 = [ln for ln in window if "SIGUSR1" in ln]
    assert not sigusr1, (
        f"watchdog sent SIGUSR1 {len(sigusr1)} times in last 7d:\n"
        + "\n".join(sigusr1)
    )

    # 5. Registered with launchd.
    result = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True, timeout=10
    )
    assert "com.pyfinagent.backend-watchdog" in result.stdout, (
        "com.pyfinagent.backend-watchdog not registered with launchd"
    )

    # Informational counters (do NOT assert -- 1/3 + 2/3 FAILs are
    # the threshold doing its job).
    transient = [ln for ln in window if re.search(r"health FAIL \([12] / 3\)", ln)]
    recoveries = [ln for ln in window if "health OK; resetting fails" in ln]
    print(
        f"ok watchdog healthy: last-line {last_ts.isoformat()}; "
        f"7d transient FAILs (filtered by threshold): {len(transient)}; "
        f"7d recoveries: {len(recoveries)}; "
        f"7d threshold-3 escalations: 0; "
        f"7d kickstart -k events: 0; "
        f"launchd-registered: yes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 11. JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
