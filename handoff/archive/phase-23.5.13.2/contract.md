---
step: phase-23.5.13.2
title: Bridge launchd job state into /api/jobs/all (launchctl print parser)
cycle_date: 2026-05-10
harness_required: true
verification: "python3 -c 'import json,urllib.request as u; r=json.load(u.urlopen(\"http://localhost:8000/api/jobs/all\")); ld=[j for j in r[\"jobs\"] if j[\"source\"]==\"launchd\"]; assert len(ld)==6, f\"want 6 launchd, got {len(ld)}\"; non_manifest=[j for j in ld if j[\"status\"]!=\"manifest\"]; assert len(non_manifest)>=4, f\"expect >=4 launchd jobs surfaced\"; print(\"OK\", len(ld), \"launchd;\", len(non_manifest), \"non-manifest\")' && python3 tests/verify_phase_23_5_13_2.py"
research_brief: handoff/current/phase-23.5.13.2-research-brief.md
---

# Contract — phase-23.5.13.2

## Hypothesis

Replacing the `_static_to_dict(entry, source="launchd")` call at
`cron_dashboard_api.py:235-236` with a merge block that consults
`launchctl print gui/$(id -u)/<label>` per launchd entry surfaces
real `status` (running/ok/failed/not_loaded/unknown) for all 6
launchd jobs. The criterion `status != "manifest"` for ≥4 jobs is
satisfied.

Per researcher (analysis verified empirically on this machine,
Sequoia, gui/501, 2026-05-10):

- **Parseable fields:** `state` (running/not running), `last exit
  code` (only when not running), `pid`, `runs`, `active count`. NO
  next-fire-time and NO last-run timestamp from launchctl. So
  `next_run` stays `None`, `last_run` stays `None`.
- **State-to-status mapping** (researcher Answer 3):
  | launchctl signal | dashboard `status` |
  |---|---|
  | `returncode != 0` (not loaded / booted out) | `"not_loaded"` |
  | `state = running` | `"running"` |
  | `state = not running`, no `last exit code` | `"ok"` |
  | `state = not running`, `last exit code = 0` | `"ok"` |
  | `state = not running`, `last exit code = -15` (SIGTERM) | `"ok"` |
  | `state = not running`, `last exit code > 0` | `"failed"` |
  | subprocess exception / timeout | `"unknown"` |
  Never emits `"manifest"` for launchd entries after this step.
- **Caching:** 30-second module-level dict
  `{label: (result_dict, monotonic_timestamp)}`. ~50-80ms per
  launchctl call × 6 = ~400ms uncached; 30s TTL keeps dashboard
  responsive without staleness for a local-only deployment.

The criterion's `>=4 non-manifest` floor allows for transient
states (e.g., the bootout'd mas-harness returns `"not_loaded"`
which IS non-manifest, satisfying the criterion). Realistically
all 6 should land non-manifest after the bridge ships.

## Research-gate summary

`researcher` agent `a961f967a76936258` ran tier=moderate and
returned `gate_passed: true`:
- 7 external sources read in full (Alan Siu 2025 launchctl,
  launchd.info, launchctl(1) man page, Python subprocess docs,
  Jonathan Levin newosxbook, TTL LRU cache Medium, masklinn
  cheat sheet).
- 7 snippet-only + 7 read-in-full = 14 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 10 internal files inspected (incl. live `launchctl print` output
  for 2 of the 6 jobs).

Brief: `handoff/current/phase-23.5.13.2-research-brief.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.13.2.verification`:

```
python3 -c 'import json,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); ld=[j for j in r["jobs"] if j["source"]=="launchd"]; assert len(ld)==6, f"want 6 launchd, got {len(ld)}"; non_manifest=[j for j in ld if j["status"]!="manifest"]; assert len(non_manifest)>=4, f"expect >=4 launchd jobs surfaced"; print("OK", len(ld), "launchd;", len(non_manifest), "non-manifest")' && python3 tests/verify_phase_23_5_13_2.py
```

Decoded:
1. The verification command returns 6 launchd entries.
2. ≥4 of them have `status != "manifest"`.
3. `tests/verify_phase_23_5_13_2.py` exits 0.

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Add module-level helper `_launchctl_state(label: str) -> dict`
      to `backend/api/cron_dashboard_api.py` (above `get_all_jobs()`).
      - Calls `subprocess.run(["launchctl", "print",
        f"gui/{os.getuid()}/{label}"], capture_output=True,
        text=True, timeout=5)`.
      - Returns `{status, last_exit_code, pid, runs, active_count,
        next_run: None, last_run: None}`.
      - Per the state-to-status mapping table above; falls back
        to `"unknown"` on subprocess exception/timeout.
   b. Add module-level cache `_LAUNCHCTL_CACHE: dict[str, tuple[dict, float]]`
      with 30s TTL.
   c. Replace `cron_dashboard_api.py:235-236` with a merge block:
      ```python
      for entry in _LAUNCHD_JOBS:
          probe = _launchctl_state(entry["id"])
          jobs.append({
              "id": entry["id"],
              "source": "launchd",
              "schedule": entry.get("schedule", "?"),
              "next_run": probe.get("next_run"),
              "last_run": probe.get("last_run"),
              "status": probe.get("status", "unknown"),
              "description": entry.get("description", entry["id"]),
          })
      ```
   d. Update existing test
      `tests/api/test_cron_dashboard.py:test_jobs_all_launchd_unaffected_by_slack_bot_bridge`
      — that test asserted launchd `status == "manifest"` (the
      guard from 23.5.2.5 saying "the slack_bot bridge didn't leak
      into launchd"). With this step's bridge, that assertion is
      no longer correct. Refactor to assert that launchd jobs
      have non-manifest status when `_launchctl_state` is mocked,
      OR delete the test if it's now redundant.
   e. Add new tests `tests/api/test_cron_dashboard_launchd_bridge.py`
      covering:
      - state=running → status="running"
      - state=not running, exit=0 → status="ok"
      - state=not running, exit=-15 → status="ok" (SIGTERM)
      - state=not running, exit>0 → status="failed"
      - returncode!=0 (bootout) → status="not_loaded"
      - subprocess timeout → status="unknown"
      - cache hit returns same dict without re-invoking subprocess
   f. Add `tests/verify_phase_23_5_13_2.py` — 4-check verifier:
      1. `_launchctl_state` symbol present in cron_dashboard_api.
      2. The new test file passes.
      3. The bridge merge is wired (grep `_launchctl_state(entry["id"])`
         in get_all_jobs body).
      4. Live `/api/jobs/all` returns ≥4 non-manifest launchd
         entries.
   g. Restart backend (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`)
      to pick up the bridge code.
   h. Run the immutable verification + sibling verifiers.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded (≥3)

1. **`pyobjc-framework-ServiceManagement`** — overkill; subprocess
   to `launchctl print` is the canonical lightweight path.
2. **Per-request fresh probes (no cache)** — would add ~400ms to
   every dashboard load × 6 launchctl forks. 30s TTL is sufficient
   for a local-only deployment.
3. **Coupling launchd path to slack_bot bridge** — keep the
   `job_status_api.get_registry_snapshot()` consumer separate;
   launchd has a different evidence source (live subprocess,
   not pushed registry).
4. **Mutating `_static_to_dict`** — leave it untouched (still used
   by `_SLACK_BOT_JOBS` legacy path if needed).
5. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Fixing autoresearch exit-code 1 (the .env-leading-space bug
  documented in phase-23.3.5; operator-fix-required).
- Bootstrapping mas-harness back (will do at session end).
- Refactoring the 6 plist files.
- Adding next-run-time discovery for KeepAlive jobs (launchd
  doesn't expose it; would need plist parsing — separate concern).

## Backwards compatibility

- `_LAUNCHD_JOBS` static manifest unchanged.
- `_static_to_dict` left in place (still callable; just not called
  by the launchd path now).
- Slack-bot bridge unchanged.
- Existing test
  `test_jobs_all_launchd_unaffected_by_slack_bot_bridge` will need
  to be updated — its current assertion `status == "manifest"`
  becomes false with this step. Refactor or delete; not a
  regression of a real invariant.

## Risk

- **launchctl invocation needs `os.getuid()`** — researcher used
  `gui/501` literally; production code MUST resolve dynamically.
- **launchctl can be slow on first invocation** — cache absorbs
  this. Timeout=5s ensures a hung subprocess can't block
  `/api/jobs/all`.
- **mas-harness was bootout'd** earlier this session, so it will
  show `status="not_loaded"` until I bootstrap it back at session
  end. This is non-manifest (satisfies criterion) and is the
  honest live state.
- **autoresearch shows `last exit code = 1`** per researcher's
  empirical probe (was 127 per phase-23.3.4; partial recovery).
  Will surface as `status="failed"`. Honest signal.

## References

- Research brief:
  `handoff/current/phase-23.5.13.2-research-brief.md` (researcher
  `a961f967a76936258`, 7 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.13.2.verification`.
- Files to edit:
  - `backend/api/cron_dashboard_api.py` (new helper + cache + merge
    block at lines 235-236).
  - `tests/api/test_cron_dashboard.py` (refactor the
    launchd-unaffected test).
- New files:
  - `tests/api/test_cron_dashboard_launchd_bridge.py` (≥6 tests).
  - `tests/verify_phase_23_5_13_2.py` (4-check verifier).
- Phase-23.5.2.5 bridge template:
  `handoff/archive/phase-23.5.2.5/contract.md`.
- Alan Siu launchctl 2025:
  https://www.alansiu.net/2025/05/28/using-new-launchctl-subcommands-to-check-for-and-reload-launch-daemons/
- launchd.info: https://www.launchd.info/
