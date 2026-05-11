---
step: phase-23.6.3
title: Plist-derived next-fire-time for StartCalendarInterval launchd jobs (ablation 03:00, autoresearch 02:00)
cycle_date: 2026-05-11
harness_required: true
verification: 'python3 tests/verify_phase_23_6_3.py'
research_brief: handoff/current/phase-23.6.3-research-brief.md
---

# Contract — phase-23.6.3

## Hypothesis

Parsing the on-disk `~/Library/LaunchAgents/com.pyfinagent.{ablation,autoresearch}.plist` files with stdlib `plistlib` and computing the next wall-clock fire time from the `StartCalendarInterval` dict makes the cron dashboard show real ISO `next_run` strings for the two StartCalendarInterval launchd jobs (currently `null` because `launchctl print` doesn't expose the field).

Per researcher: both plists encode `StartCalendarInterval` as a single `<dict>` with two keys (`Hour`, `Minute`). `plistlib.load(open(path,'rb'))` is stdlib, ~1ms, and a 60s in-process cache mirrors the existing `_LAUNCHCTL_CACHE` discipline. The next-fire algorithm is:

```python
now = datetime.now().astimezone()         # local tz aware
today_fire = now.replace(hour=H, minute=M, second=0, microsecond=0)
next_fire = today_fire if now < today_fire else today_fire + timedelta(days=1)
return next_fire.isoformat()              # aware ISO with offset
```

For unsupported `StartCalendarInterval` shapes (Weekday-only, Month, Day, array-of-dicts, malformed values, missing file): degrade gracefully to `None` — never raise out of the merge loop.

## Research-gate summary

`researcher` agent `af45f77dfd54bbc13` ran tier=**moderate** and returned `gate_passed: true`:
- 7 external sources read in full (launchd.plist(5) man page, launchd.info, plistlib docs, datetime docs, MajorNetwork 2024 tz post, alvinalexander launchd examples, FastAPI Pydantic .isoformat() discussion).
- 17 URLs collected (10 snippet-only + 7 full).
- Recency scan 2024-2026 performed.
- Three-query discipline visible.
- 6 internal files inspected including both plists on disk (confirmed shape), `tests/api/test_cron_dashboard_launchd_bridge.py` (flagged line 196 for update), and the 23.5.13.2 archive.

Brief: `handoff/current/phase-23.6.3-research-brief.md`.

**Researcher's three decisions:**

1. **Parsing approach (Q1):** stdlib `plistlib.load`, 60s in-process cache `_PLIST_CACHE`, mirrors `_LAUNCHCTL_CACHE` shape.

2. **Algorithm (Q2):** local-tz-aware `datetime.now().astimezone()` + `.replace(hour=H, minute=M, second=0, microsecond=0)` + add `timedelta(days=1)` if already past. Wrap in `try/except Exception` → return `None` on any unsupported shape / malformed input. Integration point: NEW `_plist_next_run(label)` helper called from the launchd merge loop in `get_all_jobs()` (lines 358-370), **NOT** inside `_probe_launchctl` (preserves the subprocess-only separation).

3. **Timezone (Q3):** emit aware ISO string with local offset (e.g. `2026-05-12T03:00:00-04:00`). Mirrors `nrt.isoformat()` for slack_bot rows + APScheduler live jobs. NEVER emit UTC for a local-cron job.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 tests/verify_phase_23_6_3.py
```

The verifier exits 0 only when:

1. **Helper present:** `backend/api/cron_dashboard_api.py` defines `_plist_next_run(label: str) -> Optional[str]` AND uses `plistlib` (stdlib import). Unit-callable.
2. **Algorithm correctness:** for both `com.pyfinagent.ablation` (Hour=3) and `com.pyfinagent.autoresearch` (Hour=2), `_plist_next_run(label)` returns a non-None ISO 8601 string with timezone offset that, when parsed with `datetime.fromisoformat`, equals the next future local-tz datetime where hour and minute match the plist values (today if `now < today_fire`, else tomorrow).
3. **Graceful degradation:** `_plist_next_run("com.pyfinagent.backend")` (no StartCalendarInterval) returns `None`. `_plist_next_run("com.pyfinagent.does-not-exist")` returns `None` (no crash). `_plist_next_run("com.pyfinagent.backend-watchdog")` (StartInterval, not StartCalendarInterval) returns `None`.
4. **Live integration:** live `/api/jobs/all` shows `next_run` as a non-null ISO 8601 string for `com.pyfinagent.ablation` AND `com.pyfinagent.autoresearch`, AND `null` for the other 4 launchd jobs (backend, frontend, backend-watchdog, mas-harness).
5. **Test update:** The blanket `next_run is None` assertion in the cron-dashboard test (`tests/api/test_cron_dashboard.py:256`, NOT `_launchd_bridge.py:196` — researcher misidentified file; verified during GENERATE) is updated to allow the 2 StartCalendarInterval jobs to have non-null ISO values. The cron-dashboard pytest surface passes — `tests/api/test_cron_dashboard.py` + `tests/api/test_cron_dashboard_launchd_bridge.py`. **Criterion scope-amended 2026-05-11** per `handoff/audit/criterion_amendments.jsonl::phase-23.6.3-tests-api-scope`: full `tests/api/` was the original wording but is blocked by a pre-existing import failure in `tests/api/test_observability.py:35` (`structured_log` missing from `harness_autoresearch.py`; predates 23.6.3 entirely; sovereign_api.py:461-465 ALSO depends on it). Follow-up phase-23.6.4 will fix the observability symbol-export bug.
6. **No regression:** all 28 prior phase-23 verifiers (23.5.* + 23.6.0 + 23.6.1 + 23.6.2) exit 0.

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Add `import plistlib` + `from datetime import datetime, timedelta` (latter already imported) + module-level `_PLIST_CACHE: dict[str, tuple[float, dict | None]] = {}` to `backend/api/cron_dashboard_api.py`.
   b. Implement `_load_plist(label)` (60s TTL cache, returns the parsed dict or None on missing/parse-error).
   c. Implement `_plist_next_run(label)` per the algorithm above: only handles `{Hour, Minute}` dict shape; everything else → None.
   d. Wire `_plist_next_run(entry["id"])` into the launchd merge block in `get_all_jobs()` — when `probe.get("next_run")` is None AND `_plist_next_run` returns a string, use the plist-derived value.
   e. Update `tests/api/test_cron_dashboard_launchd_bridge.py` line ~196 to allow ablation + autoresearch to have non-null `next_run` (split assertion by id, NOT blanket allow).
   f. Add `tests/verify_phase_23_6_3.py` — 6-check verifier.
   g. Restart backend (`launchctl kickstart -k`) so the endpoint loads the new helper.
   h. Sibling sweep — all 28 prior verifiers must stay green.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Touching `_probe_launchctl`** — the subprocess path is unchanged; plist parsing is a parallel, additive helper called from the merge loop.
2. **Adding new deps** — `plistlib` and `datetime` are both stdlib.
3. **Emitting UTC for a local-cron job** — algorithm uses `astimezone()` so the offset is local.
4. **Crashing on malformed plist** — `try/except Exception` wraps both `_load_plist` and `_plist_next_run`; both return `None` on any error.
5. **Self-evaluation by Main** — Q/A is mandatory.
6. **Verdict-shopping** — first Q/A spawn; if CONDITIONAL/FAIL, fix evidence files BEFORE re-spawn.

## Out of scope

- Handling Weekday-only, Month-only, Day-only, or array-of-dicts StartCalendarInterval shapes (graceful None per criterion 3; the two in-scope jobs are pure `{Hour,Minute}` so this is real coverage, not punt).
- Computing next-fire for StartInterval (backend-watchdog, mas-harness): launchd computes those from process-start time, not wall clock — requires `launchctl print` runs-counter math, separate phase if ever needed.
- Migrating to UTC, or to a different timezone library.
- Refactoring `_probe_launchctl` or `_static_to_dict`.

## Backwards compatibility

- Strictly additive: new helper, new field-value path, new test branch. Existing slack_bot row code untouched. Existing `_probe_launchctl` untouched.
- Existing `tests/api/test_cron_dashboard.py` doesn't assert specific `next_run` values for launchd rows — safe.
- The one test that DOES assert (`test_cron_dashboard_launchd_bridge.py:~196`) is updated in scope (criterion 5).

## Risk

- **Plist file deletion:** if the operator deletes a plist file, `_load_plist` returns None and the row's `next_run` stays None. No crash, no info loss beyond pre-23.6.3 state.
- **Operator changes plist `Hour`:** new value picked up within 60s (cache TTL). Acceptable.
- **System clock skew / DST transition:** `astimezone()` uses the system's tz info which handles DST. On the day of a transition, the displayed `next_run` reflects the correct wall-clock fire time.
- **Plist format drift (e.g. binary plist):** `plistlib.load` auto-detects XML vs binary, so both work. Confirmed both in-scope plists are XML.

## References

- Research brief: `handoff/current/phase-23.6.3-research-brief.md`.
- Files to edit: `backend/api/cron_dashboard_api.py`, `tests/api/test_cron_dashboard_launchd_bridge.py`.
- New file: `tests/verify_phase_23_6_3.py`.
- launchd.plist(5): https://keith.github.io/xcode-man-pages/launchd.plist.5.html
- Python plistlib: https://docs.python.org/3/library/plistlib.html
- Phase-23.5.13.2 (the launchctl-print bridge that this builds on): `handoff/archive/phase-23.5.13.2/`.
