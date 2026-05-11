---
step: phase-23.6.3
title: Plist-derived next-fire-time for StartCalendarInterval launchd jobs — Q/A critique
cycle_date: 2026-05-11
verdict: PASS
qa_agent: qa-phase-23.6.3
---

# Q/A critique — phase-23.6.3

## Harness-compliance audit (5 items)

1. **Researcher spawned BEFORE contract?** PASS. `handoff/current/phase-23.6.3-research-brief.md` exists, ends with JSON envelope: `external_sources_read_in_full=7`, `urls_collected=17`, `recency_scan_performed=true`, `internal_files_inspected=6`, `gate_passed=true`. Contract's "Research-gate summary" names researcher agent `af45f77dfd54bbc13`, tier=moderate.
2. **Contract written BEFORE generate?** PASS. `contract.md` has step id `phase-23.6.3`, six immutable success criteria, anti-patterns and out-of-scope sections present. Criterion 5 carries an explicit amendment footnote referencing `handoff/audit/criterion_amendments.jsonl::phase-23.6.3-tests-api-scope` (legitimacy audited in §1.5 below).
3. **Results written?** PASS. `experiment_results.md` frontmatter `step: phase-23.6.3`, contains verbatim verifier output `PASS (6/6) EXIT=0`, lists files changed including the git-stash incident note and masterplan.json phase-23.6.4 follow-up addition.
4. **Log-last + status-flip-last?** PASS. Grep for `phase-23.6.3` in `handoff/harness_log.md` returns 0; the masterplan log/status flip correctly held until after this Q/A verdict.
5. **No second-opinion-shopping?** PASS. First Q/A spawn for 23.6.3; no prior verdict for this step id in evaluator_critique.md or harness_log.md.

**3rd-CONDITIONAL auto-FAIL check:** 0 prior phase-23.6.3 CONDITIONALs in harness_log.md. Not applicable.

## Criterion-amendment audit (STEP 1.5 — anti-rubber-stamp)

The amendment id `phase-23.6.3-tests-api-scope` narrows criterion 5 from "full `tests/api/`" to "cron-dashboard test files only". Per the 23.5.13.3 doctrine, this is legitimate iff ALL of:

- **Pre-existing failure?** PASS. `grep structured_log backend/api/harness_autoresearch.py` returns NOTHING (symbol missing). `grep _read_audit_tail backend/api/sovereign_api.py` returns 3 call-site hits (257, 279, 296) confirming sovereign_api.py also broken-imports it. `git log --oneline -3 -- backend/api/harness_autoresearch.py` shows last commit is `22e78958 phase-10`, far predating 23.6.3.
- **23.6.3 does NOT touch the offending file?** PASS. `git diff HEAD -- backend/api/harness_autoresearch.py` is empty.
- **Real follow-up registered?** PASS. `.claude/masterplan.json` has `"id": "23.6.4"` with `status: pending`: "Restore missing observability symbols in backend/api/harness_autoresearch.py (surfaced during 23.6.3)".
- **Amendment documented in contract?** PASS. Contract criterion 5 carries the explicit footnote `Criterion scope-amended 2026-05-11 per handoff/audit/criterion_amendments.jsonl::phase-23.6.3-tests-api-scope`.

All four conditions met → amendment is LEGITIMATE under the 23.5.13.3 doctrine, not verdict-rigging.

## Deterministic checks

- (a) Verifier exit: `python3 tests/verify_phase_23_6_3.py` → `PASS (6/6) EXIT=0`. All 6 immutable checks PASS (helper present + plistlib; algorithm correctness; graceful degradation 3-of-3; live API; cron-dashboard pytest 30 passed in 0.10s; 28 sibling verifiers green).
- (b) Syntax: `cron_dashboard_api.py`, `tests/verify_phase_23_6_3.py`, `tests/api/test_cron_dashboard.py` all parse clean.
- (c) Direct helper: `_plist_next_run('com.pyfinagent.ablation')` → `2026-05-12T03:00:00+02:00` (tz-aware, hour=3); `_plist_next_run('com.pyfinagent.autoresearch')` → `2026-05-12T02:00:00+02:00` (hour=2). Both ISO strings parse cleanly.
- (d) Negative cases: `com.pyfinagent.backend` (KeepAlive) → None; `com.pyfinagent.backend-watchdog` (StartInterval) → None; `com.pyfinagent.does-not-exist` → None. All three return None gracefully, no crash.
- (e) Live API `/api/jobs/all`: ablation + autoresearch have non-null ISO `next_run`; backend, frontend, backend-watchdog, mas-harness all `None`. Exact criterion 4 shape.

## LLM-judgment review

- **Contract alignment:** Implementation matches researcher's three recommendations verbatim — `plistlib` stdlib + 60s TTL cache; `datetime.now().astimezone()` + `.replace()` + `+timedelta(days=1)` if past; aware ISO with local offset (`+02:00`). Integration is at the merge loop in `get_all_jobs()`, NOT inside `_probe_launchctl` — preserves subprocess separation as researcher specified.
- **Mutation resistance:** Verifier check 2 asserts semantic content — parses returned ISO with `datetime.fromisoformat`, checks hour/minute match the plist values, requires `tzinfo is not None`, requires the returned datetime is in the future. A no-op `return None` implementation cannot pass check 2. A naive (non-tz-aware) implementation cannot pass either. Real mutation resistance present.
- **Anti-rubber-stamp / subtle issues:** DST handling — `astimezone()` uses the OS tzdata so a transition day will report the wall-clock fire time correctly (verified by checking it's `+02:00` which is current local CEST; on the autumn switch the offset will shift to `+01:00` naturally). Cache invalidation 60s TTL is correct (mirror of `_LAUNCHCTL_CACHE`). Out-of-range Hour/Minute guarded by `try/except Exception` around `.replace()` (which raises ValueError on Hour=25).
- **Scope honesty:** `git diff HEAD --stat` shows the only NEW code edits in 23.6.3's commit window are `backend/api/cron_dashboard_api.py` + `tests/api/test_cron_dashboard.py` + new `tests/verify_phase_23_6_3.py` + handoff artifacts + masterplan 23.6.4 follow-up + amendment audit record. The other diffs visible in `git status` (job_status_api.py, slack_bot/scheduler.py) predate 23.6.3 entirely (last touched in phase-23.3.2/.3.3 commits) and are not in this step's work.
- **Research-gate compliance:** Brief has 7 sources read-in-full (>= 5 floor), 17 URLs collected (>= 10 floor), recency scan 2024-2026 explicit, three-query discipline visible per topic (current-year / last-2-year / year-less), JSON envelope `gate_passed: true`.
- **Incident transparency:** The git-stash incident note is present in `experiment_results.md` lines 112-114, honestly explains what happened (operator ran `git stash` for diagnosis which captured all in-flight phase-23.6.* mods), how recovery was done (`git show stash@{0}:<path>` per file, NOT `stash pop`), and what was lost (nothing — 39 tracked files restored verbatim, stash dropped clean). Transparency is the canonical bar.

## Violated criteria

None.

## Verdict
PASS — All 6 immutable criteria met, criterion amendment is legitimate under the 23.5.13.3 doctrine (pre-existing failure, 23.6.3 doesn't touch the file, real follow-up phase-23.6.4 registered, documented in contract), deterministic checks all green, scope honest, incident transparently disclosed. Main may now append to `harness_log.md` then flip masterplan status.
