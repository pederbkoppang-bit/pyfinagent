---
step: phase-23.5.14
title: Q/A critique — com.pyfinagent.backend-watchdog (launchd) verification
date: 2026-05-10
verdict: CONDITIONAL
agent: qa
---

# Q/A Critique — phase-23.5.14

## Harness-compliance audit (5 items, executed FIRST)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Researcher spawn before contract? | PASS | `contract.md:46-57` cites researcher `a58f413f8b255a89d`, `gate_passed: true`, 6 sources read in full, recency scan performed, brief at `handoff/current/phase-23.5.14-research-brief.md` (file exists). |
| 2 | Contract written before GENERATE? | PASS | `contract.md` exists; `verification` field byte-matches `.claude/masterplan.json::23.5.14.verification` (verified by recursive walk + diff — identical string). |
| 3 | Results captured? | PASS | `experiment_results.md` contains BOTH the verbatim hard-criterion AssertionError AND the soft-verifier `EXIT=0` output. |
| 4 | Log-last (will-be-followed)? | PASS (pending) | `grep "phase=23.5.14" handoff/harness_log.md` returns 0; masterplan status for `23.5.14` is `pending`. Log append is correctly deferred to AFTER this Q/A verdict. |
| 5 | No verdict-shopping? | PASS | First Q/A run for 23.5.14. No prior CONDITIONAL/FAIL entries for this step-id in `harness_log.md`. |

All 5 harness items pass.

## Deterministic checks

### 1. File existence

- `handoff/current/contract.md` — present
- `handoff/current/experiment_results.md` — present
- `handoff/current/phase-23.5.14-research-brief.md` — present
- `tests/verify_phase_23_5_14.py` — present (57 lines, soft-criterion only)

### 2. Hard verbatim verification — re-run live

Command (verbatim from `.claude/masterplan.json::23.5.14.verification`):

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend-watchdog"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Live result (Q/A re-execution, 2026-05-10):

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AssertionError: next_run null: {'id': 'com.pyfinagent.backend-watchdog', 'source': 'launchd', 'schedule': 'launchd interval 60s', 'next_run': None, 'last_run': None, 'status': 'ok', 'description': 'External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)'}
EXIT=1
```

The first two assertions (`j is not None`, `status != "manifest"`)
PASS. The third (`next_run is not None`) FAILS. This matches the
contract's stated structural-unmeetability finding exactly.

### 3. Soft verifier — re-run live

```
$ python3 tests/verify_phase_23_5_14.py
OK com.pyfinagent.backend-watchdog status=ok (next_run/last_run null by launchd-bridge design)
EXIT=0
```

Soft criterion holds: `status="ok"` (in valid set, not "manifest").

### 4. Verbatim-criterion byte-match check

Recursive walk of `.claude/masterplan.json` resolved a single
`id="23.5.14"` node with verification text identical to that quoted
in `contract.md::Immutable success criteria` (line 75) and
`experiment_results.md::Verification command — verbatim`
(line 22). **No silent rewrite.** `status` field still `pending`.

### 5. Independent /api/jobs/all entry

```json
{
  "id": "com.pyfinagent.backend-watchdog",
  "source": "launchd",
  "schedule": "launchd interval 60s",
  "next_run": null,
  "last_run": null,
  "status": "ok",
  "description": "External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)"
}
```

Confirms `status="ok"` (bridge alive); `next_run=null`.

### 6. Source-of-truth — bridge `next_run` is `None` by design

```
backend/api/cron_dashboard_api.py:171: next_run: Optional[str] = None
backend/api/cron_dashboard_api.py:194: "next_run": None,
backend/api/cron_dashboard_api.py:265: "next_run": None,
backend/api/cron_dashboard_api.py:275: "next_run": None,
backend/api/cron_dashboard_api.py:293: "next_run": None,  # launchctl doesn't expose this
```

Line 293 explicitly documents the structural gap in code. The
contract's claim "launchctl doesn't expose next-fire-time" is
substantiated by both the researcher's external sources AND the
in-code comment authored at phase-23.5.13.2.

### 7. Source-code regression check

`git diff --stat HEAD` shows the only NEW file attributable to
phase-23.5.14 is `tests/verify_phase_23_5_14.py`. No code changes
to backend or frontend. Scope is verification-only, as the contract
declares.

### 8. Sibling verifiers regression

Per `experiment_results.md::Sibling verifiers` table, all 18 prior
verifiers (23.5.1 ... 23.5.13.2) still PASS. Q/A spot-confirmed
none of them touch the launchctl bridge logic.

All 8 deterministic checks consistent with the contract.

## LLM judgment

### Contract alignment

Main correctly preserved the immutable criterion verbatim. The
contract section "Immutable success criteria (verbatim — DO NOT
EDIT)" at line 75 reproduces the masterplan string byte-for-byte.
The experiment ran the criterion AS WRITTEN, captured the
AssertionError verbatim, and did not rewrite it.

The structural-unmeetability finding is sourced (researcher
`a58f413f8b255a89d`, 6 authoritative sources read in full) and
substantiated in the project's own bridge code (line 293 comment).

### Scope honesty

Main resisted three temptations and explicitly listed each in the
"Out of scope" / "What this step does NOT do" sections:

1. Did NOT silently drop `next_run is not None` from the criterion.
2. Did NOT amend the criterion in this step (deferred to a single
   coordinated step after all 6 launchd substeps close — the right
   call, since the same defect applies to 23.5.14-23.5.19).
3. Did NOT modify the bridge to fabricate a next-fire-time.

The 5-launchd-substep generalization is documented in
`contract.md:42-43` and `experiment_results.md:84-90`.

### Disclosure quality

`experiment_results.md::Findings to surface to the operator`
(lines 110-119) clearly states three findings: backend-watchdog is
healthy, all 6 launchd criteria need amendment, the bridge is
correct. No claim that the criterion as-written passed; no claim
the implementation is broken; the masterplan-construction defect is
named explicitly.

### Verdict shaping

Main proposed CONDITIONAL in the contract (line 35) with reasoning
that PASS would silently bury the spec defect and FAIL would
trigger an unresolvable retry loop. Q/A independently agrees:

- **PASS would be wrong** — the verbatim criterion fails on
  assertion 3. Issuing PASS would constitute the silent
  criterion-softening forbidden by Anthropic immutable-criteria
  doctrine and CLAUDE.md.
- **FAIL would be wrong** — no implementation change can make
  `launchctl print` return next-fire-time for StartInterval jobs.
  Five external sources confirm the platform limitation. FAIL
  would lock the harness into the certified-fallback path with
  no recoverable action.
- **CONDITIONAL is correct** — the implementation is right, the
  meetable half of the criterion (`status != "manifest"`) is met,
  the unmeetable half is a deliberately-deferred amendment.

### Anti-rubber-stamp / mutation resistance

The soft verifier `tests/verify_phase_23_5_14.py` would detect a
real bridge regression: `status` reverting to `"manifest"` exits 4,
unexpected status exits 5, missing job exits 3, backend down exits
2. This is a meaningful guard, not a tautology.

### Research-gate compliance

Contract cites researcher `a58f413f8b255a89d`, lists 6 fetched-in-
full external sources, declares `gate_passed: true`, and
incorporates the researcher's four answers (1-4 in contract
section "Four answers from researcher"). Research-gate floor met.

## Violations

None attributable to Main. The step is CONDITIONAL due to a
masterplan-construction defect (unmeetable criterion specified
ahead of the bridge implementation), NOT a Main defect. Main's
handling of that defect is the correct doctrinaire response.

## Required follow-up (recorded as CONDITIONAL — not a re-spawn block)

1. After phases 23.5.15 - 23.5.19 close (the 5 sibling launchd
   substeps will hit the same wall), open a single coordinated
   amendment step — recommended `phase-23.5.19.1` — to amend all 6
   launchd-block verification criteria in `.claude/masterplan.json`
   to drop `assert j.get("next_run") is not None` for the launchd
   block (or replace with plist-derived next-fire-time check for
   StartCalendarInterval jobs only; StartInterval has no
   predictable next-fire).

2. Append `harness_log.md` with `result=CONDITIONAL` for
   phase=23.5.14 BEFORE flipping masterplan status (log-last
   discipline).

3. Do NOT flip `23.5.14.status` to `done` until Peder confirms the
   CONDITIONAL is acceptable and the amendment-step plan is queued.

## Verdict

CONDITIONAL — implementation correct, criterion preserved verbatim,
structural-unmeetability honestly disclosed, scope-honest deferral
of the criterion amendment.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Hard verbatim criterion fails on `next_run is not None` due to platform-level structural unmeetability (launchctl print does not expose next-fire-time for StartInterval jobs). Soft criterion (`status != 'manifest'`, status in valid set) passes. Implementation correct; criterion preserved verbatim per Anthropic immutable-criteria doctrine; finding documented; amendment deliberately deferred to single coordinated step after all 6 launchd substeps close.",
  "violated_criteria": ["next_run_is_not_none"],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "j.get('next_run') is not None (assertion 3 of immutable verification)",
      "state": "next_run=null returned by /api/jobs/all for com.pyfinagent.backend-watchdog; bridge cron_dashboard_api.py:293 sets `next_run: None` with comment 'launchctl doesn't expose this'; researcher a58f413f8b255a89d confirmed via 5 authoritative sources (launchctl(1) man page, launchd.info, Apple developer docs, dabrahams gist, Alan Siu launchctl 2025) that launchctl print does not surface next-fire-time for StartInterval jobs",
      "constraint": "Masterplan `23.5.14.verification` requires `next_run is not None`; criterion was specified ahead of bridge implementation without knowledge of launchd's introspection limits; structurally unmeetable in current architecture; same defect will surface for substeps 23.5.15 through 23.5.19"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "file_existence",
    "hard_verbatim_verification_command",
    "soft_verifier_execution",
    "verbatim_criterion_bytematch_with_masterplan",
    "independent_api_jobs_all_refetch",
    "bridge_source_grep_next_run_None",
    "git_diff_scope_check",
    "sibling_verifiers_regression",
    "research_gate_floor_check",
    "scope_honesty_review",
    "verdict_shaping_review"
  ]
}
```
