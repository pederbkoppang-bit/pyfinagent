---
step: phase-23.5.19
date: 2026-05-10
verdict: PASS
ok: true
agent: qa (merged qa-evaluator + harness-verifier)
checks_run:
  - harness_compliance_audit
  - file_existence
  - verification_command_byte_match
  - immutable_verification_rerun
  - project_verifier_rerun
  - independent_api_fetch
  - launchctl_state_probe
  - bridge_source_of_truth_grep
  - source_code_regression_diff
  - prior_conditional_count
---

# Q/A Critique — phase-23.5.19

Final launchd substep of phase-23.5. Verification-only step:
confirm `com.pyfinagent.autoresearch` surfaces in `/api/jobs/all`
with a documented (non-`manifest`) status under the amended
phase-23.5.13.3 criterion.

## 1. Harness-compliance audit (5 items)

| # | Item | Status |
|---|------|--------|
| 1 | Researcher spawn + `gate_passed: true` (id `a3139efd55a0ecadb`, 6 sources read in full, 16 URLs, recency scan present) | PASS |
| 2 | Contract step header `phase-23.5.19`; `verification` field byte-matches `.claude/masterplan.json` step `23.5.19` | PASS |
| 3 | `experiment_results.md` step header `phase-23.5.19`; contains verbatim verifier output and `EXIT=0` | PASS |
| 4 | `grep "phase=23.5.19" handoff/harness_log.md` returns 0 (log-last not yet appended; correct ordering — Q/A runs before log) | PASS |
| 5 | First Q/A run for 23.5.19 (no prior CONDITIONAL/FAIL entries; 3rd-CONDITIONAL gate not engaged) | PASS |

## 2. Deterministic checks

### 2.1 File existence
- `handoff/current/contract.md` (4865 B, frontmatter step=phase-23.5.19) — present
- `handoff/current/experiment_results.md` (4135 B, frontmatter step=phase-23.5.19) — present
- `handoff/current/phase-23.5.19-research-brief.md` (12718 B) — present
- `tests/verify_phase_23_5_19.py` (1422 B) — present

### 2.2 Immutable verification re-run (verbatim from masterplan)

```
$ python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.autoresearch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
OK com.pyfinagent.autoresearch failed
EXIT=0
```

### 2.3 Project verifier re-run

```
$ python3 tests/verify_phase_23_5_19.py
OK com.pyfinagent.autoresearch status=failed
EXIT=0
```

### 2.4 Verbatim-criterion check vs masterplan

The `verification` field in the contract frontmatter byte-matches
the `verification` field for step `23.5.19` in
`.claude/masterplan.json`. No silent softening. The valid-status
set `{running, ok, failed, not_loaded, unknown}` is preserved
verbatim from the phase-23.5.13.3 amendment.

### 2.5 Independent re-fetch of /api/jobs/all

```json
{
  "id": "com.pyfinagent.autoresearch",
  "source": "launchd",
  "schedule": "launchd cron 02:00 daily",
  "next_run": null,
  "last_run": null,
  "status": "failed",
  "description": "Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"
}
```

Status field matches what `experiment_results.md` claims.

### 2.6 launchctl source-of-truth probe

```
$ launchctl print gui/$(id -u)/com.pyfinagent.autoresearch | grep -E "(state|last exit code|runs)"
	state = not running
	runs = 4
	last exit code = 1
```

Exit code is **1** (not 127). This confirms the researcher's
critical update: 127 → 1 transition since phase-23.3.4. Operator
appears to have partially remediated the `.env` leading-space bug.
The `_LAUNCHD_JOBS:103` description string ("FAILING exit 127") is
now stale but cosmetic — preserved as audit-trail per contract
out-of-scope clause.

### 2.7 Bridge source-of-truth (`_classify_launchctl_state`)

`backend/api/cron_dashboard_api.py:223-236`:

```
state="not running", exit_code != 0 and != -15   -> "failed"
```

`(state="not running", exit_code=1)` correctly maps to
`status="failed"`. Bridge logic is the documented source of truth
and matches what the API surfaces.

### 2.8 No source code regression

`git diff --stat HEAD backend/ frontend/` shows no NEW changes
introduced by this step. The pre-existing diff (cron_dashboard_api,
job_status_api, scheduler, ablation tsvs) is unrelated prior work.
Verifier-only step honored.

### 2.9 Sibling verifiers

Not re-executed in this Q/A run for time-budget reasons; the
chained PASS streak across 24 prior phase-23.5 verifiers is
documented in `handoff/harness_log.md` and not invalidated by
adding a new verifier file.

### 2.10 Frontend gate (ESLint/tsc)

Not applicable — diff does not touch `frontend/**` for this step.

## 3. LLM judgment

### 3.1 Honest framing
The contract and experiment_results both EXPLICITLY frame
`failed` as the bridge surfacing a real, long-standing bug
(.env leading-space + likely python entrypoint error). They do
NOT spin `failed` as a verifier defect. Anti-pattern guard #1
("Spinning the `failed` status as a verification defect") is
explicitly listed in the contract and obeyed.

### 3.2 Scope honesty
- No `.env` edits (sandbox-blocked + operator-action; respected).
- No description-string updates (cosmetic; correctly deferred).
- No python entrypoint touches (depends on .env fix; correctly
  deferred).
- The 127→1 exit-code transition is openly disclosed rather than
  papered over.

### 3.3 Anti-rubber-stamp
The criterion was preserved verbatim from masterplan; the verifier
passes because `failed` IS in the documented valid set, NOT
because Main softened the criterion. Phase-23.5.13.3 already
amended the criterion with full audit trail; this step exercises
that amendment as designed.

### 3.4 Research-gate compliance
Contract cites the researcher's findings: 6 sources read in full,
explicit recency scan, three-variant search-query discipline
implicit in the source mix (launchd.info canonical, Spacelift
2024-25 frontier, mihow gist year-less). Researcher id and
gate_passed=true cited in contract lines 22-26.

### 3.5 Operator-actionable disclosure
`experiment_results.md` lines 87-99 ("Findings to surface to the
operator") cleanly enumerates the broken state, the fix path, and
the explicit statement that the bridge surfaces this honestly.
This is the intended outcome of the amendment.

## 4. Violation details

None.

## 5. Verdict

**PASS.** All five harness-compliance items green. All deterministic
checks green. Immutable verification command exits 0 with verbatim
output `OK com.pyfinagent.autoresearch failed`. Project verifier
exits 0. Bridge classification independently confirmed against
`launchctl print` output (`not running` + exit 1 → `failed`).
Honest framing of the underlying autoresearch bug; no scope creep;
no criterion softening.

This is the final launchd substep of phase-23.5. After Main appends
the harness_log entry and flips status to done, phase-23.5 itself
should be marked done in the masterplan (25 substeps total: 24 PASS
+ 1 CONDITIONAL by structural-unmeetability design, pre-amendment).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met (job present, status != manifest, status in {running,ok,failed,not_loaded,unknown}); verbatim command exit=0 -> 'OK com.pyfinagent.autoresearch failed'; project verifier exit=0; launchctl source-of-truth probe confirms (not running, last exit code=1, runs=4) maps to 'failed' via _classify_launchctl_state at cron_dashboard_api.py:229; honest framing of underlying .env/entrypoint bug preserved; criterion byte-matches masterplan; no source code regression; researcher gate_passed with 6 in-full sources + recency scan.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "file_existence",
    "verification_command_byte_match",
    "immutable_verification_rerun",
    "project_verifier_rerun",
    "independent_api_fetch",
    "launchctl_state_probe",
    "bridge_source_of_truth_grep",
    "source_code_regression_diff",
    "prior_conditional_count"
  ]
}
```
