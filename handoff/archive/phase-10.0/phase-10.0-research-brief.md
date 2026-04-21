# Research Brief: phase-10.0 "Retire phase-8.5.7 nightly cron; point to sprint_calendar.yaml"

**Date:** 2026-04-19
**Tier:** simple (closure-style housekeeping)
**Assumption:** No external WebFetch required. This is an internal audit and supersede-log verification gate.

---

## Objective

Verify that the already-authored supersede log at `handoff/phase-10.0-supersede-85-7.md` satisfies the immutable verification criteria:

```
test -f handoff/phase-10.0-supersede-85-7.md && grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md
```

Success criteria: `supersede_log_landed`, `phase_8_5_7_marked_superseded`.

## Output Format

Internal audit report with file:line anchors, closure-gate rationale, and JSON envelope.

## Tool Scope

Read-only. Grep, Glob, Read. No external WebFetch. No code modifications.

## Task Boundaries

This step is documentation closure only -- no code changes, no parameter updates, no schema migrations.

---

## Internal Audit Findings

### File 1: `handoff/phase-10.0-supersede-85-7.md`

Existence: confirmed on disk.

Key line anchors:
- Line 4: `**Superseded by:** phase-10.1 Sprint calendar config (\`backend/autoresearch/sprint_calendar.yaml\`)` -- satisfies `grep -q 'sprint_calendar.yaml'`.
- Line 8-12: Describes phase-8.5.7 as nightly APScheduler cron, ~100 experiments, budget-gated.
- Lines 17-19: Before/after cadence comparison -- nightly cron replaced by 2 weekly slots (Thu batch + Fri promotion gate) governed by `sprint_calendar.yaml`.
- Line 29: `phase-8.5.7 remains \`status: done\` in masterplan (historical record)` -- confirms the step is NOT being re-flipped; 10.0 is the cross-reference only.
- Lines 35-37: References section names all three load-bearing files: `cron.py`, `sprint_calendar.yaml`, `harness_log.md`.

Verification criteria: BOTH conditions pass.
- `test -f handoff/phase-10.0-supersede-85-7.md` -- file exists.
- `grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md` -- found at line 4, line 19, and line 36.

### File 2: `backend/autoresearch/cron.py` (phase-8.5.7 scaffold)

- Line 1: Docstring confirms `phase-8.5.7 Overnight autoresearch orchestration cron`.
- Line 17: `"""In-memory registration shim. Real APScheduler wiring deferred to phase-9."""` -- confirms it is a scaffold, not a live runner.
- Line 24: `cron_schedule: str = "0 2 * * *"` -- nightly 2am job being retired.
- Line 29-36: `replace_existing=True` idempotency key noted; file retained on disk as historical artifact per supersede log line 31.

Status: scaffold retained; not deleted; no active wiring beyond in-memory shim.

### File 3: `backend/autoresearch/sprint_calendar.yaml` (phase-10.1)

- Line 1: Comment: `Replaces phase-8.5.7 nightly cron with a disciplined 2-slot-per-week cadence`.
- Lines 13-14: `version: "1.0"`, `authored_at: "2026-04-20"`.
- Lines 18-27: Thursday slot (`thu_batch`, 22:00 UTC) + Friday slot (`fri_promotion`, 21:00 UTC).
- Lines 30-39: `monthly_anchor` with `hitl: true`, Sortino gate, owner approval required.
- Lines 41-44: `references` block back-links to `handoff/phase-10.0-supersede-85-7.md`, `budget.py`, `gate.py`, `harness_log.md`.

Cross-reference: `sprint_calendar.yaml` line 41 points at the supersede doc; supersede doc line 36 points at `sprint_calendar.yaml`. Bidirectional linkage confirmed.

### Masterplan status check

- `phase-8.5.7` status: `done` (confirmed via masterplan.json) -- historical record; no status flip required by 10.0.
- `phase-10.0` status: `pending` -- this gate clears the research requirement; GENERATE proceeds next.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `handoff/phase-10.0-supersede-85-7.md` | 41 | Supersede log for phase-8.5.7 retirement | Present; `sprint_calendar.yaml` cited at lines 4, 19, 36 |
| `backend/autoresearch/cron.py` | 78 | phase-8.5.7 APScheduler scaffold | Retained as historical scaffold; in-memory shim only |
| `backend/autoresearch/sprint_calendar.yaml` | 44 | phase-10.1 authoritative cadence config | Present; back-links to supersede doc at line 41 |
| `.claude/masterplan.json` | -- | Task tracker | phase-8.5.7=done, phase-10.0=pending |

---

## Closure-Gate Rationale

This step follows the same closure-gate pattern established by:
- `qa_78_v1` (phase-7.8 deferred doc)
- `qa_850_v1` (retire 2.10 stub)
- `qa_phase5_crypto_removal_v1`

In each case, the "GENERATE" artifact was authored inline during the previous session run, and the research gate's job is to audit that the artifact is complete and correct -- not to gather external literature. There is no algorithm, model, or external system being introduced. The entire step is documentation closure pointing at a config file already on disk.

External WebFetch is inapplicable here. The gate passes on internal audit evidence alone, with `external_sources_read_in_full: 0` documented honestly in the envelope.

---

## Research Gate Checklist

Hard blockers:
- [x] Recency scan performed -- N/A (closure-style; no external literature required; noted explicitly)
- [x] file:line anchors for every internal claim (see audit above)
- [x] Verification criteria confirmed satisfiable: `sprint_calendar.yaml` appears in supersede doc at lines 4, 19, 36
- [x] Masterplan cross-reference confirmed: 8.5.7=done, 10.0=pending

Soft checks:
- [x] All three load-bearing files inspected in full
- [x] Bidirectional cross-reference (supersede doc <-> sprint_calendar.yaml) confirmed
- [x] Scaffold retention (cron.py stays on disk) confirmed and explained

---

## Recency Scan (2024-2026)

Not applicable. This is internal housekeeping closure with no external literature component. No search performed; no external sources relevant.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 4,
  "gate_passed": true,
  "note": "Closure-style gate. Step is documentation housekeeping only (supersede log + cross-reference to sprint_calendar.yaml already on disk). No external literature applicable. Precedent: qa_78_v1, qa_850_v1, qa_phase5_crypto_removal_v1. gate_passed=true on internal audit evidence alone: verification criteria satisfied (sprint_calendar.yaml found at lines 4/19/36 of supersede doc), all three load-bearing files confirmed present and internally consistent, masterplan phase-8.5.7=done confirmed."
}
```
