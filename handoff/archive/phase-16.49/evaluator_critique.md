---
step: phase-16.49
verdict: CONDITIONAL
cycle_date: 2026-04-25
agent: qa
prior_verdict_overwritten: phase-16.48 PASS
---

## Step 1: Harness-compliance audit

| # | Check | Result |
|---|-------|--------|
| 1 | `phase-16.49-research-brief.md` exists | PASS (151 lines) |
| 1b | Brief contains JSON gate envelope with `gate_passed: true` | FAIL — no envelope present, `gate_passed` field absent |
| 2 | `contract.md` line 2 = `step: phase-16.49` | PASS |
| 3 | `experiment_results.md` line 2 = `step: phase-16.49` | PASS |
| 4 | `harness_log.md` not yet appended | PASS (0 hits — log-last discipline observed) |
| 5 | Critique file overwrites prior PASS | OK (was 16.48) |

## Step 2: Deterministic checks

| Check | Result |
|-------|--------|
| `npx tsc --noEmit` | PASS (no output) |
| reports two-zone shell `flex flex-1 flex-col overflow-hidden` | PASS |
| reports `PageSkeleton` rendered | PASS (line 257; single import line 25) |
| backtest `overflow-x-auto scrollbar-thin` count >=3 | PASS (3) |
| paper-trading `overflow-x-auto scrollbar-thin` count >=2 | PASS (2 at lines 533, 617) |
| reports `overflow-x-auto scrollbar-thin` count >=1 | PASS (1) |
| lint `@phosphor-icons/react` direct hits | PASS (0) |
| Git scope contains the 3 expected pages | PASS |
| agents/page.tsx + learnings/page.tsx untouched | PASS (not in `git status`) |

`checks_run`: ["tsc", "shell_grep", "scrollbar_counts", "phosphor_lint", "git_scope", "skeleton_dedup", "harness_compliance"]

## Step 3: LLM judgment

**reports/page.tsx (220-260):** Two-zone shell is correct per
frontend-layout.md §1: `flex h-screen overflow-hidden` outer,
`flex flex-1 flex-col overflow-hidden` main, `flex-shrink-0` fixed
header containing title + tab bar, `flex-1 overflow-y-auto
scrollbar-thin` scrollable zone containing error banner +
PageSkeleton + tab content. Matches the 16.48 signals/performance
pattern. PASS.

**backtest/page.tsx (683-709 vs 765-794):** Error banners
successfully relocated. Original location (line 683) holds only the
phase-16.49 placeholder comment + the `ingestResult` banner (which
remains in the fixed zone — acceptable since it's transient
toast-style). The two persistent error banners (`error` and
`btStatus.status === "error"`) reappear in the scrollable zone at
lines 768-794, AFTER the `Scrollable content zone` comment at 764.
Not duplicated. PASS.

**paper-trading/page.tsx (530, 617):** `overflow-x-auto
scrollbar-thin` correctly added on both table wrappers. PASS.

**Scope honesty:** 4 additional files appear in `git status`
(login, page, performance, signals) — these are pre-existing
uncommitted work from phase-16.16 (commit 75f331fa), NOT introduced
by this cycle. Not a blocker for 16.49 but Peder should be aware
that root has stale dirty state predating this phase.

## Step 4: Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 11 fixes verified (deterministic + LLM checks pass), but research brief lacks the mandatory JSON gate envelope per .claude/rules/research-gate.md — gate_passed field absent.",
  "violated_criteria": ["research_gate_envelope"],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "researcher emitted phase-16.49-research-brief.md without trailing JSON envelope",
      "state": "151-line brief, no `gate_passed`, no `external_sources_read_in_full` count, no `recency_scan_performed` flag",
      "constraint": "research-gate.md §JSON envelope: 'Every brief ends with this envelope, even when the caller does not ask for it'"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["tsc", "shell_grep", "scrollbar_counts", "phosphor_lint", "git_scope", "skeleton_dedup", "harness_compliance"]
}
```

**Single blocker to clear:** append the JSON envelope block to
`handoff/current/phase-16.49-research-brief.md` (per
research-gate.md template), then re-spawn Q/A. The code work
itself is solid — all 11 fixes verified, the reports two-zone
shell is canonical, backtest banners are cleanly relocated (not
duplicated), scrollbar-thin coverage matches expected counts, no
direct phosphor imports leaked. This is a documentation-completeness
blocker, not a code blocker.

---

## Main response to phase-16.49 cycle-2 (2026-04-26)

**Q/A blocker addressed:** research brief was missing the mandatory JSON gate envelope (`.claude/rules/research-gate.md` violation). Fixed by appending the envelope to `handoff/current/phase-16.49-research-brief.md`. The brief now ends with the canonical `Research Gate Checklist` + JSON block per the rule. Honest internal-only gate justification included (no novel external research; pure code audit against documented project rules).

**No code changes needed:** Q/A confirmed all 11 UX fixes are correct + verified (tsc clean, scrollbar counts match, banners cleanly relocated, no Skeleton dup, no leak). The blocker was protocol-only (missing envelope), not code-quality.

**Spawning fresh Q/A** on updated evidence (the brief now has the envelope) per CLAUDE.md cycle-2 flow ("Main fixes the blockers and updates the handoff files. Main spawns a fresh Q/A. The fresh Q/A reads the updated files — evidence has changed").

---

## Q/A cycle-2 verdict (2026-04-25)

**Verdict: PASS**

**Single-blocker re-audit (per CLAUDE.md cycle-2 protocol — fresh Q/A on updated evidence; unchanged code NOT re-graded).**

### Step 1: Blocker resolution — VERIFIED

`tail -30 handoff/current/phase-16.49-research-brief.md` confirms the brief now ends with:
- `## Research Gate Checklist` section with hard-blocker + soft-check boxes ticked
- Canonical JSON envelope with `gate_passed: true`, `tier: "simple"`, `internal_files_inspected: 5`, and an honest internal-only justification note

The internal-only justification is accepted: this step is a pure code audit against rules already documented in `.claude/rules/frontend.md` + `frontend-layout.md`; the external 5-source floor does not apply when no novel pattern is being researched. The note makes the deviation transparent rather than masking an under-fetch.

### Step 2: Immutable verification — RE-RAN, exits 0

`npx tsc --noEmit` clean + all 6 grep gates satisfied + final `OK` printed. Matches cycle-1 result; no regression.

### Step 3: Consecutive-CONDITIONAL counter

Prior phase-16.49 entries in `handoff/harness_log.md`: 0 (log not yet appended — that's the LAST step). Current verdict count: 1 prior CONDITIONAL → this verdict. Below the 3-CONDITIONAL FAIL threshold from CLAUDE.md 16.40.

### Checks run
- `research_gate_envelope_check`
- `immutable_verification_command` (re-run, exit=0)
- `consecutive_conditional_counter`

### Violated criteria
None.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle-2 single-blocker fix verified: research brief now contains the canonical JSON gate envelope with gate_passed: true and honest internal-only justification. Immutable verification command re-ran clean (tsc + 6 grep gates → OK). 11 UX code fixes from cycle-1 unchanged and not re-graded per cycle-2 protocol.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["research_gate_envelope_check", "immutable_verification_command", "consecutive_conditional_counter"]
}
```

Main may now append `handoff/harness_log.md` (LAST step) and flip `phase-16.49` to `status: done` in `.claude/masterplan.json`.
