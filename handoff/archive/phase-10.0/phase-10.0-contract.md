# Sprint Contract — phase-10 / 10.0 (Retire phase-8.5.7 nightly cron)

**Step id:** 10.0 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple (closure)

## Research-gate summary

Closure-style brief at `handoff/current/phase-10.0-research-brief.md` (`gate_passed: true`; precedent qa_78_v1, qa_850_v1, qa_phase5_crypto_removal_v1; 0 new external sources; 3 internal files inspected). The supersede doc is already on disk from the prior inline run; this cycle formalises the harness artifacts around it.

## Hypothesis

`handoff/phase-10.0-supersede-85-7.md` exists and cites `sprint_calendar.yaml` (confirmed by the immutable grep). No new edits needed; the harness artifacts (brief, contract, experiment-results, Q/A critique, log append) complete the cycle.

## Immutable criterion (verbatim)

- `test -f handoff/phase-10.0-supersede-85-7.md && grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md`

## Plan

1. Re-run the immutable command; capture verbatim output.
2. Confirm `phase-8.5.7.status == "done"` in `.claude/masterplan.json` (expected; 8.5.7 was closed during phase-8.5).
3. Write `phase-10.0-experiment-results.md` with verbatim output.
4. Spawn Q/A subagent for independent review.
5. On Q/A PASS: append cycle block to `handoff/harness_log.md`, flip 10.0 `pending -> done`.

## Out of scope

- No code changes. Cron scaffold stays at `backend/autoresearch/cron.py` (retained per supersede doc).
- No APScheduler de-registration (future housekeeping if/when launchd attaches the scaffold live).

## References

- `handoff/current/phase-10.0-research-brief.md`
- `handoff/phase-10.0-supersede-85-7.md`
- `backend/autoresearch/sprint_calendar.yaml` (authoritative; phase-10.1)
- `backend/autoresearch/cron.py` (scaffold; phase-8.5.7)
- `.claude/masterplan.json` → phase-10 / 10.0
