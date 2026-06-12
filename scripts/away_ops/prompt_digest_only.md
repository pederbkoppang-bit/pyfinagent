# DIGEST-ONLY away session (goal-away-ops) -- guardrail tripped, report-only mode

The wrapper put you in report-only mode for one of these reasons (session.log says
which): the 62.4 sentinel FAILED (possible $0-budget breach or flag-vs-token mismatch),
the sentinel is MISSING, or HALT-DEV is active for a PM slot. Authority:
docs/runbooks/away-ops-rules.md -- read FIRST; all 10 rails bind.

ABSOLUTE CONSTRAINTS THIS SESSION: no code edits, no .env reads/writes, no masterplan
changes, no flag changes, no restarts except a DEAD core service (rail 9 kickstart only,
log it). You observe and report.

## Tasks

1. Read session.log to identify the exact trigger; quote it.
2. If the trigger is a sentinel FAILURE: re-run bash scripts/away_ops/sentinel.sh,
   capture its JSON verbatim, and write a P1 entry into
   handoff/away_ops/session_notes.md + pending_tokens.json with the named gate that
   failed and the EXACT operator reply needed (e.g. "SENTINEL: ACKNOWLEDGED" after
   inspection, or a budget decision). A $0-budget breach means metered LLM spend
   happened that should not have -- identify the source from the sentinel report
   (table/endpoint it pins) before writing the ask.
3. Health snapshot: launchd state for the three services, kill-switch state, last cycle
   age, disk free -- one line each into session_notes.md.
4. Pending asks refresh (exact reply strings + ages).
5. Commit (`chore(away-digest-only): <date> <trigger>`), push, exit well under budget.

This mode is the system saying "something needs the operator's eyes". Your job is to
make the next digest so clear that a 30-second phone read tells the operator exactly
what to reply.
