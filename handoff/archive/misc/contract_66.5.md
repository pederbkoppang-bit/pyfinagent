# Contract -- 66.5 Away-backlog triage (planning-only) (goal-phase66-reactivation)

Step: 66.5 | Cycle 70 | 2026-07-07 | Operator present (asleep; sign-off async)
Sequencing: depends_on 66.0 (done); planning-only, touches NO build surface; the
66.1->66.2 P0 chain remains untouched (Q/A Cycle-69 binding boundary respected).

## Research-gate summary

research_brief_66.5.md (tier simple, gate_passed: true; 6 read-in-full / 52 URLs /
recency scan / 9 internal files). Load-bearing:
- Ground truth vs step assumptions: 22 routes still accurate; tests/e2e-functional
  ABSENT; playwright.config.ts has ONE project (:38, :67-82); defect_register.md
  ABSENT; away plists still armed and firing (operator question: disarm now that
  attended?); 62.2 token handler live-criterion still open (in-session approval
  replaces away token machinery while operator is home).
- Overlap: 65.1 (EU per-gate funnel) is SUBSUMED by 66.2's all-market funnel
  criterion -- merge, keeping the per-ticker counter residue as 66.2 input; 64.5's
  nightly-runner leg folds into 64.2 (CI leg stays); 64.4 depends_on 65.1 must
  repoint on merge. 65.3's since-06-01 baseline window is ~70% trade-freeze --
  re-anchor the window.
- 8 seed defects for 63.3's register: _resolve_claude_binary docstring mismatch;
  .env-bleed test isolation; auth-latch paged:false no-retry; auto-commit hook
  silent stalls (auto-push.log tonight: 12 INVOKED, 0 pushes); changelog
  trailing-commit race; historical_macro ~103d stale; alpaca short_market_value
  (66.2); paper_portfolio single-US-row (66.2).
- External canon: postmortem action items need owner+tracking or they become
  outages again (SRE Workbook); stabilize-before-you-pave for E2E investment
  (Fowler/Google 70/20/10); stop-starting-start-finishing grounds the one-P0-chain
  rule (WIP limits).

## Hypothesis

Nothing in phases 63-65 deserves deletion -- the away plan's substance survives; what
died was its CADENCE (PM-slot/digest wiring) and one diagnosis got superseded by
66.2's broader version. A 14-row keep/merge/drop table with re-anchored sequencing
gives the operator a one-read sign-off decision.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.5)

1. "Every step in phases 63/64/65 dispositioned keep/merge/drop with a one-line
   rationale in handoff/current/triage_phase63-65.md"
2. "Masterplan reflects the dispositions WITHOUT deleting history (dropped steps
   marked status=dropped with rationale; kept steps re-sequenced under
   attended-operation assumptions)"
3. "Operator sign-off on the triage recorded (in-session approval quoted, or token)
   BEFORE any masterplan edit takes effect; no build work performed inside 66.5"

Verification command (immutable):
test -f handoff/current/triage_phase63-65.md && jq -r '[.phases[] | select(.id=="phase-63" or .id=="phase-64" or .id=="phase-65") | .steps[].status] | group_by(.) | map({s: .[0], n: length})' .claude/masterplan.json

live_check: live_check_66.5.md with the triage table and the recorded operator
sign-off.

## Plan

1. Write triage_phase63-65.md: 14-row disposition table (rationale per row), the
   proposed EXACT masterplan edits (applied only post-sign-off), the 63.3 seed-defect
   list, and the operator questions (sign-off + away-plists disarm decision).
2. Q/A -> expected CONDITIONAL: criterion 1 closes now; criteria 2+3 are
   operator-gated by design (masterplan edits deferred until sign-off). No
   verdict-shopping risk: the CONDITIONAL is the designed intermediate state.
3. Log Cycle 70; NO status flip; NO masterplan edits this cycle.

## Scope boundaries

Planning-only: no code, no masterplan edits, no plist changes, no build work. The
away-plists question is SURFACED, not decided.

## References

research_brief_66.5.md; masterplan phases 63/64/65; goal_away_ops.md; SRE Workbook
postmortem-culture; incident.io postmortem practice; Fowler practical-test-pyramid;
Google Testing Blog 70/20/10; Planview WIP limits.
