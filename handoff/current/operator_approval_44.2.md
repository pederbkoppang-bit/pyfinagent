# Operator approval -- phase-44.2 Manage tab removal

**Date:** 2026-05-26
**Approver:** Operator (Peder Koppang)
**Subject:** Removing the Manage tab from `/paper-trading` cockpit; settings now reached via existing /settings route + sidebar entry.

---

## Verbatim approval

> "you have my approcval for the gate block by me"

(Direct quote from operator chat message timestamped 2026-05-26 ~22:55 CEST,
after Main listed the 3 gate-blocked phases needing approval. Operator
confirmed scope via follow-up AskUserQuestion selecting all 3 phases +
the misfire fix.)

## Context

Per `/goal` guardrail and `frontend_ux_master_design.md` Section 3.7:
removing a tab that has lived in the cockpit for 3 audit cycles is a
consistency violation per NN/G "tabs used right" heuristic
(researcher source #5 cycle 63 brief at `research_brief_phase_44_2.md`).
The /paper-trading cockpit refactor (cycle 63) preserved the Manage tab
as the 6th tab pending this approval; the existing manage content was
migrated to `app/paper-trading/manage/page.tsx` verbatim.

## Scope of approval

- Remove the `manage` entry from the `TABS` array in
  `frontend/src/app/paper-trading/layout.tsx`.
- Delete `frontend/src/app/paper-trading/manage/` (entire directory).
- Operator workflow change: paper-trading specific settings (lite_mode,
  paper_max_positions, paper_default_stop_loss_pct, etc.) now reached
  via the existing Settings link in the Sidebar -- which goes to the
  global /settings page.
- The "Top up fund" capability (deposit additional capital) is part of
  the deleted manage page. Will need to be re-surfaced via /settings or
  a new dedicated /paper-trading/deposit route in a follow-up cycle.

## Audit trail

| When | Who | What |
|---|---|---|
| 2026-05-25 (cycle 63) | researcher subagent `adf1469ddcbca8f37` | Cited NN/G consistency heuristic + flagged Manage-tab removal as OWNER-APPROVAL-REQUIRED |
| 2026-05-25 (cycle 63) | Main | Built ReportCompareDrawer + migrated Manage content to sub-route; left Manage as 6th tab pending approval |
| 2026-05-26 ~22:55 CEST | Operator | "you have my approcval for the gate block by me" -- AskUserQuestion confirmed scope = 44.2 + 44.3 + 44.5 + misfire fix |
| 2026-05-26 (cycle 67) | Main | Executes destructive Manage tab removal + ships the deposit-handoff note |
