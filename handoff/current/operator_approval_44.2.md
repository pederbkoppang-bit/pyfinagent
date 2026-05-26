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
- Keep the `/paper-trading/manage` sub-route reachable via a Settings
  gear button in the page header (Phosphor Gear icon, top-right action
  bar). This preserves access to the deposit + 10 paper-trading knobs
  without putting Manage back in the tablist.
- Note (2026-05-26 cycle 67 follow-up): the original scope-of-approval
  wording mistakenly said "via the existing Settings link in the
  Sidebar... global /settings page" -- but the /settings page does NOT
  expose 9 of 10 paper-trading knobs nor the deposit handler. Operator
  flagged that the Manage content was no longer reachable. Corrected
  approach: keep the sub-route + gear-button access. Manage is removed
  from the TABLIST per the original intent; it is NOT removed from the
  app.

## Audit trail

| When | Who | What |
|---|---|---|
| 2026-05-25 (cycle 63) | researcher subagent `adf1469ddcbca8f37` | Cited NN/G consistency heuristic + flagged Manage-tab removal as OWNER-APPROVAL-REQUIRED |
| 2026-05-25 (cycle 63) | Main | Built ReportCompareDrawer + migrated Manage content to sub-route; left Manage as 6th tab pending approval |
| 2026-05-26 ~22:55 CEST | Operator | "you have my approcval for the gate block by me" -- AskUserQuestion confirmed scope = 44.2 + 44.3 + 44.5 + misfire fix |
| 2026-05-26 (cycle 67) | Main | Executes destructive Manage tab removal + ships the deposit-handoff note |
| 2026-05-26 (cycle 67 follow-up) | Operator | Reported the Manage content was orphaned (/settings didn't surface the knobs). Fix: restored /paper-trading/manage sub-route + added Settings gear button in page header. Manage REMOVED FROM TABLIST per the original approval; STILL REACHABLE via the gear button. |
