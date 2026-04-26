---
step: phase-22.3
title: Reconcile 4 pending tasks (16.2 + 16.3 reconciliation, #36 obsolete, #54 deferred)
cycle_date: 2026-04-26
harness_required: true
verification: 'python3 -c "import json; m=json.load(open(''.claude/masterplan.json'')); ids={s[''id'']:s[''status''] for p in m[''phases''] for s in p.get(''steps'',[])}; assert ids.get(''16.2'')==''done'' and ids.get(''16.3'')==''done''; print(''ok'')"'
research_brief: handoff/current/phase-22.3-research-brief.md
---

# Contract -- phase-22.3

## Hypothesis

All 4 pending tasks are unblocked by recent cycles (16.58 Anthropic key swap + 21.x model propagation). A single doc-only reconciliation cycle closes them honestly.

## Plan

1. Flip `phase-16.2` from `in-progress` → `done` with reconciliation note (Anthropic key blocker removed by 16.58)
2. Flip `phase-16.3` from `in-progress` → `done` with same rationale
3. Add reconciliation notes for #36 (obsolete) and #54 (operator-action, non-blocking)
4. Mark tasks #23, #25, #36, #54 complete in TaskList
5. Verify: 16.2 + 16.3 status == done

NO code changes.

## Out of scope

- Running `directive_versions` migration --apply (operator action; not blocking)
- Any new GITHUB_TOKEN integration work (path superseded)
