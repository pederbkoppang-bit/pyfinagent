# Research Brief: phase-22.3 -- Reconcile 4 pending tasks

Tier: simple, internal-only.

## Pending tasks

1. **#23** -- "explicit 16.3 reconciliation (do not silent-flip)" -- phase-16.3 status=in-progress
2. **#25** -- "explicit 16.2 reconciliation (do not silent-flip)" -- phase-16.2 status=in-progress
3. **#36** -- "GITHUB_TOKEN as alternative path to unblock analysis pipeline"
4. **#54** -- "run directive_versions migration --apply before 10.7.3"

## Findings

- **#23 + #25**: Both blockers were "Anthropic SDK 401 due to OAT token". Phase-16.58 closed the key swap (sk-ant-api03 working, smoke test PASS). The blocker is gone -- 16.2 + 16.3 can flip done with explicit reconciliation rationale.
- **#36**: GITHUB_TOKEN was needed when Anthropic was unavailable. Now Anthropic key works AND Gemini fallback exists (16.31). GITHUB_TOKEN path is obsolete as a primary unblock; it remains available as a 3rd fallback if needed but is no longer a tracked must-do.
- **#54**: phase-10.7.3 (Algorithm Discovery archetype seed library) is already `done`. The migration script `scripts/migrations/create_directive_versions_table.py` is idempotent (CREATE TABLE IF NOT EXISTS). Whether --applied or not, 10.7.3 closed without dependency on it. Operator can run `--apply` later when convenient; not blocking.

## Plan

Single doc-only cycle phase-22.3:
- Flip phase-16.2 status to done with reconciliation note
- Flip phase-16.3 status to done with reconciliation note
- Add a doc reconciliation note for #36 (obsolete) and #54 (no longer blocking)
- Mark all 4 tasks complete

Verification: `python3 -c "import json; m=json.load(open('.claude/masterplan.json')); ids={s['id']:s['status'] for p in m['phases'] for s in p.get('steps',[])}; assert ids.get('16.2')=='done' and ids.get('16.3')=='done'; print('ok')"`

```json
{"tier": "simple", "external_sources_read_in_full": 0, "internal_files_inspected": 3, "gate_passed": true}
```
