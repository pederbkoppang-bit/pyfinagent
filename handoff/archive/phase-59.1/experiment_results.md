# Experiment Results — Step 59.1 (GENERATE)

**Step:** 59.1 — Fable 5 model adoption (both layers, quality-first). **Date:** 2026-06-11. **Mode:** operator-directed (8 in-session pre-approvals); rare-event roles only; metered paths untouched.

## What was built (7 files)

| File | Change |
|---|---|
| `.claude/agents/researcher.md` | `model: fable`, maxTurns 30→40; comment block rewritten (operator pre-approval, June-23 Max-credit economics superseding flat-fee, restart caveat, history preserved) |
| `.claude/agents/qa.md` | `model: fable`, maxTurns 12→30 (five observed stalls); same annotations |
| `backend/config/model_tiers.py` | mas_main + autoresearch_strategic → claude-fable-5 (rare-event comments); mas_qa explicitly kept (per-ticker comment); EFFORT_SUPPORTED_MODELS + claude-fable-5; MODEL_EFFORT_FALLBACK + ("claude-fable-5","xhigh") — closes the silent effort-drop trap |
| `backend/services/ticket_queue_processor.py` | agent map: main + q-and-a → claude-fable-5 (~$0.18/day, decision recorded); research stays Sonnet 4.6 |
| `CLAUDE.md` | Additive Fable 5 effort-policy block (id, $10/$50, June-23 credit change, classifier fallback, alias + v2.1.170 floor, metered-roles-stay-off warning); phase-29.2 history intact |
| `backend/tests/test_phase_59_1_fable_adoption.py` | NEW: 6 tests (resolution, metered-unchanged, effort-supported, ticket map, Layer-3 frontmatter, CLAUDE.md additive) — named to enter the `-k` net per the false-green finding |
| `backend/tests/test_agent_map_live_model.py` | mas_main assertion → claude-fable-5 (the test the repin would otherwise break) |

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests -k 'fable or model_tiers or phase_59' -q
7 passed, 773 deselected, 1 warning in 2.85s
$ test -f handoff/current/live_check_59.1.md && echo PASS
PASS
```
Full suite: `762 passed, 12 skipped, 6 xfailed` exit 0 (run explicitly because the researcher proved the `-k` selector alone could false-green).

## Key outcomes

1. Fable 5 lands exactly on the four rare-event roles + the negligible-cost ticket agents; every metered per-ticker path keeps its model (unit-tested).
2. The silent effort-drop trap is closed (EFFORT_SUPPORTED_MODELS + fallback row) — without it, fable-pinned Layer-2 roles would have lost their effort param invisibly.
3. The Layer-3 stall defect is fixed at its lever (maxTurns 12→30 / 30→40), sized from this session's observed tool-use counts.
4. The economics change (Max credits from June 23) is documented as SUPERSEDING the flat-fee rationale in all three places it lived (CLAUDE.md + both agent files) — additive, history preserved.

## Honest limitations

- The Layer-3 pins take effect NEXT session (roster snapshot); this step's own qa spawn evaluates from the OLD snapshot — protocol-identical, models differ. Roster verification is the next session's first action (`scripts/qa/verify_qa_roster_live.sh`).
- Fable 5's real-world quality delta on these roles is unmeasured here (the 59.3 stress test provides the first observation); the adoption decision is operator-preference + announcement benchmarks.
- Post-June-22 Max credit burn rates are not yet observable; the operator may want a usage review in late June.
