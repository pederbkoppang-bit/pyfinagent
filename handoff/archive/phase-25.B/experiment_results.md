---
step: phase-25.B
cycle: 85
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_B.py'
title: Remove cosmetic aliasing patch after 25.A decouples calls (P2)
audit_basis: phase-24.4 F-2 (signal_attribution.py:131-154 is_lite_dup detection became dead code after 25.A)
depends_on: 25.A (done, commit 9c5eb8ad)
---

# Experiment Results -- phase-25.B

## Code changes

### `backend/services/signal_attribution.py`
- Removed the dead `trader_rationale_trimmed = _trim(...)` line + the `is_lite_dup = (...)` boolean + the `if is_lite_dup: entry[...] = ...; entry["lite_path"] = True` branch (was lines 131-154).
- Simplified to a direct `signals.append({...})` of the RiskJudge entry.
- Inline comment notes the deletion was a post-25.A cleanup.

### `frontend/src/components/AgentRationaleDrawer.tsx`
- Removed `lite_path?: boolean` field from the `Signal` interface (was lines 12-15).
- Removed the conditional `<span>...lite-path</span>` amber badge block (was lines 168-172).
- Removed the conditional `text-amber-200/80` color class; rationale text now uses solid `text-slate-200` always.

### `tests/verify_phase_25_B.py` (new file)
- 6 immutable claims with 1 behavioral round-trip:
  - Claims 1-5: structural greps (no `is_lite_dup` token, no `"lite_path"` literal in backend, no `lite_path?:` in Signal interface, no `lite-path` badge string, no `text-amber-200/80` conditional).
  - Claim 6: **Behavioral no-regression** -- call `extract_signals_from_analysis` with a post-25.A risk_assessment shape; assert the RiskJudge entry has the expected `weight=4.5` + rationale + NO `lite_path` key.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_B.py
PASS: is_lite_dup_branch_removed_from_signal_attribution
PASS: lite_path_field_removed_from_signal_attribution
PASS: lite_path_field_removed_from_signal_interface
PASS: lite_path_amber_badge_removed_from_frontend
PASS: conditional_amber_styling_removed
PASS: behavioral_risk_judge_entry_clean_post_cleanup

6/6 claims PASS, 0 FAIL
```

## Backend + frontend gates

- `python -c "import ast; ast.parse(open('backend/services/signal_attribution.py').read())"` -- OK
- `npx tsc --noEmit` from `frontend/` -- clean except for pre-existing 25.A12 Playwright-not-installed errors (unrelated to 25.B; `@playwright/test` is declared in package.json but `npm install` hasn't run since cycle 79).
- 1 behavioral round-trip confirms no regression in `extract_signals_from_analysis`.

## Hypothesis verdict

CONFIRMED. Two immutable success criteria mapped:
- Criterion 1 (`is_lite_dup_branch_removed_from_signal_attribution`) -- claims 1 + 2 (token + literal removed).
- Criterion 2 (`lite_path_amber_badge_removed_from_frontend`) -- claims 3 + 4 + 5 (Signal interface field + badge string + amber conditional all removed).

Combined with claim 6 (behavioral RiskJudge entry is clean post-cleanup), the dead-code removal is verified end-to-end.

## Live-check

Per masterplan: "Code review: no is_lite_dup references in main branch post-25.B".

Verifier claim 1 already checks `is_lite_dup` is absent from the source file. The live-check is effectively pre-completed via the verifier; operator code review just needs to confirm no other module imports / references `is_lite_dup` (grep confirms 0 matches across backend/ + frontend/).

## Non-regressions

- `extract_signals_from_analysis` return shape preserved for the RiskJudge entry (agent / role / rationale / weight).
- No change to upstream Risk Judge call path (25.A behavior untouched).
- No change to frontend rendering of other Signal types (Trader / Synthesis / Bull / Bear unaffected).
- No new BQ schema, no API contract change.

## phase-25 cleanup status

- 19 of 19 P1 candidates done (cycles 67-84 + 85).
- 25.S (P2 attribution) done at cycle 83.
- 25.B (P2 cleanup; this cycle 85) done.
- Remaining P2 backlog: 25.C, 25.D, 25.E, 25.F, 25.L, 25.M, 25.N, 25.O, 25.P (frontend drawer toggles, Slack notifications), 25.B6, 25.D7, 25.E7, 25.B7 (smaller observability fixes), 25.B10, 25.C7 (data-freshness endpoints).
- 25.C9.1 / 25.D9.1 / 25.S.1 follow-ups (caller-side adoption of the mechanisms shipped in cycles 81/84 + per-call ticker tagging) not yet in the masterplan.

## Next phase

Q/A pending.
