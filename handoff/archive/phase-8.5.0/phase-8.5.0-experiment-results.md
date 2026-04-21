# Experiment Results — phase-8.5 / 8.5.0 (Retire phase-2 step 2.10 stub)

**Step:** 8.5.0 **Date:** 2026-04-20 **Cycle:** 1 (closure).

## What was done

Zero file changes. Both immutable criteria already hold from prior cleanup cycles:

- `handoff/phase-2.10-supersede.md` was authored during an earlier housekeeping cycle (2026-04-19, ~3.4 KB).
- `masterplan.json` already marks `phase-2.steps.2.10.status = "superseded"` (flipped in an earlier cycle).

No additional authoring required. Cycle acts as a confirmation + audit trail for the phase-8.5 sequencing.

## Verification

```
$ test -f handoff/phase-2.10-supersede.md && echo "IMMUTABLE (A) PASS"
IMMUTABLE (A) PASS

$ python3 -c "import json; mp=json.load(open('.claude/masterplan.json')); [print(s['status']) for p in mp['phases'] if p['id']=='phase-2' for s in p['steps'] if s['id']=='2.10']"
superseded

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `test -f handoff/phase-2.10-supersede.md` | PASS | File exists, 3.4 KB, authored 2026-04-19. |
| 2 (supplementary) | `2_10_status_marked_superseded_in_masterplan` | PASS | `phase-2.steps[2.10].status == "superseded"`. |

## Caveats

1. **Doc pre-existed.** This cycle did not re-author it. The harness protocol allows closure cycles against already-satisfied file criteria (precedent: qa_78_v1, qa_phase5_crypto_removal_v1). Brief's JSON envelope explicitly sets `external_sources_read_in_full: 0` + `note` on closure semantics.
2. **3 non-ASCII bytes in the existing doc** (em-dash U+2014). Not a criterion for this step; kept intact to avoid gratuitous edits. A future housekeeping cycle could strip them as a docs-linting pass.
3. **No contract/generate files authored beyond brief + contract + this results doc** — since the underlying deliverable is pre-existing, a code-shipping cycle would be misleading.
