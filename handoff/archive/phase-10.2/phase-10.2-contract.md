# Sprint Contract — phase-10 / 10.2 (Weekly ledger schema + writer)

**Step id:** 10.2 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Research-gate summary

Closure-style brief at `handoff/current/phase-10.2-research-brief.md`. TSV header on disk matches the immutable grep. `writer_unit_test_passes` requires a new writer module + pytest. `gate_passed: true`.

## Hypothesis

Ship `backend/autoresearch/weekly_ledger.py` with `append_row` + `read_rows` (idempotent by week_iso; fail-open). Plus `tests/autoresearch/__init__.py` + `tests/autoresearch/test_weekly_ledger.py` with 4 cases.

## Immutable criterion

- `test -f backend/autoresearch/weekly_ledger.tsv && head -1 ... | grep -q 'week_iso.*thu_batch_id.*fri_promoted_ids.*cost_usd'`

## Plan

1. Create `tests/autoresearch/__init__.py`.
2. Write `backend/autoresearch/weekly_ledger.py` (~80 lines): `COLUMNS`, `LEDGER_PATH`, `append_row(..., *, path=LEDGER_PATH)`, `read_rows(*, path=LEDGER_PATH)`.
3. Write `tests/autoresearch/test_weekly_ledger.py` with 4 cases (new-week append, idempotent update, read_rows, fail-open on bad path).
4. Run immutable + pytest + regression.
5. Q/A, log, flip.

## Out of scope

- No BQ write; TSV-only.
- No scheduler wiring (phase-10.3/10.4 consume this writer).
- ASCII-only.
