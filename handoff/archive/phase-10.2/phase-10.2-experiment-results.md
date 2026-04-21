# Experiment Results — phase-10 / 10.2 (Weekly ledger schema + writer)

**Step:** 10.2 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

3 new files:

1. `backend/autoresearch/weekly_ledger.py` (~105 lines): `COLUMNS` (8-tuple), `LEDGER_PATH`, `append_row(..., *, path)` (idempotent-by-week_iso; fail-open), `read_rows(*, path)` (fail-open).
2. `tests/autoresearch/__init__.py` — new package marker.
3. `tests/autoresearch/test_weekly_ledger.py` — 5 tests: new-week append, idempotent same-week, read_rows parse, fail-open on bad path, ASCII.

Existing TSV at `backend/autoresearch/weekly_ledger.tsv` (header + seed row) satisfies the `test -f ... && grep -q` immutable; its schema is formalized in `COLUMNS` and any future write via `append_row` will stay aligned.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/autoresearch/weekly_ledger.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m pytest tests/autoresearch/test_weekly_ledger.py -q
5 passed in 0.01s

$ test -f backend/autoresearch/weekly_ledger.tsv && head -1 backend/autoresearch/weekly_ledger.tsv | grep -q 'week_iso.*thu_batch_id.*fri_promoted_ids.*cost_usd' && echo "IMMUTABLE GREP PASS"
IMMUTABLE GREP PASS

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | success_criterion | Status |
|---|---|---|
| 1 | weekly_ledger_tsv_created | PASS (file on disk) |
| 2 | schema_header_stable | PASS (immutable grep + formalized `COLUMNS` tuple) |
| 3 | writer_unit_test_passes | PASS (5/5 pytest) |

## Caveats

1. Writer is text-based TSV (not csv stdlib). Matches `results.tsv` precedent.
2. `fri_promoted_ids`/`fri_rejected_ids` serialize lists as `[t1,t2]`. Readers parse back.
3. ASCII only.
