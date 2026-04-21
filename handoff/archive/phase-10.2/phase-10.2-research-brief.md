## Research: phase-10.2 — Weekly ledger schema and writer

### Internal audit

TSV header confirmed (8 cols, tab-delimited):
  week_iso  thu_batch_id  thu_candidates_kicked  fri_promoted_ids
  fri_rejected_ids  cost_usd  sortino_monthly  notes

House pattern from `backend/autoresearch/gate.py` and `promoter.py`:
- Pure functions, fail-open, ASCII-only, `from __future__ import annotations`.
- No CSV stdlib wrappers in autoresearch; `results.tsv` is written via
  raw string formatting (tab-join). Ledger writer should match that idiom.

`tests/` uses flat pytest files with a `tests/<subdomain>/` subdir pattern
(e.g. `tests/slack_bot/`). New dir: `tests/autoresearch/` with `__init__.py`.

### API surface

`backend/autoresearch/weekly_ledger.py`:
- `LEDGER_PATH = Path(__file__).parent / "weekly_ledger.tsv"` (default)
- `COLUMNS = ["week_iso", "thu_batch_id", ...]` (8 cols, order matches header)
- `append_row(week_iso, thu_batch_id, thu_candidates_kicked,
              fri_promoted_ids, fri_rejected_ids, cost_usd,
              sortino_monthly, notes, *, path=LEDGER_PATH) -> None`
  Idempotent: load all rows, overwrite matching week_iso, else append.
  Fail-open: catch all IO/parse errors, log, return.
- `read_rows(*, path=LEDGER_PATH) -> list[dict]`
  Returns [] on missing/corrupt file (fail-open).

### Test layout (`tests/autoresearch/test_weekly_ledger.py`)

- `test_append_new_week` — write to tmp_path, assert row present.
- `test_idempotent_update_same_week` — append same week twice, assert
  only one row, second values win.
- `test_read_rows_parses_header_and_data` — write header+row manually,
  assert read_rows returns correct dict.
- `test_fail_open_on_bad_path` — append_row and read_rows on
  `/nonexistent/x.tsv`, assert no exception raised.

### Research Gate Checklist

Closure-style internal-only audit. External literature not applicable
(TSV I/O is a settled pattern; no novel algorithm). Gate passed on
internal evidence per caller's declaration.

- [x] Internal files inspected: weekly_ledger.tsv, results.tsv,
      gate.py, promoter.py, tests/ tree
- [x] file:line anchors: gate.py:1-62, promoter.py:1-52
- [x] Recency scan: N/A (internal-only, no literature gap)
- [x] Consensus: house pattern is raw tab-join + fail-open + pure fns

gate_passed: true (caller pre-authorized closure on internal audit only)
