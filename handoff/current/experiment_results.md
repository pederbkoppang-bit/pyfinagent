# Phase 4.2.3 Slack Accuracy Report Formatter -- Experiment Results

**Date:** 2026-04-14
**Step:** Phase 4.2.3 -- wire signals_server accuracy aggregator into Slack formatters
**Author:** Ford remote agent (Opus 4.6)

## Summary

Added `format_accuracy_report(data, window=None) -> list[dict]` to
`backend/slack_bot/formatters.py`. Pure-Python stdlib-only Block Kit
formatter for the weekly scheduled accuracy report. Zero new imports,
zero touches to the MCP server layer, zero touches to the 5 existing
public formatters.

## File Changes

**Single file touched:** `backend/slack_bot/formatters.py`
**Diff:** +214 / -0 (appended at end of file after `format_morning_digest`)
**Budget:** 214 / 220 added-line cap (97% utilization, under bound)

### New helpers (module-private)

- `_pct(value, decimals=1, signed=False) -> str` -- coerces numeric input
  to a percent string with optional sign. Returns `"N/A"` on TypeError
  or ValueError. Never raises.
- `_coerce_int(d, key) -> int` -- dict-key -> int with None/bad-value
  fallback to 0. Never raises.
- `_coerce_float(d, key) -> float` -- dict-key -> float with fallback to
  0.0. Never raises.

### New public formatter

- `format_accuracy_report(data, window=None) -> list[dict]`

## Design (per research gate)

Block order: `header` -> `context` (window + gen timestamp) -> `section`
(TL;DR one-line summary) -> `divider` -> `section` with `fields` (headline
2-col dashboard) -> `divider` -> per-group `section` blocks (mrkdwn, not
fields) -> trailing `context` (provenance + optional "small sample" flag).

Wilson CI display rule (research-driven):
- `scored_count == 0` -> hit-rate row collapses to "Scoring pending".
  No fake `0.00%`.
- `1 <= scored_count < 5` -> CI field replaced with
  `preliminary -- n={X}`. No bracketed CI.
- `scored_count >= 5` -> inline `[low, high]` with 2-decimal fractions.

Groups: sorted by `scored_count` desc, hard-capped at 5 visible. Overflow
(>5) noted in a context block `+N more groups -- see full report`. Each
per-group line kept under 500 chars (6x safety margin on the 3000-char
section cap). Scoring-pending groups show "scoring pending" without
fake rates.

Empty/degenerate:
- `None`, non-dict, or `{}` -> "input data missing" unavailable branch.
- `total_count == 0` -> "No signals issued in {window}" branch.
- Non-numeric fields -> coerced to 0 / "N/A", never raise.

A11y (research-driven):
- Header is `plain_text` only.
- No emoji-as-label in fields.
- Percentages always carry `%` (screen reader reads "percent").
- Bold sparingly: on field labels and TL;DR only.
- Full sentences in TL;DR, not cryptic abbreviations.

Defensive bounds:
- Fields always EVEN count and `<= 10` (belt-and-suspenders truncate).
- Header text truncated at 140 chars (under the 150 Slack cap).
- TL;DR text truncated at 500 chars.
- Per-field text truncated at 180 chars.
- Per-group mrkdwn truncated at 500 chars.

## Verification Results

```
python3 -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read())"
-> OK

python3 -m py_compile backend/slack_bot/formatters.py
-> OK

git diff --stat backend/slack_bot/formatters.py
-> 1 file changed, 214 insertions(+)
```

### Behavioral SC results (pre-QA self-smoke)

All 30 contract success criteria pass:

- SC1-SC5: bad/empty inputs (None, {}, bad-type keys, non-dict) -> no raise
- SC6-SC12: normal n>=5 path -> correct block structure, fields count,
  precision, sign, CI formatting
- SC13-SC17: Wilson CI display rule (0, 1..4, 5, 100 scored; no NaN/None/inf)
- SC18-SC22: groups (empty, 3, 10-capped-to-5, overflow, lengths)
- SC23-SC24: window present/absent rendering
- SC25: 9 existing formatter functions byte-identical (AST dump equal)
- SC26: top-of-file imports byte-identical
- SC27: all string literals in new function are ASCII
- SC28: diff 214 / 220 cap
- SC29: AST parse + py_compile clean
- SC30: no `signals_server`, `backtest_server`, `data_server`,
  `mcp_servers` references inside `format_accuracy_report` body

## Key Choices and Tradeoffs

1. **`_coerce_int` / `_coerce_float` as module-level helpers** rather than
   closures inside `format_accuracy_report`. Two reasons: (a) the group
   loop reuses them, avoiding 5x nested try/except blocks; (b) cleaner
   AST for QA. Cost: two extra private names in the module namespace
   but no export risk (leading underscore convention).

2. **`scored_count >= 5` cutoff for CI display** is synthesis, not from
   a single source: Wilson stability hits at n=10 per the afit.edu PDF,
   but usability testing in practitioner blogs favors showing "some"
   signal by n=5. The contract allows the lead agent to adjust N
   without re-opening research; I chose 5 as the pragmatic middle.

3. **Trailing-context small-sample flag fires at 0 < n < 10** (not < 5).
   Reason: the CI is shown at n >= 5, but n=5..9 is still small enough
   that a reader should see the caveat. The CI itself communicates the
   uncertainty visually; the flag reinforces it in prose.

4. **No `sparkline` or emoji indicators in per-group lines.** Research
   a11y rule: emoji-as-label is forbidden; icon-as-information is
   forbidden; only prose conveys state.

5. **No interactive accessory (select menu, overflow menu) for groups**.
   The weekly scheduler posts once and does not register interaction
   handlers; an interactive element would 404 on click.

6. **Both `field` layouts (n=0, n<5, n>=5) use 6-item configurations
   except n=0 which uses 4.** The n=0 case intentionally drops both
   "Scored" and "Confidence" fields because they'd be redundant with
   "Hit rate = Scoring pending". Layout asymmetry is by design.

## Out-of-Scope Deferrals

- **Scheduler wiring**: the Phase 4.2.4 APScheduler caller that will
  invoke this formatter. Needs `scheduler.py` + httpx backend client
  which are not in the remote runner env.
- **BQ durable persistence**: Phase 4.2.4; `signal_history` is still
  in-memory.
- **SN4 since_date lex trap fix**: needs its own research gate.
- **Color coding**: explicitly warned off by CFA III(D) in research.

## Rollback

`git checkout HEAD~1 -- backend/slack_bot/formatters.py` reverts the
one-file edit. No cross-file side effects.
