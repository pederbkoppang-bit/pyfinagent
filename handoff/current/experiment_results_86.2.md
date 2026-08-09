# Experiment results — phase-86.2

**Step:** `86.2` — P1 TOTAL DISARM: one oversized JSON int aborts the entire
kill-switch audit replay and strands BOTH protective legs. **Cycle 189**, 2026-08-09.

## 1. What changed

`backend/services/kill_switch.py` (production) and a new test file. Nothing else.

| Change | Why |
|---|---|
| `_coerce_nav` except tuple → `(TypeError, ValueError, ArithmeticError)` | `OverflowError` is `ArithmeticError`'s child — a **sibling** of the old tuple, so it could never have been caught. `ArithmeticError` also covers `decimal.InvalidOperation`: the class, not the instance that bit us. |
| **PER-ROW isolation** in the apply loop, extracted to `_apply_audit_row` | The actual fix. Widening the tuple alone leaves the broad handler able to eat a whole replay on the next unanticipated fault. |
| A skipped row sets `_history_complete = False` and logs at **ERROR** | Load-bearing, not politeness: a skip without it lets a lost-history anchor claim authority again, reopening the hole phase-36.8 closed. Mirrors what `_read_audit_rows` already does per **line**. |
| The last-resort handler raised `WARNING` → `ERROR`, and sets `_history_complete = False` | Severity was inverted: a total disarm logged below a strictly lesser fault. |

**The extraction is behaviour-preserving, proven not asserted:** the event-branch
set is identical to HEAD — `['auto_resume_alert', 'baseline_anchor_on_lost_history',
'pause', 'peak_reset', 'peak_update', 'resume', 'sod_snapshot']` both sides. No
branch added, removed or reordered; the body is the previous inline loop body,
dedented.

## 2. Result

Full before/after and the mutation transcript: `live_check_86.2.md`.

```
BEFORE  armed=False  both legs 0.0%  any_breached=FALSE   (20% drawdown)
AFTER   armed=True   both legs 20.0% any_breached=TRUE
```

- new file `test_phase_86_2_replay_poison_row.py`: **7 passed**
- six adjacent kill-switch files: **172 passed**
- live journal `90e0303130fc…` **unchanged** throughout
- `ruff` gated set (`F821,F401,F811,E9`): **clean**

## 3. What the research gate corrected in my plan

1. **The immutable verification command exits 0 today with the defect present** —
   case E is printed, never asserted. So criterion 1's red test had to be a NEW
   pytest; treating the passing script as the reproduction would have been a
   green that proves nothing.
2. **The correct idiom already existed one layer up** (`_read_audit_rows`), and a
   skip omitting its `complete = False` bookkeeping reopens a closed hole.
3. **No reviewed practice permits this defect's third state** — abort halfway,
   then return normally. PostgreSQL redo PANICs; Kafka Connect defaults to
   `errors.tolerance=none`.

## 4. Two of my own expectations were wrong, and measurement corrected them

- I asserted a clean tmp replay reports `_history_complete=True`. **It reports
  False at HEAD too** — the derived archive dir does not exist and an unreadable
  *source* already sets the flag (pre-existing phase-36.8 behaviour). Test fixed
  to create the archive dir so it asserts what it claims.
- My per-row-`try` mutation replaced `try:` with `if True:`, leaving a dangling
  `except` → `SyntaxError`. An invalid mutant proves nothing; replaced with a
  syntactically valid both-guards revert.

## 5. Deviations — for Q/A, not self-cleared

Criterion 4's literal wording and criterion 1's ordering. Both stated in full in
`live_check_86.2.md` §4. Neither is argued away.
