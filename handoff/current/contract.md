# Contract: Phase 4.2.3.1 Formatter Hardening (SN1 + SN2)

**Step:** Phase 4.2.3.1 -- harden `format_accuracy_report` per prior session soft notes
**Target:** `backend/slack_bot/formatters.py` (narrow-scope edit to `_coerce_float` + n=0 fields list)
**Scope:** Two micro-fixes, one file, additive + internal.
**Date:** 2026-04-14
**Author:** Ford remote agent (Opus 4.6)
**Research gate:** PASSED -- `handoff/current/research.md` (20+ URLs, 7 categories)
**Supersedes contract for:** Phase 4.2.3 (archived in git history)

## Hypothesis

IF we apply two defensive fixes to `backend/slack_bot/formatters.py`:

1. **SN1 fix**: `_coerce_float` gains a `math.isfinite` guard that
   maps NaN / +Inf / -Inf to `0.0`, ensuring the downstream `_pct`
   and `f"{x * 100.0:.1f}%"` renders never contain `nan%` or `inf%`.
2. **SN2 fix**: in `format_accuracy_report`, the `scored_count <= 0`
   fields list replaces the `mean_forward_return` and
   `median_forward_return` field values with the canonical string
   `"Scoring pending"` (same as the hit-rate row in that branch).

THEN the weekly accuracy report never presents fake-zero forward
returns on n=0 samples and never leaks IEEE 754 non-finite values to
the Slack channel, satisfying CFA III(D) fair presentation for the
display layer.

## In-Scope Changes

- **File:** `backend/slack_bot/formatters.py`
- **Change 1 (SN1):** Add `import math` to top-of-file imports. Modify
  `_coerce_float` body to add a finiteness guard before return.
- **Change 2 (SN2):** In `format_accuracy_report`, inside the
  `if scored_count <= 0:` branch, change the field values for
  "Mean forward return" and "Median forward return" from `mean_str`
  and `median_str` to `"Scoring pending"`.

That is the entire scope. No new functions, no helper rename, no
signature changes, no branch reorganization.

## Out of Scope (Intentionally Deferred)

1. **SN4 `since_date` lex trap** in `signals_server.get_signal_history`
   -- different file, different research gate.
2. **Phase 4.2.4 BQ durable persistence** -- remote env blocker.
3. **Phase 4.2.4 scheduler wiring** -- needs APScheduler.
4. **Touching `_pct`, `_coerce_int`, the 1..4 scored branch, the >=5
   scored branch, the per-group loop** -- stable Phase 4.2.3 scaffold.
5. **Renaming `_coerce_float`** -- API contract preserved.
6. **Adding new success criteria sub-modes** (e.g., "data stale" vs
   "data pending") -- single placeholder string by design.
7. **Changes to any of the 9 pre-4.2.3 public formatters** -- must
   remain AST byte-identical.

## Anti-Leniency Rules (MUST enforce in QA)

1. **Top-of-file imports: exactly one addition.** The new import set
   is `from datetime import datetime` + `import math`. Nothing else.
2. **No touches to `signals_server.py`, `backtest_server.py`,
   `data_server.py`, or any MCP server code.**
3. **No changes to the 9 pre-4.2.3 public formatters**
   (`_truncate`, `_score_emoji`, `_rec_color`, `format_analysis_result`,
   `format_portfolio_summary`, `_signal_emoji`, `format_signal_alert`,
   `format_report_card`, `format_morning_digest`). AST byte-identical.
4. **No changes to `_pct` or `_coerce_int`.** AST byte-identical.
5. **No changes to `format_accuracy_report` branches for
   `scored_count >= 1`** (the 1..4 preliminary branch and the >=5 CI
   branch). AST byte-identical in those branches.
6. **Never raise.** Same invariant as Phase 4.2.3:
   `format_accuracy_report(None)`, `({})`, `({"hits": "bad"})`,
   `({"mean_forward_return_pct": float('nan')})`,
   `({"mean_forward_return_pct": float('inf')})` all return a
   `list[dict]` without exception.
7. **No non-ASCII in new code.**
8. **Diff budget:** `<= 20` added lines, `<= 5` deleted lines.
   This is a surgical micro-fix cycle, not a rewrite.
9. **Canonical placeholder string is `"Scoring pending"`** -- do
   NOT introduce `"N/A"`, `"--"`, `"pending"`, `"tbd"`, or any
   other synonym.

## Success Criteria (QA must run these)

### SC1-SC5: SN1 NaN/Inf Filter in `_coerce_float`

- **SC1:** `_coerce_float({"x": float('nan')}, "x") == 0.0`
- **SC2:** `_coerce_float({"x": float('inf')}, "x") == 0.0`
- **SC3:** `_coerce_float({"x": float('-inf')}, "x") == 0.0`
- **SC4:** `_coerce_float({"x": 1.5}, "x") == 1.5` (happy path
  unchanged)
- **SC5:** `_coerce_float({"x": "bad"}, "x") == 0.0` (prior bad-input
  fallback unchanged)

### SC6-SC10: Downstream rendering no longer leaks `nan%`/`inf%`

- **SC6:** Fixture with `mean_forward_return_pct = float('nan')` and
  `scored_count >= 5` -- rendered fields contain NO `nan%` substring.
  Mean field renders `"+0.00%"` (the neutral display after sanitization).
- **SC7:** Fixture with `median_forward_return_pct = float('inf')`
  and `scored_count >= 5` -- rendered fields contain NO `inf%`
  substring. Median field renders `"+0.00%"`.
- **SC8:** Fixture with `hit_rate_ci_low = float('nan')` and
  `scored_count >= 5` -- CI string contains no `nan` substring.
  (Prior clamp to `[0, 1]` still applies; the new `isfinite` guard
  pre-empts it.)
- **SC9:** Group with `mean_forward_return_pct = float('nan')` --
  rendered group line contains no `nan%` substring.
- **SC10:** 10 consecutive invocations with random non-finite
  values never raise and never produce `"nan"` or `"inf"` substrings
  anywhere in the serialized block text.

### SC11-SC15: SN2 n=0 "Scoring pending" expansion

- **SC11:** Fixture with `total_count=5, scored_count=0,
  mean_forward_return_pct=0.0` -- the "Mean forward return" field
  value is exactly `"Scoring pending"`, NOT `"+0.00%"`.
- **SC12:** Same fixture -- the "Median forward return" field value
  is exactly `"Scoring pending"`, NOT `"+0.00%"`.
- **SC13:** Same fixture -- the "Hit rate" field value is still
  exactly `"Scoring pending"` (unchanged from Phase 4.2.3).
- **SC14:** Same fixture -- the fields list length is still 4 (even),
  <= 10.
- **SC15:** `scored_count=1` fixture -- "Mean forward return" and
  "Median forward return" fields still render as signed percents
  (e.g., `"+1.24%"`), NOT `"Scoring pending"`. SN2 fix is strictly
  gated to the n=0 branch.

### SC16-SC20: Byte-identity preservation

- **SC16:** All 9 pre-4.2.3 public formatters are AST byte-identical
  to `origin/main` HEAD (`eeea983`).
- **SC17:** `_pct` and `_coerce_int` are AST byte-identical.
- **SC18:** `format_accuracy_report` branch body for
  `elif scored_count < 5:` is AST-equivalent (field labels + values
  unchanged).
- **SC19:** `format_accuracy_report` `else:` branch (`scored_count >= 5`)
  is AST-equivalent.
- **SC20:** Top-of-file imports are exactly `from datetime import
  datetime` + `import math`, in that order or module-pep8 order.

### SC21-SC25: Contract bounds + defensive invariants

- **SC21:** Diff `<= 20` added lines, `<= 5` deleted lines against
  `origin/main` HEAD.
- **SC22:** `ast.parse` and `py_compile` both clean.
- **SC23:** `format_accuracy_report(None)` still returns `list[dict]`.
- **SC24:** AST walk of every string literal in changed code: 0
  non-ASCII.
- **SC25:** No references to `signals_server`, `backtest_server`,
  `data_server`, or `mcp_servers` in changed code.

## Verification Command Block

```bash
# Sanity
python3 -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read())"
python3 -m py_compile backend/slack_bot/formatters.py

# Contract smoke
python3 - <<'PY'
import importlib.util, ast
spec = importlib.util.spec_from_file_location("fmt", "backend/slack_bot/formatters.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# SN1: _coerce_float filters non-finite
assert m._coerce_float({"x": float('nan')}, "x") == 0.0
assert m._coerce_float({"x": float('inf')}, "x") == 0.0
assert m._coerce_float({"x": float('-inf')}, "x") == 0.0
assert m._coerce_float({"x": 1.5}, "x") == 1.5
assert m._coerce_float({"x": "bad"}, "x") == 0.0

# SN2: n=0 branch shows "Scoring pending" for mean/median
blocks = m.format_accuracy_report({"total_count": 5, "scored_count": 0,
    "hits": 0, "mean_forward_return_pct": 0.0,
    "median_forward_return_pct": 0.0})
ser = str(blocks)
assert "Scoring pending" in ser
# Prior fake-zero substrings must not appear in the n=0 fields
fields = [b for b in blocks if b.get("type") == "section"
          and "fields" in b]
for fb in fields:
    for f in fb["fields"]:
        if "Mean forward return" in f["text"] or "Median forward return" in f["text"]:
            assert "Scoring pending" in f["text"], f["text"]

# Gate SC15: scored_count=1 still renders percents (not "Scoring pending")
blocks2 = m.format_accuracy_report({"total_count": 5, "scored_count": 1,
    "hits": 1, "hit_rate": 1.0, "mean_forward_return_pct": 1.24,
    "median_forward_return_pct": 0.85})
ser2 = str(blocks2)
assert "+1.24%" in ser2 or "1.24%" in ser2
assert "Mean forward return" in ser2

# SN1 downstream: nan doesn't leak
blocks3 = m.format_accuracy_report({"total_count": 20, "scored_count": 12,
    "hits": 7, "hit_rate": 0.5833,
    "hit_rate_ci_low": 0.3056, "hit_rate_ci_high": 0.8043,
    "mean_forward_return_pct": float('nan'),
    "median_forward_return_pct": float('inf')})
ser3 = str(blocks3)
assert "nan" not in ser3.lower() or "nan" not in ser3  # explicit
assert "inf%" not in ser3
print("smoke PASS", len(blocks), "+", len(blocks2), "+", len(blocks3), "blocks")
PY
```

## Rollback

`git revert <commit>` on the GENERATE commit. Single-file, small-diff
reversion is clean.
