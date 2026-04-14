# Contract: Phase 4.2.3 Slack Accuracy Report Formatter

**Step:** Phase 4.2.3 -- wire `signals_server.get_accuracy_report()` into Slack
**Target:** `backend/slack_bot/formatters.py` (ADD one new public function)
**Scope:** One pure-Python stdlib-only formatter, no touches to signals_server or Phase 4 scaffold.
**Date:** 2026-04-14
**Author:** Ford remote agent (Opus 4.6)
**Research gate:** PASSED -- `handoff/current/research.md` (34 URLs, 5 categories)

## Hypothesis

IF we add `format_accuracy_report(data, window=None) -> list[dict]` to
`backend/slack_bot/formatters.py` as a pure-Python Block Kit formatter that:

1. Consumes the exact return shape of `SignalsServer.get_accuracy_report()`
   (Phase 4.2.2): `total_count, scored_count, hits, misses, neutral, unscored,
   hit_rate, hit_rate_ci_low, hit_rate_ci_high, mean_forward_return_pct,
   median_forward_return_pct, groups: dict[str, dict]`.
2. Renders a header -> context -> TL;DR -> divider -> fields -> divider ->
   per-group sections -> context layout (research lock-in).
3. Hides the Wilson CI when `scored_count < 5` (replaces with "preliminary")
   and hides the hit-rate row entirely when `scored_count == 0`.
4. Caps groups at 5, overflow shown in a context block.
5. Tolerates `None`, `{}`, and missing keys without raising.

THEN the weekly scheduler (Phase 4.2.4 future work) can post an accurate,
accessible, sample-size-honest weekly accuracy report to the Slack channel
without any further backend wiring.

## In-Scope Changes

- **File:** `backend/slack_bot/formatters.py`
- **Change 1:** Add helper `_pct(value, decimals=1, signed=False) -> str`
  for consistent percent rendering. Stdlib only.
- **Change 2:** Add public `format_accuracy_report(data: dict | None,
  window: tuple[str, str] | None = None) -> list[dict]`. Pure function.
  No new imports. Uses existing `_truncate` + `datetime`.
- **Location:** Appended after `format_morning_digest` at end of file.

## Out of Scope (Intentionally Deferred)

1. **Scheduler wiring** -- the Phase 4.2.4 weekly-scheduler caller.
2. **BQ durable persistence of `signal_history`** -- Phase 4.2.4 territory.
3. **`get_signal_history` since_date lex trap (SN4 from Phase 4.2.2)**.
4. **Additional fields on `get_accuracy_report` return shape**.
5. **Interactive overflow / select menu for group drill-down**.
6. **Color-coding of hit rates** -- CFA III(D) warning from research.
7. **New emojis or emoji-as-label** -- a11y rule.
8. **Changes to existing formatters** (5 public funcs must be byte-identical).

## Anti-Leniency Rules (MUST enforce in QA)

1. **No new imports.** The only top-of-file import today is
   `from datetime import datetime`. No additions.
2. **No touches to `signals_server.py`, `backtest_server.py`, `data_server.py`.**
3. **No changes to the 5 existing public formatters.** AST byte-identical.
4. **Never raise.** `format_accuracy_report(None)`,
   `format_accuracy_report({})`, `format_accuracy_report({"hits": "bad"})`
   all return `list[dict]` without exception.
5. **No non-ASCII in string literals** in the new function or helpers.
6. **Fields array size is EVEN AND <= 10.**
7. **Per-section mrkdwn text <= 2500 chars; header <= 140 chars.** Truncate,
   do not raise.
8. **Diff budget:** <= 220 added lines total (including docstring),
   <= 10 deleted lines.

## Success Criteria (QA must run these)

### SC1-SC5: Return-shape basics

- **SC1:** Empty-but-present fixture (total_count=0) returns `list[dict]`
  length >= 2 with at least a header and a section or context explaining the
  empty state. No headline fields rendered.
- **SC2:** `format_accuracy_report(None)` returns `list[dict]`, never raises.
- **SC3:** `format_accuracy_report({})` returns `list[dict]`, never raises.
- **SC4:** `format_accuracy_report({"total_count": "bad"})` returns
  `list[dict]`, never raises, coerces/ignores bad values.
- **SC5:** Every block is a dict with a `type` key.

### SC6-SC12: Normal-path rendering (n >= 5 scored)

- **SC6:** Full fixture (`total_count=20, scored_count=12, hits=7, misses=5,
  hit_rate=0.5833, CI=[0.3056, 0.8043], mean=1.24, median=0.85, groups={}`)
  produces: (a) one `header`, (b) a `context`, (c) at least one `section`
  with `fields`, (d) at least one `divider`, (e) a trailing `context`.
- **SC7:** Header text is `plain_text` and <= 150 chars.
- **SC8:** Fields array length is EVEN and <= 10.
- **SC9:** Hit rate appears as `58.3%` (one decimal, percent suffix).
- **SC10:** Wilson CI appears as `[0.31, 0.80]` (two decimals fraction).
- **SC11:** Mean / median forward returns appear signed with `%`
  (`+1.24%`, `+0.85%` or negative equivalents).
- **SC12:** Total signals and scored counts are integers, no float artifacts.

### SC13-SC17: Wilson CI display rule

- **SC13:** `scored_count=0` -> hit-rate row replaced with `Scoring pending`
  (or hit-rate field is absent). No `0.00%` rendered as real.
- **SC14:** `1 <= scored_count < 5` -> CI field replaced with
  `preliminary -- n={X}`. No bracketed CI.
- **SC15:** `scored_count=5` -> CI IS shown.
- **SC16:** `scored_count=100` -> CI IS shown.
- **SC17:** CI strings never contain literal `None`, `nan`, or `inf`.

### SC18-SC22: Groups / drill-down

- **SC18:** Empty `groups={}` -> no per-group blocks.
- **SC19:** `groups` with 3 entries -> 3 per-group `section` blocks (mrkdwn,
  not fields).
- **SC20:** `groups` with 10 entries -> exactly 5 per-group blocks (top by
  `scored_count` desc) plus a context block noting `+5 more groups`.
- **SC21:** Each per-group mrkdwn contains the group label, hit rate with
  `.1f%`, and `(hits/scored)` fraction.
- **SC22:** No per-group block text exceeds 500 chars.

### SC23-SC27: Defensive / byte-identity

- **SC23:** `window=("2026-04-07", "2026-04-14")` renders
  `2026-04-07 to 2026-04-14` in the early context block.
- **SC24:** `window=None` still produces a valid context block.
- **SC25:** All 5 existing public formatters AST byte-identical pre/post.
- **SC26:** Top-of-file import list byte-identical.
- **SC27:** AST walk of every string literal in new code: 0 non-ASCII.

### SC28-SC30: Contract bounds

- **SC28:** Diff <= 220 added lines, <= 10 deleted.
- **SC29:** `ast.parse` and `py_compile` both clean.
- **SC30:** No references to `signals_server`, `backtest_server`,
  `data_server`, or `mcp_servers` in new code.

## Verification Command Block

```bash
python3 -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read())"
python3 -m py_compile backend/slack_bot/formatters.py

python3 - <<'PY'
import ast, importlib.util
spec = importlib.util.spec_from_file_location("fmt", "backend/slack_bot/formatters.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f = m.format_accuracy_report
assert isinstance(f(None), list)
assert isinstance(f({}), list)
assert isinstance(f({"total_count": "bad"}), list)
full = {"total_count": 20, "scored_count": 12, "hits": 7, "misses": 5,
        "neutral": 3, "unscored": 5, "hit_rate": 0.5833,
        "hit_rate_ci_low": 0.3056, "hit_rate_ci_high": 0.8043,
        "mean_forward_return_pct": 1.24,
        "median_forward_return_pct": 0.85, "groups": {}}
blocks = f(full, window=("2026-04-07","2026-04-14"))
assert any(b["type"] == "header" for b in blocks)
assert any(b["type"] == "section" and "fields" in b for b in blocks)
print("smoke PASS", len(blocks), "blocks")
PY
```

## Rollback

`git checkout HEAD~1 -- backend/slack_bot/formatters.py`.
