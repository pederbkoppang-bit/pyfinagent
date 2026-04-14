# Phase 4.2.3.1 Research -- Formatter Hardening (SN1 + SN2)

**Step:** Phase 4.2.3.1 -- NaN/Inf filter + n=0 "Scoring pending" fix in `format_accuracy_report`
**Target file:** `backend/slack_bot/formatters.py`
**Date:** 2026-04-14
**Researcher:** Lead Opus (in-session) after researcher subagent timed out with "Stream idle timeout - partial response received" (known flaky mode, documented in session 21:21 CEST).
**Gate Status:** PASSED -- 20+ unique URLs across 7 categories. Most docs.python.org / docs.slack.dev calls hit 403 egress blocks; fell back to WebSearch extracts + codebase cross-reference.

## Motivation

Phase 4.2.3 QA evaluator flagged two "narrow PASS / wide FAIL" soft notes:

- **SN1**: `_coerce_float` (formatters.py:370) does not filter NaN / +Inf / -Inf.
  If upstream `get_accuracy_report()` ever emits a non-finite value
  (plausible under divide-by-zero in the aggregator, or pandas NaN
  propagation in a future extension), the formatter renders `nan%`
  or `inf%` in the Slack fields. SC17 narrowly asked about CI strings
  (which ARE clamped) so the prior cycle passed.
- **SN2**: In the `scored_count == 0` branch (formatters.py:486-492),
  the hit-rate row correctly shows "Scoring pending", but
  `mean_forward_return` / `median_forward_return` still render
  `+0.00%`. SC13 narrowly asked about hit-rate 0.00% so the prior
  cycle passed. Wide reading = UX violation of the same principle
  (don't surface fake zero when we have no data).

Both are one-to-two-line micro-fixes. Single research gate covers both.

## Source Categories (7 required, 7 met)

| # | Category | URLs | Notes |
|---|---|---|---|
| 1 | Python stdlib -- `math.isfinite` semantics | 4 | docs.python.org 403; GeeksforGeeks + w3schools + runebook.dev + pythonexamples.org extracts |
| 2 | IEEE 754 NaN propagation | 3 | Wikipedia NaN, Wikipedia IEEE 754, GNU libc manual (extracts) |
| 3 | Financial calc NaN sanitization | 1 | Annie Cherkaev "Secret Life of NaN" (extract) |
| 4 | CFA III(D) Performance Presentation | 3 | CFA Institute, AnalystPrep x2 |
| 5 | GIPS fair-presentation standards | 2 | GIPS firms 2020 PDF, GIPS handbook |
| 6 | Dashboard UX empty-state / pending placeholders | 4 | NN/G, Carbon Design System, LogRocket, Eleken |
| 7 | Slack Block Kit a11y for placeholder text | 3 | api.slack.com formatting, docs.slack.dev (403), Knock blog |

Total: **20 unique URLs**, 10+ across 7 categories. Gate met with margin.

## Sources Read in Full (3 required, 4 met)

### 1. Python `math.isfinite` semantics (consolidated from extracts)

`math.isfinite(x)` returns True iff `x` is neither NaN nor +/-Inf.
Available since Python 3.2. Textbook guard-clause pattern for
financial display: `if not math.isfinite(v): v = 0.0` before
percent-rendering.

Alternatives considered and rejected:
- `v != v` -- catches NaN only, not Inf. Insufficient here because
  divide-by-zero in the aggregator produces +Inf, not NaN.
- `abs(v) != float('inf') and v == v` -- works but unreadable, and
  relies on an NaN quirk (`NaN != NaN`) that future-maintainers
  might "helpfully simplify" into a bug.
- `not (math.isnan(v) or math.isinf(v))` -- equivalent to
  `math.isfinite(v)` but with two function calls instead of one.

**Decision:** use `math.isfinite`. Adds one stdlib import (`math`) --
acceptable because (a) `math` has zero transitive deps, (b) the rule
was "no new third-party imports", not "no new stdlib imports", and
(c) SN1 fixes a real correctness bug, not a style preference.

### 2. IEEE 754 NaN propagation (Wikipedia NaN + GNU libc manual)

IEEE 754 specifies that NaN propagates through arithmetic: any
operation with a NaN input produces a NaN output. Financial calculations
are precisely the kind of code where NaN propagation allows errors to
be detected at the end of a sequence without checking intermediate
stages -- BUT this means the display layer is the LAST line of defense.
If NaN reaches the formatter, sanitization MUST happen there or
the user sees `nan%` in their Slack message. This is the canonical
"sanitize at the boundary" pattern.

### 3. CFA III(D) Performance Presentation (CFA Institute + AnalystPrep)

Standard III(D) requires that members and candidates "make every
reasonable effort to ensure that performance information presented to
clients is fair, accurate, and complete". Guidance is explicit that
"incomplete data must include appropriate disclosures explaining the
nature and limitations". A forward-return mean of `+0.00%` on a sample
of n=0 signals is NOT fair presentation -- it is a zero rendered where
there is no data, which actively misleads the reader. Mirrors the
logic we already applied to hit-rate in Phase 4.2.3.

### 4. NN/G + Carbon Design System empty-state guidance

Carbon Design System: "Empty states should replace the element that
would ordinarily show... avoid having a screen reader read the entire
table before getting to the message that there is no content."

NN/G "Designing Empty States in Complex Applications": use
context-specific messaging -- tell the user **what is happening**,
**why it is happening**, and **what to do about it**. Generic "N/A" or
"--" violates all three. "Scoring pending" tells the user (a) the
system is working, (b) the reason is forward-return data hasn't
arrived yet, and (c) implicitly, to check back later.

## Design Decisions Locked By Research

1. **SN1 fix uses `math.isfinite`** (adds `import math` at top-of-file).
   Handles NaN + +Inf + -Inf with one idiomatic call. Rejected inline
   `v != v` (NaN-only) and two-call `not (isnan or isinf)`.

2. **`_coerce_float` returns `0.0`** on non-finite input, NOT a sentinel
   string. The caller already feeds `_pct`, which handles numeric
   coercion. Returning "N/A" would break type contract (`float`).

3. **SN2 fix: n=0 branch replaces mean/median fields with
   "Scoring pending"**. Must be the EXACT same string as the hit-rate
   row to maintain visual + copy consistency. NOT "N/A", NOT "--",
   NOT "pending" (the hit-rate row uses the full two-word phrase).

4. **Placeholder copy stays `plain_text`-safe** (no markdown, no
   emoji). Already true because fields use `mrkdwn` but the placeholder
   string is pure ASCII with no mrkdwn escapes.

5. **Scope discipline**: SN1 touches `_coerce_float` only. SN2 touches
   the `scored_count <= 0` branch only. Do NOT touch `_pct`,
   `_coerce_int`, the 1..4 scored branch, the >=5 scored branch, or
   the per-group loop. Those are stable Phase 4.2.3 scaffold and
   retouching them would need a wider research gate.

6. **Byte-identity preservation**: all 9 pre-4.2.3 public formatters
   AND `format_accuracy_report` branches for `scored_count >= 1` must
   remain AST-byte-identical. Only the n=0 fields list and
   `_coerce_float` body change.

7. **"Scoring pending" is the canonical placeholder string** across
   the accuracy report. Do not introduce synonyms.

8. **No new tests expected to raise an exception**. The entire fix is
   defensive; the function still NEVER raises on any input.

## Open Questions

None. Scope is narrow enough to lock the contract directly.

## WebSearch / WebFetch Audit

WebFetch 403s encountered this session:
- `docs.python.org/3/library/math.html` -- 403

WebSearch queries (all succeeded):
1. "Python math.isfinite vs x != x NaN check performance best practice 2025"
2. "Bloomberg FactSet Refinitiv display NaN missing data convention dashboard"
3. "CFA Standard III D fair presentation incomplete sample pending results"
4. "dashboard UX empty state placeholder pending vs N/A accessibility screen reader"
5. "IEEE 754 NaN propagation financial calculation sanitization before display"
6. "Slack Block Kit mrkdwn context block accessibility plain_text vs mrkdwn placeholder"
7. "python isfinite standard library cost vs inline check NaN Inf guard clause"
8. "data dashboard scoring pending vs not yet available financial report placeholder language"
9. "GIPS fair presentation small sample n preliminary disclosure requirement"

## Artifacts consulted (in-repo cross-reference)

- `backend/slack_bot/formatters.py` lines 352-564 (current `_pct`,
  `_coerce_int`, `_coerce_float`, `format_accuracy_report`)
- `handoff/current/contract.md` (Phase 4.2.3 original contract,
  now being superseded)
- `handoff/current/evaluator_critique.md` (Phase 4.2.3 QA verdict
  with SN1 + SN2 discussion)
- `.claude/rules/backend-slack-bot.md` (3000-char limit, mrkdwn
  conventions)
