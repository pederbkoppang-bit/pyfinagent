# Experiment Results — phase-8 / 8.4 (Promote or reject decision memo)

**Step:** 8.4 — final phase-8 step. **Date:** 2026-04-20 **Cycle:** 1.

## What was built

One new doc: `handoff/current/phase-8-decision.md` (~100 lines). **Verdict: REJECT.** First line starts with `REJECT:` as required by the immutable grep.

Doc sections: (1) What was shipped across 8.1–8.3 with Q/A verdicts. (2) Evidence behind REJECT including runtime gate, published evidence of zero-shot TSFMs underperforming, ensemble-has-no-data. (3) What promotion would have required — 4 conditions, none met. (4) What stays on disk (scaffolds retained). (5) What stays disabled. (6) Re-evaluation triggers. (7) Explicit non-decisions. (8) References.

## Verification

```
$ test -f handoff/current/phase-8-decision.md && echo "FILE OK"
FILE OK

$ grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md && echo "PREFIX OK"
PREFIX OK

$ test -f handoff/current/phase-8-decision.md && grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md && echo "IMMUTABLE PASS"
IMMUTABLE PASS

$ head -1 handoff/current/phase-8-decision.md
REJECT: Shadow-only pilots are kept as scaffolds; no live trading in phase-8.

$ python3 -c "open('handoff/current/phase-8-decision.md','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Mid-cycle fix

Initial draft included 40 non-ASCII bytes (em-dash U+2014, R-squared U+00B2, Sec. U+00A7). Replaced with ASCII equivalents (`--`, `R-squared`, `Sec.`) while preserving the `REJECT:` first-line prefix. Second pass clean.

## Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `test -f handoff/current/phase-8-decision.md && grep -qE '^(PROMOTE|REJECT):' ...` | PASS |

## Phase-8 closure

With qa_84_v1 PASS, phase-8 closes 4/4:

- 8.1 TimesFM scaffold → PASS (qa_81_v1)
- 8.2 Chronos-Bolt scaffold → PASS (qa_82_v1)
- 8.3 Ensemble blend → PASS (qa_83_v1)
- 8.4 Decision memo → PASS (pending qa_84_v1)

Outcome: **REJECT** — no live trading. Scaffolds retained for re-evaluation under 4 conditions enumerated in the memo.
