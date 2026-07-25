---
name: probe-self-contamination-shared-module
description: A probe that monkeypatches a shared module attribute (yf.Ticker) poisons its own later fetches; identical rows across different inputs is the tell
metadata:
  type: feedback
---

When an old-vs-new comparison probe patches a module attribute to inject a
fixture (`NEW.yf.Ticker = _Fake`), remember that `NEW.yf` is the SAME module
object the probe itself fetches with. Capture the real callable FIRST
(`REAL = yf.Ticker`) and use it for every fetch, restoring after each
comparison.

**Why:** on 2026-07-25 (phase-80.31 cycle 3) my live criterion-4 probe patched
`NEW.yf.Ticker` inside the loop, so tickers 2-8 all re-ran AAPL's cached frame.
The table printed eight confident rows — 251/250/251/250, dtype int64, identical
signals — that looked like strong 8-ticker corroboration and were one ticker of
real data. The corrected run showed SAP.DE 254/254/254/254 and 005930.KS
243/243/243/243 (no malformed row on those markets at all) and MSFT NORMAL, not
ANOMALY_OPPORTUNITY.

**How to apply:** the tell is *suspiciously identical rows across inputs that
should differ*. Before reporting a multi-input table, check that at least one
column varies where the underlying data must vary; if every row is identical,
suspect the harness before believing the result. This is the evaluator's own
version of [[measure-dont-assert]] — my instrument was the unmeasured claim.
Related: [[survivor-needs-behavioural-differential]] (check the baseline row is
real before scoring against it).
