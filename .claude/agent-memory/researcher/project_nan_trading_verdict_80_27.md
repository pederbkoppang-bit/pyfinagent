---
name: project-nan-trading-verdict-80-27
description: phase-80.27 measurements — yfinance NaN tail row is PERMANENT + universal (not market-hours), quant_model absent from _SOURCE_CRITICALITY, ERROR path proven crash-free, 18 enumerated NaN-vulnerable ladders incl. 4 risk-control bypasses
metadata:
  type: project
---

Measured 2026-07-25 while researching masterplan step 80.27 (the trading-side
half of [[project-nan-json-leak-80-1]]).

**1. The NaN is NOT a forming-session artifact — it is permanent and universal.**
Measured on a SATURDAY with US markets closed: `yf.Ticker('AAPL').history(period='3mo')`
returns 62 rows whose LAST row (the most recent completed session) has NaN
Open/High/Low/Close and a **real** Volume (47,402,209). yfinance 1.2.0 /
pandas 3.0.1. Real volume is why yfinance's own `keepna` mask (`.all(axis=1)`)
does not drop it. 3/3 tickers affected, weekend included.

**Why:** every prior write-up (including the step body) assumed a market-hours
forming-session row that clears after the close. It does not. Any "ship it after
close" or "only affects intraday" reasoning is wrong.

**How to apply:** never gate a NaN-related fix on market hours. `screener.py:169-170`
`.dropna()` is the reason the trading funnel escapes this; `price_quality.py`
does NOT catch it (US fast-path no-op at `:55-56`, and `is_bad_bar` returns
False for all-NaN because `NaN is not None` and every comparison is False).

**2. `quant_model` is invisible to info-gap.** `_SOURCE_CRITICALITY`
(`backend/agents/info_gap.py:19-31`) has 11 keys; `enrichment_raw`
(`orchestrator.py:2008-2015`) has 12. `detect_info_gaps` iterates the
criticality dict, so quant_model is never assessed, never counted in the
denominator, and can never enter `critical_gaps`. Returning ERROR from
quant_model changes nothing until that key is added.

**3. `signal: "ERROR"` is provably safe to emit.** There are ZERO field-level
dereferences of the sector/quant payloads outside the tools — only
`.get("signal")` / `.get("summary")` at `orchestrator.py:2167`, `:2171`, `:2291`.
So an ERROR payload missing every data key cannot KeyError. ERROR is excluded
from session memory (`:2003`), not merged on retry (`:2046`), stripped from
compacted debate prompts (`:2181` `_DEAD_SIGNALS`), and raises a
confidence-lowering bias flag (`bias_detector.py:231-246`). `data_quality_min`
is 0.5 (`settings.py:234`) and dq only falls 1.0 -> 0.91, so nothing is gated off.

**Cost:** the failure is deterministic, so `retry_critical_gaps` burns both
retries every time — one `get_sector_analysis` is ~14 yfinance round-trips, so
roughly 500-900 wasted calls per cycle. Budget for HTTP 429.

**4. Four NaN-suppressed RISK CONTROLS found during the ladder audit** (each
needs its own step; all are "NaN makes the comparison False so the control never
arms"): `orchestrator.py:347-374` sector-concentration warning,
`mcp_servers/signals_server.py:926` per-ticker limit, `:935` total exposure
limit, `:1285-1287` kill-switch / de-risk.

**How to apply:** the full 18-ladder enumeration with the membership rule lives
in `handoff/archive/phase-80.27/research_brief_80.27.md` (or
`handoff/current/` while the step is open). Reuse the membership rule rather
than re-deriving it. See also [[feedback-measure-dont-assert-claims]].
