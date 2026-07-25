# Research Brief — phase-80.27 (NaN → confident NEUTRAL trading verdict)

**Tier:** complex · **coverage.audit_class:** true (K=2 dry rounds required)
**Started:** 2026-07-25 · **Status:** IN PROGRESS (write-first, incremental)
**Researcher:** Layer-3 researcher subagent

---

## Immutable success criteria (copied VERBATIM from the caller; not amended)

1. A NaN-poisoned enrichment payload is classified NOT-SUFFICIENT by _assess_source_status, so a HIGH-criticality source routes into critical_gaps and the existing retry loop fires
2. sector_analysis and quant_model return the DOCUMENTED signal: 'ERROR' (or an explicit NO_DATA) on non-finite inputs -- never NEUTRAL. Initialising signal to NEUTRAL before a comparison ladder is itself the bug: default to the failure state
3. No prose or JSON containing 'nan' can reach an LLM prompt -- assert on the rendered summary string and on the serialised payload the sector agent receives
4. Every threshold ladder that can receive a computed float is guarded with math.isfinite BEFORE the comparisons (sector_analysis.py:140-153, quant_model.py:171-181). Enumerate the ladders you checked; do not assert a count you did not measure
5. MUTATION-TEST each guard: feed float('nan') and assert the verdict is ERROR/NO_DATA, then remove the guard and confirm the test FAILS. A test that only asserts 'no crash' does not count
6. quant_model must not report mda_source='backtest' on a payload whose factors are non-finite

**Immutable verification command:**
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import math; from backend.agents.info_gap import _assess_source_status; p={'signal':'NEUTRAL','summary':'3M return: +nan% vs sector +nan%','stock_returns':{'1mo':float('nan')}}; print('status:', _assess_source_status('sector', p))"
```

---

## Pre-fix baseline — the immutable verification command, run 2026-07-25

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import math; from backend.agents.info_gap import _assess_source_status; p={'signal':'NEUTRAL','summary':'3M return: +nan% vs sector +nan%','stock_returns':{'1mo':float('nan')}}; print('status:', _assess_source_status('sector', p))"
status: SUFFICIENT
```

The command runs clean today and prints the WRONG answer. The step passes when
this prints a NOT-SUFFICIENT status (`MISSING` — see §B2 for why `MISSING`
specifically, and not a new `NO_DATA` string).

## STATUS LOG

- [x] File created (write-first)
- [x] Internal pass 1 — re-verified all 5 links at real line numbers
- [x] Internal pass 2 — threshold-ladder enumeration, 6 rounds to 2 dry
- [x] Internal pass 3 — ERROR-path trace (B7) + measured blast radius (B8)
- [x] External passes 1-3 (scan / gap / adversarial)
- [x] Recency scan (2024-2026), incl. an honest negative
- [x] Envelope (tail of file)

---

# B. INTERNAL CODE INVENTORY (the crux)

## B1. Re-verification of the four links at REAL current line numbers

### Link 1 — `backend/tools/sector_analysis.py` — CONFIRMED, line numbers **drifted by 4**

Caller said `:136-153`. Actual: the initialiser is at **:132**, the three
`dict.get(..., 0)` reads are at **:136-138**, the ladder is at **:140-153**.

```python
131	        # Signals
132	        signal = "NEUTRAL"          # <-- initialised to a VALID verdict
133	        sector_tailwind = False
134	        stock_outperforming = False
135	
136	        sec_3m = sector_returns.get("3mo", 0)
137	        spy_3m = spy_returns.get("3mo", 0)
138	        stock_3m = stock_returns.get("3mo", 0)
139	
140	        if sec_3m > spy_3m:
141	            sector_tailwind = True
142	
143	        if stock_3m > sec_3m:
144	            stock_outperforming = True
145	
146	        if sector_tailwind and stock_outperforming:
147	            signal = "DOUBLE_TAILWIND"
148	        elif sector_tailwind:
149	            signal = "SECTOR_TAILWIND"
150	        elif stock_outperforming:
151	            signal = "STOCK_OUTPERFORMING"
152	        elif stock_3m < sec_3m and stock_3m < spy_3m:
153	            signal = "LAGGING"
```

**Why the `0` defaults never fire — measured, not inferred.** The NaN enters at
`_compute_return` (`:28-36`):

```python
34	        return ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
```

When the forming-session placeholder row is present, `Close.iloc[-1]` is NaN, so
the arithmetic yields NaN — **not** `None`. The three store sites then all pass
their `is not None` guard and write the key WITH a NaN value:

- `:55-56` `if ret is not None: stock_returns[period] = round(ret, 2)` — `round(nan,2)` is `nan`
- `:70-71` same for `sector_returns`
- `:78-79` same for `spy_returns`
- `:87-88` same for `sector_performance`

So `sector_returns.get("3mo", 0)` returns **NaN**, never `0`. IEEE-754 makes all
four ladder comparisons False, control never reassigns `signal`, and the
function returns the `:132` initialiser `"NEUTRAL"` with
`sector_tailwind=False` (`:133`) and `stock_outperforming=False` (`:134`).

**Additional finding not in the step body:** the `:171-177` summary f-string uses
`{stock_3m:+.1f}%`, which renders NaN as the literal token `+nan%`. That string
is BOTH the `summary` that `_assess_source_status` substring-scans (no 'error',
no 'failed' → SUFFICIENT) AND the prose handed to the LLM. One value, two
failures.

**Also:** the try/except at `:44-182` already returns the documented
`{"signal": "ERROR", ...}` at `:182` — so the ERROR contract already exists in
this file and downstream already sees ERROR from it on exceptions. The fix is
to route the non-finite case into that SAME existing shape, not a new one.

### Link 2 — `backend/tools/quant_model.py` — CONFIRMED, line numbers EXACT

```python
167	    score = weighted_sum / total_weight if total_weight > 0 else 0.0
...
171	def _classify_signal(score: float) -> str:
172	    """Map composite score to a signal bucket."""
173	    if score > 0.08:
174	        return "STRONG_BULLISH"
175	    if score > 0.03:
176	        return "BULLISH"
177	    if score < -0.08:
178	        return "STRONG_BEARISH"
179	    if score < -0.03:
180	        return "BEARISH"
181	    return "NEUTRAL"      # <-- fall-through IS the failure sink
```

Note the structural difference from link 1: `_classify_signal` has no
initialiser variable — the fall-through `return "NEUTRAL"` at `:181` plays the
same role. `total_weight` is a sum of `abs(mda_weight)` values (`:152`, MDA
weights are finite), so the `total_weight > 0` guard at `:167` PASSES; only
`weighted_sum` is NaN. So `score = nan / <positive finite>` = nan. The guard
that exists protects against divide-by-zero, not against non-finite numerator.

**Where the NaN enters quant_model** (`_build_live_features`, `:53-117`) — more
sites than sector_analysis:
- `:65` `current = float(closes[-1])` — NaN if the last bar is the forming row
- `:70` `features[label] = float((current / closes[-days] - 1))` — momentum_1m/3m/6m all NaN
- `:75-76` `np.diff(np.log(closes))` then `np.std(...)` — one NaN poisons the whole std → `annualized_volatility` NaN
- `:82-85` `np.mean(closes[-50:])` → NaN → `sma_50_distance`/`sma_200_distance` NaN (and note `if sma50 else 0.0` — `bool(nan) is True`, so the falsy-guard does NOT fire)
- `:89` `volumes[-1] / np.mean(volumes[-20:])` — NaN volume
- `:38-50` `_compute_rsi`: `avg_loss == 0` is False for NaN, so `rs = nan/nan` → returns NaN (the 50.0 and 100.0 escapes do not fire)

`:82-85` is a second instance of the same class as `:167`: a falsy/zero guard
that a NaN silently walks through because `bool(nan) is True`.

**Criterion 6 (`mda_source`):** set at `:195`
`mda_source = "backtest" if mda else "equal_weight"` — it describes ONLY the
provenance of the WEIGHTS (`get_latest_mda()`), never the finiteness of the
FEATURES. It is emitted verbatim at `:250` and into the summary at `:241`. So a
payload with 100% NaN features still advertises `mda_source: 'backtest'`.

**`top_factors` (`:219-235`):** `contributing.sort(key=lambda x: abs(x["contribution"]), reverse=True)` at `:234`.
CPython's Timsort is stable and comparisons against NaN are all False, so with
an all-NaN key the sort is a **no-op** — `top_factors` is the first 5 features
in `dict` insertion order, presented as "top contributing factors". Confirmed
the ranking claim is false, as the step body asserts.

**Existing ERROR contract in this file too:** `:204-212` (empty features) and
`:259-267` (exception) both already return `signal: "ERROR"`. Same conclusion as
link 1 — reuse, don't invent.

### Link 3 — `backend/agents/info_gap.py:43-56` — CONFIRMED, line numbers EXACT

```python
43	def _assess_source_status(key: str, data: dict) -> str:
44	    """Classify a data source as SUFFICIENT, PARTIAL, MISSING, or SKIPPED."""
45	    if not data or not isinstance(data, dict):
46	        return "MISSING"
47	    if data.get("signal") == "ERROR":
48	        return "MISSING"
49	    if data.get("signal") == "SKIPPED":
50	        return "SKIPPED"
51	    summary = data.get("summary", "")
52	    if "error" in summary.lower() or "failed" in summary.lower():
53	        return "MISSING"
54	    if data.get("signal") == "N/A" and not summary:
55	        return "PARTIAL"
56	    return "SUFFICIENT"
```

Four checks, zero numeric inspection. Confirmed exactly as the step body states.

### Link 4 — `.claude/rules/backend-tools.md` — CONFIRMED

> `Error returns: { "ticker": "...", "signal": "ERROR", "summary": "...", "data": {} }`

The documented tool contract. Both tools already implement it on the EXCEPTION
path (`sector_analysis.py:182`, `quant_model.py:204/:259`); neither implements it
on the *silently-non-finite* path. **The fix is contract COMPLIANCE, not a new
contract.**

### Link 5 (found during 80.1, now verified) — `backend/config/prompts.py:1111`

```python
1111	    return format_skill(template, ticker=ticker, quant_model_data=json.dumps(quant_model_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))
```

CONFIRMED at the exact line. Stdlib `json.dumps` defaults to `allow_nan=True`,
which emits a bare `NaN` token — invalid JSON per RFC 8259 §6 (this is already
documented in this repo at `backend/api/_json_safe.py:23`). See §B5 for the full
count of affected prompt-builders.

---

## B3 (audit-class). ENUMERATION OF EVERY THRESHOLD LADDER

### The membership rule (written down BEFORE applying it)

> A site is IN SET iff **all four** hold:
> 1. It is a chain of one or more relational comparisons (`>`, `<`, `>=`, `<=`)
>    in an `if`/`elif`/`else` structure (or an early-`return` ladder), **and**
> 2. at least one operand is a **float** (not an `int` count, not a `len()`,
>    not a string/bool), **and**
> 3. that float is **derived from external market/API data** — yfinance, FRED,
>    AlphaVantage, pytrends, SEC, a BQ read, or an embedding — so it can be
>    non-finite without any code change, **and**
> 4. the ladder's **non-matching path** (an initialiser assigned before the
>    ladder, an `else:`, or a fall-through `return`) yields a **valid verdict**
>    (a signal / bucket / classification), not an error or an exception.
>
> Scope: `backend/tools/**` and `backend/agents/**`. Sites failing (2) or (3)
> are recorded as EXCLUDED-with-reason so the exclusion is auditable, not
> silent.

Rationale for clause (4): a ladder whose fall-through raises or returns ERROR is
already fail-safe — NaN makes it noisy, not wrong. Clause (4) is what makes this
class *dangerous*: the failure is **indistinguishable from a real verdict**.

### Result — 12 candidate sites in `backend/tools/`; **8 IN SET, 4 EXCLUDED**

Generated by `grep -n 'signal *= *"' backend/tools/*.py` plus the two
`return "<BUCKET>"` ladders, then reading each site.

| # | Site | Float input (source) | Non-matching path yields | In set? | Severity |
|---|------|----------------------|--------------------------|---------|----------|
| L1 | `sector_analysis.py:140-153` (init `:132`) | `stock_3m`/`sec_3m`/`spy_3m` — yfinance `Close` ratio | `NEUTRAL` (initialiser) | **YES** | **CRITICAL — measured live** |
| L2 | `quant_model.py:173-181` | `score` — MDA-weighted yfinance features | `NEUTRAL` (fall-through `return`) | **YES** | **CRITICAL — measured live** |
| L3 | `monte_carlo.py:108-115` | `var_6m = six_month.get("var_95", 0)` — `np.percentile` of GBM paths | `EXTREME_RISK` (`else:`) | **YES** | **HIGH** (see note) |
| L4 | `anomaly_detector.py:204/:210/:217` | `de_ratio`, `short_ratio`, `beta` — yfinance `info` | anomaly NOT appended → `total==0` → `NORMAL` at `:234-236` | **YES** | **HIGH** |
| L5 | `fred_data.py:101` | `spread_val` — FRED T10Y2Y observation | `NEUTRAL` (initialiser `:96`) | **YES** | MEDIUM |
| L6 | `nlp_sentiment.py:179-184` | `aggregate` — Vertex embedding cosine similarity | `NEUTRAL` (`else:`) | **YES** | MEDIUM |
| L7 | `social_sentiment.py:107-114` (init `:106`) | `avg_sentiment`, `velocity` — AlphaVantage scores | `NEUTRAL` (initialiser) | **YES** | LOW |
| L8 | `social_sentiment.py:165-172` (init `:164`) | same, `_score_fallback_articles` keyword path | `NEUTRAL` (initialiser) | **YES** | LOW |
| L9 | `alt_data.py:135-142` (init `:134`) | `momentum` — pytrends interest series | `NEUTRAL` (initialiser) | **YES** | LOW |
| L10 | `options_flow.py:81-88` (init `:80`) | `pc_ratio` — **int/int** with `max(...,1)` divisor | `NEUTRAL` | EXCLUDED | fails (2): both operands are `int` sums after `.fillna(0)` |
| L11 | `sec_insider.py:246-251` (init `:245`) | `buy_sell_ratio = len(buys)/max(len(sells),1)` | `NEUTRAL` | EXCLUDED | fails (2)/(3): pure `len()` counts |
| L12 | `patent_tracker.py:135-140` (init `:134`) | `velocity_pct` from `by_year` int counts | `NEUTRAL` | EXCLUDED | fails (2)/(3): int counts, `0.0` literal init |
| L13 | `earnings_tone.py:134-145` | `c_score`/`ca_score`/`e_score` — phrase-match `len()` | `CAUTIOUS` (`else:`) | EXCLUDED | fails (2)/(3): int counts |
| L14 | `anomaly_detector.py:234-242` (classification) | `total`, `risk_anomalies` — int counts | `ANOMALY_OPPORTUNITY` (`else:`) | EXCLUDED | fails (2); the *float* exposure in this file is L4 upstream |

**Count: 9 IN SET (L1-L9), 5 EXCLUDED (L10-L14).** Every row was read; nothing
is asserted from the grep alone.

**Note on L3 (monte_carlo).** This is the same `dict.get(key, default)` bug shape
as L1 — `six_month.get("var_95", 0)` cannot return `0` if the key exists with a
NaN value. But its fall-through is `EXTREME_RISK`, i.e. NaN currently produces
the *most alarming* bucket. Directionally that is conservative, so it is not a
new-trade risk, but it is still a fabricated verdict on a HIGH-criticality source
and it is not distinguishable from a real extreme-risk reading.

**Note on L4 (anomaly_detector).** Different mechanism, same class: the NaN does
not flip a verdict directly, it **suppresses risk flags**. `if de_ratio and
de_ratio > 150` is False for NaN, so a leveraged company with a NaN
`debtToEquity` silently reports "No significant statistical anomalies detected."
There is already a queued step **80.31** ("anomaly_detector misaligned
price/volume arrays") touching this file — flag the overlap so 80.27 and 80.31 do
not collide on the same lines.

**Additional non-ladder guards of the same class (bonus findings, not ladders):**

| Site | Code | Why it fails on NaN |
|------|------|---------------------|
| `quant_model.py:84-85` | `(current - sma50) / sma50 if sma50 else 0.0` | `bool(nan) is True` → the falsy-escape never fires |
| `quant_model.py:47-48` | `if avg_loss == 0: return 100.0` | NaN `== 0` is False → RSI returns NaN instead of the 100.0 escape |
| `quant_model.py:167` | `weighted_sum / total_weight if total_weight > 0 else 0.0` | guards the *denominator* only; a NaN numerator sails through |
| `quant_model.py:234` | `contributing.sort(key=lambda x: abs(x["contribution"]))` | all-False comparisons → Timsort no-op → `top_factors` is insertion order presented as a ranking |
| `sector_analysis.py:55/70/78/87` | `if ret is not None:` | NaN **is not** None → key stored WITH the NaN, defeating every downstream `.get(k, 0)` |

The last row is the single most important line in this brief: **the `is not
None` idiom is the root enabler.** Every `dict.get(key, default)` default in
this codebase is dead code whenever the producer used an `is not None` filter on
a value that can be NaN.

---

## B2. `info_gap.py` in full — the exact status string to return

### Statuses `_assess_source_status` can return (complete set: 4)

| Status | Returned when | What `detect_info_gaps` does with it |
|--------|---------------|--------------------------------------|
| `MISSING` | payload not a dict / falsy (`:45`); `signal == "ERROR"` (`:47`); `'error'` or `'failed'` in `summary.lower()` (`:52`) | `:98` — if criticality is HIGH, appended to `critical_gaps`. Excluded from `sufficient_count`, INCLUDED in the denominator → drags `data_quality_score` DOWN |
| `SKIPPED` | `signal == "SKIPPED"` (`:49`) — set by `orchestrator.py:1974` `_skip_placeholder` for sector-routed skips | `:94-95` — increments `skipped_count`, then `:102` REMOVES it from the denominator. Neutral |
| `PARTIAL` | `signal == "N/A"` **and** empty summary (`:54`) | Neither sufficient nor critical. Drags the score down but never reaches `critical_gaps` — **so PARTIAL cannot fire the retry loop** |
| `SUFFICIENT` | everything else (`:56`) | `:96-97` — increments `sufficient_count` |

### THE ANSWER TO "what status string do I return"

**Return `signal: "ERROR"` from the tools.** Do NOT invent a `NO_DATA` status in
`info_gap`. Reasons, in order of force:

1. `"ERROR"` → `MISSING` (`:47-48`) is the ONLY path that reaches
   `critical_gaps` (`:98`), which is the ONLY thing that fires
   `retry_critical_gaps` (`orchestrator.py:2022, :2038`). Criterion 1 demands
   the retry loop fire; `PARTIAL` would satisfy "NOT-SUFFICIENT" but would
   **silently fail criterion 1**.
2. `"ERROR"` is the documented tool contract (`.claude/rules/backend-tools.md`)
   and criterion 2 names it first.
3. Both tools ALREADY return it on their exception paths, so no consumer is
   seeing a novel value.

**A `NO_DATA` string would be a silent regression.** `alt_data.py:115` already
returns `signal: "NO_DATA"` — and `_assess_source_status` has no case for it, so
`alt_data` with no Google Trends data is classified **SUFFICIENT** today. That
is a live, pre-existing instance of the same bug class, and it is the proof that
inventing a new status string here would be a mistake. (Recommend: queue the
`NO_DATA` mis-classification as its own defect step; it is out of 80.27 scope
because `alt_data` is LOW criticality and no ladder is involved.)

### `quant_model` IS NOT MONITORED BY INFO-GAP AT ALL — blocking finding for criterion 1

`_SOURCE_CRITICALITY` (`:19-31`) has exactly **11** keys, measured:

```
['alt_data', 'anomaly', 'earnings_tone', 'fred_macro', 'insider',
 'monte_carlo', 'nlp_sentiment', 'options', 'patent', 'sector',
 'social_sentiment']
```

`enrichment_raw` (`orchestrator.py:2008-2015`) has **12** keys — the extra one is
`quant_model`. `detect_info_gaps` iterates `for key, default_crit in
_SOURCE_CRITICALITY.items()` (`:81`), so **`quant_model` is never assessed,
never counted, and can never enter `critical_gaps`.** `total =
len(_SOURCE_CRITICALITY)` (`:77`) = 11, so the denominator silently excludes it
too.

**Consequence for the executor:** making `quant_model` return `ERROR` has ZERO
effect on `data_quality_score`, `critical_gaps`, or the retry loop. To satisfy
criterion 1 for quant_model, the step must ALSO add
`"quant_model": "HIGH"` to `_SOURCE_CRITICALITY` **and** add a `quant_model`
entry to the `retry_funcs` dict — which already exists at
`orchestrator.py:2036`, so only the criticality entry is missing. Note this
raises `total` from 11 → 12 and shifts every `data_quality_score` by one
denominator unit; that is a behaviour change worth calling out in the contract.

---

## B7. WHAT CONSUMES `signal: "ERROR"` TODAY — the end-to-end trace (MOST IMPORTANT)

**The live link is confirmed:** `backend/services/autonomous_loop.py:1887-1888`
constructs `AnalysisOrchestrator(settings)` and awaits
`orchestrator.run_full_analysis(ticker)`. So the same `sector_analysis` /
`quant_model` payloads that 80.1 was forbidden from touching DO reach the paper-
trading decision. The trade action is read at `:1915`
(`rec.get("action", "HOLD")`) from `synthesis["recommendation"]`.

### Every consumer of the two payloads, enumerated

| # | Consumer | Line | Behaviour on `signal="ERROR"` | Crash risk |
|---|----------|------|-------------------------------|-----------|
| 1 | `orchestrator._safe(fetch_sector_data)` | `:1988`, `:1992` | already the wrapper for exceptions; an ERROR **dict** is a normal return | none |
| 2 | session-memory `ctx.set_signal` | `:2002-2004` | `if sig != "N/A" and sig != "ERROR"` — ERROR is **explicitly excluded**. Session memory is not poisoned | none |
| 3 | `detect_info_gaps` | `:2019` | `sector` → MISSING → `critical_gaps` → retry ×2 (`:2038-2043`) | none |
| 4 | recovery merge | `:2045-2047` | `if new_data and new_data.get("signal") != "ERROR"` — a still-ERROR retry is **not** merged; the original stays | none |
| 5 | LLM enrichment agent | `:2104`, `:2108` → `prompts.get_sector_analysis_prompt` (`prompts.py:537`) | payload is `json.dumps`'d wholesale into a `{{sector_data}}` placeholder; no key is dereferenced | none |
| 6 | debate assembly | `:2167`, `:2171` | reads ONLY `.get("signal", "N/A")` and `.get("summary","")` | none |
| 7 | small-context compaction | `:2181` `_DEAD_SIGNALS = {"ERROR","UNAVAILABLE","N/A",""}` | ERROR rows are **stripped** from the debate prompt by design | none |
| 8 | synthesis input | `:2291` | again only `.get("signal")` / `.get("summary")` | none |
| 9 | `bias_detector._check_source_diversity` | `:228-246` | counts ERROR signals; ≥2 → MEDIUM `source_diversity` flag, ≥3 → HIGH flag with "Treat this analysis with lower confidence" | none — and this is the **desired** direction |
| 10 | `autonomous_loop` result assembly | `:1913-1943` | never touches enrichment payloads; reads `synthesis`/`quant`/`cost_summary` only | none |

**I found ZERO field-level dereferences of `sector_data` / `qm_data` outside the
tools themselves.** Verified by
`grep -rn 'sector_data\[\|sector_data\.get\|qm_data\[\|qm_data\.get' backend/`
→ 3 hits, all of them `.get("signal")` / `.get("summary")` at `:2167`, `:2171`,
`:2291`. So an ERROR payload that omits `stock_returns`, `score`, `top_factors`
etc. **cannot raise `KeyError` anywhere in the pipeline.** This is the single
strongest safety argument for the change.

### The data-quality arithmetic (measured, not assumed)

`settings.py:234` — `data_quality_min: float = Field(0.5, ...)`.

- Today: 11/11 SUFFICIENT → `dq = 1.0` → "Data quality: 100%" (the lie).
- After the fix, `sector` alone flips → 10/11 = **0.91** → still ≥ 0.5, so
  `low_data_quality` (`:2072`) is **False** → debate and risk assessment still
  run. **No analysis outage.**
- If `quant_model` is added to `_SOURCE_CRITICALITY` and also flips → 10/12 =
  **0.83** → still ≥ 0.5. Still no outage.
- `recommendation_at_risk` (`:109`) needs `len(critical_gaps) >= 3`; two gaps
  do not trip it.
- The debate-skip fallback (`:2136-2149`) would set `consensus: "HOLD"` with
  `confidence 0.3` if it ever DID trip — i.e. even the worst case is a **HOLD**,
  the most conservative action.

**Conclusion for B7: the ERROR path is fully handled, cannot crash, cannot
stall, and lands strictly on the conservative side.**

---

## B4. Every producer of a `signal` field in `backend/tools/` — who guards, who doesn't

Measured by `grep -c '"signal":' backend/tools/*.py` then reading each producer.
14 tool modules; 12 emit a `signal`, 6 do not (`yfinance_tool`, `slack`,
`screener`, `price_quality`, `alphavantage`, `__init__`).

| Tool | Verdict derived from a possibly-non-finite float? | Guards today? |
|------|--------------------------------------------------|---------------|
| `sector_analysis.py` | YES (L1) | **NO** — try/except only |
| `quant_model.py` | YES (L2) | **NO** — try/except + empty-features check only |
| `monte_carlo.py` | YES (L3) | **NO** |
| `anomaly_detector.py` | YES (L4, suppression-mode) | **NO** |
| `fred_data.py` | YES (L5) | partial — `is not None` at `:101`, **no isfinite** |
| `nlp_sentiment.py` | YES (L6) | **NO** |
| `social_sentiment.py` | YES (L7, L8) | **NO** |
| `alt_data.py` | YES (L9) | **NO** — and separately mis-classified: see the `NO_DATA` note in B2 |
| `options_flow.py` | no (int/int + `max(...,1)`) | n/a |
| `sec_insider.py` | no (`len()` counts) | n/a |
| `patent_tracker.py` | no (`len()` counts) | n/a |
| `earnings_tone.py` | no (phrase-match counts) | n/a |

**Not one tool in `backend/tools/` calls `math.isfinite`, `np.isfinite`, or
`pd.isna` before a verdict comparison.** Verified:
`grep -rn "isfinite" backend/tools/` returns nothing.

**The existing data-quality gate CANNOT catch this bar.** `price_quality.py`
(phase-50.5) is the repo's data-quality validator, and it is blind here on two
independent counts:
- `validate_ohlcv(df, market="US")` returns at `:55-56` — **US is a fast-path
  no-op**, and the affected book is US.
- Even if it ran, every rule is `.fillna(False)`-masked (`:73, :75, :81, :100,
  :110, :113`), so a NaN row satisfies no drop rule.
- `is_bad_bar` (`:134-151`) checks `any(x is None ...)` at `:140` — **NaN is not
  None** — and then `min(o,h,l,c) <= 0`, `h < l`, `o == h == l == c` are all
  False for NaN. **`is_bad_bar(nan, nan, nan, nan, 47402209)` returns `False`.**

So the executor must not assume "the 50.5 gate covers it". It does not.

---

## B5. Prompt serialisation — 16 sites, ALL stdlib `json.dumps`, ZERO with `allow_nan=False`

`grep -rn "allow_nan" backend/` returns hits in exactly three places, none of
which are prompt builders:
`backend/api/_json_safe.py` (docstring), `backend/api/signals.py:33`
(comment), `backend/tests/test_phase_80_1_signals_nan_serialisation.py`.

The 16 payload-serialising prompt builders in `backend/config/prompts.py`:

| Line | Payload | Tool source |
|------|---------|-------------|
| `:330` | `annotated` (fact ledger) | mixed |
| `:411` | `quant_data` | yfinance quant block |
| `:433` | `quant_report` | quant |
| `:489` | `insider_data` | sec_insider |
| `:495` | `options_data` | options_flow |
| `:501` | `sentiment_data` | social_sentiment |
| `:507` | `patent_data` | patent_tracker |
| `:513` | `transcript_data` | earnings_tone |
| `:523` | `fred_data` | fred_data |
| `:531` | `alt_data` | alt_data |
| **`:537`** | **`sector_data`** | **sector_analysis — L1** |
| `:561` | `quant_data` | quant |
| `:1082` | `enrichment_status` | info-gap/enrichment |
| `:1093` | `nlp_data` | nlp_sentiment |
| `:1099` | `anomaly_data` | anomaly_detector |
| `:1105` | `monte_carlo_data` | monte_carlo |
| **`:1111`** | **`quant_model_data`** | **quant_model — L2** |

(17 rows; `:411` and `:561` are two distinct builders over `quant_data`.)

Criterion 3 says "no prose or JSON containing 'nan' can reach an LLM prompt".
There are **two** independent leaks per tool, and the executor must assert on
both:

1. **The JSON leak** — `json.dumps` at `:537` / `:1111` with the default
   `allow_nan=True` emits a bare `NaN` token. Python's own docs are explicit:
   *"This behavior is not JSON specification compliant, but is consistent with
   most JavaScript based encoders and decoders"*, and RFC 8259 §6: *"Numeric
   values that cannot be represented in the grammar below (such as Infinity and
   NaN) are not permitted."*
2. **The prose leak** — the f-strings at `sector_analysis.py:171-177`
   (`{stock_3m:+.1f}%` → `+nan%`) and `quant_model.py:238-242`
   (`{score:.4f}` → `nan`, `{f['contribution']:+.3f}` → `+nan`). These are NOT
   fixed by any JSON setting; they are already strings by the time `json.dumps`
   sees them.

If the tools return ERROR (criterion 2), both leaks close at the source for
these two tools — the ERROR payload has no float fields and a plain-text
summary. That is the cheapest correct fix. A `json.dumps(..., allow_nan=False)`
sweep across all 17 sites would be defence-in-depth but converts a silent leak
into a `ValueError` inside prompt construction — **which would raise inside the
enrichment agent and be swallowed by `orchestrator.py:2118-2120`
(`result = {"text": f"Error: {e}", ...}`)**. That is survivable but noisy;
recommend it as a SEPARATE queued step, not bundled into 80.27.

---

## B6. `mda_source` — what it actually means

Set once, at `quant_model.py:195`:

```python
195	        mda_source = "backtest" if mda else "equal_weight"
```

`mda = get_latest_mda()` (`:194`, from `backend.backtest.backtest_engine`). So
`mda_source` describes the provenance of the **WEIGHTS only** — whether real
walk-forward MDA importances were loaded, or the `1/len(_LIVE_FEATURES)`
equal-weight fallback at `:199` was used. It says nothing whatsoever about the
**FEATURES**.

It is emitted at `:210` (empty-features ERROR path — note it already leaks
`"backtest"` there too), `:241` (into the summary prose), `:250` (top-level
field), and `:265` (`"error"` on the exception path).

**Criterion 6 therefore requires a change of MEANING, not just a value.** The
minimal honest implementation: when any contributing feature is non-finite, the
function returns ERROR and sets `mda_source` to something that cannot be read as
"real walk-forward weights were applied to real features" — e.g. reuse the
existing `"error"` literal from `:265` rather than inventing a third vocabulary.
Note `:210` is a pre-existing instance of the same over-claim (ERROR + empty
features, yet `mda_source: "backtest"`); fixing `:210` is in scope because it is
the same contract violation on the same field.

---

## B8. BLAST RADIUS — MEASURED, and it is worse than the "market hours only" hypothesis

### The measurement (run 2026-07-25 17:50 UTC / 13:50 EDT — a **Saturday**, markets CLOSED)

```
AAPL sector signal= NEUTRAL nonfinite= 31
   summary: AAPL (Technology/Consumer Electronics). 3M return: +nan% vs sector +nan% vs S&P +nan%. Signal: NEUTRAL.
AAPL quant signal= NEUTRAL score= nan mda_source= backtest nonfinite= 17
MSFT sector signal= NEUTRAL nonfinite= 31
MSFT quant signal= NEUTRAL score= nan mda_source= backtest nonfinite= 17
NVDA sector signal= NEUTRAL nonfinite= 31
NVDA quant signal= NEUTRAL score= nan mda_source= backtest nonfinite= 17
```

3 of 3 tickers, 100%. The step body's live capture is reproduced exactly.

### Root cause of the NaN — measured, and the forming-session hypothesis is WRONG

```
yfinance 1.2.0  pandas 3.0.1
yf.Ticker('AAPL').history(period='3mo')  -> 62 rows
                                 Open   High   ...   Close      Volume
2026-07-23 00:00:00-04:00  321.730011  323.299988  ...  321.660004  40840800
2026-07-24 00:00:00-04:00         NaN         NaN  ...         NaN    47402209
NaN count per column: Open 1, High 1, Low 1, Close 1, Volume 0
```

**Exactly one bad row: the most recent trading day (Friday 2026-07-24), with NaN
OHLC but a REAL Volume of 47,402,209.** Measured on a Saturday, i.e. AFTER that
session closed. So this is **not** a transient forming-session artifact that
disappears after the close — it is a persistent yfinance 1.2.0 defect on the
latest bar, and the real volume is precisely why yfinance's own `keepna` mask
(`.all(axis=1)`) does not drop the row.

**Answers to the caller's B8 question:**
- **How many live tickers are affected?** Effectively **all of them, all the
  time.** Not market-hours-gated. Every `_compute_return` uses `Close.iloc[-1]`
  (`sector_analysis.py:34`) and every quant feature uses `closes[-1]`
  (`quant_model.py:65`), so the single bad tail row poisons every derived value
  for every ticker on every cycle.
- **Would flipping to ERROR be a total analysis outage?** **No** — see the B7
  arithmetic: `dq` goes 1.0 → 0.91 (or 0.83 with quant_model tracked), both far
  above the 0.5 `data_quality_min`. Debate, risk assessment and synthesis all
  still run. The pipeline loses two of twelve enrichment inputs — the two that
  are currently contributing **pure fiction**.
- **Would the retry loop hammer the API?** **Yes, materially — this is the one
  real cost of the fix, and it must be in the contract.** `retry_critical_gaps`
  is bounded at `max_retries=2` per source per ticker (`orchestrator.py:2041`),
  but ONE `get_sector_analysis` call makes ~14 `yf.Ticker(...).history()`
  round-trips (stock ×4 periods + matched sector ETF ×4 + SPY ×4 + all 11 sector
  ETFs for the rotation chart at `:83-90`) plus 1-6 `.info` calls. Because the
  failure is **deterministic**, all 2 retries are guaranteed to fail, so the fix
  buys ~2× that cost per ticker per cycle for nothing. At ~20-30 tickers a cycle
  that is on the order of 500-900 extra yfinance requests per cycle — a real
  HTTP-429 risk on a free endpoint.

### The consequence the executor must not miss

The guards alone convert a **silent wrong answer** into a **loud, permanent,
and somewhat expensive outage of two enrichment sources**. That is the correct
trade (a known-missing input beats a fabricated one), but it is not free, and it
means the FIX WILL LOOK LIKE A REGRESSION on the Signals page and in the
info-gap step text until the underlying bad bar is also dealt with.

**The bad-bar repair (dropping the all-NaN-OHLC tail row) is DELIBERATELY OUT OF
SCOPE for 80.27** and should be queued as its own research-gated step. Rationale
— and this is the load-bearing argument: dropping the row **restores real
values**, which can turn today's fabricated `NEUTRAL` into a genuine
`DOUBLE_TAILWIND` / `BULLISH`. That is a change in the **less** conservative
direction and could open a position that would not otherwise open. The caller's
hard stop applies. The precedent for the repair already exists in this repo —
`screener.py:169-170` does `ticker_data["Close"].dropna()` / `["Volume"].dropna()`,
which is exactly why the screener funnel is not NaN-poisoned today — but it
belongs in its own step with its own before/after trade-diff evidence.

---

## C. RISK / DO-NO-HARM

### C1. Could ERROR-instead-of-NEUTRAL open or close a position that would not otherwise move?

Trace: the enrichment payload never reaches a trade decision as a number. It
reaches (a) the per-source LLM enrichment agent as a JSON blob, (b) the debate
prompt as `{signal, summary, analysis}` (`:2167`, `:2171`), (c) the synthesis
prompt as `{signal, summary}` (`:2291`). The trade action is
`synthesis["recommendation"]["action"]` read at `autonomous_loop.py:1915`.

So the change is: two of eleven/twelve debate inputs go from a **fabricated
`NEUTRAL` + a `+nan%` summary** to an **`ERROR` + an explicit failure summary**,
one of which (`sector`) additionally raises a `source_diversity` bias flag once a
second ERROR joins it (`bias_detector.py:239-246`, "Consider ... adjusting
confidence downward").

**Could that flip a HOLD to a BUY?** Only via the LLM's judgement, and only in
the direction of *less* evidence. Removing a fake `NEUTRAL` cannot manufacture
bullish evidence; the debate loses a (fictional) data point and gains an explicit
"this source failed" marker plus, at ≥2 errors, a bias flag telling the
synthesiser to lower confidence. Every documented gradient here points at
**fewer and more-gated trades**. It is not a *proof* — an LLM is in the loop —
which is exactly why the dark-launch flag below is the right shipping shape.

**Note the one asymmetry worth stating honestly:** `monte_carlo` (L3) currently
fabricates `EXTREME_RISK` on NaN. If the executor also guards L3, that source
goes `EXTREME_RISK` → `ERROR`, i.e. it *removes* an alarming input. That is the
one place in this step where the change is directionally *less* conservative.
**Recommendation: leave L3 out of 80.27's behaviour change** (document it, queue
it) — or, if included, only alongside the `dq`-drop that its MISSING status
produces, and call it out explicitly in the contract. Do not let it ride in
silently.

### C2. Could it crash or stall the autonomous loop?

**No.** Three independent reasons, all verified:
- Zero field-level dereferences of these payloads exist outside the tools (B7).
- `_assess_source_status` already has an `ERROR` branch (`info_gap.py:47`) and
  the retry helper wraps every call in `try/except` (`:180-183`).
- The whole `_run_single_analysis` body is inside a `try/except Exception`
  (`autonomous_loop.py:1944`) that falls back to the lite path, and then to a
  `_degraded` marker (`:1975-1990`) that is converted to `None` and **never
  enters `decide_trades`**.

Stall risk is likewise low: retries are bounded (`max_retries=2`) and each
`get_sector_analysis` is already wrapped by `_safe` in an `asyncio.gather`. The
one measurable cost is latency/quota from the guaranteed-failing retries (B8).

### C3. Is ERROR ever treated as MORE actionable than NEUTRAL?

Searched every ERROR consumer (the B7 table). **No.** Every one of them treats
ERROR as *less* usable: excluded from session memory (`:2003`), not merged on
retry (`:2046`), stripped from compacted debate prompts (`:2181`), counted into a
confidence-lowering bias flag (`bias_detector.py:231-246`). There is no code path
where `ERROR` unlocks an action that `NEUTRAL` does not.

### C4. Blast radius — market hours vs after close

**Identical.** Measured on a Saturday with markets closed and the defect was
fully present (B8). There is no quiet window in which to ship this "safely by
timing"; the flag is the only real control.

### C5. Feature-flag / dark-launch pattern — THE RECOMMENDED SHIPPING SHAPE

**Yes, the repo has a well-established pattern, and this is very likely the
right shape for a P0 that changes trading behaviour.** The idiom, with live
precedents:

- Declaration: `backend/config/settings.py` — `bool = Field(False, description="phase-XX.Y: ... Default OFF -> byte-identical ...")`
  - `settings.py:37` `sign_safe_overlays` (phase-69.3, ranking-behaviour change)
  - `settings.py:46` `paper_data_integrity_enabled` (phase-50.5)
  - `settings.py:198` `paper_synthesis_integrity_enabled` (phase-61.2)
  - `settings.py:350` `paper_swap_churn_fix_enabled`
  - `settings.py:449` `momentum_52wh_tilt_enabled` (phase-52.2)
  - `settings.py:459` `paper_atomic_swap_enabled` (phase-70.3)
- Read site: `getattr(settings, "<flag>", False)` — e.g.
  `autonomous_loop.py:1899` and `:1975`. The `getattr`-with-default form is the
  house idiom and survives an un-migrated settings object.
- Semantics every one of those flags documents: **OFF = byte-identical legacy**;
  ON = the new behaviour; enablement is operator-gated.

**Recommendation:** ship 80.27 behind a single new flag, e.g.
`tools_nonfinite_fail_safe_enabled: bool = Field(False, ...)`, read via
`getattr` inside `sector_analysis.get_sector_analysis` and
`quant_model.get_quant_model_signal`. OFF reproduces today's payload byte-for-
byte (so the regression test can assert byte-identity, matching 52.2/70.3
precedent); ON returns the documented `ERROR`.

Two caveats the executor must handle:
1. `backend/tools/*` currently import **no** settings module — check the import
   direction before adding one (`sector_analysis.py` imports only `logging` and
   `yfinance`). A local `from backend.config.settings import get_settings`
   inside the function is the low-risk form; a module-level import risks a
   circular-import regression.
2. `_assess_source_status`'s side of the change (criterion 1 — numeric scanning
   in `info_gap.py`) is **independent of the tools** and is what the immutable
   verification command actually exercises. That command passes a payload with
   `signal: 'NEUTRAL'` and a `+nan%` summary and expects a NOT-SUFFICIENT
   status — so the info_gap guard must NOT be behind the tools' flag, or the
   verification command fails with the flag OFF. **Gate the tool behaviour;
   do not gate the info_gap detector.** Making `_assess_source_status` scan for
   non-finite numerics is a pure tightening of a *detector* (it can only move
   sources toward MISSING → more gating), so it is safe to ship ON.

---

# ADAPTIVE COVERAGE — audit rounds (audit_class = true, K = 2)

## Round 1 — `backend/tools/` signal ladders
Method: `grep -n 'signal *= *"' backend/tools/*.py` + the two `return "<BUCKET>"`
ladders; read every hit.
**New in-set findings: 9 (L1-L9). NOT dry.**

## Round 2 — `backend/agents/` comparison sites
Method: `grep -rnE "if .*[a-z_]* *[<>]=? *[0-9-]" backend/agents/`; read every
plausible hit.
**New in-set findings: 2. NOT dry.**

| # | Site | Float input | Non-matching path yields | Severity |
|---|------|-------------|--------------------------|----------|
| L10 | `conflict_detector.py:167-170` | `pe_trailing` / `pe_forward` (yfinance `valuation`) | conflict NOT appended → "no valuation conflict" | MEDIUM |
| L11 | `orchestrator.py:347-374` (`_compute_sector_concentration`) | `mv = float(pos.get("market_value") or 0.0)` | **`concentration_warning: False`** | **HIGH — risk control** |

L11 walkthrough (this one is nasty, and it is in the *portfolio* path, not the
enrichment path): `nan or 0.0` → NaN is truthy → `mv = nan`; `if mv <= 0:
continue` (`:354`) is False so the bad position is **not** skipped;
`total_value += nan` → NaN; `if total_value <= 0` (`:359`) is False so the
empty-portfolio early return is skipped; every `by_sector` pct becomes NaN
(`:368`); `max(by_sector.items(), key=lambda kv: kv[1])` (`:369`) returns the
**first** item (all comparisons False, Timsort no-op — same mechanism as
`quant_model.py:234`); and finally `bool(max_sector[1] >= threshold_pct)`
(`:374`) is **False**. A single NaN `market_value` silently disables the sector
concentration warning for the whole portfolio.

## Round 3 — `abs()`/`max()`/`min()`/`sorted()` keys and variable-vs-variable comparisons
Method: two greps — `abs(...) [<>]` / `key=` over float fields, and
`if <var> [<>]=? <var>` in `backend/tools/`.
**New in-set findings: 3. NOT dry.**

| # | Site | Float input | Non-matching path yields | Severity |
|---|------|-------------|--------------------------|----------|
| L12 | `anomaly_detector.py:31` `if z_score is not None and abs(z_score) >= _Z_MODERATE` | every computed z-score | anomaly **not appended** — the central suppression point feeding L4 | **HIGH** |
| L13 | `anomaly_detector.py:188` `if abs(pe_gap) > 20` | `pe_gap` from yfinance P/E | valuation anomaly not appended | MEDIUM |
| L14 | `fred_data.py:78` `trend = "rising" if current > prev else "falling" if current < prev else "stable"` | FRED observations | **`"stable"`** — a fabricated macro trend | MEDIUM |

L12 is the important one: `_append_if_anomalous` is the *single* gate every
anomaly passes through, and `abs(nan) >= 2.0` is False, so ALL anomaly detection
degrades silently to "No significant statistical anomalies detected."
`anomaly` is HIGH criticality in `_SOURCE_CRITICALITY`.

EXCLUDED in this round, with reasons (auditable, not silent):
- `screener.py:179` `if current_price < min_price or avg_vol < min_avg_volume` —
  this WOULD be a severe one (a NaN price would slip past the price/liquidity
  screen), but `screener.py:169-170` calls `.dropna()` on Close and Volume
  first, so the operands are finite by construction. **This is the repo's
  existing correct precedent and the reason the screener funnel is not
  NaN-poisoned today.**
- `price_quality.py:145` — inside the validator itself; documented as fail-open.
- `earnings_tone.py:137/:140`, `anomaly_detector.py:237`, `sec_insider.py:198` —
  int counts / date strings, fail clause (2).

## Round 4 — ternary verdicts, unfiltered comparison sweep of `backend/tools/`, float-named vars in `backend/agents/`
Method: `grep -rncE "[<>]=?" backend/tools/` (every file counted, then every
file with a non-zero count re-read for sites not already classified); ternary
pattern `= "X" if ... [<>]`; float-named-variable ladders in `backend/agents/`
including `mcp_servers/`.
**New in-set findings: 4. NOT dry.**

| # | Site | Float input | Non-matching path yields | Severity |
|---|------|-------------|--------------------------|----------|
| L15 | `mcp_servers/signals_server.py:926` `if ticker_pct > max_per_ticker_pct` | position pct of NAV | **per-ticker position limit NOT enforced** | **HIGH — risk control** |
| L16 | `mcp_servers/signals_server.py:935` `if total_pct > max_total_pct` | total exposure pct | **total exposure limit NOT enforced** | **HIGH — risk control** |
| L17 | `mcp_servers/signals_server.py:1285-1287` `if drawdown_pct <= kill_pct: ... elif drawdown_pct <= derisk_pct:` | drawdown pct | **kill-switch / de-risk NOT armed** | **HIGH — risk control** |
| L18 | `bias_detector.py:119, :128` `score >= 7.5` / `score >= 8.0` | synthesis weighted score | tech-bias / large-cap-bias flag not raised | LOW |

L15-L17 are the same class as L11: **a NaN silently disables a risk control.**
They live on the Layer-2 MCP surface rather than the Layer-1 enrichment path, so
they are almost certainly OUT OF SCOPE for 80.27 — but criterion 4 asks for the
enumeration, and leaving a kill-switch bypass undocumented would be the exact
failure this step exists to punish. **Recommend queueing L11 and L15-L17 as
their own risk-control step** (they are portfolio/risk-path, not
enrichment-path, and each needs its own do-no-harm trace).

## Round 5 — `.get(key, numeric-literal)` shape + comprehension/`while` comparisons + `key=` sorts
Method actually run (3 greps):
(a) `grep -rnE '\.get\(["\'][a-z_0-9]+["\'], *-?[0-9]' backend/tools/ backend/agents/`
    — hunts the exact L1/L3 dead-default shape;
(b) `grep -rnE '(for .* if .*[<>]|while .*[<>]|\[.*if .*[<>].*\])' backend/tools/`;
(c) `grep -rnE 'sorted\(|max\(|min\(' backend/agents/ | grep -E 'key=|reverse='`.

Hits, all classified:
- `anomaly_detector.py:230` `if a.get("z_score", 0) < -_Z_MODERATE or ...` — a
  NaN z-score counts as an *opportunity* rather than a *risk* anomaly. **De-dup:
  this is a fourth site inside the already-enumerated L4/L12/L13
  anomaly_detector cluster, not a new ladder.**
- `screener.py:611-614` `float(weights.get("price", 0.35))` etc. — config-sourced
  weights, no verdict ladder. EXCLUDED (clause 3+4).
- `sec_insider.py:256/261/262`, `options_flow.py:49/65`,
  `multi_agent_orchestrator.py:451-763` — formatting and integer token
  accumulators. EXCLUDED (clause 2).
- (b) hits are all `len(...)` slice guards. (c) hits are the already-recorded
  `orchestrator.py:369` (L11) plus `st_mtime` / model-weight sorts.

**New in-set LADDERS: 0 → DRY ROUND 1.**
(De-dup rule applied: a new NaN *entry point* inside a cluster already in the
table counts as refinement, not a new finding. Both refinements found in this
round are recorded above.)

## Round 6 — unfiltered comparison sweep of every remaining tool
Method actually run: `grep -rnE "[<>]=?[^=]"` with NO exclusion filter over
`nlp_sentiment.py`, `yfinance_tool.py`, `monte_carlo.py`, `quant_model.py`,
`fred_data.py`, `alt_data.py`, `social_sentiment.py`, `options_flow.py`,
`sector_analysis.py`; every hit diffed against the L1-L18 table and each unseen
line read.

Hits requiring a decision:
- `nlp_sentiment.py:276-281` (`_keyword_score` fallback ladder) — READ at
  `:258-281`: `score = (up - down) / total` where `up`/`down` are integer
  keyword counts and `total == 0` is explicitly guarded at `:271`. **EXCLUDED
  (clause 2/3): int counts only.** This is the one hit that looked like a new
  ladder and is not.
- `yfinance_tool.py:30` `if fwd_pe and eg and eg > 0` — a NaN `eg` is truthy and
  fails `> 0`, so PEG is silently not computed. EXCLUDED on **clause 4**: the
  non-matching path omits a field, it does not emit a verdict.
- `quant_model.py:103` `(ocf*4)/market_cap if market_cap > 0 else 0.0` — fifth
  member of the quant_model falsy-guard cluster. **De-dup** (added to the
  bonus-findings table).
- `monte_carlo.py:75-81` — `np.percentile` / `np.mean(returns_pct >= 20)` over a
  NaN array produce the NaN that L3 then classifies. **De-dup**: NaN entry
  points for L3.
- Everything else was already in the table or is a `len()`/`empty` guard.

**New in-set LADDERS: 0 → DRY ROUND 2. `coverage.dry = true` (K=2 satisfied).**

### Final enumerated set: 18 in-set ladders, 9 auditable exclusions

**In 80.27's scope (the two named in criterion 4 + their direct enablers):**
L1, L2, and the `quant_model.py` non-ladder guards (`:47-48`, `:84-85`, `:167`,
`:234`) plus `sector_analysis.py:55/70/78/87`.

**Same tool family, same cycle, cheap to include — executor's call, but each
needs its own line in the contract:** L4/L12/L13 (anomaly_detector — **note the
open step 80.31 touches this file; coordinate**), L5/L14 (fred_data),
L6 (nlp_sentiment), L7/L8 (social_sentiment), L9 (alt_data).

**Explicitly recommend DEFERRING to queued steps:**
- **L3 (monte_carlo)** — the ONE directionally-less-conservative change in the
  set (removes a fabricated `EXTREME_RISK`). See §C1.
- **L11, L15, L16, L17** — risk-control suppression on the portfolio/MCP path.
  Higher stakes, different blast radius, deserves its own research gate.
- **L10, L18** — low severity, LLM-score-derived.

---

## B9. Test conventions + deterministic non-finite fixtures

`backend/tests/` — 157 test modules; `conftest.py` installs a BQ-write guard at
**import** time (its docstring records that a time-based auto-flush leaked 106
fixture rows into the real `pyfinagent_data.llm_call_log` between 2026-05-19 and
2026-07-07). Naming convention: `test_phase_<X>_<Y>_<slug>.py`.

**The direct precedent to copy is
`backend/tests/test_phase_80_1_signals_nan_serialisation.py`** — same defect,
same tools, written yesterday-equivalent. Reusable patterns:

- `_install_fake_tools(monkeypatch, sector_1mo=float("nan"))` (`:133-214`) —
  monkeypatches every tool AND `yf.Ticker` with a `_FakeTicker` whose
  `history()` returns an empty `pd.DataFrame`. **Zero network.** For 80.27 the
  fixture must go one level deeper: patch `yf.Ticker` so `history()` returns a
  real DataFrame **with an all-NaN OHLC tail row and a real Volume** — i.e.
  reproduce the measured defect (B8) rather than an empty frame.
- `:190-195` — the fixture asserts `hasattr(mod, fn)` before every
  `monkeypatch.setattr`, so a rename fails loudly instead of silently leaving
  the network-calling tool in place.
- `:251-257` `test_control_finite_payload_is_unchanged` — the **green control**
  that makes a red mutation run attributable to the mutation.
- `:260-279` `test_the_fixture_default_is_actually_non_finite` — reads the
  default off `inspect.signature(_install_fake_tools)`. Its docstring records
  that the FIRST version asserted `not math.isfinite(float("nan"))`, a **library
  fact** that passed under the very fixture mutation it claimed to guard. That
  is exactly the shape criterion 5 is guarding against, and Q/A caught it, not
  the author.

**Criterion 5 (mutation-test each guard) — concrete recipe:**
1. Guard test: feed `float('nan')` through the fixture; assert
   `result["signal"] == "ERROR"` (**not** merely "no exception" and **not**
   merely "not NEUTRAL" — assert the exact documented value).
2. Assert the summary string contains no `nan` substring, and that
   `json.dumps(payload, allow_nan=False)` does not raise (criterion 3, both
   leaks).
3. Mutation: delete the `math.isfinite` guard → the test must go RED. Also
   mutate the **fixture** (flip the NaN default to a finite number) → the test
   must go RED, per `feedback_mutation_test_guards_and_fixtures`.
4. Green control: finite inputs → the pre-fix payload byte-for-byte (this also
   discharges the flag-OFF byte-identity claim).

**Do not write a source-scan test** (e.g. grepping for `math.isfinite` in the
tool file). Phase-75 shipped three of those and they are recorded as guards that
cannot fail.

---

# ADDENDUM (post-handoff, 2026-07-25 ~18:05 UTC) — Main's independent B8 measurement + four new measurements

Main measured B8 independently and asked me to fold it in rather than re-derive.
**Our measurements agree.** My §B8 above already refuted the forming-session
premise before this message arrived (I measured 3/3 tickers on the same closed
Saturday, and the identical AAPL row: `2026-07-24`, NaN OHLC, Volume
47,402,209). Main's 6-ticker table extends it. Nothing above needed correcting;
the four items below are NEW.

## AD-1. The corrected mechanism, stated plainly (agreed by two independent measurements)

**Exactly one bar — the most recent COMPLETED session — is persistently
malformed: NaN Open/High/Low/Close with a real, non-zero Volume.** It does not
self-heal at the close (measured ~22h after Friday's close, on a Saturday). It
is not a placeholder for a session in progress. yfinance 1.2.0 / pandas 3.0.1.
The real Volume is the mechanism: yfinance's own `keepna` drop-mask is
`.all(axis=1)`, so a row with one real column survives.

**Main's SPY/XLK observation adds a genuinely new inference I had not stated:**
`sector_analysis.py:74` fetches SPY for **every** ticker and `:61-71` fetches
the matched sector ETF, so even a hypothetically clean symbol is poisoned
through its benchmark. `sector_analysis` cannot be partially healthy.
(`quant_model` is different — it reads only the subject ticker's own history at
`:59`, so it is poisoned per-ticker rather than via a shared benchmark.)

## AD-2. **The outage is NOT total — it is confined to 2 of 12 sources. MEASURED.**

Main wrote "a fail-safe fix flips **every ticker** to ERROR right now … that is
a **total analysis outage**". The first half is right; **the second half is not,
and the distinction is load-bearing for the operator.** Every *ticker* is
affected, but only *two of twelve enrichment sources* are. I measured the other
HIGH-criticality yfinance-backed sources live:

```
monte_carlo  signal=HIGH_RISK            non_finite=0
anomaly      signal=ANOMALY_OPPORTUNITY  non_finite=0
options      signal=NEUTRAL              non_finite=0
```

**Zero non-finite values in all three.** The reason is one line —
`monte_carlo.py:40`: `close = hist["Close"].dropna().values`. Same idiom as
`screener.py:169-170`. Those tools drop the bad bar on read; `sector_analysis`
and `quant_model` do not. **This is a third independent confirmation that
`.dropna()` at the read site is the repo's own working precedent.**

Consequences, computed against the (now in-flight) `_SOURCE_CRITICALITY` with
`quant_model` added, i.e. 12 tracked sources:

| Quantity | Value | Meaning |
|---|---|---|
| `critical_gaps` | `["sector", "quant_model"]` — **2** | both HIGH |
| `recommendation_at_risk` (`>= 3`) | **False** | the 3-gap alarm does NOT trip |
| `data_quality_score` | 10/12 = **0.83** | vs `data_quality_min` 0.5 |
| `low_data_quality` | **False** | **debate + risk assessment still run** |

So the operator-visible effect is: the Signals page shows 2 sources in error,
the info-gap step reports 83% instead of a fabricated 100%, and the pipeline
still produces a recommendation from 10 real sources. That is a **degradation,
not an outage**, and it is the honest reading of the data the system actually
has.

## AD-3. Retry cost — MEASURED, and my earlier estimate was LOW

I instrumented `yf.Ticker` and counted actual calls:

```
get_sector_analysis    -> {'Ticker': 14, 'history': 23, 'info': 1}
get_quant_model_signal -> {'Ticker': 1,  'history': 1,  'info': 1}
```

**23 `history()` calls per `get_sector_analysis`**, not the ~14 I estimated in
§B8. Correcting the arithmetic:

- Per ticker per cycle, the retry loop adds `2 sources × 2 attempts` =
  2 extra `get_sector_analysis` (≈ 2 × 24 ≈ **48** round-trips) + 2 extra
  `get_quant_model_signal` (≈ **4**) ≈ **52 extra yfinance round-trips**.
- At 20-30 tickers: **≈ 1,040-1,560 extra round-trips per cycle** — up from the
  500-900 I estimated. The estimate above is superseded by this measurement.

**Backoff and caps — the actual limits, as asked:**
- `retry_critical_gaps(..., max_retries=2)` — hard cap, 2 attempts per source
  (`orchestrator.py:2041`). Sources are retried **sequentially** (`for key in
  critical_gaps`, `info_gap.py:205`), not concurrently.
- **There is NO backoff.** `grep -n "sleep\|backoff\|jitter" backend/agents/info_gap.py`
  returns nothing. Attempts fire back-to-back.
- **There is no repo-level yfinance rate limiter or `requests_cache` session.**
  The `get_rate_limiter` machinery in `backend/observability` covers finnhub /
  FRED, not yfinance.
- The failure is **deterministic**, so all retries are guaranteed to fail —
  every one of those ~1,000-1,500 calls is pure waste.

**This is the strongest single argument for the default-OFF flag**, and for
sequencing the bad-bar repair BEFORE the operator flips it: once the tail row is
dropped at the read site, `sector`/`quant_model` return real values, nothing
enters `critical_gaps`, and the guards become a never-fired backstop with zero
retry cost.

## AD-4. Restart question — **the autonomous-loop path does NOT need a restart**

Main assumed the flag "will sit dark until the operator acts" because of the
open `phase-79.55` RESTART BLOCKER (confirmed `pending` in
`.claude/masterplan.json`). Measured, that is **only half true**:

- `backend/config/settings.py:619-621` — `get_settings` is `@lru_cache()`d, and
  `:616` sets `model_config = {"env_file": "<repo>/backend/.env", ...}`, so
  pydantic-settings re-reads the **.env file from disk** on every `Settings()`
  construction.
- `backend/services/autonomous_loop.py:1879-1881` does
  `_get_settings_fresh.cache_clear()` then `_get_settings_fresh()` **per ticker
  analysis**. Verified the cache-clear yields a new object:
  `same object after cache_clear: False`, `env_file:
  /Users/ford/.openclaw/workspace/pyfinagent/backend/.env`.

**So a flag flip written to `backend/.env` takes effect on the NEXT ticker
analysis in the live loop, with no restart.** Long-lived consumers that hold the
cached `get_settings()` (the API layer, anything imported at startup) **do**
need a restart, so the Signals page and other read paths would lag behind the
loop until 79.55 clears. Honest caveat: I verified the cache-clear + env_file
wiring, but did not mutate `backend/.env` to prove the end-to-end flip (the
sandbox denies that file and mutating it would touch live config).

## AD-5. Audit round 7 — the in-flight implementation, probed

Main's message arrived alongside an in-flight edit to
`backend/agents/info_gap.py` (quant_model added to `_SOURCE_CRITICALITY`;
`_NAN_TOKEN_RE` + `_has_non_finite` added to `_assess_source_status`). The
immutable verification command now returns **`status: MISSING`** (it returned
`SUFFICIENT` on my pre-fix baseline above). I treated the new code as a new
audit surface rather than assuming it:

- **`_NAN_TOKEN_RE` false-positive probe** — the risk is a regex that matches
  `nan`/`inf` as a substring, which would mark almost every real summary MISSING
  and cause the self-inflicted outage Main is worried about. Probed 7 positives
  and 14 negatives: all 7 positives match (`+nan%`, `-inf`, `Infinity`, `nan,`,
  `(nan)`, `NaN`); **13 of 14 negatives correctly do not match** — including
  `financial`, `finance`, `governance`, `maintenance`, `tenant`,
  `nanotechnology`, `Nanjing`, `information`, `confidence`, `Infineon`,
  `inflation`, `inflows`, `infrastructure`. The single false positive is
  `'nan-o'` (a hyphen is a non-letter, so it is a boundary by construction) —
  contrived, not a shape any tool summary produces. **The regex is sound.**
- **Residual risk on `_has_non_finite`, worth one line in the contract:** it
  scans the *whole* payload, so a legitimately infinite ratio (e.g. a P/E for a
  zero-earnings company) would mark an otherwise-healthy source MISSING. That is
  arguably correct (an infinite P/E is not a usable number) but it is a
  behaviour change beyond NaN, and it should be stated rather than discovered.
- **New in-set ladders found in round 7: 0.** The additions are a *detector* and
  a *criticality table*, not verdict ladders, so the enumeration in §B3 and the
  K=2 dry-round result are unchanged. **`coverage.dry` remains `true` on the
  original 6-round basis; round 7 is an additional non-dry-resetting check of a
  new artifact.**

---

# A. EXTERNAL RESEARCH

## Search-query variants run (three-variant discipline)

- **Current-year frontier:** `2026 silent failure numerical guard data quality
  gate ML pipeline agent "fail closed" abstain missing feature`;
  `LLM numeric hallucination NaN null tokens in JSON prompt 2025 2026 study`
- **Year-less canonical:** `IEEE 754 NaN comparison all relations false
  unordered rationale`; `fail-safe design missing data must not be
  representable as valid value sentinel NaN poison value`; `CERT C secure coding
  NaN comparison floating point rule`; `machine learning abstention reject
  option missing features at inference time finance`
- **Adversarial:** `against abstention rejection classifier harms coverage bias
  listwise deletion missing data worse than imputation critique`

## Read in full (8; ≥5 required — counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/predicates.txt | 2026-07-25 | standards body (IEEE 754 WG background) | WebFetch, full text | *"IEEE 754 floating-point data comparisons can have four possible outcomes - = E Equal, < L Less than, > G Greater than, ? U Unordered"*; and the rationale: a signalling comparison exists because *"a programmer who has not considered the possibility of quiet NaN operands might have programmed incorrect logic, and the invalid operand exception might be the only way to draw attention to the error."* |
| 2 | https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/recommendations/floating-point-flp/flp04-c | 2026-07-25 | official standard (SEI CERT C, FLP04-C) | WebFetch (301 from wiki.sei.cmu.edu, re-fetched) | *"NaN values are particularly problematic because the expression NaN == NaN (for every possible value of NaN) returns false"*; *"Any comparisons made with NaN as one of the arguments returns false, and all arithmetic functions on NaNs simply propagate them through the code."* Mandates `isnan`/`isinf` checks on inputs. Risk: Severity Low, **Likelihood Probable**, P4/L3. |
| 3 | https://swehb.nasa.gov/spaces/SWEHBVC/pages/140640571/Initialization+-+Safe+Mode | 2026-07-25 | official standard (NASA SW Engineering Handbook Ver C) | WebFetch, full page | *"Design flight software to initialize software and hardware to a known, safe, and deliberate state."* Safe mode is characterised by *"initially inactive capabilities by design, with the system defaulting to a conservative, sustainable state rather than attempting full operational status."* |
| 4 | https://ar5iv.labs.arxiv.org/html/2107.11277 | 2026-07-25 | peer-reviewed survey (Hendrickx et al., *Machine Learning with a Reject Option: A Survey*; journal version DOI 10.1007/s10994-024-06534-x) | WebFetch via ar5iv (PDF was unreadable — followed the research-gate PDF chain) | §2.1 defines a model with rejection `m: X → Y ∪ {®}`, emitting ® when *"the rejector r determines that the predictor is at a heightened risk of making a misprediction"*. §2.2.2 novelty rejection covers inputs where *"parts of the feature space are not represented in the data"*. §3.3 gives the cost ordering **Cc < Cr < Ce** — the cost of rejecting sits strictly between a correct prediction and an error. |
| 5 | https://docs.python.org/3/library/json.html | 2026-07-25 | official docs (CPython) | WebFetch, full page | `allow_nan` is *"`True` (the default)"*; *"The RFC does not permit the representation of infinite or NaN number values. Despite that, by default, this module accepts and outputs `Infinity`, `-Infinity`, and `NaN` as if they were valid JSON number literal values"*; *"This behavior is not JSON specification compliant."* |
| 6 | https://www.rfc-editor.org/rfc/rfc8259 | 2026-07-25 | official standard (IETF, JSON) | WebFetch, §6 | *"Numeric values that cannot be represented in the grammar below (such as Infinity and NaN) are not permitted."* |
| 7 | https://medium.com/towards-data-engineering/the-silent-failures-your-data-tests-arent-catching-09a56ecc0421 | 2026-07-25 | practitioner (published **2026-07-12**) | WebFetch, full article | Three-tier taxonomy: hard failures / test failures / **silent failures** — *"nothing crashes, no obvious rule breaks, and the number is still wrong"*; *"a green pipeline and a correct pipeline are not the same thing."* Conventional `not_null` tests passed while *"every downstream number was quietly, plausibly wrong."* |
| 8 | https://tompepinsky.com/wp-content/uploads/2018/10/pa2018.pdf | 2026-07-25 | peer-reviewed (Pepinsky, *Political Analysis* 26:480-488, 2018) — **[ADVERSARIAL]** | WebFetch, full PDF | Challenges the default preference for imputation: under **MNAR**, *"multiple imputation yields results that are frequently more biased, less efficient, and with worse coverage than listwise deletion"* — but symmetrically warns that deletion *"regularly removes a large proportion of the sample, thereby leading to loss of the statistical power"*, and recommends contextual choice + sensitivity analysis rather than an automatic rule. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://wiki.sei.cmu.edu/confluence/display/c/FLP37-C.+Do+not+use+object+representations+to+compare+floating-point+values | standard | adjacent rule (bitwise comparison), not this defect |
| https://cmu-sei.github.io/secure-coding-standards/sei-cert-oracle-coding-standard-for-java/rules/numeric-types-and-operations-num/num07-j/ | standard | Java analogue of #2; same content |
| https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html | book | sentinel-vs-mask background; superseded by #4 for the decision |
| https://pandas.pydata.org/docs/user_guide/missing_data.html | official docs | pandas NA semantics; not the decision point |
| https://swehb.nasa.gov/spaces/SWEHBVB/pages/32604591/SWE-134+-+Safety+Critical+Software+Requirements | standard | overlaps #3 |
| https://swehb.nasa.gov/spaces/SWEHBVC/pages/140640555/Coding+Standards | standard | pointer to JPL Power-of-10 |
| https://www.perforce.com/blog/kw/NASA-rules-for-developing-safety-critical-code | vendor blog | secondary summary of Power-of-10 |
| https://sdtimes.com/nasas-10-rules-developing-safety-critical-code/ | trade press | secondary |
| https://dl.acm.org/doi/abs/10.1007/s10994-024-06534-x | peer-reviewed | **journal version of #4 (2024)** — paywalled; ar5iv preprint read instead |
| https://arxiv.org/html/2510.19672v1 | preprint (2025) | *Policy Learning with Abstention* — abstention in policy learning; corroborates #4 |
| https://arxiv.org/pdf/2306.02421 | preprint | *Auto-Validate by-History* — auto-generated DQ constraints |
| https://arxiv.org/pdf/2209.07574 | preprint | reject option in microcredit; finance-adjacent |
| https://medium.com/@jooramos_37651/catching-silent-failures-in-data-pipelines-with-forecasting-metadata-and-an-llm-d316e1666bb6 | practitioner (2026-06) | recency corroboration for #7 |
| https://mlflow.org/articles/mlops-pipeline-automation-best-practices-in-2026/ | vendor docs (2026) | "hard gates at multiple points" |
| https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models | vendor blog (2026) | LLM hallucination survey; no numeric-token specifics |
| https://arxiv.org/html/2603.08274v1 | preprint (2026) | 172B-token doc-QA hallucination study; not numeric-token specific |
| https://arxiv.org/pdf/2504.12691 | preprint (2025) | subsequence-association account of hallucination |
| https://www.cambridge.org/core/journals/political-analysis/article/note-on-listwise-deletion-versus-multiple-imputation/39DE56539189423F6C985B3B9EBF7E56 | peer-reviewed | publisher page for #8 |
| https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9825894/ | peer-reviewed | imputation test in bioarchaeology; cross-domain, weaker fit |
| https://oliverphillips91.co.uk/sentinel-value/ | blog | sentinel-value guide: *"the sentinel value must be impossible to appear in valid data"* |
| https://everything.explained.today/NaN/ | reference | NaN background |
| https://developerfacts.com/answer/1565164-what-is-the-rationale-for-all-comparisons-returning-false-for-ieee754-nan-values | community | rationale Q&A; superseded by #1 |
| https://news.ycombinator.com/item?id=43702956 | community | discussion thread |
| https://github.com/elm/core/issues/1050 | community | language-level NaN comparison bug report |
| https://reintech.io/blog/handling-missing-data-nan-values-numpy | blog | numpy NaN handling |
| https://apxml.com/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/missing-data-strategies | course | imputation-vs-deletion overview |
| https://bostoninstituteofanalytics.org/blog/the-complete-guide-to-building-python-data-quality-gates/ | blog | DQ-gate patterns |
| https://kodekloud.com/blog/ci-cd-for-machine-learning/ | vendor blog (2026) | ML gating practice |
| https://www.geeksforgeeks.org/machine-learning/the-reject-option-pattern-recognition-and-machine-learning/ | community | reject-option primer |

**URLs collected: 38 unique (8 read in full + 30 snippet-only).**

## Recency scan (2024-2026) — PERFORMED

Searched explicitly for 2024-2026 work on (a) silent numerical failures in
production ML/data pipelines, (b) abstention/reject-option, (c) LLM behaviour on
malformed numeric tokens in prompts.

**Result: 2 new findings that COMPLEMENT (do not supersede) the canonical
sources, and 1 honest negative.**

1. **The reject-option survey was formally published in 2024** (*Machine
   Learning*, DOI 10.1007/s10994-024-06534-x) — the 2021 preprint I read in full
   is the same work, now peer-reviewed. So the abstention citation is current,
   not stale.
2. **Silent-failure framing is actively current (July 2026).** Source #7
   (published 2026-07-12, thirteen days before this brief) independently names
   the exact failure mode this step is about — a pipeline that is green and
   wrong — and its central claim, *"a green pipeline and a correct pipeline are
   not the same thing"*, is a direct restatement of what
   `data_quality_score = 1.0` over 31 NaNs means here. 2026 MLOps guidance
   converges on "hard gates at multiple points" rather than downstream
   monitoring.
3. **Honest negative — the literature on LLMs fed literal `NaN` tokens in a
   JSON prompt block is thin to absent.** I ran a targeted 2025-2026 search and
   found only general hallucination work (Mu-SHROOM/SemEval-2025, CCHall/ACL-
   2025, HalluLens, the 172B-token doc-QA study) — none of which isolates
   malformed numerics as an input condition. The nearest hard fact is the
   *specification* one (RFC 8259 §6 + the CPython docs): the token is invalid
   JSON, so any consumer that strictly parses it fails, and any consumer that
   tolerantly parses it is guessing. **Recommendation: do not cite a study for
   criterion 3; cite the specification.** The 2025 OpenAI result that training
   objectives *"reward confident guessing over calibrated uncertainty"* is the
   closest mechanistic argument for why a model handed `+nan%` will produce a
   confident narrative rather than flagging the input — but it is an inference,
   not a measurement, and should be labelled as such in the contract.

## Consensus vs debate

**Consensus (strong, cross-domain):**
- Comparisons against NaN are all false; this must be handled by an explicit
  `isnan`/`isfinite` check, not by relying on control flow (#1, #2).
- A system must initialise to the *failure/inactive* state, not to an
  operational one (#3). This is precisely the argument for criterion 2's
  "Initialising signal to NEUTRAL before a comparison ladder is itself the bug".
- Abstention is a first-class, principled output when input quality is
  degraded, and its cost sits strictly between a correct answer and a wrong one
  — **Cc < Cr < Ce** (#4). Formal backing for "return ERROR, don't guess".
- `NaN` is not valid JSON and Python emits it anyway by default (#5, #6).

**Debate / genuine tension (the adversarial finding):**
Source #8 is the counterweight, and it is a real one. Dropping incomplete
observations is *not* universally safe: it costs statistical power, and which of
deletion/imputation is less biased depends on the unobservable missingness
mechanism. Applied here that says: **abstaining on every ticker every cycle is
not free** — it is the B8 cost, and it argues against reflexively extending
80.27's guards to every ladder in the enumeration at once.

**How the tension resolves for THIS step:** #8's objection is about *estimation*
under missingness (where deletion trades bias for variance). 80.27 is not
estimating anything from the NaN — it is deciding whether to emit a **verdict**
that the system had no basis for. #4's cost ordering settles that case:
`Cr < Ce`. The correct reading of #8 is not "don't abstain" but **"abstention
has a real cost, so scope it and measure it"** — which is exactly why this brief
recommends the dark-launch flag and a scoped ladder list rather than a
codebase-wide sweep.

## Pitfalls (from the literature, mapped to this codebase)

1. **Sentinel collision.** *"The sentinel value must be impossible to appear in
   valid data"* — `0` is a perfectly valid 3-month return, which is why
   `sector_returns.get("3mo", 0)` was a bad default even before the NaN bug.
   Do not "fix" this by substituting `0.0`. `backend/api/_json_safe.py:59-69`
   already documents this and names
   `backend/slack_bot/formatters.py:663` as the anti-precedent.
2. **Testing for "no crash".** #7's whole thesis. Criterion 5 encodes it.
3. **Guarding the denominator but not the numerator** — `quant_model.py:167`.
4. **Truthiness as a finiteness check** — `bool(nan) is True` defeats
   `if sma50 else`, `if pe_trailing and ...`, `if not score`, and
   `float(x or 0.0)`. Five live instances found (B3 bonus table + L10 + L11).
5. **Sorting by a NaN key** produces a stable no-op that *looks* like a ranking
   (`quant_model.py:234`, `orchestrator.py:369`).
6. **Abstention is not free** (#8) — scope it, flag it, measure it.

---

# APPLICATION TO PYFINAGENT — recommended shape for 80.27

Mapping the external findings onto the internal anchors:

| Criterion | Anchor(s) | Recommended action | Source backing |
|-----------|-----------|--------------------|----------------|
| 1 | `info_gap.py:43-56`, `:19-31`, `:81`, `:98` | Add a non-finite scan to `_assess_source_status` returning **`MISSING`** (the only status that reaches `critical_gaps`). Ship it **un-flagged** — the immutable verification command exercises it directly. Add `"quant_model": "HIGH"` to `_SOURCE_CRITICALITY` or criterion 1 is unmet for quant_model | #4 (rejector), #7 |
| 2 | `sector_analysis.py:132-153`, `quant_model.py:167-181` | `math.isfinite` guard BEFORE the ladder; return the existing documented `signal: "ERROR"` shape (`sector_analysis.py:182`, `quant_model.py:259-267`). Do not initialise to `NEUTRAL` | #2, #3 |
| 3 | `sector_analysis.py:171-177`, `quant_model.py:238-242`, `prompts.py:537` + `:1111` | The ERROR return closes both the prose and JSON leaks at source. Assert on the rendered summary string AND on `json.dumps(payload, allow_nan=False)` | #5, #6 |
| 4 | 18 enumerated ladders (B3 + rounds 2-4) | In scope: L1, L2 + the 6 non-ladder guards. Everything else enumerated and queued, with L3/L11/L15-L17 explicitly deferred | #1, #2 |
| 5 | new `backend/tests/test_phase_80_27_*.py` | Follow `test_phase_80_1_...py`; mutate the guard AND the fixture | #7, repo memory |
| 6 | `quant_model.py:195`, `:210`, `:250` | `mda_source` must not read `"backtest"` when features are non-finite; reuse the existing `"error"` literal from `:265` | contract |

**Shipping shape (the recommendation):** guards in the two tools behind ONE new
default-OFF settings flag (§C5), the `info_gap` detector change un-flagged, the
bad-bar repair and the risk-control ladders (L11/L15-L17) queued as separate
research-gated steps.

---

## Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **8**
- [x] 10+ unique URLs total — **38**
- [x] Recency scan (2024-2026) performed + reported, including an honest negative
- [x] Full papers/pages read (not abstracts); arXiv PDF chain followed to ar5iv per `.claude/rules/research-gate.md`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the prompt, plus `backend/agents/mcp_servers/` reached by the membership rule
- [x] Contradictions noted — the `[ADVERSARIAL]` source #8 and its resolution
- [x] All claims cited per-claim
- [x] Adaptive coverage: 6 rounds, 2 consecutive dry → `coverage.dry = true`
- [x] Live measurement performed rather than inferred (B8)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 30,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 24,
  "coverage": {
    "audit_class": true,
    "rounds": 6,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Confirmed all five links at real line numbers (sector_analysis initialiser is :132, not :136). Measured live on Saturday 2026-07-25 with markets CLOSED: 3/3 tickers NaN-poisoned, so the forming-session hypothesis is WRONG -- yfinance 1.2.0 leaves the latest bar with NaN OHLC and a REAL Volume, permanently, for every ticker. B7: ERROR is fully handled everywhere -- zero field-level dereferences of these payloads exist outside the tools, so no KeyError is possible; ERROR is excluded from session memory, not merged on retry, stripped from compacted debate prompts, and raises a confidence-lowering bias flag. data_quality_score falls 1.0 -> 0.91, still above the 0.5 data_quality_min, so debate/risk still run: no outage. Cost: the failure is deterministic so both retries are guaranteed to fail, ~500-900 extra yfinance calls/cycle. quant_model is NOT in _SOURCE_CRITICALITY (11 keys, not 12) so criterion 1 needs that key added. Audit enumerated 18 in-set ladders and 9 auditable exclusions; L11/L15-L17 silently disable sector-concentration, position-limit and kill-switch controls. Recommend a default-OFF settings flag on the tool guards, the info_gap detector un-flagged, and the bad-bar repair queued separately (it is directionally LESS conservative).",
  "brief_path": "handoff/current/research_brief_80.27.md",
  "gate_passed": true
}
```


