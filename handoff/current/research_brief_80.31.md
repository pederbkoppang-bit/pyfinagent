# Research Brief — step 80.31 (P2: misaligned price/volume arrays in the anomaly detector)

Tier: **moderate**. `coverage.audit_class = false`. Written 2026-07-25.
All internal claims carry `file:line` anchors read at the working-tree HEAD
(`a88bb0fb` + uncommitted phase-80 work). All measurements were run, not inferred.

## Immutable success criteria (VERBATIM from the caller / masterplan — do not amend)

1. close/high/low/volume arrays are guaranteed equal-length and index-aligned -- assert the lengths are equal in a test using a fixture with a trailing NaN-OHLC/real-Volume row (the exact yfinance shape)
2. The volume z-score is computed over completed sessions only, and the choice about the in-progress session is stated explicitly
3. MUTATION-TEST: restore the per-column dropna and confirm the alignment test FAILS
4. The module still returns 200 / serialises cleanly (do not regress what already works)

**Immutable verification command:**
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import yfinance as yf; h=yf.Ticker('AAPL').history(period='1y'); print('rows',len(h),'close',len(h['Close'].dropna()),'volume',len(h['Volume'].dropna()))"
```

---

## Headline answers (read these first)

**1. The 80.27 line-collision map.** 80.31 must touch **`:57` and `:66-69` ONLY**
(plus the volume block `:90-95` if criterion 2 is implemented there). The three
ladders 80.27 deferred are **`:31`** (`_append_if_anomalous` threshold),
**`:38`** (the severity ternary) and **`:188`** (`if abs(pe_gap) > 20`). None of
them are in 80.31's blast radius; do not edit them. Zero overlap — the two steps
are cleanly separable.

**2. "Does the fix restore real values?" — NO. It DISCARDS one.** This is the
opposite of the direction 80.27 hard-stopped on. Row-wise `dropna` throws away the
malformed session's **real int64 Volume** (AAPL 47,402,209 on 2026-07-24). The
payload is 0-non-finite before the fix and 0-non-finite after it (measured), so
nothing previously NaN-suppressed is un-suppressed. **But the numbers the LLM
debate sees DO change** (measured Δz +0.047 … +0.338 across 6 tickers), so this is
not byte-identical and needs criterion-4 evidence, not a dark flag. See §C.

**3. The caller's correction (a) is CONFIRMED.** Measured 2026-07-25 on a
Saturday: the NaN-OHLC/real-Volume bar is **2026-07-24 (Friday) — a COMPLETED
session**, on 3/3 tickers. It is not the forming session. Criterion 2's wording
("the in-progress session") must be answered on the corrected mechanism.

---

## Search queries run (3-variant discipline)

| Variant | Query |
|---|---|
| year-less canonical | `pandas dropna subset preserve alignment across columns how any` |
| year-less canonical | `volume z-score anomaly detection exclude incomplete current bar trading data` |
| year-less canonical | `overlapping windows rolling z-score bias recent window included in baseline statistics` |
| current-year | `yfinance history last row NaN OHLC real volume 2026` |
| last-2-year window | `"2025" OR "2026" pandas 3.0 copy-on-write .values to_numpy migration gotchas dataframe` |

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html | 2026-07-25 | official doc | WebFetch (full) | `subset` = "Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include"; `how` default `'any'`; `axis` default 0; `ignore_index` default **False** — so the DatetimeIndex is PRESERVED, which is what makes one row-wise drop yield four co-indexed columns |
| https://pandas.pydata.org/docs/whatsnew/v3.0.0.html | 2026-07-25 | official doc | WebFetch (full) | pandas 3.0.0 (2026-01-21). CoW is default and `mode.copy_on_write` "no longer has any impact". `.to_numpy()`/`.values` change ONLY under the opt-in `future.distinguish_nan_and_na` flag and only for **nullable** dtypes. **No change to `dropna` semantics is listed.** |
| https://github.com/ranaroussi/yfinance/issues/2622 | 2026-07-25 | library issue tracker | WebFetch (full) | Missing/blank most-recent-session OHLC starting 2025-11-03, attributed **upstream to Yahoo**, reproduced on Yahoo's own website. "The problematic row contains valid volume data (700,560 shares), indicating it's a **completed trading session** rather than an incomplete intraday formation." Independent corroboration of correction (a). No maintainer workaround. |
| https://optimumsportsperformance.com/blog/rolling-mean-and-sd-not-including-the-most-recent-observation/ | 2026-07-25 | practitioner (named analyst, Patrick Ward PhD) | WebFetch (full) | The canonical statement of criterion 4's defect: "including it in the mean and standard deviation calculation creates artificial look-ahead bias"; the fix is `lag()` around `rollapplyr()` so the scored observation is excluded from its own baseline |
| https://ar5iv.labs.arxiv.org/html/2004.04013 | 2026-07-25 | peer-reviewed preprint (Bias-optimal vol-of-vol estimation) | WebFetch (full, ar5iv per research-gate PDF chain) | **[ADVERSARIAL / qualifying]** overlapping windows are not automatically a defect: "allowing for the overlap of consecutive local windows to pre-estimate the spot variance **may correct for this bias**"; "window overlapping is crucial in order to optimize the relative bias". Qualifies the naive "overlap = bug" reading in criterion 4. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://pandas.pydata.org/docs/dev/user_guide/migration.html | official doc | superseded by the 3.0.0 whatsnew read in full |
| https://pandas.pydata.org/docs/user_guide/copy_on_write.html | official doc | CoW read-only-array gotcha captured via the 2025-26 recency search |
| https://github.com/ranaroussi/yfinance/issues/134 | issue tracker | 2019-era NaN-on-split issue; different mechanism |
| https://github.com/ranaroussi/yfinance/issues/515 | issue tracker | bulk-download NaN tail; different mechanism (download failure), not our shape |
| https://www.tradingview.com/script/WPyvzh8u-Z-Score-Volume-Range-Anomaly-Detector/ | community indicator | practitioner corroboration of the `[1]` offset only |
| https://www.mql5.com/en/articles/23217 | community article | modified-z-score on OHLCV; useful prior art, lower tier |
| https://anomiq.io/blog/zscore-trading/ | community blog | z>=2 ≈ top-5%; threshold folklore, matches `_Z_STRONG` |
| https://rdrr.io/cran/roll/man/roll_scale.html | package doc | R `roll_scale` confirms trailing-window convention |
| https://phofl.github.io/cow-adaptions.html | authoritative blog (pandas core dev) | CoW deep-dive; not load-bearing here |
| https://arxiv.org/pdf/2606.20079 | preprint | ensemble anomaly-detection framework; adjacent, not needed for a P2 |
| https://www.systemoverflow.com/learn/ml-timeseries-forecasting/timeseries-feature-engineering/rolling-statistics-and-window-aggregations | tutorial | rolling-stat leakage framing |
| https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html | official lib doc | `PriceHistory` API surface; no NaN-tail contract documented |

URLs collected: **17**.

## Recency scan (2024-2026)

Searched the last-2-year window explicitly (`"2025" OR "2026" pandas 3.0
copy-on-write .values to_numpy migration gotchas`, plus the 2026-scoped yfinance
query). Result: **two new findings that materially bear on this step**, and one
non-finding.

1. **pandas 3.0.0 shipped 2026-01-21** and CoW is now the only mode. The installed
   interpreter is **pandas 3.0.1** (measured). The relevant new gotcha is that
   **`.values` / `.to_numpy()` now return READ-ONLY arrays** — "Arrays returned
   from `.to_numpy()` are read-only… `ValueError: assignment destination is
   read-only`". `anomaly_detector.py` never mutates the extracted arrays in place
   (every operation is `np.mean` / `np.diff` / `np.std` / slicing, all read-only),
   so the fix is unaffected — but this is the one pandas-3 behaviour that could
   have bitten a naive `.values` refactor, so it is stated here explicitly.
2. **The yfinance blank-most-recent-bar defect is a documented, still-open,
   upstream-Yahoo condition** first reported 2025-11-03 (issue #2622), not a
   local artefact and not self-healing. This is the recency evidence for treating
   the bad bar as permanent rather than transient.

Non-finding: **no change to `dropna` semantics** appears anywhere in the pandas
3.0.0 whatsnew. The `subset` / `how='any'` contract relied on by the fix is
unchanged from pandas 1.x through 3.0.1.

---

## A. External findings

**A1 — `df.dropna(subset=[...])` is the correct and documented idiom, and it
preserves cross-column alignment by construction.** The doc defines `subset` as
"Labels along other axis to consider, e.g. if you are dropping rows these would
be a list of columns to include" with `axis=0` and `how='any'` as defaults
(https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html,
accessed 2026-07-25). A single row-wise drop returns **one** DataFrame; every
column extracted from it afterwards is a view of the same surviving index, so
positional correspondence is guaranteed. `ignore_index` defaults to `False`, so
the DatetimeIndex survives — useful if a later step wants to report *which*
session was dropped.

`how='any'` (drop the row if ANY listed column is NA) is the right choice here,
not `'all'`. On the observed shape all four OHLC columns are NaN together, so the
two agree today — but `'any'` is what makes the fix robust to a future partial
NaN pattern (see A5), which is most of its value.

**A2 — `.values` after a row-wise drop is positionally faithful; pandas 3.0 does
not change that.** The only `.values`/`.to_numpy()` change in the 3.0.0 whatsnew
is gated behind the opt-in `future.distinguish_nan_and_na` option and applies to
**nullable** dtypes (`Float64`, `int64[pyarrow]`)
(https://pandas.pydata.org/docs/whatsnew/v3.0.0.html, accessed 2026-07-25).
yfinance returns NumPy-backed `float64`/`int64` (measured, §B3), so neither the
`NA`-vs-`NaN` unification nor the `object`-dtype coercion applies. The one live
pandai-3 hazard is the **read-only array** from CoW — irrelevant here because the
module never writes into the extracted arrays.

**A3 — Excluding the scored observation from its own baseline is standard
practice, and there is a clean citation for it.** "including it in the mean and
standard deviation calculation creates artificial look-ahead bias… the mean and
standard deviation of the full season data is not information one would have had
on the day" (Patrick Ward PhD,
https://optimumsportsperformance.com/blog/rolling-mean-and-sd-not-including-the-most-recent-observation/,
accessed 2026-07-25). The implementation pattern is `lag(rollapplyr(...))`. The
same convention appears independently in the trading-practitioner corpus as the
`[1]` bar-offset — "use a `[1]` offset so the current bar doesn't contaminate the
statistics, which reduces flicker"
(https://www.tradingview.com/script/WPyvzh8u-Z-Score-Volume-Range-Anomaly-Detector/,
snippet) — and in R's `roll_scale`. **Cross-domain triangulation:** sports
science and market microstructure converge on the identical rule, which raises
confidence that it is a property of z-scoring, not a domain convention.

Note what this citation does and does not support. It supports **"do not include
the scored observation in its own baseline"** (criterion 4's overlap). It does
**not** by itself support "never compare a partial session's volume to completed
sessions" — I found no authoritative source stating that as a named rule. The
partial-session concern is a data-completeness argument (a half-day of volume is
mechanically ~half of a full day's), which is arithmetic rather than a citable
finding. **State it as reasoning, not as a citation.**

**A4 — Overlap is NOT automatically a defect. [ADVERSARIAL]** The obvious reading
of criterion 4 is "the last 5 bars are inside the 60-bar baseline, therefore the
z-score is contaminated, therefore it is a bug." The vol-of-vol literature
qualifies this: "allowing for the overlap of consecutive local windows to
pre-estimate the spot variance **may correct for this bias**… window overlapping
is crucial in order to optimize the relative bias of the PSRV"
(https://ar5iv.labs.arxiv.org/html/2004.04013, accessed 2026-07-25). Overlap
trades a small self-contamination against a large variance reduction from the
longer effective sample, and in that setting the overlapping estimator is
*preferred*. So: report the overlap, quantify its direction, and **do not
"fix" it inside 80.31** — the step's own instruction ("note it if so; do not
silently fix it") is the right call and now has a literature basis.

Quantified direction (arithmetic, not a citation): with 5 of the 60 baseline bars
being the scored window, the baseline mean is pulled toward the recent value,
which **shrinks |z|**. That biases the detector toward **not** flagging — a
*suppressive*, therefore conservative, error.

**A5 — the real defect class here is broader than one column.** yfinance issue
#2622 documents the malformed-most-recent-bar condition as an upstream Yahoo
condition, first observed 2025-11-03 and still open, with "valid volume data…
indicating it's a completed trading session"
(https://github.com/ranaroussi/yfinance/issues/2622, accessed 2026-07-25). There
is no contractual guarantee that a future malformed bar will null *all four* OHLC
columns together. The current per-column code is only accidentally safe for
high/low/close because today's NaN pattern happens to be uniform across them
(§B2). That is the structural argument for the row-wise fix, independent of
today's one-session volume offset.

---

## B. Internal code inventory

### B1 — `backend/tools/anomaly_detector.py`, real line numbers (file is 259 lines)

| Line(s) | Content | 80.31 action |
|---|---|---|
| `:16-18` | `_Z_STRONG = 2.0`, `_Z_MODERATE = 1.5` | **DO NOT TOUCH** (80.27-deferred cluster) |
| `:21-23` | `_z(value, mean, std)` — returns `None` iff `std <= 0` | leave |
| `:26-42` | `_append_if_anomalous` | — |
| `:31` | `if z_score is not None and abs(z_score) >= _Z_MODERATE:` | **DO NOT TOUCH — this is 80.27's deferred L12** |
| `:38` | `"severity": "high" if abs(z_score) >= _Z_STRONG else "moderate"` | **DO NOT TOUCH — 80.27's deferred severity ladder (L13 cluster)** |
| `:55` | `hist = stock.history(period="1y")` | leave |
| `:57` | `if hist.empty or len(hist) < 20:` | **TOUCH** — must run AFTER the row-wise drop |
| `:66` | `close = hist["Close"].dropna().values` | **REPLACE** |
| `:67` | `volume = hist["Volume"].dropna().values` | **REPLACE** |
| `:68` | `high = hist["High"].dropna().values` | **REPLACE** |
| `:69` | `low = hist["Low"].dropna().values` | **REPLACE** |
| `:90-95` | volume anomaly block | **TOUCH only if criterion 2 is implemented as a baseline change** |
| `:184-191` | trailing-vs-forward PE block | — |
| `:188` | `if abs(pe_gap) > 20:` | **DO NOT TOUCH — 80.27's deferred L13** |
| `:204/:210/:217` | `de_ratio > 150`, `short_ratio > 4`, `beta > 1.8 or beta < 0.3` | **DO NOT TOUCH — 80.27's deferred L4** |
| `:230` | `if a.get("z_score", 0) < -_Z_MODERATE or a["metric"] in risk_metrics` | **DO NOT TOUCH — 80.27 de-duped this into the L4/L12 cluster** |
| `:234-242` | signal classification ladder | leave |
| `:250` | `"current_price": round(float(close[-1]), 2) if len(close) > 0 else None` | leave (value unchanged — verified §C2) |

The caller's audit anchors were close but drifted: `:74` (`close[-1]/close[-21]`)
and `:91-95` (volume block) are **correct**; `:17-18` for the thresholds is
**correct**; `:66-69` and `:57` are **correct**. `:188` is correct for the PE
ladder. No anchor in the prompt was wrong.

### B2 — Blast radius: every indexed/sliced use of the four arrays

**Membership rule (written down BEFORE applying it):** include every expression
in `get_anomaly_scan` that (i) reads `close`, `high`, `low`, or `volume` via
integer index, negative index, or slice, or (ii) reads a derived array whose
length is a function of one of those four (`daily_returns`, `deltas`, `gains`,
`losses`, `daily_pct`, `returns_20d`, `returns_5d`, `vols_20d`, `sma_devs`,
`sma_devs_200`). For each, ask: **does a one-element length offset in `volume`
relative to `close/high/low` change the computed value?** Pure `len()` guards are
listed separately because they change control flow, not values.

| Line | Expression | Arrays read | Offset-sensitive? |
|---|---|---|---|
| `:72` | `len(close) >= 60` | close | guard only |
| `:74` | `close[-1] / close[-21]` | close | **No** — close-internal |
| `:75` | `close[i]/close[i-20]` comprehension | close | No |
| `:82` | `close[-1] / close[-6]` | close | No |
| `:83` | `close[i]/close[i-5]` comprehension | close | No |
| `:90` | `len(volume) >= 60` | volume | guard only |
| `:91` | `np.mean(volume[-5:])` | volume | **YES** |
| `:92` | `np.mean(volume[-60:])` | volume | **YES** |
| `:93` | `np.std(volume[-60:])` | volume | **YES** |
| `:98` | `len(close) >= 60` | close | guard only |
| `:99` | `np.diff(np.log(close))` | close | No |
| `:100` | `daily_returns[-20:]` | derived(close) | No |
| `:102-103` | `daily_returns[i:i+20]` | derived(close) | No |
| `:111` | `len(close) >= 50` | close | guard only |
| `:113` | `np.diff(close)` | close | No |
| `:116-117` | `gains[-14:]`, `losses[-14:]` | derived(close) | No |
| `:132` | `np.mean(close[-50:])` | close | No |
| `:133` | `close[-1]` | close | No |
| `:136-138` | `close[i-50:i]`, `close[i]` | close | No |
| `:145` | `len(close) >= 200` | close | guard only |
| `:147-152` | `close[-200:]`, `close[-1]`, `close[i-200:i]`, `close[i]` | close | No |
| `:160` | `len(close) >= 60` | close | guard only |
| `:161` | `np.diff(close) / close[:-1]` | close | No |
| `:162` | `daily_pct[-5:]` | derived(close) | No |
| `:169` | `len(close) >= 200` | close | guard only |
| `:170` | `np.max(high[-252:])` / `np.max(high)` | high | **No, today only** — see caveat |
| `:171` | `np.min(low[-252:])` / `np.min(low)` | low | **No, today only** — see caveat |
| `:174` | `(close[-1] - low_52w) / range_52w` | close+low | **No, today only** — see caveat |
| `:250` | `close[-1]` | close | No |

**Result: exactly 3 offset-sensitive expressions, all in the volume block
(`:91`, `:92`, `:93`).** The step's claim about the true blast radius is
CONFIRMED, not merely asserted.

**The "today only" caveat — this is the finding worth carrying into the
contract.** `high`, `low` and `close` are mutually aligned right now *only
because* Open/High/Low/Close carry NaN on the **same** row (measured, §B3). The
per-column code has no mechanism enforcing that; it is a coincidence of the
current Yahoo defect shape. If Yahoo ever emits (say) NaN Close with real
High/Low, then `:174` would compare a `close[-1]` from session T-1 against a
52-week range including session T, and `:170-171` would silently shift — with no
NaN and no exception. So the row-wise fix buys **structural immunity**, and the
"only volume is affected" statement should be scoped as *"today, under the
currently-observed defect shape."*

Also worth stating: `volume` is `int64` (measured). **An int64 Series cannot hold
NaN**, so `hist["Volume"].dropna()` at `:67` is a structural no-op — `len(volume)
== len(hist)` unconditionally. That is the mechanism that guarantees the
misalignment whenever any OHLC NaN exists; it is not probabilistic.

### B3 — Live measurement (2026-07-25, `.venv/bin/python`)

```
python 3.14.4 | pandas 3.0.1 | numpy 2.4.4 | yfinance 1.2.0
AAPL rows 251 close 250 high 250 low 250 volume 251 open 250
MSFT rows 251 close 250 high 250 low 250 volume 251 open 250
NVDA rows 251 close 250 high 250 low 250 volume 251 open 250
   tail idx: ['2026-07-23', '2026-07-24']
   AAPL tail: [{Open 321.73, High 323.30, Low 319.35, Close 321.66, Volume 40840800},
               {Open nan,    High nan,    Low nan,    Close nan,    Volume 47402209}]
   dtypes: {Open float64, High float64, Low float64, Close float64, Volume int64, ...}
```

- **The step's 251/250/250/250/251 figure is CONFIRMED** on 3/3 tickers.
- **`Open` is also 250** — the prompt did not mention Open; it shares the defect.
- **The bad bar is 2026-07-24 = Friday**, measured on Saturday 2026-07-25 with US
  markets closed ~22h. **A COMPLETED session.** Caller's correction (a) confirmed
  independently of the caller, and corroborated by yfinance #2622 (A5).
- `Volume` dtype is `int64` on all three → its `dropna()` cannot ever drop.

### B4 — Consumers of `get_anomaly_scan`

| Consumer | Anchor | What it does with it |
|---|---|---|
| Signals API (bundle) | `backend/api/signals.py:116` | `_safe(anomaly_detector.get_anomaly_scan, "anomalies", ticker)` → one of 12 keys in the `/api/signals/{ticker}` bundle (the endpoint 80.1 fixed) |
| Signals API (single) | `backend/api/signals.py:210` | `asyncio.to_thread(...)` → dedicated endpoint |
| Layer-1 orchestrator | `backend/agents/orchestrator.py:1267-1269` | `fetch_anomaly_scan` wrapper |
| Layer-1 orchestrator | `:1990` | `_safe(self.fetch_anomaly_scan, "Anomaly", ticker)` in the enrichment fan-out |
| Layer-1 orchestrator | `:2034` | `"anomaly": lambda: self.fetch_anomaly_scan(ticker)` (retry/lite map) |
| Audit scripts | `scripts/audit/data_sources.py:45`, `scripts/audit/score_current_state.py:43` | metadata only, no numbers consumed |
| Tests | `backend/tests/test_phase_80_1_signals_nan_serialisation.py:144,186` | monkeypatched to a canned `ok("anomalies")` — **so the 80.1 suite is insensitive to this change** |

There is **no direct paper-trading consumer**. The path to a trade is:
`get_anomaly_scan` → enrichment dict → (a) `info_gap` status classification and
(b) the prose handed to the Layer-1 LLM debate → synthesis verdict → the
autonomous loop. So the numbers are an **LLM input**, never an arithmetic input to
sizing or a gate threshold. No screener/backtest/optimizer path reads it.

### B5 — Does it feed `info_gap`, and is 80.27 already affecting it?

Yes it feeds it, and **no, 80.27 has not degraded it.** Measured just now against
the shipped-ON detector:

```
anomaly criticality: HIGH | _SOURCE_CRITICALITY keys: 12
AAPL signal=ANOMALY_OPPORTUNITY count=2 non_finite=False status=SUFFICIENT
     metrics: ['realized_volatility_20d', 'near_52w_high']
MU   signal=ANOMALY_OPPORTUNITY count=3 non_finite=False status=SUFFICIENT
     metrics: ['max_daily_move_5d', 'pe_trailing_vs_forward_gap', 'beta_extreme']
```

`_has_non_finite` (`backend/agents/info_gap.py:58-73`) returns **False** and
`_assess_source_status` (`:76-122`) returns **SUFFICIENT**. The per-column
`dropna` is precisely why — it strips the NaN before any number is computed. This
reproduces 80.27's own `anomaly non_finite=0` measurement. Note `anomaly` is
**HIGH** criticality, so if a future change did introduce a non-finite into this
payload it would immediately become a `critical_gap` at `:164-165`. **Criterion 4
must therefore assert `_has_non_finite(payload) is False` after the fix, not just
"HTTP 200"** — that is the sharper, cheaper guard, and it binds the two steps
together correctly.

Also note `_NAN_TOKEN_RE` at `info_gap.py:55` scans the *summary prose*. This
module's summaries (`:236/:239/:242`) are built from ints, so they cannot render
`nan` — but `_append_if_anomalous`'s `note=` strings at `:126/:129/:143/:157/:166`
are f-strings over floats and **would** render `nan` if a NaN ever reached them.
That is 80.27's L12 territory, not 80.31's. Do not fix it here; it is already
deferred.

### B6 — Test conventions to follow

Existing tests: `backend/tests/test_phase_80_1_signals_nan_serialisation.py`,
`test_phase_80_2_error_response_contract.py`,
`test_phase_80_27_nonfinite_fail_safe.py`. **There is no existing test that calls
`get_anomaly_scan` for real** — 80.1 monkeypatches it away (`:186`). So 80.31's
test is greenfield for this module and must build its own synthetic frame.

House conventions, from reading both files:

- Module docstring states the MEASURED defect with a date and a `file:line`, names
  what ships ON vs DARK, and states why each assertion exists individually
  (`test_phase_80_1_...py:1-21`).
- `_REPO_ROOT = Path(__file__).resolve().parents[2]` + `sys.path.insert(0, ...)`
  preamble (`:27-30`), `from __future__ import annotations`.
- Deterministic, network-free: monkeypatch the tool layer, never call Yahoo.
- **A green control** so a red mutation run is attributable to the mutation and
  not a broken fixture (`test_control_finite_payload_is_unchanged`, `:251-257`).
- **A fixture pin that binds to the actual subject.** The caller's warning about
  `:260-279` is well-placed: the first version asserted
  `not math.isfinite(float("nan"))` — a LIBRARY FACT that passes under the very
  fixture mutation it claimed to guard. The corrected version introspects the
  fixture itself:
  ```python
  default = inspect.signature(_install_fake_tools).parameters["sector_1mo"].default
  assert isinstance(default, float)
  assert not math.isfinite(default)
  ```
  **Apply the same shape here.** For 80.31 the fixture pin must assert that the
  synthetic frame *actually has* the trailing NaN-OHLC/real-Volume row — e.g.
  `assert frame["Close"].isna().sum() == 1 and frame["Volume"].notna().all()` and
  `assert frame["Volume"].dtype == "int64"` — read off the fixture builder, so it
  dies if someone "cleans up" the fixture. An assertion like
  `assert len(a) == len(b)` on arrays built by the fixture is NOT a pin; it is the
  thing under test.

The mutation for criterion 3 is exact and easy: restore `:66-69` verbatim and
confirm the length-equality assertion goes red. Per
`feedback_mutation_test_guards_and_fixtures`, **also mutate the fixture** (drop
the trailing bad row) and confirm the alignment test then passes vacuously —
that proves the fixture, not just the code, is load-bearing.

---

## C. Risk / do-no-harm

### C1 — Measured before/after, 6 tickers (2026-07-25)

Ran the real `:66-69` extraction against the proposed
`hist.dropna(subset=["Open","High","Low","Close"])` extraction on the same fetched
frame:

| Ticker | lens OLD (c,v,h,l) | lens NEW | z OLD | z NEW | Δz | verdict flip |
|---|---|---|---|---|---|---|
| AAPL | 250,**251**,250,250 | 250,250,250,250 | -0.3680 | -0.2591 | +0.1089 | no (none→none) |
| MSFT | 250,**251**,250,250 | 250,250,250,250 | -0.5289 | -0.4820 | +0.0469 | no |
| NVDA | 250,**251**,250,250 | 250,250,250,250 | -1.1926 | -1.0468 | +0.1458 | no |
| AMD | 250,**251**,250,250 | 250,250,250,250 | -0.6358 | -0.5679 | +0.0679 | no |
| MU | 250,**251**,250,250 | 250,250,250,250 | -0.9420 | -0.6044 | **+0.3376** | no |
| JPM | 250,**251**,250,250 | 250,250,250,250 | -0.4687 | -0.3536 | +0.1151 | no |

`position` in the 52-week range (`:174`) was **byte-identical** on 6/6, confirming
§B2's finding that only the volume block moves today.

**Reading.** No verdict flips on 6/6 today, but MU's +0.34 is the same order as
the gap between `_Z_MODERATE 1.5` and `_Z_STRONG 2.0`. So a flip is entirely
possible on a different day/ticker; "measured no flips" is not "cannot flip". Do
not let the contract claim byte-identity.

Direction is **not** systematically signed — Δz was positive on all 6 today only
because the dropped Friday volume happened to be below the replacement bar's, and
the 60-bar baseline shifts too. Do not generalise the sign.

### C2 — Does the fix restore real values? (the key question)

**No. It removes one.** Precisely:

- The malformed row carries a **real, completed-session int64 Volume**
  (AAPL 47,402,209 for 2026-07-24). Row-wise `dropna` **discards it**.
- Nothing that was previously NaN-suppressed becomes visible. The payload is
  `_has_non_finite == False` before (§B5) and stays `False` after — there is no
  NaN anywhere in this module's output in either state.
- `current_price` at `:250` is **unchanged**: `close[-1]` is the same
  session in both paths (the NaN close was already dropped by the old code too).
  Verified by the identical `position` figures in C1.

So this is categorically **not** the 80.27 hazard. 80.27's dark flag exists
because enabling it lets a *previously-suppressed failure* become a visible
`ERROR`, changing what the debate sees in a direction that needed operator
consent. Here the change is: **the volume statistics stop being computed over a
window one session offset from the price window.** That is a correctness repair
with no suppression/un-suppression component.

**However — this is not byte-identical, and it does change what the LLM sees.**
Three honest statements for the contract:

1. The Δz is real and up to +0.34 today; near a threshold it could add or remove
   a `volume_5d_vs_60d` entry.
2. An added anomaly increments `total` at `:223` and can move the classification
   at `:234-242`. `volume_5d_vs_60d` is **not** in `risk_metrics` (`:226-227`), so
   a newly-firing volume anomaly with a **positive** z counts as an
   *opportunity* — the **less-conservative** direction. (A negative z counts as
   risk via `:230`.)
3. That said, the output is an LLM debate input, not a gate or a size. There is
   no arithmetic path from an anomaly entry to a position size or a stop.

**Ruling: do NOT dark-flag this.** A dark flag here would be the wrong instrument
— it would preserve a *known-wrong* computation as the default, and the correct
value is not operator-discretionary the way 80.27's ERROR-vs-NEUTRAL verdict was.
Instead: state the Δz measurement in `experiment_results.md`, and satisfy
criterion 4 with the `_has_non_finite is False` + HTTP-200 evidence. **If Main
disagrees, the cheap middle path is to ship it un-flagged but record the 6-ticker
before/after table in the live_check so the operator can audit the delta** — the
same instrument 80.27 used for its measurements, without the flag.

### C3 — Live book

No paper-trading, screener, optimizer, or backtest path reads `get_anomaly_scan`
(§B4). The book cannot move as a direct consequence. The only channel is the
LLM debate's read of the enrichment prose, which is indirect and unquantifiable —
disclose it, do not claim it is zero.

### C4 — Criterion 2: the recommendation, with the choice stated explicitly

The corrected mechanism (§B3) means "exclude the in-progress session" is the
wrong frame — the malformed bar is a *completed* session, and the row-wise drop
already removes it. So criterion 2 has to be answered as **two separate
statements**, and the contract should say both:

> **(i) The malformed bar.** The most recent bar returned by yfinance carries
> NaN OHLC with a real int64 Volume and is a **COMPLETED** session (measured
> 2026-07-24, on a Saturday with markets closed; corroborated upstream by
> yfinance issue #2622). Row-wise `dropna(subset=["Open","High","Low","Close"])`
> excludes it from **all four** arrays, so the volume z-score is computed over
> completed sessions whose OHLC is intact. The cost is that one real volume
> observation is discarded; that is accepted deliberately, because a volume
> figure that cannot be aligned to a price is not usable by this detector.
>
> **(ii) The genuinely in-progress session.** During US market hours yfinance
> returns a partial current-day bar with real (partial) OHLC and partial volume.
> Row-wise `dropna` does **not** remove it — nothing is NaN. Recommendation:
> **do not add a special case in 80.31.** Rationale: (a) the entire module
> already treats the last bar as final for every close-based metric
> (`:74/:82/:133/:148/:162/:174/:250`); special-casing volume alone would make
> the module internally inconsistent; (b) the module's own scheduled caller runs
> outside market hours; (c) an in-session guard needs a market-calendar
> dependency this module does not have (`backend/services/markets.py` is the
> house calendar). **Queue it as its own step** per
> `feedback_queue_discovered_defects_in_masterplan`.

The `[1]`-offset / `lag()` practice from A3 is the right instrument for (ii) if
and when it is done, but it is a *different* change from alignment and should not
be smuggled into 80.31.

### C5 — `subset` composition: one judgment call to make explicitly

The step prescribes `subset=["Open","High","Low","Close"]`. The module never
reads `Open`. Including it means a hypothetical future row with NaN Open but
usable HLC would be dropped unnecessarily. Excluding it means such a row survives
with a usable HLC. **Recommendation: keep all four as the step specifies** — it
matches the measured defect shape exactly (Open is NaN on the same row, §B3), it
is the more conservative choice (drop a suspect bar rather than trust a partially
malformed one), and deviating from the step's stated fix direction on a P2 is not
worth the divergence. State the reasoning in the contract rather than leaving it
silent.

### C6 — Discovered defects to QUEUE, not fix here

Per `feedback_queue_discovered_defects_in_masterplan`, each of these needs its own
research-gated step, written for an executor with no memory of this discovery:

- **D1 — units mismatch in the volume z-score (`:91-94`).** `recent_vol` is the
  **mean of 5** daily volumes, but it is divided by the standard deviation of
  **individual daily** volumes (`np.std(volume[-60:])`). Under an iid assumption
  the standard error of a 5-day mean is σ/√5, so the computed |z| is
  systematically **≈2.24× too small**. Direction: **suppressive** — anomalies are
  systematically under-detected. Fixing it makes the detector fire MORE, i.e. the
  less-conservative direction, so it is **out of scope for 80.31** and needs its
  own gate. This is the single largest numerical defect in this block, an order of
  magnitude bigger than the alignment offset it sits next to.
- **D2 — self-overlapping baseline (`:91` vs `:92-93`).** The 5 scored bars are
  inside the 60-bar baseline. Also suppressive (shrinks |z|). Qualified by A4 —
  overlap is not automatically wrong — so this needs analysis, not a reflex fix.
- **D3 — the in-progress-session bar (C4-ii).** As above.
- **D4 — `high`/`low` alignment is coincidental (§B2 caveat).** 80.31's row-wise
  fix closes this, so if 80.31 ships as designed D4 is closed by construction —
  record it as *closed by 80.31*, and make sure the contract says so, otherwise a
  later audit will re-discover it.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: 2 pandas official docs, 1 library issue tracker, 1 named-practitioner blog, 1 peer-reviewed preprint via ar5iv)
- [x] 10+ unique URLs total (17)
- [x] Recency scan (last 2 years) performed + reported — 2 findings + 1 explicit non-finding
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (anomaly_detector, info_gap, signals API, orchestrator, both phase-80 test files)
- [x] Contradictions / consensus noted (A4 is an explicit qualifier against the naive overlap reading)
- [x] All claims cited per-claim
- [x] Every numeric claim MEASURED, not asserted (§B3, §B5, §C1) per `feedback_measure_dont_assert_claims`

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "80.31 and 80.27 do not collide: 80.31 touches :57 and :66-69 (plus the :90-95 volume block), while 80.27's deferred ladders are :31, :38, :188, :204/:210/:217 and :230. Measured on 6 tickers: 251 raw rows, close/high/low/open all 250, volume 251 -- the step's figure is confirmed, and the malformed bar is Fri 2026-07-24, a COMPLETED session (caller's correction (a) confirmed, corroborated by yfinance issue #2622). Exactly 3 expressions are offset-sensitive, all in the volume block; high/low are aligned only coincidentally, which is the structural argument for the row-wise fix. The fix does NOT restore real values -- it DISCARDS the malformed session's real int64 volume; the payload is 0-non-finite before and after, so no dark flag is warranted, unlike 80.27. It is not byte-identical: dz +0.047..+0.338, no verdict flips on 6/6 today but a flip is possible near the 1.5/2.0 thresholds. No paper-trading path consumes this tool; it is an LLM debate input only. Two suppressive statistical defects found and queued, not fixed: a sqrt(5) units mismatch (std of daily volume used to score a 5-day mean) and the self-overlapping baseline.",
  "brief_path": "handoff/current/research_brief_80.31.md",
  "gate_passed": true
}
```
