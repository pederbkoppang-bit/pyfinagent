# live_check — phase-80.1

**Required (masterplan, verbatim):** *verbatim curl output showing 200 + the sector block
with its 1mo value, plus the pytest output for the NaN regression test.*

Captured 2026-07-25. All output verbatim.

---

## §A. METHOD + the same honest limitation as 80.2

The operator's backend on `:8000` was **NOT restarted** — `phase-79.55` is `status:
pending` and carries `[RESTART BLOCKER -- answer BEFORE the next backend restart]`.
So the after-fix evidence comes from an isolated second uvicorn:

```
DEV_LOCALHOST_BYPASS=1 PYFINAGENT_TEST_NO_BQ=1 \
  .venv/bin/python -m uvicorn backend.main:app --port 8001 --lifespan off --log-level warning
```

`--lifespan off` because `backend/main.py`'s lifespan starts an APScheduler paper-trading
scheduler — a second full instance could have run a second trading loop. The middleware
and routing under test are built at app construction, independent of lifespan.

**Unlike the in-process test suite, this rig hit the REAL network and the REAL yfinance
data**, so §B/§C below are the genuine end-to-end reproduction, not a fixture.

**Consequence:** this fix is **inert on `:8000` until the operator restarts.** On the
un-restarted process `/api/signals/AAPL` still returns 500 — measured below as the "before"
control.

Playwright: `@playwright/mcp@0.0.76` as connected this session, 1440x900, isolated
skip-auth `:3100` with `PLAYWRIGHT_DIST_DIR=.next-audit-3100` and
`NEXT_PUBLIC_API_URL=http://localhost:8001`. The operator's `:3000` was never driven.

---

## §B. The immutable verification command — BEFORE vs AFTER

Command (verbatim): `curl -s -m 120 -o /dev/null -w '%{http_code}\n' .../api/signals/AAPL`

```
########## BEFORE — operator's :8000 (pre-fix process) ##########
GET /api/signals/AAPL -> HTTP 500  (18.958060s)

########## AFTER — :8001 rig, phase-80.1 code, REAL network + REAL yfinance ##########
200
```

**Criterion 1 MET** — and note the "before" is not a historical quote: it was re-measured
against the live `:8000` at capture time.

## §C. Criterion 2 — all 12 signal keys present

```
=== criterion 2: all 12 signal keys present? ===
  present 12/12; missing=NONE
```

(`insider, options, social_sentiment, patent, earnings_tone, fred_macro, alt_data, sector,
nlp_sentiment, anomalies, monte_carlo, quant_model`)

## §D. Criterion 3 — the sector block, verbatim: keys PRESENT, values `null`

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "sector_etf": "XLK",
  "stock_returns":      { "1mo": null, "3mo": null, "6mo": null, "1y": null },
  "sector_returns":     { "1mo": null, "3mo": null, "6mo": null, "1y": null },
  "spy_returns":        { "1mo": null, "3mo": null, "6mo": null, "1y": null },
  "relative_vs_sector": { "1mo": null, "3mo": null, "6mo": null, "1y": null },
  "relative_vs_market": { "1mo": null, "3mo": null, "6mo": null, "1y": null },
  "sector_performance": {
    "Technology": null, "Financial": null, "Energy": null, "Healthcare": null,
    "Communication Services": null, "Industrials": null, "Consumer Staples": null,
    "Utilities": null, "Real Estate": null, "Materials": null, ...
  }
}
```

`"1mo"` is **present** and its value is **`null`** — not dropped, not `0.0`, not a 500.
That is exactly what criterion 3 and the step's additional criterion demand.

**Worth stating rather than glossing:** it is not just `1mo`. **Every** return in the block
is null, on every period, for the stock AND the sector AND SPY. That corroborates the
research finding that `sector_analysis.py:74` fetches SPY for every ticker, so the
placeholder-row NaN poisons the whole analysis regardless of the symbol requested. The
scale of the data outage was previously invisible — it presented as a single opaque 500.

## §E. Criterion 4 — the regression suite

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_1_signals_nan_serialisation.py -q
..............                                                           [100%]
14 passed, 40 warnings in 2.44s
```

Mutation matrix (driver `scratchpad/mutate_80_1.py`) — **the 5 mutations run were all
killed**, including the fixture mutation. Full table in `experiment_results_80.1.md`.

*(Cycle-2 correction: this line previously read "5/5 guards held, **0 vacuous**". A
matrix licenses only "these N mutations were killed" — never a suite-level no-vacuity
claim. Q/A falsified the stronger claim by finding a genuinely vacuous test in the same
file; see `experiment_results_80.1.md` §3.1.)*

---

## §F. 80.27's evidence is INTACT — this step deliberately did not fix it

Measured on the same rig, immediately after the 200 above:

```
=== 80.27 EVIDENCE — deliberately NOT fixed by this step ===
  sector.signal          = 'NEUTRAL'   <- a data outage rendered as a TRADING VERDICT
  sector.summary         = 'AAPL (Technology/Consumer Electronics). 3M return: +nan% vs
                            sector +nan% vs S&P +nan%. Signal: NEUTRAL.'
  quant_model.signal     = 'NEUTRAL'
  quant_model.score      = None
  quant_model.mda_source = 'backtest'

=== non-finite floats remaining in the RAW tool output (pipeline path) ===
  sector_analysis.get_sector_analysis('AAPL') -> 31 non-finite floats
  raw signal = 'NEUTRAL'  <- 80.27: NEUTRAL from an all-NaN input
```

**31 non-finite floats** independently reproduces the audit's measured count exactly.

The sanitiser sits at the far edge of the pipe, so:
- the **tool layer still returns NaN** — the pipeline consumes it unchanged;
- the **summary string still literally reads `+nan%`** (it is a `str`, not a float, so the
  sanitiser correctly does not touch it) and that prose is what gets fed to the Gemini
  sector agent;
- `mda_source: 'backtest'` still advertises real walk-forward weights over non-finite
  factors.

This is the intended outcome. A Q/A or operator investigating 80.27 will still find every
piece of evidence.

## §G. Playwright — the operator's reported symptom is gone, and 80.27 became VISIBLE

`/signals` → typed `AAPL` → **Fetch Signals**. Screenshot:
`handoff/current/captures_80.1/80.1_signals_page_renders_200.png`.

The page renders fully — Signal Consensus bar (`0 bullish · 11 neutral · 1 bearish`) and
all 12 signal cards (Insider Activity, Options Flow, Social Sentiment, Innovation,
Earnings Tone, Macro Climate, Alt Data, Sector Strength, NLP Sentiment, Anomaly Scan, Risk
Scenario, Quant Model). The dead feature works.

**And the fix made the 80.27 defect operator-visible rather than masked.** Two cards read,
on screen:

- **Sector Strength — `NEUTRAL`**: *"AAPL (Technology/Consumer Electronics). 3M return:
  **+nan%** vs sector **+nan%** vs S&P **+nan%**. Signal: NEUTRAL."*
- **Quant Model — `NEUTRAL`**: *"Quant model score: **nan** → NEUTRAL. MDA source:
  backtest."*

Before this step, that was hidden behind an opaque 500 that the frontend reported as
"backend unreachable". An operator can now see a data outage being presented as a
confident NEUTRAL — which is precisely the argument for prioritising 80.27, and precisely
why this step must not be mistaken for its fix.

## §H. Teardown + operator-instance integrity

```
:3100 listeners: 0
:8001 listeners: 0
operator :3000/      -> 302   (healthy authed signature)
operator :3000/login -> 200
:8000 pid -> 70791            (same pid as session start => NOT restarted)
```

`frontend/tsconfig.json` + `frontend/next-env.d.ts` were rewritten by `next dev` and
restored from HEAD; md5s back to `cecfaa5d04f97bf443b8750d944606f9` /
`ba64ff7d54714a8f64db89b1003207d8`, `git status` clean.

**CORRECTION (cycle 2, Q/A finding 2).** An earlier revision of this line said
"`.next-audit-3100` removed." **That was false.** Measured now:

```
$ du -sh frontend/.next-audit-3100      -> 208M
$ find frontend/.next-audit-3100 -type f | wc -l   -> 165
$ git check-ignore -v frontend/.next-audit-3100
  .gitignore:25:.next-*/	frontend/.next-audit-3100
$ git status --short | grep -c next-audit   -> 0
```

**It is still on disk and it was NOT removed.** `rm -rf frontend/.next-audit-3100` is
policy-denied to this session (attempted three times, denied each time — it is on the
operator ask list). There is **no commit-pollution risk**: `.gitignore:25` (`.next-*/`)
covers it, so it is invisible to `git status` and cannot be swept by the auto-commit
hook's `git add -A`. The cost is 208M of disk, which the operator can reclaim with one
command.
