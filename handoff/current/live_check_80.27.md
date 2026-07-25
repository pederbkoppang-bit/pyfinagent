# live_check — phase-80.27

**Required (masterplan, verbatim):** *(a) `_assess_source_status` returning a
non-SUFFICIENT status for a poisoned payload, (b) sector_analysis and quant_model
returning ERROR/NO_DATA rather than NEUTRAL on non-finite input, (c) the mutation runs
proving each guard fails without the fix.*

Captured 2026-07-25. All output verbatim.

---

## §A. (a) The immutable verification command — run VERBATIM

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import math; from backend.agents.info_gap import _assess_source_status; p={'signal':'NEUTRAL','summary':'3M return: +nan% vs sector +nan%','stock_returns':{'1mo':float('nan')}}; print('status:', _assess_source_status('sector', p))"
status: MISSING
```

**Pre-fix baseline for the identical payload was `SUFFICIENT`** (independently
re-measured by the researcher before the change). `MISSING` is the specific status that
routes a HIGH-criticality source into `critical_gaps` and fires the existing retry loop —
criterion 1. `PARTIAL` would also be "not SUFFICIENT" but would silently fail the
criterion, so the test pins the exact string.

**This half ships ON (un-flagged)** and therefore works on the operator's box as soon as
the process reloads the module — it is a pure *detector* tightening that can only move a
source toward MISSING, i.e. more gating and fewer trades.

## §B. (b) Both tools, on LIVE data, in BOTH flag states

Real `yf.Ticker("AAPL")`, no fixtures:

```
===== flag OFF (default; legacy) =====
  sector.signal      = 'NEUTRAL'   info_gap -> MISSING
  sector.summary     = 'AAPL (Technology/Consumer Electronics). 3M return: +nan% vs sector +nan% vs S&P +nan%. S'
  sector non-finite  = 31
  quant.signal       = 'NEUTRAL'    info_gap -> MISSING
  quant.score        = nan
  quant.mda_source   = 'backtest'

===== flag ON =====
  sector.signal      = 'ERROR'   info_gap -> MISSING
  sector.summary     = 'Error: sector analysis unavailable for AAPL -- price history returned non-finite values,'
  sector non-finite  = 0
  quant.signal       = 'ERROR'    info_gap -> MISSING
  quant.score        = None
  quant.mda_source   = 'non_finite_inputs'
```

with the warnings the guards emit:

```
sector_analysis AAPL: non-finite inputs (stock_3m=np.float64(nan) sector_3m=np.float64(nan)
  spy_3m=np.float64(nan)) -- returning ERROR instead of a fabricated NEUTRAL (phase-80.27)
quant_model AAPL: non-finite score=nan or features ['momentum_1m', 'momentum_3m',
  'momentum_6m', 'annualized_volatility', 'sma_50_distance', 'sma_200_distance']
  -- returning ERROR instead of a fabricated NEUTRAL (phase-80.27)
```

Reading that table:

- **Criterion 2** — `ERROR`, never `NEUTRAL`, on non-finite input.
- **Criterion 3** — sector non-finite floats `31 → 0`, and `nan` is gone from the prose,
  **on this all-non-finite input**.

  > **CORRECTION (cycle 2, Q/A finding D1).** An earlier revision of this line
  > generalised that measurement to *"neither the rendered summary nor the serialised
  > payload **can** carry nan into an LLM prompt"*. **That was false**, and Q/A
  > demonstrated it rather than argued it. The original ladder guard inspected only the
  > three **3mo** operands, so a *partial* outage — a bad bar at the START of one window,
  > which poisons that horizon alone because `_compute_return` is `Close[-1]/Close[0]` —
  > still produced `NEUTRAL` with literal `NaN` in the payload that
  > `json.dumps(sector_data, indent=2)` at `backend/config/prompts.py:537` hands to the
  > sector agent. Fixed in cycle 2 by a payload-completeness guard; measured below.
- **Criterion 6** — `mda_source` stops claiming `'backtest'` over non-finite factors.
- **The safety net fires in BOTH states.** `info_gap -> MISSING` on the OFF row too,
  because the detector half is un-flagged. So gap detection and the retry loop are live
  immediately; only the *verdict* change waits for the operator.

## §C. Root-cause measurement — the step's stated mechanism is WRONG

Measured 2026-07-25 17:45 UTC — a **Saturday**, US markets closed ~22h:

```
ticker  rows  lastIdx      lastClose      lastVol   NaN-close?
AAPL     125  2026-07-24         nan     47402209   YES
SPY      125  2026-07-24         nan     41480799   YES
XLK      125  2026-07-24         nan      7725125   YES
NVDA     125  2026-07-24         nan    110844026   YES
MSFT     125  2026-07-24         nan     27544509   YES
JPM      125  2026-07-24         nan      7513084   YES
```

AAPL `history(period="1mo")` — NaN-Close rows: **1 of 21**; rows with NaN Close AND
Volume > 0: **1**.

The step body (and 80.1's research) describe a placeholder for the *still-forming current
session*. **That is not what this is.** Exactly one bar — the most recent **completed**
session — is persistently malformed, and it does not self-heal at the close. The real
non-zero Volume is precisely why yfinance's own `keepna` mask (`.all(axis=1)`) keeps the
row. **Consequence: universal and permanent, not market-hours-gated — there is no quiet
window in which to ship this "safely by timing". The flag is the only real control.**

Measured independently twice (Main, then the researcher) before either of us saw the
other's numbers; the results agree exactly.

## §D. (c) Mutation matrix — 11/11 killed

Driver `scratchpad/mutate_80_27.py`. Each mutation applied to the real file, guard run,
file restored from an in-memory snapshot (never `git stash`).

| # | Mutation | Result |
|---|---|---|
| M1 | Detector: remove the payload non-finite scan | **killed** |
| M2 | Detector: remove the nan-in-prose check | **killed** |
| M3 | Detector: widen the nan regex to a SUBSTRING match | **killed** |
| M4 | Detector: drop `NO_DATA` handling | **killed** |
| M5 | Detector: remove `quant_model` from `_SOURCE_CRITICALITY` | **killed** |
| M6 | `sector_analysis`: remove the fail-safe guard | **killed** |
| M7 | `quant_model`: remove the fail-safe guard | **killed** |
| M8 | `quant_model`: keep advertising `mda_source='backtest'` | **killed** |
| M9 | Flag defaults **ON** | **killed** |
| **M10** | **FIXTURE** mutated: poisoned returns become finite | **killed** |
| M11 | Guard becomes **fail-ALWAYS** (ERROR even on finite data) | **killed** |

`11/11 mutations killed; 0 survived.` Working tree byte-identical afterwards.

Three of these are worth naming individually:
- **M3** is the self-inflicted-outage mutation. `"financial"`, `"governance"`,
  `"covenant"`, `"tenant"`, `"nanotech"` and `"finance"` all contain the substring `nan`;
  a naive `"nan" in summary.lower()` would mark nearly every real summary MISSING —
  strictly worse than the bug being fixed. The token regex uses non-letter boundaries.
- **M9** guards the dark-launch contract itself: if the flag ever defaults `True`, live
  trading behaviour changes without an operator token.
- **M11** guards against fail-ALWAYS. A guard that returns ERROR on *finite* data is a
  denial-of-service on the pipeline, not a safety feature.

## §E. Suites + lint

```
$ pytest test_phase_80_27 + test_phase_80_1 + test_phase_80_2 -q
56 passed

$ ruff check --select F401,F811,F821 (the step's derived scope)
backend/agents/info_gap.py:12:8: F401 `json` imported but unused
```

That single F401 **reproduces byte-identically against `git show HEAD:`** — pre-existing,
not introduced by this step.

**One regression-sweep failure, proven pre-existing rather than asserted:**
`test_phase_23_2_6_backend_log_has_skipping_buy_evidence` asserts `skip_count >= 1`
against the **live `backend.log`**. I reverted all four of my changed files to their HEAD
contents and re-ran it:

```
reverted all 4 changed files to HEAD; re-running the failing test...
FAILED backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
1 failed in 0.16s
RESTORED all 4 files
```

Identical failure with my changes absent. It is a live-log-content test whose evidence has
rotated out. 106 other tests in that selection pass.

## §F. Deployment state

`tools_nonfinite_fail_safe_enabled` defaults **False** — verified live:

```
$ python -c "from backend.config.settings import get_settings; print(get_settings().tools_nonfinite_fail_safe_enabled)"
False
```

So **Half B ships completely dark**: byte-identical legacy verdicts until the operator
sets it. The flip token is on the ask list. Half A (the detector) is live on the next
module load.

Operator `:8000` was **not** restarted (`phase-79.55` remains an open RESTART BLOCKER).
Per the researcher's measurement of `settings.py:619-621` + `autonomous_loop.py:1879-1881`
(`cache_clear()` then `get_settings()` per ticker, with pydantic-settings re-reading the
`.env` file), a flag flip would reach the **autonomous loop** on the next ticker analysis
without a restart, while the API layer would lag until one. That wiring was verified by
reading; an end-to-end flip was **not** performed, because mutating `backend/.env` is out
of scope for this session.

---

## §G. CYCLE 2 — the D1 leak, closed and measured

Q/A's exact input classes, re-run after the payload-completeness guard:

```
[PARTIAL: only 1mo NaN, 3mo finite] flag=False
   signal='NEUTRAL'  json has NaN: True   leaked=['"1mo": NaN,', '"1mo": NaN,']
   info_gap -> MISSING
[PARTIAL: only 1mo NaN, 3mo finite] flag=True
   signal='ERROR'    json has NaN: False  leaked=[]
   info_gap -> MISSING

[PARTIAL: only 1y NaN] flag=False
   signal='NEUTRAL'  json has NaN: True   leaked=['"1y": NaN', '"1y": NaN']
   info_gap -> MISSING
[PARTIAL: only 1y NaN] flag=True
   signal='ERROR'    json has NaN: False  leaked=[]
   info_gap -> MISSING

[ALL finite (control)] flag=True
   signal='NEUTRAL'  json has NaN: False  leaked=[]
   info_gap -> SUFFICIENT
```

with the guard's own line:

```
sector_analysis AAPL: 5 non-finite value(s) in the assembled payload --
returning ERROR so no NaN reaches an LLM prompt (phase-80.27)
```

**Criterion 3 now holds in general, not just on the all-non-finite case**, and the
all-finite control still produces a real `NEUTRAL` classified `SUFFICIENT` — so the
widened guard is not fail-always. The flag-OFF rows are unchanged, preserving the
byte-identical dark-launch contract.

## §H. CYCLE 2 — Q/A's five surviving mutations, now killed

Q/A authored 10 mutations of its own; 7 survived my cycle-1 suite. The five substantive
ones (the other two were NOTE-level coverage gaps, addressed by new tests) re-run against
the cycle-2 suite:

```
[QM12-sector]  KILLED -- flag helper hard-wired to return False (dead flag read)
[QM12b-sector] KILLED -- flag helper reads a MISSPELLED settings attribute
[QM12-quant]   KILLED -- quant flag helper hard-wired to return False
[QM13]         KILLED -- sector payload-completeness guard removed (D1 leak returns)
[QM14]         KILLED -- quant scans only the score, not every feature

Q/A's surviving mutations now killed: 5/5
```

**QM12/QM12b is the one that mattered most.** Every Half-B test monkeypatched
`_nonfinite_fail_safe_enabled` itself, so the *production* flag-read executed in **zero**
tests — the whole guard could have shipped wired to a misspelled settings key and the
suite would have stayed green. `test_the_real_flag_read_path_executes` now drives the real
helper against a stubbed `get_settings`, and a companion test pins the fail-open behaviour.

Suite: **29 passed** (was 24).
