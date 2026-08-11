# Experiment results -- step 86.9

**Step**: `86.9` (phase-86, **P1**) | **Phase**: GENERATE | **Date**: 2026-08-11
**Driver**: Main (`pyfinagent-06`) | **Contract**: `26037c1e` (written BEFORE any code)

**NOTHING WAS CHANGED.** Every finding is a measurement; every recommendation is an
ask. No timeout, flag, `.env` value or setting was modified.

---

## 1. Criterion 1 -- MET. Read from the RUNNING process.

```
$ curl -s http://127.0.0.1:8000/api/settings/
  paper_cycle_max_seconds = 10800.0     <- live from pid 66306
  paper_analyze_top_n     = 5
  paper_screen_top_n      = 10
```

> **MY OWN CLAIM THAT THIS WAS UNSATISFIABLE WAS WRONG.** I reported that no
> endpoint exposed the budget. `GET /api/settings/` has exposed it since step 38.12
> (`settings_api.py:123/:171/:308/:383`). **I probed `/api/settings` without the
> trailing slash, got an empty response, and treated absence-of-result as evidence
> of absence.** The research gate refuted it.

**Caveat that matters for interpretation**: `_cycle_timeout` is captured once at
`autonomous_loop.py:507`, so this endpoint reports the **next** cycle's budget.
Separately, `get_settings()` is `lru_cache`d but `autonomous_loop.py:2136-2138`
clears it **per ticker**, so `.env` is live for this key **without a restart** --
unusual in this codebase and not to be generalised.

## 2. Criterion 2 -- MET by an ALREADY-COMPLETED post-raise cycle

The raise landed **2026-08-09T13:50Z**. The cycle below started **2026-08-10
20:00:02**, more than a day later, and **completed**:

```
CYCLE  started=2026-08-10 20:00:02.593000  terminal=completed  wall=4532.113s
  screening      : 224.153s
  analysis phase : 4267.658s  (end reason: reached_mark_to_market)
  tickers        : planned=6 dispatched=6 finished=6 unfinished=[]
```

**Wall-clock 4,532.113s.** It did **not** time out.

> I had assumed criterion 2 required waiting for tonight's 20:00 cycle and said so.
> It did not -- a qualifying cycle already existed. Tonight's run adds a **second**
> sample, recorded in the day report, not the only one.

## 3. Criterion 3 -- RE-DERIVED by me with the named script

`scripts/diagnostics/measure_analysis_phase.py`, run against post-raise data:

| quantity | value |
|---|---|
| per-ticker wall | CRWD 961.4 / DELL 1705.5 / HPE 958.1 / HUM 1067.7 / NTAP 1672.8 / PANW 1525.6 |
| **per-ticker mean** | **1,315.2s** |
| median | 1,296.6s |
| serial ticker-seconds | 7,891.1s |
| effective parallelism | **1.85** (cap 3) |
| **projected cycle** | **4,492s** |
| cc_rail | started 152, timed_out 1, **rate 0.0066** |

**SAMPLE-SIZE HONESTY**: my run reports `cycles_with_analysis_phase=1` -- the live
`backend.log` rotated at 08-11 08:41 and holds one cycle. The gate's **n=7** spans
6 rotated archives. **I am not claiming n=7 as my own measurement**; the figures in
this table are from the single cycle I re-derived, and the n=7 distribution is
attributed to the gate.

## 4. Criterion 4 -- ANSWERED: there is NO per-ticker timeout

Verified in source. `autonomous_loop.py:514` is the **only** `asyncio.timeout` and
it wraps the **entire cycle**:

```python
507:  _cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
514:  async with asyncio.timeout(_cycle_timeout):
```

The sole inner cap is a per-call 150s at `claude_code_client.py:593`.

**So a longer budget delays a hung ticker's failure by 3,600s; it does not remove
it.** With effective parallelism 1.85 and a mean of 1,315s/ticker, one wedged
ticker still burns the whole deadline exactly as before.

## 5. Criterion 5 -- #24 RECOMMENDED, #25 deferred. Both as ASKS.

**ASK #24 (rail timeout 150 -> 210): RECOMMENDED, and the data is the argument.**
p90 = 134s and the **longest SUCCESS = 145s** against a **150s cap**. That is a
**censored distribution** by definition -- calls that would have succeeded are being
cut at the cap. Raising a **per-ITEM** cap against censored data is the *endorsed*
remedy in the literature; raising the **global batch deadline** is the rejected one.

**ASK #25 (merged dispatch): DEFERRED, not withdrawn.** Effective parallelism is
1.85 against a cap of 3, so there is real headroom -- but the measured binding
constraint is the rail timeout rate, not dispatch shape, and changing two things at
once would make neither attributable.

## 6. Criterion 6 -- MET

Key-by-key diff against the retained backup `backend/.env.bak.20260809T155016`:
key set **identical**, **exactly one changed value**
(`PAPER_CYCLE_MAX_SECONDS: '7200.0' -> '10800.0'`). `paper_analyze_top_n` is **5**,
confirmed live on the same endpoint, **not lowered**.

## 7. THE CONCLUSION THE STEP ASKED FOR: the raise was the WRONG FIX

The step's second question is *"whether a longer budget is the right fix at all."*
**It is not**, and the evidence is arithmetic rather than rhetorical:

- **Both 7200s overruns PRE-DATE the raise.** No post-raise cycle has overrun.
- The post-raise cycle finished **2,708s INSIDE THE OLD BUDGET**. The budget was
  never the binding constraint for it.
- Overruns track the **rail timeout rate** -- 18.1% and 14.9% on the overrun cycles
  against **0.66%** on the healthy one -- **not batch size**.
- **32 x 150s = 4,800s** of rail-timeout waste against a **1,329s** overrun. The
  waste is 3.6x the overrun it produced.

**The raise (ask #23) treated a symptom of ask #24.** Nothing needs reverting -- a
larger ceiling is harmless when it is not reached -- but the open item is the rail
cap, and closing this step without saying so would be the real failure.

## 8. NEW DEFECT FOUND -- config drift across FOUR sites

One concept, four values:

| site | value |
|---|---|
| `autonomous_loop.py:507` **consumer fallback** | **1800.0** |
| `backend/config/settings.py:33` | 7200.0 |
| `backend/api/settings_api.py:123` | 7200.0 |
| `backend/.env:70` (live) | **10800.0** |
| `scripts/diagnostics/measure_analysis_phase.py:263` `--budget-sec` default | 7200.0 |

**The consumer fallback is the dangerous one**: if `settings` ever lacks the
attribute, the cycle budget silently becomes **30 minutes** -- a sixth of the
authorized value -- and the `getattr` default means that failure is **silent**.

The diagnostic's stale default is the visible one: it printed *"within budget
7200s"* while the live budget was 10800.0. It happened not to change the verdict
here; it would have flipped it for a cycle between 7,200s and 10,800s.

**To be filed as its own step**, not fixed here.

## 9. What is NOT claimed

- **Not** that the budget is now correct -- only that it is live, and that it was
  not the binding constraint on any measured cycle.
- **Not** n=7 as my own measurement (§3).
- **Not** that tonight's cycle will complete; it is a second sample and its outcome
  is reported whatever it is.
