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

**SAMPLE-SIZE HONESTY, CORRECTED.** My run reports
`cycles_with_analysis_phase=1` -- the live `backend.log` rotated **2026-08-10
08:41** and holds one cycle. The gate's **n=7** is that cycle plus **ONE rotated
archive** (`backend.log.20260810T064130Z.gz`) holding **6 further cycles**.

> An earlier revision said the gate's n=7 "spans **6 rotated archives**". Wrong: six
> *cycles* in **one** archive. Six archives do exist in `handoff/logs/`, which is
> exactly what made the misstatement look checkable. The rotation date was also
> wrong (08-11, actually 08-10). **A step being careful to attribute a figure to the
> gate has to describe the gate's evidence correctly, or the attribution is itself
> unverifiable.**

I am not claiming n=7 as my own measurement; the table above is the single cycle I
re-derived.

## 4. Criterion 4 -- ANSWERED: there is NO per-ticker timeout

> **PATH DISAMBIGUATION**: two files are named `autonomous_loop.py` --
> `backend/autonomous_loop.py` and `backend/services/autonomous_loop.py`. **Every
> line number in this artifact resolves against `backend/services/`**; the
> top-level file carries unrelated code at those lines.

Verified in source. **`backend/services/autonomous_loop.py:514`** is the **only**
`asyncio.timeout` and
it wraps the **entire cycle**:

```python
507:  _cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
514:  async with asyncio.timeout(_cycle_timeout):
```

The sole inner cap is a per-call 150s at `claude_code_client.py:593`.

**So a longer budget delays a hung ticker's failure by 3,600s; it does not remove
it.** With effective parallelism 1.85 and a mean of 1,315s/ticker, one wedged
ticker still burns the whole deadline exactly as before.

## 5. Criterion 5 -- #24 RECOMMENDED, #25 DEFERRED. Both as ASKS.

**ASK #24 (rail timeout 150 -> 210): RECOMMENDED -- but read the provenance first.**

> **THE DECISIVE FIGURES ARE PRE-FIX AND I PRESENTED THEM AS CURRENT.** p90 = 134s
> and longest-success = 145s trace to **`research_brief_85.4.md:321`**, dated to
> phase-85.4. They **cannot be re-derived from the post-fix window**:
> `measure_analysis_phase.py` computes `p90_s` and `n_within_5s_of_150s_cap`
> (`:249/:251`), but both my run and the Q/A's print **`agent latency : None`** for
> the 08-10 cycle. Criterion 5 asks for post-fix data, and on this leg I do not have
> it.

**AND THE POST-FIX DATUM THAT DOES EXIST CUTS AGAINST URGENCY**: 1 timeout in 152
calls, **0.66%**. On that night alone, #24 would have changed almost nothing.

**Why I still recommend it:** the 0.66% night is one sample, and five other measured
cycles ran **9.9%-23.4%**. The honest case for #24 is not "the last cycle was bad" --
it is that the rate is **highly variable** and the cap sits **5s above the longest
observed success**, so on a bad night the cap censors work that would have completed.
That is the argument, and it rests on the pre-fix distribution, which I now say
plainly.

**ASK #25 (merged dispatch): DEFERRED -- and "deferred" is a third value against a
criterion worded "recommended or withdrawn", so let me be unambiguous: NOT
recommended now, NOT withdrawn.** Effective parallelism is 1.85 against a cap of 3,
so headroom exists, but the measured binding constraint is the rail rate. Changing
dispatch shape and the rail cap together would make neither attributable. Revisit
**after** #24 lands.

## 6. Criterion 6 -- MET

Key-by-key diff against the retained backup `backend/.env.bak.20260809T155016`:
key set **identical**, **exactly one changed value**
(`PAPER_CYCLE_MAX_SECONDS: '7200.0' -> '10800.0'`). `paper_analyze_top_n` is **5**,
confirmed live on the same endpoint, **not lowered**.

## 7. THE CONCLUSION THE STEP ASKED FOR -- restated after the cycle-1 Q/A

> **AN EARLIER REVISION SAID FLATLY "IT IS NOT [the right fix]" AND OMITTED THE
> ARITHMETIC THAT MOST DIRECTLY REBUTS THAT.** The two overrun cycles project to
> **8,554s and 8,529s** (`research_brief_86.9.md:397`) -- **both fit inside the new
> 10,800s budget with ~2,250s to spare.** So the raise **would have converted both
> observed failures into completions**. Those figures were in the brief I
> commissioned; `grep` over my own artifacts returned zero hits for them. I had the
> counter-evidence and did not carry it.
>
> The flat form was also the dangerous one: **"the raise was the WRONG fix" is the
> one framing that could invite reverting an operator-authorised value.**

**THE ACCURATE ANSWER, both halves true:**

**(a) The raise IS an effective mitigation for the observed overrun magnitude.**
8,554s and 8,529s both land inside 10,800s. Had it been in force, neither cycle
would have been cut off, and each would have analysed the ticker it dropped.

**(b) It is aimed at the WRONG CAUSAL TARGET.** The overruns were produced by rail
timeouts, not by batch size:

- overrun cycles ran a **9.9%-23.4%** rail-timeout rate; the healthy one ran **0.66%**
- **32 x 150s = 4,800s** of rail-timeout waste against a **1,329s** overrun -- the
  waste is **3.6x** the problem it caused
- the post-raise cycle finished **2,708s inside the OLD budget**, so the budget was
  not its binding constraint

**So the honest reading is: ask #23 buys headroom that works, while ask #24 addresses
the thing generating the need for headroom.** Nothing should be reverted -- an
unreached ceiling is harmless and this one is operator-authorised.

**AND THE POST-RAISE EVIDENCE IS n=1**, on what was the healthiest rail night in the
measured set. One completion under a raised ceiling, on the quietest night, is weak
evidence that the ceiling is right-sized. Tonight's cycle is a second sample.

## 8. NEW DEFECT FOUND -- config drift, population DERIVED

> **CORRECTED.** An earlier revision said "across FOUR sites" above a table with
> **five** rows, and I propagated that undercount into 86.53's `audit_basis`. The
> count was typed, not derived. Below is the output of
> `grep -rn "paper_cycle_max_seconds|_CYCLE_BUDGET_FALLBACK_SEC" backend/ scripts/`.

| site | value / role |
|---|---|
| `backend/config/settings.py:33` | `Field(7200.0, ...)` |
| `backend/api/settings_api.py:123` | `= 7200.0` |
| `backend/api/settings_api.py:171` | validation bounds `ge=300.0, le=21600.0` |
| `backend/api/settings_api.py:308` | env-name mapping (phase 38.12) |
| `backend/api/settings_api.py:383` | `getattr(s, ..., 7200.0)` |
| **`backend/services/autonomous_loop.py:507`** | **`getattr(settings, ..., 1800.0)`** |
| **`backend/services/cycle_lock.py:63`** | **`_CYCLE_BUDGET_FALLBACK_SEC = 7200.0`** |
| `backend/services/cycle_lock.py:82,84,86` | three separate returns of that fallback |
| `scripts/diagnostics/measure_analysis_phase.py:263` | `--budget-sec default=7200.0` |
| `backend/.env:70` | **10800.0 (live)** |

**`cycle_lock.py` was missing from my table entirely** -- and its own comment at
`:57` already documents the drift: *"paper_cycle_max_seconds (1800s) ... while the
budget in force had moved to"*. Someone had already noticed and left a note.

**The consumer fallback remains the hazard**: a missing attribute silently yields a
**30-minute** cycle budget, a sixth of the authorised value, with no error or alert.

Filed as **86.53**, whose criterion 1 requires a grep-derived enumeration precisely
so the executor does not inherit a typed count.

## 9. What is NOT claimed

- **Not** that the budget is now correct -- only that it is live, and that it was
  not the binding constraint on any measured cycle.
- **Not** n=7 as my own measurement (§3).
- **Not** that tonight's cycle will complete; it is a second sample and its outcome
  is reported whatever it is.
