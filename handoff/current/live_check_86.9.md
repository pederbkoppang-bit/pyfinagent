# live_check -- step 86.9

Captured 2026-08-11 by Main. All output verbatim from execution against the
RUNNING system (backend pid 66306).

## 1. Criterion 1 -- the budget read from the RUNNING PROCESS
```
$ curl -s http://127.0.0.1:8000/api/settings/
  paper_cycle_max_seconds = 10800.0
  paper_analyze_top_n = 5
  paper_screen_top_n = 10
```

Not from .env, not from a fresh interpreter. The endpoint is served by pid 66306.

## 2. Criteria 2 + 3 -- the post-raise cycle, re-derived
```
$ python scripts/diagnostics/measure_analysis_phase.py
log=backend.log  lines_parsed=73231  cycles_with_analysis_phase=1
budget_sec=7200

==============================================================================
CYCLE  started=2026-08-10 20:00:02.593000  terminal=completed  wall=4532.113s
  screening        : 224.153s
  analysis phase   : 4267.658s  (end reason: reached_mark_to_market)
  tickers          : planned=6 dispatched=6 finished=6 unfinished=[]
  concurrency cap  : 3
  per-ticker wall  : {'CRWD': 961.4, 'DELL': 1705.5, 'HPE': 958.1, 'HUM': 1067.7, 'NTAP': 1672.8, 'PANW': 1525.6}
  per-ticker mean  : 1315.2s  median=1296.6s
  serial ticker-s  : 7891.1s   effective parallelism=1.85
  PROJECTED analysis if all dispatched tickers finished : 4268s
  PROJECTED cycle   (screening + analysis)              : 4492s  vs budget 7200s
  VERDICT          : within budget (delta -2708s)
  cc_rail calls    : started=152 timed_out=1 rate=0.0066 subprocess_timeout_s=150
  agent latency    : None
```

The raise landed 2026-08-09T13:50Z; this cycle started 2026-08-10 20:00:02 and
COMPLETED. Note the script's own budget_sec=7200 is its stale default, not the
live 10800.0 -- see section 8 of experiment_results.

## 3. Criterion 4 -- the only asyncio.timeout wraps the WHOLE cycle
```
426:    # wrapped in `async with asyncio.timeout(...)` -- a timeout mid-cycle
507:    _cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
509:        # phase-23.2.18: outer asyncio.timeout ceiling so a stuck
514:        async with asyncio.timeout(_cycle_timeout):
```

No per-ticker cap exists. **CORRECTED: there is not one inner cap, there are
several** -- my "sole" was a citation I had not derived:

```
claude_code_client.py:302   timeout_s: int = 120            <- this client's subprocess default
claude_code_client.py:591   recommended_step_timeout = 150  <- deliberately ABOVE the 120s
claude_code_client.py:593   def __init__(..., timeout_s: int = 150)
orchestrator.py:398         timeout = 180
orchestrator.py:1118/:1135  httpx timeout=900 / 300         <- HTTP client, not a step cap
```

**None of them is per-TICKER**, so criterion 4's answer is unchanged -- but the
count was wrong.

## 4. Criterion 6 -- exactly one setting changed
```
$ diff (backup) (current), values compared key-by-key
  keys added/removed : 0
  keys changed       : 1
    PAPER_CYCLE_MAX_SECONDS: '7200.0' -> '10800.0'
```

## Criterion 1's OTHER half: the pid's START TIME -- and what it exposes

```
$ ps -o pid=,lstart=,etime= -p 66306        # -o without -e; `ps -e` overrides -p
  66306   man. 10 aug. 21.33.01 2026   18:35:23
$ lsof -nP -iTCP:8000 -sTCP:LISTEN  ->  listener pid: 66306
$ curl -s http://127.0.0.1:8000/api/settings/  ->  paper_cycle_max_seconds = 10800.0
```

**pid 66306 started 2026-08-10 21:33:01 CEST.** The criterion asks for the start
time *"since the setting is read at cycle start"*, and the reason is now concrete:

| event | time | source |
|---|---|---|
| qualifying cycle START | 2026-08-10 20:00:02 | cycle line below |
| qualifying cycle END | 2026-08-10 21:15:34 | start + 4532.113s |
| **pid 66306 START** | **2026-08-10 21:33:01** | `ps -o lstart=` |

**The process now serving 10800.0 came up 1,046s AFTER the qualifying cycle
finished. A PREDECESSOR process ran that cycle, and it is gone.**

The restart was **not** the watchdog: `backend-watchdog.log` never reaches 3/3 and
its last entry is 2026-08-10T18:07:04Z. I do not know what caused it and am not
guessing.

**Can I recover the budget that predecessor held? YES -- and my first answer here
was NO, which was wrong.**

The predecessor is identifiable. `grep -c "Application startup complete" backend.log`
returns **exactly 1** (21:33:04, pid 66306), and the archive's **last** startup is
`Started server process [43839]` at **2026-08-09 22:11:55**, with **no startup
between it and the cycle**:

```
2026-08-09 17:08:08   Started server process [84494]
2026-08-09 18:56:03   Started server process [6644]
2026-08-09 22:11:55   Started server process [43839]   <- ran the 08-10 20:00 cycle
2026-08-10 21:33:04   Started server process [66306]   <- current, started AFTER it
```

**pid 43839 started 6h21m AFTER the `.env` write** (15:50 CEST, corroborated by the
backup stamp `.bak.20260809T155016`). A freshly-started process builds `Settings`
from `backend/.env` on its first `get_settings()`; `_scheduled_run`
(`paper_trading.py:1485-1487`) calls it at fire time and hands that object to
`run_daily_cycle`, which uses it at `:406` and reads `paper_cycle_max_seconds` at
`:507`. **So the cycle ran under 10800.0, as a measurement.**

**What IS true**: there is no cycle-**START** budget record. The value is logged only
on the **timeout** path (`autonomous_loop.py:1896`), which emitted three `Paper
trading cycle TIMED OUT after 7200s` records on 2026-08-04/06/07. **My blanket
"never logged" was wrong**; a failure-only record is poor observability, not absence.
86.54 stands on the narrower ground.
