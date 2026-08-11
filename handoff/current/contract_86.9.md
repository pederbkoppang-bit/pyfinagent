# Contract -- step 86.9

**Step**: `86.9` (phase-86, **P1**, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~16:3x CEST, read from `date`) | **Driver**: Main (`pyfinagent-06`)
**Written BEFORE any code.** No production file is modified at this moment.

---

## 1. Research gate

**PASSED** -- `wf_6f5558d5-56b`, tier `moderate`. Script-enforced and recomputed:
**8 sources read in full** (floor 5), **21 URLs** (floor 10), recency scan present,
all 8 claimed URLs verified in the brief, `brief_status: COMPLETE`,
`rail_dropped: null`, 14 internal files inspected.

Sources: Google SRE (cascading failures, handling overload, configuration design),
gRPC deadlines, Prometheus histograms, Azure Bulkhead + Circuit Breaker, Spring Boot
Actuator endpoints.

> **DISCLOSED GAP, and it is a real one.** The session-shared **WebSearch budget was
> exhausted 200/200 before this gate spawned**, so the mandatory
> **three-variant search discipline could NOT run**. Recency was established from
> revision metadata instead (2 in-window findings). The external half of this gate
> is therefore narrower than the tier implies; the load-bearing findings below are
> **internal measurements**, which I re-derived.

## 2. MY OWN CRITERION-1 CLAIM WAS WRONG, AND THE GATE REFUTED IT

I reported that **no endpoint exposes the budget** and that criterion 1 might be
unsatisfiable. **False.** `GET /api/settings/` has exposed it since step 38.12
(`settings_api.py:123/:171/:308/:383`). Verified by me against the running process:

```
$ curl -s http://127.0.0.1:8000/api/settings/
  paper_screen_top_n      = 10
  paper_analyze_top_n     = 5
  paper_cycle_max_seconds = 10800.0     <- LIVE from pid 66306
```

**I probed `/api/settings` without the trailing slash, got an empty response, and
concluded the endpoint did not exist.** An empty result from an incomplete probe is
not evidence of absence -- the same "suspect the clean check" failure, inverted.

### Two corrections that follow, and one caveat that matters

- **`.env` is LIVE WITHOUT A RESTART for this setting.** `get_settings()` is
  `lru_cache`d, but `autonomous_loop.py:2136-2138` **clears the cache per ticker**.
  My "committed but not in force" framing was wrong here. That is a genuinely
  unusual property in this codebase and worth not generalising from.
- **The endpoint reports the NEXT cycle's budget, not the running one.**
  `_cycle_timeout` is captured once at `autonomous_loop.py:507`. So a mid-cycle read
  describes what the *following* cycle will use.
- **CONFIG DRIFT, unresolved**: the consumer falls back to **1800.0** while
  `settings.py:33` / `settings_api.py:123` carry **7200.0**. Three different numbers
  for one concept.

## 3. THE HEADLINE: THE RAISE WAS THE REJECTED FIX, AND THE DATA SAYS SO

Measured over **n=7 cycles** (the gate re-ran `measure_analysis_phase.py` post-raise):

| finding | value |
|---|---|
| cycles that overran 7200s | **2 -- BOTH PRE-RAISE**, by ~1330s |
| the post-raise cycle | **4,532s** -- i.e. **2,708s INSIDE the OLD budget** |
| what overruns track | the **rail timeout rate** (18.1% / 14.9% vs a 0.66% baseline) -- **not batch size** |
| rail-timeout waste | 32 x 150s = **4,800s**, against a 1,329s overrun |

**The literature is unambiguous and this project applied the wrong half.** Raising a
**per-ITEM** cap against a censored distribution is endorsed; raising the **GLOBAL
batch deadline** is the rejected anti-pattern (Google SRE cascading-failures,
deadline propagation). **Ask #23 shipped the rejected one. Ask #24 -- the endorsed
one -- is still open**, and the numbers make its case: **p90 = 134s and the longest
SUCCESS = 145s against a 150s cap**, which is a censored distribution by definition.

## 4. Immutable success criteria -- VERBATIM from `.claude/masterplan.json`

> **CORRECTED after the cycle-2 Q/A, and this was the blocker.** The previous
> revision of this section was headed *"VERBATIM"* while **5 of the 6 criteria
> differed from the masterplan**, two of them materially. **The block below is now
> generated programmatically from `.claude/masterplan.json` rather than typed**, so
> it cannot drift again.
>
> **The most damaging drift was in criterion 1.** The masterplan requires *"...and
> record the pid and its start time, since the setting is read at cycle start"*. My
> copy replaced that with *", not from .env or a new import"* -- **dropping the exact
> clause whose measurement produced this step's most important finding.** Had I
> worked from my own copy, I would never have measured the start time, never found
> that a predecessor ran the qualifying cycle, and never traced it to pid 43839. The
> criterion was carrying information I deleted before reading it.
>
> Criterion 3 said *"AFTER the raise"* where the masterplan says *"AFTER the rail was
> repaired 2026-08-09"* -- **a different qualifying event**. Criterion 4 substituted a
> *latency* question ("merely delays the same failure") for the masterplan's
> *detection* question ("increases the window in which a hang goes UNNOTICED"), and
> §4 of the results answered only the substituted one. Criteria 2 and 5 each lost a
> qualifying clause.
>
> **Cycle 1 did not catch this**: its check verified the masterplan SOURCE was
> unamended, which is a different proposition from verifying that my COPY matches it.
> An unamended source and a faithful copy are independent facts and need independent
> checks.

> 1. the effective running value is 10800.0 in the BACKEND PROCESS, not merely in a fresh interpreter -- read it from the running process (or an endpoint it serves) and record the pid and its start time, since the setting is read at cycle start
> 2. at least one cycle completes end-to-end under the new budget and its wall-clock is recorded; if it still times out, the step reports that the raise was INSUFFICIENT rather than closing on the config change alone
> 3. the measured per-ticker mean and projected total are RE-DERIVED with scripts/diagnostics/measure_analysis_phase.py against cycles run AFTER the rail was repaired 2026-08-09 -- the 2310-2320s figure predates that fix and may no longer hold
> 4. the hung-cycle caveat is addressed explicitly: state whether _run_single_analysis still lacks an inner per-ticker timeout, and if so whether a longer outer budget increases the window in which a hang goes unnoticed
> 5. asks #24 (rail timeout 150 -> 210) and #25 (merged dispatch) are each re-evaluated against post-fix data and explicitly recommended or withdrawn -- a budget raise that leaves 26% of rail time being discarded is treating the symptom
> 6. no other setting changed; paper_analyze_top_n is NOT lowered; the .env backup is retained and referenced
## 5. What is already measured

- **Criterion 1 -- MET** (§2): 10800.0 read live from pid 66306 via `/api/settings/`.
- **Criterion 6 -- MET**: key-by-key diff against `backend/.env.bak.20260809T155016`
  shows the key set identical and **exactly one changed value**,
  `PAPER_CYCLE_MAX_SECONDS: '7200.0' -> '10800.0'`. `paper_analyze_top_n` is
  untouched at 5, confirmed live on the same endpoint.
- **Criterion 4 -- ANSWERED**: there is **no per-ticker timeout**. One
  `asyncio.timeout` at `autonomous_loop.py:514` wraps the entire cycle; the only
  inner cap is a per-call 150s at `claude_code_client.py:593`. So a single hung
  ticker still consumes the whole budget -- **a longer budget delays that failure by
  3600s, it does not remove it.**

## 6. Plan

**P1 -- CRITERION 2 IS TONIGHT'S CYCLE.** The 20:00 CEST run supplies the
end-to-end wall-clock. I will observe and record it, not simulate it. **If it
overruns, the step reports the raise did not fix it** -- that outcome is explicitly
permitted by the criterion and I will not soften it.

**P2 -- CRITERION 5: RECOMMEND #24, RE-EVALUATE #25.** The censored-distribution
evidence (p90 134s, max-success 145s, cap 150s) is exactly the endorsed
per-item-cap case. I will recommend it **as an ask, not a change** -- it is a
timeout on the live analysis rail.

**P3 -- SAY PLAINLY THAT THE RAISE TREATED A SYMPTOM.** Both overruns predate it;
the post-raise cycle finished 2,708s inside the *old* budget. The honest reading is
that the budget was never the binding constraint -- the rail timeout rate was.

**P4 -- FILE THE CONFIG DRIFT.** 1800.0 vs 7200.0 vs 10800.0 across consumer,
settings and API is its own defect and gets its own step rather than a footnote.

**P5 -- CARRY THE GATE'S OWN GAP FORWARD.** The three-variant search discipline did
not run. Any future step leaning on this brief's external half must know that.

### Explicitly NOT doing

- **Not** changing any timeout, including the rail 150s (criterion 5 says
  *recommend*, and the standing goal forbids unasked config change).
- **Not** writing `backend/.env`. **Not** lowering `paper_analyze_top_n`.
- **Not** restarting the backend to test anything -- restarts batch to session end,
  and the cycle runs at 20:00.

### Risk

This step touches the live analysis rail's budget on the day of a cycle. The
mitigation is that it **changes nothing** -- every finding is a measurement and
every recommendation is an ask.

## 7. References

- `handoff/current/research_brief_86.9.md` (gate `wf_6f5558d5-56b`)
- Google SRE: addressing cascading failures, handling overload, configuration
  design; gRPC deadlines; Prometheus histograms; Azure Bulkhead + Circuit Breaker
- `backend/services/autonomous_loop.py:507,514,2136-2138`;
  `backend/api/settings_api.py:123,171,308,383`;
  `backend/agents/claude_code_client.py:593`; `backend/config/settings.py:33`;
  `backend/.env.bak.20260809T155016`
