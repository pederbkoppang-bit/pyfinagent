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

## 2. Criterion 2 -- MET, and the evidence is STRONGER than I claimed

The raise landed **2026-08-09T13:50Z (15:50 CEST)**. The cycle below started
**2026-08-10 20:00:02** and **completed**:

```
CYCLE  started=2026-08-10 20:00:02.593000  terminal=completed  wall=4532.113s
  screening      : 224.153s
  analysis phase : 4267.658s  (end reason: reached_mark_to_market)
  tickers        : planned=6 dispatched=6 finished=6 unfinished=[]
```

> **I WROTE AN UNDER-CLAIM AND THE Q/A REFUTED IT. Both of these were FALSE:**
> *"the value in force is unrecoverable post-hoc"* and *"`_cycle_timeout` is **never**
> logged"*. I asked for that finding to be graded hardest and it came back inverted --
> not "you over-claimed" but **"you stopped one seam short of the query that settles
> it."** Re-derived by me:
>
> - `grep -c "Application startup complete" backend.log` -> **exactly 1**
>   (2026-08-10 21:33:04, pid 66306) -- so nothing in the live log covers the cycle.
> - The archive's **last** startup is `Started server process [43839]` at
>   **2026-08-09 22:11:55**, with **no startup between it and the cycle**.
>
> **So pid 43839 ran the 20:00 cycle -- and it started 6h21m AFTER the `.env` write**
> (corroborated independently by the backup stamp `.bak.20260809T155016` = 15:50
> CEST). A freshly-started process constructs `Settings` from `backend/.env` on its
> first `get_settings()`; `_scheduled_run` (`paper_trading.py:1485-1487`) calls it at
> fire time and passes that object into `run_daily_cycle`, whose `:406` uses it and
> `:507` reads `paper_cycle_max_seconds` from it.
>
> **The predecessor held 10800.0. That is now MEASURED, not inferred.**

**The claim-strength table, corrected:**

| claim | strength |
|---|---|
| a cycle completed end-to-end, wall 4,532.113s | **MEASURED** |
| it did not time out | **MEASURED** |
| the running process serves 10800.0 (pid 66306, up 21:33:01) | **MEASURED** |
| **the process that RAN the cycle (pid 43839) held 10800.0** | **MEASURED** -- was INFERRED |

**What remains true, and it is the narrower point 86.54 rests on:** the budget is
logged **only on the timeout path** (`autonomous_loop.py:1896`, which produced three
`Paper trading cycle TIMED OUT after 7200s` records on 2026-08-04/06/07). There is
**no cycle-START budget record**. A failure-only log is not observability -- but it is
not *nothing*, and my bolded "never logged" was wrong.

**Those three 7200s timeout records also independently corroborate that both pre-raise
overruns ran under the 7,200s budget** -- direct evidence for §7 that I had in the
archive and never used.

**A 4,532s cycle would still have completed under the old 7,200s budget**, so it does
not demonstrate that the *headroom* was needed. But criterion 2 does not ask that; it
asks that a cycle complete under the new budget with its wall-clock recorded, and
that is now established by measurement at both ends.

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

**The inner caps, corrected -- there is more than one and my "sole" was wrong:**

| site | value | scope |
|---|---|---|
| `claude_code_client.py:302` | `timeout_s: int = 120` | this client's own subprocess default |
| `claude_code_client.py:591` | `recommended_step_timeout = 150` | per-step budget, deliberately ABOVE the 120s so the CLI fails first and is retried |
| `claude_code_client.py:593` | `def __init__(..., timeout_s: int = 150)` | the instance default I originally cited |
| `orchestrator.py:398` | `timeout = 180` | per-step |
| `orchestrator.py:1118 / :1135` | `httpx` 900 / 300 | HTTP client, not a step cap |

**None of them is a per-TICKER timeout**, which is why criterion 4's answer is
unchanged -- but "the sole inner cap is 150s" was a citation I did not derive.

**So a longer budget delays a hung ticker's failure by 3,600s; it does not remove
it.** With effective parallelism 1.85 and a mean of 1,315s/ticker, one wedged
ticker still burns the whole deadline exactly as before.

### The half of criterion 4 I had not answered: does a hang go UNNOTICED for longer?

> The masterplan asks whether a longer outer budget *"increases the window in which a
> hang goes **UNNOTICED**"* -- a **detection** question. My contract had substituted a
> **latency** question, and §4 answered only that. Restored and answered:

**MEASURED ANSWER: yes, but by 3.8%, and the dominant clock is unaffected.**

`backend/services/cycle_health.py` runs two staleness clocks:

| clock | value | measured from |
|---|---|---|
| `_CYCLE_HEARTBEAT_STALE_SEC` | **93,600s (26h)** | cycle **START** |
| `_CYCLE_COMPLETED_STALE_SEC` | **345,600s (96h / 4 days)** | last **COMPLETED** cycle |

The raise moves a hung cycle's failure from 7,200s to 10,800s -- **+3,600s (1h)**,
which is **3.8%** of the 26h heartbeat clock and **1.0%** of the 96h completed clock.

**And the second clock is the one that matters here, for a reason phase-85.4 wrote
down in the source**: the heartbeat is measured from cycle START, so *"a cycle that
times out every single day resets the heartbeat age to ~0 daily and
`cycle_heartbeat_alarm` never goes stale"* -- which the comment says *"is exactly what
happened"*. The completed-age clock exists precisely to catch that, and **a longer
budget does not move it at all**, because it is keyed to completions, not to how long
each attempt is allowed to run.

**So the honest answer to the restored criterion 4 is: the unnoticed window grows by
one hour against detection thresholds measured in days.** That is a real but minor
cost, and it is bounded by a mechanism that was built for this exact failure.

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

**Why I still recommend it, with the population DERIVED this time:** of the six
cycles other than #7, **FOUR ran 9.9%-23.4%** (#1 23.4%, #3 18.1%, #2 14.9%, #6 9.9%)
and **two ran 0.0%** (#4 and #5) -- and those two are not comparable cycles: **340.4s
and 322.1s wall, with 20 and 33 rail calls**, against 124-177 calls in the others.

> **CORRECTED: I had written "five other measured cycles".** It is four. Worse, I
> **adopted that figure from the cycle-2 critique without re-deriving it** -- taking a
> reviewer's number at face value is the same failure as taking my own on faith, and
> it overstated prevalence 4/6 -> 5/6 in the direction supporting my recommendation.

The honest case for #24 is not "the last cycle was bad" -- it is that the rate is
**highly variable across comparable cycles** and the cap sits **5s above the longest
observed success**, so on a bad night it censors work that would have completed.
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

**(b) It is aimed at a DIFFERENT CAUSAL TARGET than batch size -- but the evidence is
weaker than I twice claimed, and the honest version contains a counterexample to my
own thesis.**

> **CORRECTED after the cycle-3 FAIL. All three bullets below were wrong, in the
> direction that flattered the argument.** Every figure is now read off the table at
> `research_brief_86.9.md:380-388`:
>
> | # | terminal | wall / projected | rail rate |
> |---|---|---|---|
> | 1 | (none) | 5,670s projected | **0.2339** |
> | 2 | **timeout** | 7,200.117s | **0.1486** |
> | 3 | **timeout** | 7,200.077s | **0.1808** |
> | 4 | completed | 340.4s | 0.0 |
> | 5 | completed | 322.1s | 0.0 |
> | 6 | completed | 5,942.7s | 0.0988 |
> | 7 | completed | 4,532.1s | 0.0066 |

- **The two cycles that actually overran ran 14.9% and 18.1%** -- not the
  "9.9%-23.4%" I wrote. That range's endpoints are cycles **#6 and #1, neither of
  which overran.** My own contract at §3 states the correct pair; the results section
  widened it.
- **AND THE WIDENING CONCEALED A COUNTEREXAMPLE TO MY OWN CLAIM.** Cycle **#1 carries
  the HIGHEST rail-timeout rate in the entire set (23.4%) and did NOT overrun** --
  6/6 finished, 5,670s projected. So "overruns are produced by rail timeouts" is
  **not** supported as a sufficient condition: the worst rail night in the set
  completed comfortably. The defensible claim is narrower -- **both overruns coincided
  with elevated rail-timeout rates, and the two cycles that overran were also the only
  two that dropped a ticker (5/6, NTAP both times)**.
- **The waste multiple is ~1.95x, not 3.6x.** `32 x 150s = 4,800` is **subprocess**
  seconds; the 1,329s overrun is **wall** seconds. Dividing them compares different
  units. At the measured parallelism of 1.85, 4,800 subprocess-seconds is **~2,595s
  of wall time**, giving **1.95x**. The brief performed this conversion *and* stated a
  caveat against its own interest; **I carried the number forward and dropped both.**
  The conclusion survives -- even at 1.95x the recovered time clears the 1,329s
  overrun -- but the stated multiple did not reproduce.
- The post-raise cycle finished **2,708s inside the OLD budget**, so the budget was
  not its binding constraint. *(This bullet was correct.)*

**So the honest reading is: ask #23 buys headroom that works, while ask #24 addresses
the thing generating the need for headroom.** Nothing should be reverted -- an
unreached ceiling is harmless and this one is operator-authorised.

**AND THE POST-RAISE EVIDENCE IS n=1**, on what was the healthiest rail night in the
measured set. One completion under a raised ceiling, on the quietest night, is weak
evidence that the ceiling is right-sized. Tonight's cycle is a second sample.

## 8. NEW DEFECT FOUND -- config drift. Census REGENERATED, not curated.

> **CORRECTED TWICE, AND THE SECOND CORRECTION IS THE INSTRUCTIVE ONE.** Cycle 1
> caught that "FOUR sites" was **typed** above a five-row table. My fix replaced it
> with a table captioned as the output of a `grep`. **That caption was false in two
> independent ways**, and cycle 2 caught both:
>
> 1. **Run literally as I published it, the command returns ZERO rows.** I wrote it
>    without `-E`, and basic `grep` treats `|` as a literal character -- so it
>    searched for a string containing a pipe. The command I published could not have
>    produced any table at all.
> 2. **The table was still curated.** Against the real output the difference is
>    non-empty in BOTH directions: it dropped eight rows and contained two the grep
>    cannot produce.
>
> **So I replaced a typed count with a curated table wearing a derivation's label --
> the same defect one level up, dressed as its own fix.** Below is the command's
> actual stdout, captured to a file and pasted unedited.
>
> **A THIRD THING SURFACED WHILE PROVING THE PASTE MATCHES, AND IT AFFECTS EVERY
> "CAPTURED OUTPUT" IN THIS PROJECT.** My first equality check failed with a
> symmetric difference of 8 -- because `grep` in this shell is a **function wrapping
> `ugrep`**, while `subprocess.run(["grep", ...])` resolves `/usr/bin/grep` (BSD grep
> 2.6.0-FreeBSD). Same pattern, same paths, **different programs**: 18 rows vs 26,
> the extra 8 being `Binary file ... .pyc matches` notices. Neither was wrong; they
> are different tools.
>
> **So a command published without naming its binary is not a reproducible
> derivation** -- a reader running it gets a different answer than the one printed
> above it, and would reasonably conclude the artifact was curated. The command is
> therefore pinned to `/usr/bin/grep` with `-I`, which yields **18 on both
> implementations**.

```console
$ /usr/bin/grep -rnIE "paper_cycle_max_seconds|_CYCLE_BUDGET_FALLBACK_SEC" backend/ scripts/
backend/config/settings.py:33:    paper_cycle_max_seconds: float = Field(7200.0, description="phase-34.2 corrective + cycle-7 (38.12) bump: hard wall-clock budget for one autonomous paper-trading cycle. Read by backend/services/autonomous_loop.py:219 via asyncio.timeout. Default raised from 1800 -> 7200 (2h) because cycle 6 (2026-05-26) found the Claude Code CLI rail (paper_use_claude_code_route=True; ~30s per claude_code_invoke) + serial enrichment-debate-risk-synthesis dependencies push a 13-ticker full-orchestrator cycle past 3600s. Cycle 6 timed out with 7 of 13 tickers analyzed; 7200s gives headroom for the full 13 + Step 6-9 (trade decide / execute / snapshot / outcome). When `paper_use_claude_code_route=False` AND Anthropic-direct rail is available, the lower 1800s remains adequate -- operator can lower via Settings UI.")
backend/tests/test_phase_85_4_cycle_loudness.py:244:    s = _settings().model_copy(update={"paper_cycle_max_seconds": 10.0})
backend/tests/test_phase_85_5_cycle_lock_split_brain.py:356:        self.paper_cycle_max_seconds = budget
backend/tests/test_phase_85_5_cycle_lock_split_brain.py:363:    settings.paper_cycle_max_seconds moved to 7200s, so the TTL became 0.75x
backend/tests/test_phase_85_6_anchor_deadlock.py:374:        autonomous_loop.run_daily_cycle(settings=_settings(paper_cycle_max_seconds=10.0))
backend/tests/test_phase_38_6_restart_survivable.py:161:# constant froze at 5400s while settings.paper_cycle_max_seconds moved to
backend/api/settings_api.py:123:    paper_cycle_max_seconds: float = 7200.0
backend/api/settings_api.py:171:    paper_cycle_max_seconds: Optional[float] = Field(None, ge=300.0, le=21600.0)
backend/api/settings_api.py:308:    "paper_cycle_max_seconds": "PAPER_CYCLE_MAX_SECONDS",  # phase-cycle-7 (38.12)
backend/api/settings_api.py:383:        paper_cycle_max_seconds=float(getattr(s, "paper_cycle_max_seconds", 7200.0)),
backend/services/autonomous_loop.py:507:    _cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
backend/services/cycle_lock.py:28:- The TTL is derived at call time from settings.paper_cycle_max_seconds
backend/services/cycle_lock.py:57:# paper_cycle_max_seconds (1800s)" while the budget in force had moved to
backend/services/cycle_lock.py:63:_CYCLE_BUDGET_FALLBACK_SEC = 7200.0
backend/services/cycle_lock.py:82:        value = float(getattr(get_settings(), "paper_cycle_max_seconds",
backend/services/cycle_lock.py:83:                              _CYCLE_BUDGET_FALLBACK_SEC))
backend/services/cycle_lock.py:84:        return value if value > 0 else _CYCLE_BUDGET_FALLBACK_SEC
backend/services/cycle_lock.py:86:        return _CYCLE_BUDGET_FALLBACK_SEC
$ # 18 rows
```

**Eight rows the curated table had dropped**: `cycle_lock.py:28,:57,:83` plus five
test sites (`test_phase_38_6_restart_survivable.py:161`,
`test_phase_85_4_cycle_loudness.py:244`,
`test_phase_85_5_cycle_lock_split_brain.py:356,:363`,
`test_phase_85_6_anchor_deadlock.py:374`). **The tests matter**: they PIN the drift,
so changing it is a test-visible change rather than a silent one.

**Two rows I had listed that this command cannot produce, removed rather than quietly
kept:**

- `scripts/diagnostics/measure_analysis_phase.py:263` -- **that file contains the
  token zero times**. Its `:263` is `ap.add_argument("--budget-sec", type=float,
  default=7200.0)`: a related 7200.0 default under a *different name*. Worth knowing,
  not a hit for this census.
- `backend/.env:70` -- the live value, outside the searched paths.

**The defect is unchanged and the regenerated census confirms it**: one concept,
several values -- `settings.py:33` and `settings_api.py:123` at **7200.0**,
`autonomous_loop.py:507` falling back to **1800.0**, `cycle_lock.py:63` at **7200.0**,
the live value **10800.0**.

**The consumer fallback is the hazard**: a missing attribute silently yields a
**30-minute** budget -- a sixth of the authorised value -- with no error or alert.
`cycle_lock.py:57`'s own comment already documents the drift.

Filed as **86.53**, whose criterion 1 demands a grep-derived enumeration. **That
criterion is now doing visible work**: it is exactly what would have caught this.
## 9. What is NOT claimed

- **Not** that the budget is now correct -- only that it is live, and that it was
  not the binding constraint on any measured cycle.
- **Not** n=7 as my own measurement (§3).
- **Not** that tonight's cycle will complete; it is a second sample and its outcome
  is reported whatever it is.
