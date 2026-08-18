# Research Brief -- step 86.109

**Topic:** Calendar-aware staleness/freshness alarming for batch data pipelines
**Tier:** moderate (caller-specified) | **Audit-class:** YES (loop-until-dry, K=2)
**Started:** 2026-08-17
**Role:** Layer-3 Researcher (external literature + internal code exploration)

## Envelope (born inert -- phase-86.37; flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 40,
  "snippet_only_sources": 94,
  "urls_collected": 134,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": true,
    "rounds": 23,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.109.md",
  "gate_passed": true
}
```

**Every number above was re-derived mechanically from this file after the
final edit** (`40 read-in-full + 94 snippet-only = 134` distinct URLs; 16
internal artifacts enumerated in §C5; 23 rounds with dry rounds 22-23 in
§C6). Cycle 1 failed precisely because a count was carried forward by hand
instead of re-derived, so re-deriving is the fix, not a formality.

**RECONCILIATION RUN COMPLETE (2026-08-17, cycle 2).** The cycle-1 envelope
claimed `urls_collected: 120` / `snippet_only_sources: 81` against a brief that
records **119** distinct URLs -- an over-claim of exactly one, and the ONLY
violation the enforced gate flagged (sources floor, URL floor, recency scan,
audit-class dry requirement, `brief_status` marker and all 39 read-in-full URLs
were independently corroborated against the file on disk). This envelope was
re-born `INCOMPLETE` for the duration of the reconciliation and flipped to
`COMPLETE` as the run's final act. See "Envelope reconciliation (cycle 2)"
below for the root cause, the arithmetic, and the resolution.

**Envelope provenance:** born `INCOMPLETE` with zeroed counts in the first
write, updated in place after rounds 1, 8 and 15, flipped to `COMPLETE` at the
end of cycle 1, and re-born `INCOMPLETE` at the start of the cycle-2
reconciliation run. One failed fetch (`tradinghours.com/data`, HTTP 403) is
excluded from `external_sources_read_in_full` and recorded in the
snippet-only table. One URL initially logged as snippet-only
(`docs.datadoghq.com/.../data_observability/`) was later read in full and has
been struck from the snippet table so it is counted exactly once.

---

## Status log (append-only, write-first discipline)

- [t0] Brief created; envelope born INCOMPLETE. Reading researcher.md +
  research-gate.md done. Starting internal exploration + external round 1.
- [t1] Internal round 1 complete: cycle_health.py (754 lines, read in full),
  freshness_cron.py (268, full), markets.py::is_trading_day, alerting.py
  AlertDeduper, slack_bot/scheduler.py digest guard. CRITERION-2 PREMISE
  RESOLVED -- see below, it is TRUE and Main's grep undercounted.
- [t2, cycle 2] Reconciliation run opened. Envelope re-born `INCOMPLETE`.
  Root cause of the cycle-1 over-claim isolated by structural enumeration of
  both tables (below). No source content changed; the counts were wrong, not
  the research.
- [t3, cycle 2] Duplicate rows annotated in place (`[DUP -- not counted]`);
  §C7 checklist figure corrected from the hand-waved `~104` to the enumerated
  figure; envelope counts corrected.
- [t4, cycle 2] Audit-class dry-check rounds 21-22 run fresh in this session
  (see §C6). Envelope flipped to `COMPLETE` as the final act.

---

## Envelope reconciliation (cycle 2) -- the one enforced violation, resolved

The cycle-1 enforced gate FAILED on **exactly one** violation, and it was an
arithmetic over-claim, not a research shortfall:

> `over-claim: urls_collected=120 but only 119 distinct URLs appear in the brief`

Everything else was independently corroborated against this file on disk: the
>=5 sources floor (39), the >=10 URL floor, the recency scan section, the
audit-class `coverage.dry` requirement, the `brief_status` marker, and all 39
read-in-full URLs. **Nothing in Parts A/B/C is retracted by this run.**

### Root cause -- structural enumeration of both tables

Counted mechanically over the two tables (not by hand):

| Quantity | Count | How derived |
|---|---|---|
| Read-in-full rows (§B1 + B1b-B1f) | **39** | rows matching `\| <n> \|`; numbering is contiguous 1..39 with no gaps |
| Distinct URLs among them | **39** | zero duplicates |
| Snippet-only table rows (§B2) | **85** | every row carries a URL |
| ...minus rows repeating a URL **already listed earlier in the same table** | **-3** | `cronalert.com/blog/http-health-check-endpoints`; `reliableuptime.com/blog/rest-api-health-check-endpoint-design`; `github.com/rsheftel/pandas_market_calendars/blob/master/docs/change_log.rst` -- each appears twice in §B2 |
| Distinct URL strings in §B2 | **82** | 85 - 3 |
| ...minus rows whose URL is **identical to a read-in-full row** | **-3** | `docs.datadoghq.com/monitors/types/data_observability/` (struck, = #16); `docs.tecton.ai/docs/monitoring/alerting` (= #29); `docs.aws.amazon.com/wellarchitected/latest/devops-guidance/anti-patterns-for-continuous-monitoring.html` (= #34) |
| **Snippet-only distinct URLs not already read in full** | **79** | 82 - 3 |
| **TOTAL distinct URLs the brief records** | **118** | 39 + 79 |

**Where cycle 1 went wrong.** Its `snippet_only_sources: 81` came from
`85 rows - 3 intra-table duplicates - 1 struck Datadog row = 81`. That
derivation caught the Datadog upgrade (it is called out in the envelope
provenance) but **missed two further overlaps with the read-in-full table**,
because both were written with a parenthetical suffix that made them *look*
like different pages while the URL string is byte-identical to a row already
counted:

- `https://docs.tecton.ai/docs/monitoring/alerting (stream-lag page)` -- same
  URL as read-in-full **#29**. The annotation is about a *section* of the page
  that was not fetched; it is not a second URL.
- The AWS Well-Architected `anti-patterns-for-continuous-monitoring.html` row
  annotated `(index)` -- same URL as read-in-full **#34**. Same shape of
  mistake. *(Deliberately named here without its scheme. Writing an elided,
  ellipsis-bearing copy of the address would itself be counted as a distinct
  pseudo-URL by a naive extractor -- the exact class of artifact this section
  exists to remove, and a mistake this run made once and corrected. The real
  address appears in its two proper places: read-in-full #34, and its tagged
  duplicate row in §B2.)*

So `39 + 81 = 120` double-counted two URLs against a true 118.

**Why the enforcing verifier said 119 and not 118.** A naive
`https?://\S+`-style extraction captures the trailing strikethrough marker on
the struck Datadog row (`...data_observability/~~`) as part of the URL, which
yields a 119th *pseudo*-URL that is not a distinct source. That is a
counting artifact of the extractor, not a real URL, and it is the entire
difference between 118 and 119.

### Resolution chosen: lower the claim, do not invent a URL

The cycle-1 baseline is therefore **118**, not 120 and not 119. Nothing was
added to reach a number; the claim came down to what the artifact holds.

- The gate rejects `urls_collected > distinct_urls_in_brief` and explicitly
  permits claiming fewer (`.claude/workflows/research-gate.js`: *"claiming
  fewer is fine (a brief may cite extra)"*).
- The artifact-free figure corroborates under **both** counting conventions,
  which the artifact-preserving figure does not: claiming 119 would have been
  correct only for an extractor that keeps the `~~` and would fail the moment
  that extractor is tightened.
- The floor is 10. There is no incentive whatsoever to reach for a bigger
  number, and an honest count is strictly better than one the artifact cannot
  corroborate.

`snippet_only_sources` is corrected **81 -> 79** on the cycle-1 baseline so
that `39 + 79 = 118` is internally consistent, and the six double-listed rows
in §B2 are annotated in place with `[DUP -- not counted]` rather than
deleted, so a later auditor can replay the arithmetic against the rows
themselves instead of taking this table's word for it.

### Final counts after the cycle-2 dry-check rounds

The reconciliation run also re-ran the audit-class critic (§C6 rounds 21-23),
and round 21 was **not** dry -- it added read-in-full **#40** plus the URLs
rounds 21-23 evaluated and rejected (§B2b). Those are genuine collections
from this session, recorded because the snippet-only table is exactly where
"what was evaluated but not read" belongs. Final enumeration:

| Quantity | Cycle-1 baseline | Cycle-2 delta | **Final** |
|---|---|---|---|
| Read-in-full | 39 | +1 (#40) | **40** |
| §B2 rows / distinct | 85 / 82 | -- | 85 / 82 |
| §B2b rows / distinct (new) | -- | +15 / +15 | 15 / 15 |
| Snippet-only distinct, excluding the 3 that duplicate a read-in-full row | 79 | +15 | **94** |
| **TOTAL distinct URLs** | 118 | +16 | **134** |

`urls_collected` is set to **134** and `snippet_only_sources` to **94**, so
`40 + 94 = 134` holds exactly. Re-derived mechanically after the final edit,
not carried forward by hand -- which is the discipline whose absence caused
the cycle-1 failure in the first place.

**A naive extractor will report 135, and the one-URL difference is again an
artifact, not a source:** it keeps the trailing strikethrough marker on the
struck Datadog row as part of that URL. `134 <= 134` under a
de-duplicating count and `134 <= 135` under the naive one, so the claim
corroborates either way.

*(Self-inflicted trap worth recording, because it is the same class of error:
while writing this very section I twice introduced ellipsis-bearing "URLs"
into the prose -- e.g. an elided AWS address, and a bare scheme inside a
parenthetical -- each of which a naive extractor counts as a distinct
pseudo-URL and each of which would have inflated the count I was in the
middle of correcting. Caught by re-running the enumeration after every edit.
Prose that discusses URLs is itself a source of fake URLs.)*

### A third figure in the brief was also wrong (found while reconciling)

§C7's checklist line read `~104 (39 read-in-full + ~65 snippet-only)`. That
was a third, hand-waved number disagreeing with both the envelope's 120 and
the true 118. It is corrected to the enumerated figure. The brief now carries
**one** URL count in **all** the places that state one -- the envelope, the
§B2 header, this table, and §C7.

---

# PART A -- INTERNAL CODE INVENTORY (the Explore half)

All line numbers RE-DERIVED at 2026-08-17 against the working tree; the
numbers in the spawn prompt were treated as unverified.

## A1. CRITERION-2 PREMISE CHECK -- the headline internal finding

**The criterion's premise is TRUE. Main's repo-wide grep UNDERCOUNTED.**

The spawn prompt states: *"Main's repo-wide grep found exactly ONE production
consumer of is_trading_day (backend/services/autonomous_loop.py:~674) and
otherwise only backend/tests/test_phase_50_4_calendar.py."*

Measured -- command used (note the QUOTING; an unquoted `--include=*.py` in
zsh raises `no matches found` and prints NOTHING, which is the most likely
cause of the undercount -- I hit exactly that failure on my first attempt):

```
grep -rc "is_trading_day" backend/ frontend/src/ scripts/ 2>/dev/null \
  | grep -v ":0$" | grep -v "__pycache__" | sort
```

Result (7 files, not 2):

| File | Hits | Role |
|---|---|---|
| `backend/backtest/markets.py:192` | 1 | **the definition** |
| `backend/slack_bot/scheduler.py:373,375` (+:369 docstring) | 3 | **THE 51.3 DIGEST APPLICATION** |
| `backend/services/autonomous_loop.py:663,674` | 2 | entry gate (multi-market) |
| `backend/news/fetcher.py:162` | 1 | comment: deliberately does NOT use it |
| `frontend/src/lib/format.ts:243` | 1 | comment: UI hint only |
| `backend/tests/test_phase_50_4_calendar.py` | 13 | tests |
| `backend/tests/test_phase_51_3_digest_guard.py` | 3 | tests |

**A 51.3 digest-job application DOES exist.** It is
`backend/slack_bot/scheduler.py:365-375`:

```python
def _is_us_trading_day_now() -> bool:
    """phase-51.3: True iff TODAY (ET -- the digest cron tz) is a US trading
    session. Gates the morning/evening digests so they skip weekends AND market
    holidays ... Fail-open: is_trading_day returns True if
    exchange_calendars is unavailable ... APScheduler has no holiday support, so
    this in-body guard (not day_of_week='mon-fri') is required to cover holidays too."""
    from backend.backtest.markets import is_trading_day
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    return is_trading_day(et_today, "US")
```

Consumed at two sites, both an early `return` (hard skip, no Slack message):
- `backend/slack_bot/scheduler.py:565` in `_send_morning_digest`
- `backend/slack_bot/scheduler.py:610` in `_send_evening_digest`

**Why the grep missed it:** the call is behind a *wrapper name*
(`_is_us_trading_day_now`) and the import is **function-local**
(`from backend.backtest.markets import is_trading_day` inside the body at
`:373`), so a grep scoped to module-level imports or to the literal call
shape would miss the two real consumers at `:565`/`:610`.

**Design consequence for Main:** the pattern the criterion wants already
exists and is proven in production -- a **market-local ET date + fail-open
`is_trading_day` + hard skip**, with a unit-test fixture at
`backend/tests/test_phase_51_3_digest_guard.py:45-49` that monkeypatches
`backend.backtest.markets.is_trading_day` to drive both branches. 86.109
should REUSE `_is_us_trading_day_now`'s shape, not invent a second one --
but see A2, because reuse-by-copy is exactly what creates the drift risk
the prompt asks about.

### A1b. Cycle-2 re-verification of the criterion-2 premise (and a sharpening)

Re-derived live at 2026-08-17 in the cycle-2 run, because the carried-forward
claim "`is_trading_day` spans 7 files not 2" is load-bearing for criterion 2.
**Confirmed at 7 -- and the breakdown matters more than the number:**

| # | File:line | What it actually is |
|---|---|---|
| 1 | `backend/backtest/markets.py:192` | The **definition** (`def is_trading_day(date, market=DEFAULT_MARKET) -> bool`) |
| 2 | `backend/slack_bot/scheduler.py:365-375` | **Production caller** -- the phase-51.3 digest guard `_is_us_trading_day_now`, used at `:565` and `:610` |
| 3 | `backend/services/autonomous_loop.py:663,674` | **Production caller** -- `from backend.backtest.markets import is_trading_day, ...` at `:663`, called at `:674` as `is_trading_day(local_date, mk)` |
| 4 | `backend/news/fetcher.py:162` | **Comment only, and it is an ANTI-usage:** "NOT use markets.is_trading_day, which fails OPEN when the calendar lib..." |
| 5 | `backend/tests/test_phase_50_4_calendar.py` | Test |
| 6 | `backend/tests/test_phase_51_3_digest_guard.py` | Test |
| 7 | `frontend/src/lib/format.ts:243` | **Comment only** (TypeScript), a UI hint disclaiming authority |

So: **7 files mention it, 2 are production call sites, 2 are comments (one of
them an explicit refusal to use it), 2 are tests, 1 is the definition.**

**The sharpening:** the reason a grep undercounts is not specific to the
digest guard -- **BOTH production callers use a function-local import**
(`scheduler.py:373` inside `_is_us_trading_day_now`; `autonomous_loop.py:663`
inside the calling function). A search for a module-level
`from backend.backtest.markets import is_trading_day` finds **neither**. The
premise of criterion 2 is TRUE, and the mechanism is the function-local
import idiom, which is the house style for this symbol.

Verbatim from `backend/slack_bot/scheduler.py:365-375`, because the docstring
states the fail-open polarity and the APScheduler rationale that objective
(d) turns on:

```python
def _is_us_trading_day_now() -> bool:
    """phase-51.3: True iff TODAY (ET -- the digest cron tz) is a US trading
    session. Gates the morning/evening digests so they skip weekends AND market
    holidays -- they fired 7 days/week and re-sent the prior trading day's data on
    Sat/Sun (operator-reported). Fail-open: is_trading_day returns True if
    exchange_calendars is unavailable, so a calendar-lib error never suppresses a
    digest. APScheduler has no holiday support, so this in-body guard (not
    day_of_week='mon-fri') is required to cover holidays too."""
    from backend.backtest.markets import is_trading_day
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    return is_trading_day(et_today, "US")
```

Also re-verified in this run: `cycle_health.py:42 CRITICAL_RATIO = 2.0`,
`:101 def _band(...)` with the `ratio >= CRITICAL_RATIO` test at `:105`, and
`:262 is_weekday_et = now.astimezone(_NYSE_TZ).weekday() < 5` -- **a bare
`weekday() < 5`, holiday-BLIND**, exactly as carried forward. Note `:202`
documents the existing contract as "`should_alarm` is True iff
`stale AND is_weekday_et`", so a weekday term already gates the alarm; what
is missing is the *holiday* half, not the *weekday* half.

**Fail-open polarity, and why it points the same way as the literature.**
`is_trading_day` fails **OPEN** (returns True when the calendar library is
unavailable -- three such paths, §A3). Fail-open on a *suppression* gate is
the safe polarity: a calendar-lib outage degrades to "treat every day as a
session", i.e. **alarm as today**, never silence. That is the same direction
as PagerDuty #23 (calendar governs urgency, never detection) and the
severity-whitelist rule in #40 -- and it means adopting the 51.3 idiom does
not introduce a new silent-failure mode. `news/fetcher.py:162` refuses the
symbol for the opposite reason: there, fail-open means *fetching on a
non-session day*, which is the costly direction for that caller. **Same
polarity, opposite desirability -- the polarity is only safe relative to what
the caller does with the answer.**

## A2. Weekday/calendar notions ALREADY inside cycle_health.py

The prompt asks: would adding a trading-day term to `_band` create a SECOND,
independently-drifting definition inside one module? **Yes -- it would be the
THIRD notion in that file, and the file already carries two that disagree.**

Every day/calendar notion in `backend/services/cycle_health.py`:

| Anchor | Notion | Semantics |
|---|---|---|
| `:62` | `_NYSE_TZ = ZoneInfo("America/New_York")` | the tz anchor |
| `:262` | `is_weekday_et = now.astimezone(_NYSE_TZ).weekday() < 5` | **Notion 1: Mon-Fri only. NO holiday awareness.** |
| `:295`, `:308` | `should_alarm* = stale and is_weekday_et` | consumers of Notion 1 |
| `:56-58` | comment: "cron fires Mon-Fri once per day at settings.paper_trading_hour ET" | prose model |
| `:76-79` | comment: "largest legitimate gap ... Fri -> Mon = 72h", 96h threshold | **Notion 2: the weekend is absorbed into the THRESHOLD, not a calendar term** |
| `:48-53` `_TABLE_MAX_AGE_SEC` | 26h / 95d / 35d per-table ceilings | **Notion 3 (implicit): calendar-blind absolute ages** |
| `:101-109` `_band()` | `ratio = age_sec / interval_sec`; `>=2.0 red`, `>=1.5 amber` | **NO day term at all** |

So the module already contains **two different weekend strategies**:
`cycle_heartbeat_alarm` gates the *alarm* on `is_weekday_et` (Notion 1),
while `_CYCLE_COMPLETED_STALE_SEC` (`:80`) instead *widens the threshold* to
96h so the Fri->Mon 72h gap is swallowed (Notion 2). Neither is holiday-aware:
`is_weekday_et` is a raw `weekday() < 5`, so **it fires on July 4th, Memorial
Day, Thanksgiving and Christmas** whenever those fall Mon-Fri -- and the
authoritative holiday-aware helper `markets.is_trading_day` is **not imported
anywhere in cycle_health.py** (grep: zero hits for `is_trading_day`,
`markets`, `exchange_calendars` in that file).

**Verdict for Main:** adding a day-of-week term to `_band` would create a
FOURTH notion. The consolidating move is to introduce **one** module-level
trading-day predicate in `cycle_health.py` that delegates to
`markets.is_trading_day` and have `is_weekday_et` at `:262` delegate to it too
-- otherwise 86.109 ships a holiday-aware `_band` sitting beside a
holiday-blind `is_weekday_et` in the same file, which is precisely the
"second, independently-drifting definition" the prompt fears. Note the two
have DIFFERENT correct answers ~9-10 weekdays/year.

## A3. `is_trading_day` fail-open behaviour (`backend/backtest/markets.py:192-213`)

```python
def is_trading_day(date, market: str = DEFAULT_MARKET) -> bool:
    cal = get_trading_calendar(market)
    if cal is None:
        return True  # fail-open: calendar unavailable
    try:
        import pandas as pd
        ts = pd.Timestamp(date)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)  # is_session rejects tz-aware labels
        return bool(cal.is_session(ts.normalize()))
    except Exception as e:
        logger.warning("Calendar check failed for %s/%s: %s; assuming trading day", market, date, e)
        return True
```

**THREE fail-open paths, all returning `True` (= "it IS a trading day"):**
1. `markets.py:203-204` -- `exchange_calendars` not installed / calendar
   object is None (`get_trading_calendar:178-180` logs a warning, returns None).
2. `markets.py:211-213` -- any exception during `is_session`.
3. `get_trading_calendar:187-189` -- unknown exchange -> `logger.error` -> None
   -> path 1.

**The direction of fail-open is load-bearing and it FLIPS with the use case.**
For the digest guard and the entry gate, `True` is the safe default (never
block a trade / never suppress a digest because a library is missing). For an
**alarm suppressor** the direction inverts: `True` there means "market open, so
DO evaluate freshness" -- which is also the loud/safe direction. But if Main
writes the predicate as `is_market_closed()` and fails open to `True`, the
alarm goes **silent** on library failure. **Main must write the calendar term
so that a calendar failure keeps the alarm LOUD, i.e. reuse `is_trading_day`'s
polarity (fail-open to "trading day") rather than negating it.** Independent
corroboration inside the repo: `backend/news/fetcher.py:162` explicitly
declines to use `markets.is_trading_day` *because* "it fails OPEN when the
calendar lib" is missing -- so the repo already has one caller for whom the
fail-open direction was wrong.

## A4. The three HTTP call sites that reach `_fire_freshness_alarm`

`compute_freshness(bq, cycle_interval_sec, *, emit_alarm=True)` at
`backend/services/cycle_health.py:640-642`; the level-triggered fire is at
`:732-733`:

```python
    if emit_alarm and overall_band == "red":
        _fire_freshness_alarm(sources)
```

`_fire_freshness_alarm` is `:123-158` and fires **one P1 per red table** on
**every call** -- there is **no state-transition gate anywhere in
cycle_health.py**. Its only suppression is `AlertDeduper` inside
`raise_cron_alert_sync`, which for P1 is a *repeat window*, not an edge.

**None of the three HTTP call sites has a state-transition gate** (verified
below in A6). They are the default `emit_alarm=True` path.

## A5. The state-transition gate that already exists (quoted)

It is in `backend/services/freshness_cron.py`, **not** under `scripts/`
(prompt said "scripts/.../freshness_cron.py"; the real path is
`backend/services/freshness_cron.py`). Module-level state at `:64-69`:

```python
_last_red_sources: Optional[set[str]] = None
```

and the gate itself at `:146-151`:

```python
        payload = compute_freshness(bq, cycle_interval_sec, emit_alarm=False)

        red_now = _red_sources(payload)
        baseline = _last_red_sources
        newly_red = red_now if baseline is None else (red_now - baseline)
        _last_red_sources = red_now
```

Then `:155-176` pages only for `newly_red`; `:184-192` logs still-red
**without paging** ("Deliberately log-only. Re-paging a source that has been
red since the last tick is the alert-fatigue failure mode this module exists
to avoid"); `:193-198` logs recovery. Cadence `DEFAULT_INTERVAL_HOURS = 6`
(`:62`), registered as an APScheduler interval job (`:229-258`, `JOB_ID =
"freshness_evaluator"`, `replace_existing=True`).

The sibling pattern in `backend/slack_bot/scheduler.py` -- four
module-level transition flags, all `bool | None` with `None` = no baseline:
`_watchdog_last_was_healthy:101`, `_cycle_heartbeat_last_was_stale:113`,
`_cycle_completed_stale_last_was_stale:121`,
`_ingestion_silence_last_was_stale:125`; consumed at `:690-691`, `:751-752`,
`:808-809`, `:837-838`.

**So the repo already contains the edge-triggered answer twice over.** The
gap 86.109 addresses is that the *HTTP read path* still runs the
level-triggered path.

## A6. `AlertDeduper` (`backend/services/observability/alerting.py:63-107`)

P1 branch (`:83-93`) -- **bypasses the consecutive threshold, keeps the
repeat window**:

```python
        if severity in _CRITICAL_SEVERITIES:
            with self._lock:
                st = self._state.setdefault((source, error_type), _AlertState())
                st.occurrences.append(now)
                fire = (
                    st.last_fired_at is None
                    or (now - st.last_fired_at) >= self.repeat
                )
                if fire:
                    st.last_fired_at = now
            return fire
```

Defaults `:66-71`: `window_minutes=5, repeat_hours=1,
consecutive_threshold=3`; live values read from settings at `:124-130`
(`alert_debounce_minutes`, `alert_repeat_hours`,
`alert_consecutive_failure_threshold`). `_CRITICAL_SEVERITIES` at `:54` =
`{"P0","P1","critical","CRITICAL"}`.

**So the P1 repeat window is 1h per `(source, error_type)`.** A permanently
red table therefore pages **~24x/day forever** from any level-triggered
caller -- which is the arithmetic behind the phase-66 hotfix comment at
`:46-53` ("~120 pages/hour the moment a dashboard tab was open against a red
table") and the freshness_cron docstring's "~512 pages over 128 days".
Critically, `error_type` is `f"freshness_critical_{table_name}"`
(`cycle_health.py:144`), so **each red table gets its own independent 1h
window** -- N red tables = N pages/hour.

## A7. The three HTTP call sites -- confirmed, and NONE is gated

`grep -rn "compute_freshness" backend/ scripts/ frontend/src/`:

| Call site | Anchor | Gate? |
|---|---|---|
| `GET /api/paper-trading/freshness` | `backend/api/paper_trading.py:477` (route), **`:498` the call** -- `return await asyncio.to_thread(compute_freshness, bq, cycle_interval_sec)` | **NO** -- positional args only, so `emit_alarm` defaults to `True` |
| `backend/api/observability_api.py:36` | `from backend.services.cycle_health import compute_freshness as _cf` (function-local import) | **NO** |
| `backend/api/observability_api.py:55` | second alias, same import shape | **NO** |
| `backend/services/freshness_cron.py:146` | `compute_freshness(bq, cycle_interval_sec, emit_alarm=False)` | **YES** -- the only gated caller |

So exactly **one of four production callers** suppresses the level-triggered
alarm, and it is the cron. **The three that DO page are all HTTP GET
handlers on the dashboard read path.** `backend/api/paper_trading.py:26`
imports `compute_freshness` at module level; the two `observability_api`
sites use function-local imports, which is why a naive module-level-import
grep undercounts here too.

## A8. MEASURED: what the alarm is actually doing to Slack

Command used (rotated `.gz` + the LIVE log; the live log is **NOT** under
`handoff/logs/` -- per `~/Library/LaunchAgents/com.pyfinagent.backend.plist`
`StandardOutPath`/`StandardErrorPath` it is
`/Users/ford/.openclaw/workspace/pyfinagent/backend.log`, 29,576,152 bytes at
capture time. A grep scoped to `handoff/logs/*.log` alone returns **ZERO** --
that was my first attempt and it was wrong):

```bash
for f in handoff/logs/backend.log.*.gz; do gzcat "$f" | grep "Data freshness critical"; done > fresh_rot.txt
grep "Data freshness critical" backend.log > fresh_live.txt
cat fresh_rot.txt fresh_live.txt > fresh_all.txt
```

Captured 2026-08-17 ~22:50 local.

| Measure | Value |
|---|---|
| Total `Data freshness critical` log lines | **1,149** (1,071 rotated + 78 live) |
| ...of which **DELIVERED to Slack** (`delivered=True`, bot-token fallback) | **991** |
| ...of which dropped (`slack_webhook_url not configured`) | 158 (all in the two OLDEST rotated logs) |
| Distinct message shapes | exactly 2 (verified by normalising + `uniq -c`) |
| `freshness_critical_` (the `error_type` string) in logs | **0** -- the error_type never reaches the log; only the human `title` does |

**The filing's figure of 58 is stale by ~20x on the delivered count.** (58 is
close to the per-table count for `historical_prices`, which is 58 exactly --
so the filing may have measured one table.)

Per-table split of all 1,149:

| Table | Lines |
|---|---|
| `paper_trades` | 499 |
| `historical_macro` | 348 |
| `paper_portfolio_snapshots` | 115 |
| `historical_prices` | 58 |
| `signals_log` | 51 |

Cadence, over the **430 lines that carry a parseable date** (only the newer
JSON log format has one; the 719 older plain lines carry `HH:MM:SS` with no
date, so they are excluded from the timing stats and this is an
**undercount**, not a sample):

| Measure | Value |
|---|---|
| First dated | **2026-07-24 08:39:43** |
| Last dated | **2026-08-17 21:21:46** (i.e. TODAY, still firing) |
| Span | 24.53 days |
| Mean rate | **17.5 pages/day** |
| Inter-arrival median | 2,461 s (41 min) |
| Inter-arrival p75 | 3,641 s (61 min) |
| Gaps < 70 min | **392 of 429 (91.4%)** |

The 61-minute p75 is the **AlertDeduper 1h repeat window rendered in the
data** -- the alarm is re-firing at essentially the maximum rate the deduper
permits, per table, indefinitely.

### The weekend result -- this is the step's core evidence

| Day (log-local) | Dated alerts |
|---|---|
| Mon | **165** |
| Tue | 38 |
| Wed | 27 |
| Thu | 35 |
| Fri | 36 |
| **Sat** | **46** |
| **Sun** | **83** |
| **Sat+Sun** | **129 = 30.0%** of all dated alerts |

**Three facts Main should design to:**
1. **30% of pages fire on Saturday or Sunday**, when the paper-trading cron
   (`Mon-Fri`, `cycle_health.py:56-58`) is not scheduled to write at all --
   i.e. they are pages for an expected absence.
2. **Monday is the single loudest day at 165 (38.4%)** -- 4.3x the Tue-Fri
   mean of 34. That is the weekend's accumulated age carried into Monday
   before the first cycle writes. So a naive "suppress on Sat/Sun" fix would
   NOT remove the Monday spike; the ratio is still >2.0 on Monday morning
   because `age_sec` measured from Friday's write is ~72h against a 24h
   `cycle_interval_sec`. **The weekend problem is a Monday problem too.**
3. Consequently, **suppression alone is the wrong shape** -- it hides the
   Friday-died writer for 48-72h, which is exactly the danger the objective
   names. See Part B.

---

# PART B -- EXTERNAL LITERATURE

## Search-query composition (3-variant discipline, per research-gate.md)

- **Year-less canonical:** `business calendar aware SLO data freshness
  alerting weekend holiday suppression`; `edge-triggered vs level-triggered
  alerting state transition deduplication alert storm`; `health check endpoint
  side effects anti-pattern read path should not trigger alerts`
- **Last-2-year / current-year:** `data observability freshness monitoring
  seasonality aware anomaly detection expected next arrival time 2025`;
  `"exchange_calendars" OR "pandas_market_calendars" holiday data accuracy
  incorrect caveat 2025 2026`; plus the round-2/3 queries listed in the
  Recency-scan and Coverage sections below.

## B1. Read in full (WebFetch) -- counts toward the gate

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 1 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-17 | Official doc (Google SRE Workbook ch.5) | WebFetch full | Defines the 4 axes: **Precision** = "The proportion of events detected that were significant"; **Recall** = "The proportion of significant events detected"; **Detection time**; **Reset time** = "How long alerts fire after an issue is resolved. Long reset times can lead to confusion or to issues being ignored." Single long-window threshold has terrible reset time: an alert "will fire shortly after 2 minutes, and continue to fire for the next 36 hours". Explicitly **recommends against duration/`for` clauses** as SLO alerting criteria. Multiwindow multi-burn-rate is the recommended shape. Names the suppression problem: "To avoid multiple alerts from firing if all conditions are true, you need to implement alert suppression." Page vs ticket by budget-exhaustion horizon. |
| 2 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-17 | Official doc (Google SRE Book ch.6) | WebFetch full | "Every page should be actionable"; "**Every page response should require intelligence. If a page merely merits a robotic response, it shouldn't be a page**"; "Pages should be about a novel problem or an event that hasn't been seen before"; "I can only react with a sense of urgency a few times a day before I become fatigued." The rule-design test: "**Will you ignore this alert as benign? If so, redesign it.**" Bigtable/Gmail case studies of over-alerting; "Pages with rote, algorithmic responses should be a red flag." |
| 3 | https://prometheus.io/docs/practices/alerting/ | 2026-08-17 | Official doc | WebFetch full | "keep alerting simple, alert on symptoms, have good consoles to allow pinpointing causes, and **avoid having pages where there is nothing to do**"; "Aim to have as few alerts as possible"; "**Allow for slack in alerting to accommodate small blips**"; metamonitoring (monitor the monitor). |
| 4 | https://grafana.com/docs/grafana/latest/alerting/fundamentals/alert-rules/state-and-health/ | 2026-08-17 | Official doc | WebFetch full | The production **state machine**: `Normal / Pending / Alerting / NoData / Error`, with an explicit **pending period** and transitions `Normal -> Pending -> Alerting`. NoData is a **first-class, separately configurable state** with four options (`No Data` / `Alerting` / `Normal` / **`Keep last state`**), and produces its own `DatasourceNoData` instance with distinct labels -- so existing silences may not apply. Directly relevant: "no data" is modelled as a state, not conflated with "bad data". |
| 5 | https://docs.getdbt.com/docs/build/sources | 2026-08-17 | Official doc | WebFetch full | The canonical batch-freshness config: `warn_after`/`error_after` as `{count, period}`, `loaded_at_field`, `filter`; freshness computed as `max(loaded_at) vs current_timestamp`. **CRITICAL NEGATIVE RESULT:** the docs contain **no** weekend/non-business-day handling, no calendar term, no guidance for schedule-gapped sources. The industry-standard tool models freshness as a **pure absolute-age threshold** -- i.e. pyfinagent's `_band` is not unusual, and there is no off-the-shelf calendar-aware primitive to adopt. |
| 6 | https://ar5iv.labs.arxiv.org/html/2204.09670 | 2026-08-17 | Peer-reviewed (arXiv/ICSE-SEIP, Huawei Cloud) | WebFetch full (ar5iv per gate chain -- 2022 paper, pre-Dec-2023) | Empirical taxonomy over **4M+ alerts / 2 years / 11 services / 192 microservices** + 18 on-call engineers. Names our exact defect: **[A5] Repeating Alerts** = "Alerts from the same alert strategy appear repeatedly" -- **94.4% of OCEs** agreed it has significant impact; in one storm a single strategy produced **~30% of 2,751 alerts**. Also **[A3] Improper and Outdated Generation Rule** (72.2% rated high impact) and **[A4] Transient and Toggling Alerts** (94.4%). Mitigations: **[R1] rule-based alert blocking, [R2] alert aggregation**, [R3] correlation, [R4] emerging-alert detection. Only **22.2%** found SOPs helpful. |
| 7 | https://learn.microsoft.com/en-us/azure/architecture/patterns/health-endpoint-monitoring | 2026-08-17 | Official doc (Azure Architecture Center) | WebFetch full | **The decisive quote for objective (c/e):** "**Consider caching the endpoint status. Running the health check frequently might be expensive. For example, if the health status is reported through a dashboard, you don't want every request to the dashboard to trigger a health check. Instead, periodically check the system health, and cache the status. Expose an endpoint that returns the cached status.**" Also: "Performing excessive processing during the check can overload the application and affect other users"; and "ensure that the monitoring system performs checks on itself... to prevent the monitoring system from issuing false positive results." |
| 8 | https://github.com/gerrymanoim/exchange_calendars | 2026-08-17 | Official repo/doc | WebFetch full | 59 ISO-10383 calendars incl. XNYS/XETR/XKRX. `is_session()` is the session predicate; schedule carries `open/break_start/break_end/close`. **Accuracy disclaimer:** "***All*** of the exchange calendars are maintained by user contributions" -- correctness is community-sourced, fixed by PR. **No stated release cadence and no statement of how far ahead sessions are computed** -- so a forward-dated holiday can silently be wrong until someone files it. |
| 9 | https://pandas-market-calendars.readthedocs.io/en/latest/usage.html | 2026-08-17 | Official doc | WebFetch full | The alternative library. `valid_days()`, `schedule()`, `early_closes(schedule)`, `late_opens()`, `open_at_time()`; `interruptions` for trading halts; `regular_market_times` with historical change tuples (NYSE close 3pm->3:30pm 1952->4pm 1974). **Caveats:** "times will NOT be adjusted to special_opens/special_closes if market_open/market_close are not requested"; `force_special_times` tri-state; `UserWarning: [...] are discontinued`; helper functions only support the 4 standard columns. Holidays are **computed projections** (returns dates out to 2200), not a curated record. |

### B1b. Read in full -- rounds 2-4 (audit-class loop continuation)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 10 | https://healthchecks.io/docs/configuring_checks/ | 2026-08-17 | Official doc | WebFetch full | **The prior art for objective (a): expected-next-write scheduling.** Three modes -- Simple (Period + Grace Time), **Cron** ("Use 'Cron' for monitoring cron jobs and other processes with more complex schedules. This monitoring mode ensures that jobs run **at the correct time** and not just at the correct time intervals"), and OnCalendar. Cron mode additionally requires the **Server's Time Zone** because "The cron daemon typically uses the system's local time". **Grace Time** is a separate knob from the schedule. This is the exact decomposition pyfinagent lacks: the schedule says *when a write is due*; the grace time says *how late is tolerable*; the ratio-of-age says neither. |
| 11 | https://prometheus.io/docs/alerting/latest/configuration/ | 2026-08-17 | Official doc (Alertmanager) | WebFetch full | **The official calendar-aware muting primitive.** `time_intervals` schema: `times{start_time,end_time}`, **`weekdays`**, `days_of_month` (supports negative indices), `months`, `years`, **`location` (IANA tz)**. "For an instant of time to match a complete time interval, **all fields must match**"; missing fields match any value. `mute_time_intervals` (route "will not send any notifications" but "otherwise acts normally") vs `active_time_intervals` (send ONLY during). **"The root node cannot have any mute times."** Timing defaults: `group_wait` 30s, `group_interval` 5m, **`repeat_interval` 4h** (vs pyfinagent's 1h). Inhibit rules have an explicit **anti-self-suppression clause**. |
| 12 | https://arxiv.org/html/2502.05392v1 | 2026-08-17 | Peer-reviewed preprint (2025, industry perspective) | WebFetch full (arXiv HTML per gate chain) | **States our exact failure mode academically:** a store closing on July 4th produces zero sales, which "looks anomalous statistically but represents expected behavior". §5.3 Threshold Setting: "In an alerting setting, the choice of threshold for converting an anomaly score into an anomaly prediction is critical", and benchmark evaluation "often uses an ideal threshold that not only requires knowledge of the full time series, but also of the ground truth" -- i.e. **static thresholds are not defensible from benchmarks**. §4: "false alerts about irrelevant behavior are sent to fridge operators, who are sent to rescue food that is perfectly safe." The **"Application Specific"** tenet: whether an event is an anomaly depends on downstream usage, not the signal. **Honest gap:** the paper does NOT treat "expected absence of data" as a distinct class -- see Consensus vs debate. |
| 13 | https://arxiv.org/html/2510.24452v1 | 2026-08-17 | Peer-reviewed preprint (2025, Google BigQuery ARIMA_PLUS) | WebFetch full | **Canonical ordering for calendar-aware detection, from the vendor of pyfinagent's own warehouse.** Pipeline order: (1) **holiday/event adjustment**, (2) spike/dip outlier removal, (3) seasonality extraction, (4) change-point detection, (5) trend/ARIMA, (6) forecast+intervals. Rationale quoted verbatim: "**To avoid these from being treated as outliers, holiday and event adjustment is done before cleaning.**" Holiday effects are "usually on a smaller temporal scale compared to seasonal effects" and "quite temporary", hence "easier to screen out at a local level". Maintains "fixed date holidays, floating date holidays and manual events" (Easter, Lunar New Year). `ML.DETECT_ANOMALIES` is a TVF. |
| 14 | https://oneuptime.com/blog/post/2026-01-30-batch-processing-alerting/view | 2026-08-17 | Industry blog (**2026-01-30**) | WebFetch full | Batch-alerting design with an explicit **`business_days_only: true  # Only runs Mon-Fri`** SLA field: when set, "the system skips compliance evaluation on weekends and holidays, preventing false alerts for intentionally dormant periods." Recommends **schedule-aware expectations over blanket suppression**, with three complementary filters: **deduplication (exact-match), throttling (rate limit), aggregation (summary)**. `SLAManager.get_approaching_deadlines()` = proactive pre-miss detection. Motivating quote: "A failed nightly data sync might go unnoticed until a business user complains about stale dashboards the next morning." |
| 15 | https://oneuptime.com/blog/post/2026-02-06-alert-suppression-maintenance-opentelemetry/view | 2026-08-17 | Industry blog (**2026-02-06**) `[PARTIAL-NEGATIVE]` | WebFetch full | Read in full specifically to test the "suppression hides real failure" claim. **It does NOT make that argument** -- it covers manual/CI/recurring silences and says "Use Alertmanager silences when you want alerts to still be evaluated and visible in the UI but not dispatched", alerts remaining "Visible in UI as Silenced". It explicitly does *not* discuss suppression hiding genuine failures, critical-severity bypass, or post-window audit. **Recorded as a negative result:** the strongest form of the "weekend suppression is dangerous" claim is NOT well-supported in the vendor literature; the support for it in this brief comes from SRE-book recall reasoning + the internal Monday-spike measurement, not from a source asserting it. |

### B1c. Read in full -- rounds 5-8 (audit-class loop continuation)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 16 | https://docs.datadoghq.com/monitors/types/data_observability/ | 2026-08-17 | Official vendor doc | WebFetch full | Freshness = "time elapsed since a table was last updated" (table-level) / "the most recent date seen in a datetime column" (column-level). **NEGATIVE: no business-calendar / weekday / custom-seasonality option is documented.** The only escape hatch is post-hoc **"Annotate Bounds"** -- `Expected` ("Expand bounds to include the marked behavior permanently"), `Reset for now`, `Missed alert`, `Ignore`. Anomaly detection needs a **3-7 day training window**. |
| 17 | https://docs.getmontecarlo.com/docs/exclusion-windows | 2026-08-17 | Official vendor doc | WebFetch full | **The closest thing to "suppress on non-trading days" that a major vendor ships -- and it confirms the danger.** Exclusion windows do BOTH: training removal ("the specified timeframe is removed from the training data that informs Monte Carlo's anomaly detection") AND **alert suppression** ("**no incidents or notifications from the above detectors will be generated for the duration of the exclusion window**"). Scope: warehouse / project / dataset / table. Caveats: up to **9 hours** to propagate into training; does not apply to SQL Rules. **The docs do NOT warn about what an exclusion window can hide** -- i.e. the danger is real and undocumented even in the shipping product. |
| 18 | https://docs.getmontecarlo.com/docs/freshness-rules | 2026-08-17 | Official vendor doc | WebFetch full | Freshness Rules = "select the lookback range within which you expect the table(s) to receive an update"; guidance is to **match the run schedule** -- "run every 6 hours looking back 6 hours" for regular updates, "run at 9am UTC every day looking back 2 hours" for scheduled updates. **NEGATIVE: no business-day-aware option; no weekday/weekend logic documented.** Note the feature is **being deprecated** ("can no longer create new Freshness Rules through the UI"). |
| 19 | https://www.anomalo.com/blog/defining-data-freshness-measuring-and-monitoring-data-timeliness/ | 2026-08-17 | Industry blog | WebFetch full | Useful vocabulary split: **Timeliness** = "the time it takes for data to be made available after an event occurs"; **Freshness** = "the age of the data at any given point in time"; **Latency** = "the delay between data generation and its availability". "timely data ... does not necessarily guarantee freshness if the data is not regularly updated". **NEGATIVE: no expected-schedule / business-day mechanism at all.** |
| 20 | https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/freshness/ | 2026-08-17 | Official doc | WebFetch full | Freshness via `ExpectColumnMaxToBeBetween` ("the maximum, or the most recent, timestamp, in a column is within an expected range of time") and `ExpectColumnMinToBeBetween`. Threshold-setting process says to "incorporate temporal patterns (daily, weekly cycles, known gaps)" and gives a **time-of-day-conditional** example: "During peak hours: ... within the last 30 minutes. During off-hours: ... within the last 2 hours." **NEGATIVE on weekends/holidays specifically** -- the user must hand-encode it. |
| 21 | https://www.rfc-editor.org/rfc/rfc9110.html#name-safe-methods | 2026-08-17 | **IETF standard (highest non-peer-reviewed tier)** | WebFetch full | **The normative basis for objective (e).** §9.2.1: "Request methods are considered '**safe**' if their defined semantics are essentially **read-only**; i.e., **the client does not request, and does not expect, any state change on the origin server** as a result of applying a safe method to a target resource." GET/HEAD/OPTIONS/TRACE are safe. "A request method is safe if the intended semantics of the method definition, in all cases, should not change the state of the origin server **in ways that are significant to the client**." **The crucial nuance, which cuts BOTH ways:** the RFC explicitly permits server-side side effects (logging, analytics) but says "**the client cannot be held responsible for these side effects**" -- so a GET *may* have side effects, but the dashboard poller cannot be blamed for the page it caused. §9.2.2 idempotency: "the side effects of N > 0 identical requests is the same as for a single request" -- which the freshness alarm violates, because N polls produce N pages (bounded only by the deduper). |
| 22 | https://github.com/apache/airflow/issues/38099 | 2026-08-17 | Official issue tracker (Apache Airflow) | WebFetch full | Independent corroboration that the "age vs schedule" gap is a recognised industry problem, not a pyfinagent quirk. Opened **2024-03-13**, labels `area:deadline-alerts`, **`AIP-86`**, `kind:feature`, now **Closed**. Proposal wording: an SLA "could take the form of **a timedelta since 'the last successful run'** or some other more complex expression"; motivation: "any dag that's data-aware scheduled has little to none direct monitoring/observability capabilities." Note pyfinagent has independently arrived at the same primitive -- `cycle_health.py:80` `_CYCLE_COMPLETED_STALE_SEC` measures exactly "age since the last SUCCESSFUL run". |

### B1d. Read in full -- rounds 9-13 (audit-class loop continuation)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 23 | https://support.pagerduty.com/main/docs/dynamic-notifications | 2026-08-17 | Official vendor doc | WebFetch full | **THE literature's answer to objective (b): DEMOTE, DO NOT SUPPRESS.** Support Hours are defined by "the **days** of the week, **hours**, and **time zone**". Outside them, incidents are **not** suppressed -- "**incidents are created regardless of support hours, but how responders are notified varies based on the configured urgency setting**". Two independent knobs: "Under **During support hours, use**, select the type of notification urgency" and "Under **Outside support hours, use**, ...". So the calendar governs **routing/urgency**, never **detection**. |
| 24 | https://training.promlabs.com/training/monitoring-and-debugging-prometheus/metrics-based-meta-monitoring/end-to-end-watchdog-alerts/ | 2026-08-17 | Authoritative training (PromLabs / Prometheus core) | WebFetch full | The **inversion** pattern. An always-firing rule `expr: vector(1)` routed to an external receiver with a short repeat interval; the receiver alerts on the **ABSENCE** of the expected notification. Verbatim kube-prometheus rule annotation: "This is an alert meant to ensure that the entire alerting pipeline is functional. This alert is always firing, therefore it should always be firing in Alertmanager and always fire against a receiver." Catches the class normal alerts cannot: "the Alertmanager through which notifications are supposed to get delivered to you might be down, or your entire organization's network might be disconnected from the internet!" |
| 25 | https://dataintellect.com/blog/stale-data-measuring-what-isnt-there/ | 2026-08-17 | Industry practitioner (capital-markets data engineering) `[ADVERSARIAL/limitation]` | WebFetch full | A **statistically principled alternative to age thresholds**: model trade inter-arrival as a **Poisson process** ("events are relatively evenly distributed throughout the day (with a spike at close)") and alert at a fixed improbability -- "we can probably stick with **1-in-a-billion** as our alert threshold ... this would generate an alert **42 milliseconds** after our last message" for NYSE trades. Splits detection into **Time** ("stopped flowing ... or arriving late") vs **Quality** ("if rows are coming in, are they correct?"). **Adversarial value:** despite being a trading-domain staleness paper, it contains **no** exchange-calendar or session awareness at all -- it simply assumes you are inside market hours. Evidence that the calendar term is routinely *omitted* even by domain experts. |
| 26 | https://eodhd.com/financial-academy/fundamental-analysis-examples/real-time-market-data-reliability-stale-price-detection-rest-fallback-and-websocket-recovery | 2026-08-17 | Industry (market-data vendor) | WebFetch full | **The state-machine shape Main should copy.** Session awareness is modelled as an explicit STATE `MARKET_CLOSED`, not as alert suppression: the monitor "should not mark the quote as stale when no live updates are expected". Bands `LIVE <=3s / DEGRADED 3-10s / STALE >10s`, with the explicit caveat "**These thresholds should be adjusted by exchange, asset class, update frequency, market session, and the consequence of showing a stale price.**" Also: track freshness "**per symbol, not only per connection**" -- "one symbol can freeze while the feed stays connected". |
| 27 | https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | 2026-08-17 | Industry blog (**2026-02-06**) | WebFetch full | Heartbeat/dead-man's-switch parameters: heartbeat "every 30s", alert `for: 2m`, detection `absent_over_time([5m])`. Explicitly favours **detecting absence rather than staleness**: `absent_over_time(metric[5m]) == 1`. Recommends "a layered defense against silent telemetry failures" = collector heartbeats + per-service freshness + an **external** dead man's switch. **NEGATIVE: does not address jobs that run only on certain days.** |
| 28 | https://oneuptime.com/blog/post/2026-02-09-monitor-cronjob-missed-schedules/view | 2026-08-17 | Industry blog (**2026-02-09**) | WebFetch full | **A concrete, documented instance of the exact bug 86.109 is about.** Computes `success_age = now - last_successful` and compares against `expected_window = interval * 2` (x3 for successful runs) -- i.e. a **grace multiplier**, structurally the same as pyfinagent's `CRITICAL_RATIO = 2.0`. Its `parse_cron_schedule()` handles only `@hourly/@daily/@midnight` and `*/N`, and **"Weekday-only schedules (like `0 2 * * 1-5`) are NOT properly handled -- the parser would incorrectly default to daily"**, generating exactly our weekend false positives. Its own comment: "This is a basic parser - production code should use croniter". |
| 29 | https://docs.tecton.ai/docs/monitoring/alerting | 2026-08-17 | Official vendor doc (feature store) | WebFetch full | Freshness as an **expectation**, not an age: `monitor_freshness` (bool), `alert_email`, `expected_feature_freshness` (optional override). "**If the expected freshness is less than the actual freshness, the Feature View is considered to be serving stale data.**" Noise remedy is explicitly to widen the expectation: "If the Expected Feature Freshness is too low resulting in noisy freshness alerts, specifying a higher value for `expected_feature_freshness` might help." **NEGATIVE: the docs do not connect the expectation to the materialization schedule.** |
| 30 | https://github.com/pallets-eco/croniter | 2026-08-17 | Official library repo | WebFetch full | **The concrete implementation primitive** if Main goes the expected-next-write route. `get_next(datetime)` / `get_prev(datetime)` (prev since v0.2.0). `day_or` semantics: default **OR** ("fires on every Wednesday OR on the 1st of the month"), `day_or=False` gives **AND**. `W` = nearest weekday (Sat->preceding Fri, Sun->following Mon, never crosses month boundary; cannot combine with ranges/lists). **Critical caveat quoted:** "**Be sure to init your croniter instance with a TZ aware datetime for this to work!**" Guards: `max_years_between_matches` (default 50) raising `CroniterBadDateError`; `strict=True` needed to reject impossible dates like `0 0 31 2 *`. **NOTE: croniter models WEEKDAYS, not exchange HOLIDAYS** -- it does not replace `is_trading_day`. |
| 31 | https://oneuptime.com/blog/post/2026-02-17-how-to-configure-metric-absence-alerting-policies-to-detect-missing-data-in-cloud-monitoring/view | 2026-08-17 | Industry blog (**2026-02-17**) | WebFetch full | **The cleanest statement of the design principle for objective (b).** Why absence beats threshold: "A service that stops reporting metrics is often in worse shape than one reporting elevated error rates" -- when the emitter crashes, "there are no data points to evaluate - the error rate is not high, it is **nonexistent**." Names our exact trap: "A service that gets no requests between 2 AM and 6 AM would trigger a request-count absence alert every night." **The prescribed fix is NOT suppression** but a **dedicated heartbeat metric**: "Your application sends a heartbeat metric periodically **regardless of traffic**." Duration guidance: "set the absence duration to **at least 3x** the metric's reporting interval"; "For heartbeat metrics: 2x to 3x the expected reporting interval". |

### B1e. Read in full -- rounds 14-15 (audit-class loop continuation)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 32 | https://www.nyse.com/trade/hours-calendars | 2026-08-17 | **Primary source (the exchange itself)** | WebFetch full | Answers the "how far ahead is the calendar knowable" question for objective (d): **NYSE publishes three years ahead (2026/2027/2028)**. 10 full-closure holidays. **Early closes at 1:00 p.m. ET** -- 2026: Fri **Nov 27**, Thu **Dec 24**, and **Jul 3**; options close 1:15 p.m. ET. Weekend-observance rule quoted verbatim: "**Because the holiday falls on Saturday, January 1, 2028, no New Year's Day holiday is observed.**" (Independence Day 2027 observed Mon Jul 5.) So a half-day IS a session -- `is_session` returns True -- and the observance rule is exchange-specific, not derivable from a weekday test. |
| 33 | https://authzed.com/blog/fail-open | 2026-08-17 | Industry blog `[NEGATIVE -- claim NOT corroborated]` | WebFetch full | **Fetched specifically to source the striking claim the search-result summary attributed to it** -- "Most systems have a failure behavior that nobody chose -- it emerged from caught exceptions, default timeouts, or retry policies". **Read in full, the article does NOT contain that claim.** It argues only that "Fail-open code can end up awful to read: sometimes so awful that you might be more likely to write bugs because it's too hard to read", that developers must "decide for yourself where to risk writing something fail-open or fail-closed", and that fail-closed suits authorization/financial contexts. **Recorded as an explicit correction: do not cite that quote.** (Cheap illustration of why the gate requires full fetches -- the snippet was persuasive and wrong.) |
| 34 | https://docs.aws.amazon.com/wellarchitected/latest/devops-guidance/anti-patterns-for-continuous-monitoring.html | 2026-08-17 | Official doc (AWS Well-Architected) | WebFetch full | Two named anti-patterns land on this step. **"Noisy and unactionable alarms:** If alarms frequently sound without actionable cause, trust in the alerting system diminishes, risking slower response times or overlooked genuine alerts... Implement mechanisms to mute false positives and adjust overly sensitive alarms." And, cutting the OTHER way, **"Inadequate monitoring coverage"** whose symptom is named "**no dogs barking**", where "**the absence of expected alerts or metrics itself can indicate an issue**". Those two anti-patterns are precisely the two horns of objective (b). |
| 35 | https://sre.google/workbook/on-call/ | 2026-08-17 | Official doc (Google SRE Workbook ch.8) | WebFetch full | **The hard number to benchmark pyfinagent against.** "**We target a maximum of two incidents per on-call shift, to ensure adequate time for follow-up**" (12-hour shifts) -- and one incident = one *problem*, however many alerts fire for it. "**All alerts should be immediately actionable. There should be an action we expect a human to take immediately after they receive the page that the system is unable to take itself.**" New alerts need review + "a corresponding playbook entry" + ~a week of production vetting before being upgraded to paging. **Measured comparison: pyfinagent's freshness alarm alone runs 17.5 pages/day = ~8.75 per 12-hour shift, i.e. ~4.4x Google's ceiling for ALL causes combined -- and it is one alarm for one already-known problem.** |

### B1f. Read in full -- rounds 16-18 (final new findings before the dry pair)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 36 | https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/anomaly-detection | 2026-08-17 | Official vendor doc | WebFetch full | The **learned** version of the expected-next-write idea: freshness = "how recently a table has been updated"; the system "**analyzes the history of commits to a table and builds a per-table model to predict the time of the next commit**"; "**If a commit is unusually late, the table is marked as stale.**" **NEGATIVE:** no documented handling of weekday-only/irregular schedules, no "unusually late" threshold published, and no configuration to exclude expected gaps -- and manual job-parameter configuration (`table_threshold_overrides`, `static_table_threshold_override`) is **no longer supported for new customers from 2025-07-21**. |
| 37 | https://docs.siffletdata.com/docs/data-freshness | 2026-08-17 | Official vendor doc | WebFetch full | **Static Mode** = fixed time slots ("If data is not found within a particular time interval, an incident is created"). **Dynamic Mode** = learned: "**An anomaly is detected only if data is expected to have arrived in the last time slot but has not**", illustrated with data arriving "twice weekly on Monday and Thursday" alerting "only if data arrives on unexpected days or fails to arrive on expected days". **This is the closest thing in the market to a Mon-Fri-aware freshness monitor -- and it is LEARNED, not calendar-driven; the docs contain no business-day/holiday configuration at all.** |
| 38 | https://grafana.com/docs/grafana/latest/alerting/configure-notifications/mute-timings/ | 2026-08-17 | Official doc | WebFetch full | **The decisive design distinction for 86.109, stated normatively.** Mute timing = "A **recurring** interval that stops notifications ... **It suppresses notifications but does not interrupt alert evaluation.**" Silence = label-matched, fixed start/end, **one-off**. "Mute timings take precedence over active time intervals when they overlap." Fields identical to Alertmanager's (time range, `monday:thursday` ranges, days-of-month incl. negatives, months, years, **location**); "if a field is left blank, any moment of time matches the field." Alerts "continue being processed and remain visible in the UI; only the notification output is suppressed." |
| 39 | https://oneuptime.com/blog/post/2026-01-30-alert-suppression-rules/view | 2026-08-17 | Industry blog (**2026-01-30**) | WebFetch full | Fetched to test the "suppression hides real failure" claim after #15 came back negative. **Partially corroborates it:** names the anti-pattern "**Silencing instead of fixing**" where "**Alert debt accumulates**", warns that long-running unsupervised suppression can "miss cascading failures", and prescribes "**Monitor what you suppress. And always, always document why**" with per-silence `Creator / Comment / Duration` metadata plus "Set Conservative Durations". It does **not** use the phrase "powerful and dangerous". |

### B1g. Read in full -- round 21 (cycle-2 dry-check; round was NOT dry)

The cycle-2 reconciliation run re-ran the loop-until-dry critic rather than
inheriting cycle-1's `dry` flag. Round 21 was **not** dry: it closed a gap
§C2 had explicitly recorded as unsourced.

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 40 | https://www.devopsschool.nl/alert-suppression/ | 2026-08-17 | **Community tier (lowest weight -- see caveat below)** | WebFetch full | Fetched specifically because §C2 recorded that the strong "suppression hides real failure" claim could NOT be sourced (#15 came back negative, #39 only partial). This source **does** state it, and adds two mechanisms the brief did not have. (i) **The claim, stated:** "Incorrect suppression can mask outages and violate SLOs"; the named failure mode "**Over-suppression: Missing critical pages** [Symptom] / **Broad rule matches** [Cause]"; and the postmortem narrative "After an outage, postmortem shows a critical alert had been suppressed by mistake. **Mistaken suppression masked root cause and delayed recovery.**" (ii) **Stale suppression** as a distinct failure: "**Alerts never resume** [Symptom] / **Suppression state never expires** [Cause]" -- i.e. an expiry/renewal obligation, which a *recurring* calendar rule structurally lacks. (iii) **Source-vs-delivery suppression**, an axis absent from the rest of the brief: "**Source-level suppression: Suppress before ingestion -- Saves cost -- Loses raw event context**" vs "**Delivery suppression: Suppress notifications at routing/delivery stage -- Flexible but late in pipeline -- Leaves storage of raw alerts intact**". (iv) The severity carve-out, in normative form: "**Suppression must never silence alerts tied to SLO-critical indicators unless explicit executive-approved maintenance windows exist**", remedied by "Add **severity whitelist** and test with alert replay". (v) An audit obligation: "**Auditability: every suppression event must be logged for postmortem.**" (vi) The decision rule: "If alert is non-actionable AND recurring -> consider suppression. **If alert maps to user-impacting SLI degradation -> do not suppress.**" **NEGATIVE, and it is the fifth independent one: business-calendar / weekend / holiday / business-hours-aware suppression is ABSENT** -- the doc's only time construct is the one-off maintenance window / blackout. |

**Tier caveat, stated rather than buried.** `devopsschool.nl` is **community
tier** -- the lowest weight in the source hierarchy
(`.claude/rules/research-gate.md` §"Source quality hierarchy"). It is
recorded because it *closes a gap the brief had honestly flagged as open*,
and because its **negative** on calendar-aware suppression independently
replicates the negatives from dbt (#5), Datadog (#16), Monte Carlo (#18),
Anomalo (#19) and Great Expectations (#20). It must **not** be used as the
sole support for a design decision: the load-bearing citations for
"demote, don't suppress" remain PagerDuty (#23, official) and Grafana
(#38, official). Treat #40 as corroboration, not as authority.

## B2. Snippet-only (context; does NOT count toward the gate)

**85 rows; 82 distinct URL strings; 79 that are snippet-only** (the other 3
are the same URL as a read-in-full row). Six rows are double-listings and are
tagged `[DUP -- not counted]` in place: three repeat an earlier row in this
same table; two carry a URL byte-identical to read-in-full #29 / #34 behind a
parenthetical suffix; one is the struck Datadog row upgraded to read-in-full
#16. **§B2b below adds 15 more snippet-only URLs from the cycle-2 dry-check
rounds, giving 94 snippet-only in total and `40 + 94 = 134` distinct URLs
brief-wide.** Arithmetic and root cause: §"Envelope reconciliation (cycle 2)"
above.

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://atlan.com/know/data-quality-alerts/ | Vendor blog | Adaptive-threshold claim already covered by #5 + #11; vendor tier |
| https://www.metabase.com/metrics/data-freshness | Vendor doc | Duplicate of dbt freshness model |
| https://uptrace.dev/blog/sla-slo-monitoring-requirements | Vendor blog | Subsumed by SRE Workbook (#1) |
| https://www.getdbt.com/blog/data-product-slas-and-slos | Vendor blog | Subsumed by dbt docs (#5) |
| https://support.freshservice.com/support/solutions/articles/156462-setting-default-business-hours | Vendor doc | Business-hours-vs-calendar-hours SLA distinction; ITSM tier, not data-pipeline |
| https://venue.cloud/news/insights/from-slos-to-status-pages-your-e-commerce-thanksgiving-uptime-playbook | Industry blog | Holiday-aligned error-budget windows; low authority |
| https://hackernoon.com/building-data-observability-monitoring-nulls-drift-freshness-and-business-impact | Community | Community tier |
| https://sreschool.com/blog/alert-deduplication/ | Community | Dedup taxonomy; superseded by #6 |
| https://support.squadcast.com/services/alert-deduplication-rules/alert-deduplication-rules | Vendor doc | Dedup-key mechanics; same content as our AlertDeduper |
| https://garystringham.com/level-triggered-vs-edge-triggered-interrupts/ | Blog | Edge/level from the hardware-interrupt side; conceptual only |
| https://mechatronicslab.net/courses/c-programming-for-embedded-systems/lessons/edge-vs-level-triggered/ | Course | ditto |
| https://learn.netdata.cloud/docs/alerts-&-notifications | Vendor doc | Alert state model; subsumed by Grafana (#4) |
| ~~https://docs.datadoghq.com/monitors/types/data_observability/~~ | -- | **UPGRADED in round 5 -- read IN FULL as #16. Struck from this table so it is not double-counted.** |
| https://www.synq.io/blog/data-observability-guide | Vendor blog | 2025 guide; vendor tier |
| https://www.dqlabs.ai/blog/the-definitive-guide-for-data-observability-2026/ | Vendor blog | 2026 guide; vendor tier |
| https://www.siffletdata.com/blog/data-freshness | Vendor blog | vendor tier |
| https://www.actian.com/blog/data-observability/data-observability-metrics/ | Vendor blog | vendor tier |
| https://www.conduktor.io/glossary/data-freshness-monitoring-sla-management | Vendor glossary | vendor tier |
| https://www.thedataops.org/the-ultimate-guide-to-predictive-alerts-and-data-observability/ | Community | Prophet/Holt-Winters for freshness bounds |
| https://cronalert.com/blog/http-health-check-endpoints | Vendor blog | "A health check should not write to the database, increment counters, or trigger any state change. It is read-only." |
| https://reliableuptime.com/blog/rest-api-health-check-endpoint-design | Vendor blog | "read state, not modify it"; "alerts on consecutive failures... rather than on every read" |
| https://learn.microsoft.com/en-us/previous-versions/msp-n-p/dn589789(v=pandp.10) | Official (archived) | Superseded by the live Azure page (#7) |
| https://github.com/rsheftel/pandas_market_calendars/blob/master/docs/change_log.rst | Official changelog | CME no longer mirrors NYSE -> "some holidays missing until this calendar is fully tested and vetted"; BSE 2025/2026 + JPX 2026/2027 holiday fixes -- evidence calendars are patched *in arrears* |
| https://github.com/rsheftel/pandas_market_calendars/issues | Issue tracker | Live evidence of calendar-correctness bug reports |
| https://arxiv.org/pdf/2606.19386 | Preprint | "Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence" -- surfaced in the edge/level search; adjacent, not on-topic |
| https://healthchecks.io/docs/monitoring_cron_jobs/ | Official doc | Same content as the configuring_checks page read in full (#10) |
| https://healthchecks.io/about/ | Official doc | Marketing page |
| https://docs.thousandeyes.com/product-documentation/alerts/alert-clearing/alert-suppression-windows | Vendor doc | Suppression-window mechanics; duplicate of Grafana (#38) |
| https://upstat.io/blog/alert-suppression-best-practices | Vendor blog | "critical alerts bypass suppression entirely"; low authority, same claim as PagerDuty (#23) |
| https://hyperping.com/blog/maintenance-windows-monitoring-guide | Vendor blog | Maintenance-window how-to |
| https://dohost.us/index.php/2026/04/28/intelligent-suppression-silencing-alerts-during-scheduled-maintenance-windows/ | Low-authority blog | "If 90% of alerts are just maintenance noise, the 10% that are real outages will be ignored" -- rhetorically apt, source tier too low to cite |
| https://www.servicenow.com/community/itom-blog/alert-suppression-during-maintenance-windows-and-change/ba-p/2273469 | Vendor community | ITSM framing |
| https://arxiv.org/html/2510.11141v1 | Preprint 2025 | Forecasting-based TSAD benchmarking on NAB; method-level, not alerting-policy |
| https://arxiv.org/html/2508.21128 | Preprint 2025 | Collective/point anomalies under trend + seasonality; method-level |
| https://arxiv.org/pdf/2510.17562 | Preprint 2025 | TSAD evaluation metrics; method-level |
| https://arxiv.org/pdf/2112.03196 | Preprint | Online FDR control for TS anomalies; a principled alternative to a fixed threshold, but heavy for this step |
| https://arxiv.org/abs/2607.02046 | Preprint 2026 | "Fast and Accurate Anomaly Detection in Time Series"; method-level |
| https://github.com/apache/airflow/discussions/47040 | Official tracker | `sla_miss_callback` not triggered when SLA missed -- evidence Airflow's SLA leg is itself unreliable |
| https://airflow.apache.org/docs/apache-airflow/2.2.4/core-concepts/tasks.html | Official doc | "Only scheduled tasks will be checked against SLA" |
| https://blog.devops.dev/detecting-silent-failures-tracking-down-missing-metrics-in-prometheus-56c5746b1a26 | Community | `absent()` vs `absent_over_time()` mechanics |
| https://github.com/sapcc/absent-metrics-operator/blob/master/README.md | OSS repo | Auto-generates absence alerts from existing rules |
| https://grafana.com/blog/2021/04/08/how-we-use-metamonitoring-prometheus-servers-to-monitor-all-other-prometheus-servers-at-grafana-labs/ | Vendor eng blog | Metamonitoring at scale; duplicate of #24 |
| https://www.limod.de/posts/02-deadmanswatch-alertmanager/ | Blog | Deadmanswatch wiring |
| https://support.pagerduty.com/main/docs/configurable-service-settings | Official vendor doc | Support-hours settings surface; duplicate of #23 |
| https://support.pagerduty.com/main/docs/event-orchestration-examples | Official vendor doc | Time-based suppression via orchestration rules |
| https://dev.to/pdcommunity/better-sleep-with-pagerduty-dynamic-notifications-and-support-hours-4jkp | Community | Walkthrough of #23 |
| https://community.pagerduty.com/ask-a-product-question-2/suppress-specific-incidents-between-hours-of-10pm-6am-761 | Community | Practitioner Q&A on hour-based suppression |
| https://medium.com/@senthilcegptech/the-ghost-quote-problem-identifying-stale-data-in-trading-analytics-238ec57adfe5 | Community | "Ghost quote" framing; "Your staleness monitor should be aware of Market Hours" |
| https://dev.to/kalos889/how-to-auto-detect-forex-market-holidays-with-api-data-streams-3nje | Community | Inferring holidays from feed silence -- the inverse (and circular) approach |
| https://www.tradinghours.com/data | Vendor API | **FETCH FAILED -- HTTP 403.** Would have sourced the "markets publish schedules 1-3 years in advance" claim; superseded by the NYSE primary source (#32) which states three years |
| https://www.tradinghours.com/ | Vendor | ditto |
| https://www.financecalendar.com/stock-market-holidays/ | Aggregator | 2026 NYSE/Nasdaq holiday list; superseded by #32 |
| https://mrtopstep.com/calendars/nyse-holidays/ | Aggregator | ditto |
| https://howmanytradingdays.com/ | Aggregator | "Scheduled early-close days ... are counted as 0.5 trading days" -- the half-day-weighting convention |
| https://github.com/wilsonfreitas/python-bizdays | OSS repo | Business-day calc; no exchange sessions |
| https://github.com/gocardless/business-python | OSS repo | ditto |
| https://github.com/botant/py-business-calendar | OSS repo | ditto |
| https://pypi.org/project/bizcal | OSS package | Claims holiday-awareness that says "not just when not trading, but also why" -- interesting but immature |
| https://github.com/quantopian/trading_calendars | OSS repo (archived) | The ancestor of `exchange_calendars`; unmaintained |
| https://github.com/apptastic-software/trading-calendar | OSS repo | REST API for 60+ exchange calendars -- a network dependency, wrong shape for a monitor |
| https://github.com/rsheftel/pandas_market_calendars/blob/master/docs/change_log.rst | Official changelog | `[DUP -- not counted]` Second listing of the changelog row already in this table above; content identical (CME "may cause some holidays missing until this calendar is fully tested and vetted"; BSE 2025/26 + JPX 2026/27 fixes in arrears) |
| https://apscheduler.readthedocs.io/en/latest/modules/triggers/cron.html | Official doc | APScheduler cron trigger -- has `day_of_week` but **no holiday support** (the 51.3 rationale) |
| https://pypi.org/project/croniter/ | Package index | Duplicate of #30 |
| https://docs.cloud.google.com/monitoring/alerts/concepts-indepth | Official doc | Metric-absence vs metric-threshold condition semantics |
| https://medium.com/@nishchithaspirangi07/when-silence-isnt-golden-handling-no-data-in-grafana-alerting-b10f1a7144fd | Community | "No Data" handling; duplicate of #4 |
| https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-mute-rules.html | Official doc | CloudWatch alarm mute rules -- a third vendor with the same "mute notification, keep evaluating" shape |
| https://github.com/prometheus/alertmanager/issues/2778 | Official tracker | Feature request: mute when time is OUTSIDE an interval (i.e. `active_time_intervals`) |
| https://github.com/prometheus/alertmanager/discussions/3822 | Official tracker | "Repeating silences during specific time periods?" -- the community keeps asking for recurring suppression |
| https://github.com/prometheus/alertmanager/issues/3368 | Official tracker | Live bug reports against `time_intervals` |
| https://blog.anomalyarmor.ai/data-pipeline-monitoring-how-to-stop-silent-failures-before-they-hit-production/ | Vendor blog | The narrative twin of our defect: "a pipeline that ran perfectly Monday through Friday but failed every weekend ... paged every Saturday and Sunday morning", root cause a hard-coded row-count floor |
| https://www.elementary-data.com/post/data-freshness-best-practices-and-key-metrics-to-measure-success | Vendor blog | "A fixed freshness SLA ... must have data no older than a specified time **during business days**" |
| https://www.paradime.io/guides/blog-dbt-source-freshness-best-practices | Vendor blog | "set the SLA slightly longer than the expected update interval -- a table that updates every hour should have a 2-hour SLA, not a 61-minute one" |
| https://oneuptime.com/blog/post/2026-01-30-freshness-slos/view | Industry blog 2026 | Freshness-SLO how-to; same family as #14, no new mechanism |
| https://oneuptime.com/blog/post/2026-03-05-alert-fatigue-ai-on-call/view | Industry blog 2026 | Alert-fatigue-and-AI framing |
| https://pingfatigue.com/research | Citation aggregator | Aggregates the 2024-2026 alert-fatigue numbers (PagerDuty 2025: ~50 alerts/wk, 2-5% actionable; incident.io 2025: 67% of engineers dismiss alerts, 85% of teams say most alerts are false positives; Catchpoint 2024: 70% rank alert fatigue top-3; Splunk 2025: 73% had outages linked to ignored alerts). **Second-hand -- cited as context only, NOT as a primary measurement** |
| https://incident.io/blog/on-call-best-practices-guide-2026 | Vendor blog 2026 | On-call practice; superseded by #35 |
| https://docs.tecton.ai/docs/monitoring/alerting (stream-lag page) | Official doc | `[DUP -- not counted]` **Same URL as read-in-full #29** -- the parenthetical names an unfetched *section*, not a second page. Note: the `expected_feature_freshness` default derivation was not read. This row is one of the two that caused the cycle-1 `urls_collected` over-claim |
| https://www.ibm.com/docs/en/apptio-gov/tbm-studio/saas?topic=architecture-data-freshness-validation | Official doc | Monthly-refresh freshness tracking |
| https://streamkap.com/resources-and-guides/data-freshness-monitoring | Vendor guide | Streaming-freshness SLO examples |
| https://www.sciencedirect.com/science/article/pii/S0304407623002385 | Peer-reviewed (J. Econometrics) | "Systematic staleness" -- **econometrics of stale PRICES, a different meaning of the word**; recorded so a later reader does not mistake it for pipeline staleness |
| https://sre.google/sre-book/example-postmortem/ | Official doc | Google's example postmortem; no read-path-alerting case |
| https://www.datadoghq.com/blog/reduce-alert-storms-datadog/ | Vendor blog | Alert-storm reduction |
| https://docs.aws.amazon.com/wellarchitected/latest/devops-guidance/anti-patterns-for-continuous-monitoring.html (index) | Official doc | `[DUP -- not counted]` **Same URL as read-in-full #34** -- the parenthetical was meant to mean "the parent index", but the URL string is byte-identical to #34, so it is not a second page. The other of the two rows that caused the cycle-1 over-claim |
| https://cronalert.com/blog/http-health-check-endpoints | Vendor blog | `[DUP -- not counted]` Second listing of a row already in this table above |
| https://reliableuptime.com/blog/rest-api-health-check-endpoint-design | Vendor blog | `[DUP -- not counted]` Second listing of a row already in this table above |

### B2b. Snippet-only added by the cycle-2 dry-check rounds 21-23

Recorded so an auditor can see what rounds 21-23 actually *evaluated* before
being declared dry -- a dry round with no visible evaluation is unfalsifiable.

| URL | Kind | Round | Why not fetched in full |
|---|---|---|---|
| https://oneuptime.com/blog/post/2026-01-27-alertmanager-silences/view | Industry blog 2026 | 22 | Silence-expiry hygiene ("do not forget to expire silences"; "if you find yourself renewing the same silence repeatedly, either fix the underlying issue or delete the alert"). **Same mechanism as #39 + #40** -- de-dup, no new finding |
| https://oneuptime.com/blog/post/2026-02-09-prometheus-alert-silences-maintenance/view | Industry blog 2026 | 22 | Maintenance-window silences; duplicate of #38/#39 |
| https://github.com/oneuptime/blog/tree/master/posts/2026-01-27-alertmanager-silences | Repo mirror | 22 | Source repo of the post above; same bytes |
| https://grafana.com/docs/grafana/latest/alerting/configure-notifications/create-silence/ | Official doc | 22 | The *silence* half of #38, which already contrasts silences (one-off) with mute timings (recurring). No new mechanism |
| https://github.com/prometheus/alertmanager/issues/1057 | Official tracker | 22 | "Auto-expire silences upon alert resolution" -- a 4th instance of the same class already represented by trackers 2778 / 3822 / 3368; feature request, not shipped behaviour |
| https://brtkwr.com/posts/2025-12-10-alertmanager-silences/ | Personal blog | 22 | Practitioner walkthrough; community tier, content subsumed by #38/#39 |
| https://medium.com/@subham11/health-check-is-my-service-alive-pattern-for-beginners-70f917afe1fe | Community | 22 | Health-check primer; subsumed by #7 + #21 |
| https://www.geeksforgeeks.org/system-design/health-endpoint-monitoring-pattern/ | Community | 22 | ditto |
| https://www.jeisystems.co.uk/tech-blog/code-examples/health-endpoint-monitoring-pattern/ | Community | 22 | ditto |
| https://medium.com/@manikumarthati/health-check-api-pattern-ensuring-service-availability-in-microservices-1e83dadbde98 | Community | 22 | ditto |
| https://www.tradingsim.com/blog/us-stock-exchange-calendar | Retail blog | 21 | 2026 trading-day count (252); superseded by the NYSE primary source #32 |
| https://www.coppclark.com/ | Vendor | 21 | Commercial business-day/settlement calendar feed; a paid network dependency, wrong shape for a monitor (same objection as `apptastic-software/trading-calendar`) |
| https://www.marketholidays.com/todaysHolidays.aspx | Aggregator | 21 | Holiday aggregator; superseded by #32 |
| https://dohost.us/index.php/tag/maintenance-windows/ | Low-authority blog | 21 | Tag index of a post already recorded and already judged too low-tier to cite |
| https://sreschool.com/blog/downtime/ | Community | 21 | Downtime definitions; no suppression-risk mechanism beyond #40 |

**Round-22 negative worth recording:** the query
`exchange_calendars holiday data incorrect 2026 bug report calendar accuracy
python` returned **only Microsoft Exchange Server / Outlook results** -- the
search engine collides on the word "Exchange". Zero relevant hits. The
library-accuracy evidence in this brief therefore still rests on the two
primary artifacts already recorded: the `exchange_calendars` README's own
"***All*** of the exchange calendars are maintained by user contributions"
(#8) and the `pandas_market_calendars` changelog showing holiday corrections
landing **in arrears** (§B2). **This is a coverage limitation, stated: I
could not find independent third-party auditing of exchange-calendar
correctness, and the absence of such auditing is itself the finding.**

## B3. Recency scan (last 2 years, 2024-2026) -- MANDATORY, and non-empty

Searched explicitly for 2024-2026 work on calendar-aware freshness alerting,
seasonality-aware anomaly detection, and alert-suppression risk.
**Result: 12 findings in the window, and they CHANGE the recommendation.**

1. **arXiv:2502.05392 (2025)** -- states the July-4th-store-closure false
   positive as a named open challenge and argues static thresholds are not
   benchmark-defensible. Supersedes nothing, but it means "just tune the
   ratio" is a known dead end.
2. **arXiv:2510.24452 (2025, Google BigQuery ARIMA_PLUS)** -- establishes
   **holiday adjustment BEFORE outlier cleaning** as the canonical ordering.
3. **Databricks anomaly detection (config frozen for new customers
   2025-07-21)** -- the industry is moving from operator-set thresholds to
   **learned next-commit prediction**, with manual thresholds actively
   withdrawn.
4. **Sifflet Dynamic Mode** -- learned injection pattern; the Mon/Thu example
   is the nearest shipping analogue of a weekday-aware freshness monitor.
5-8. **OneUptime 2026-01-30 (batch alerting, `business_days_only`),
   2026-02-06 (heartbeat/dead-man's-switch), 2026-02-09 (cronjob missed
   schedules), 2026-02-17 (metric-absence policies)** -- four independent
   2026 posts converging on **heartbeat + absence detection + schedule-aware
   expectation**, and one of them documents the weekday-cron parser bug that
   IS our defect.
9. **OneUptime 2026-01-30 (alert suppression rules)** -- names "Silencing
   instead of fixing" / "Alert debt".
10. **pandas_market_calendars changelog** -- BSE 2025/2026 and JPX 2026/2027
   holiday corrections landed **in arrears**; CME "may cause some holidays
   missing until this calendar is fully tested and vetted".
11. **NYSE (primary)** -- calendar published three years out (2026-2028),
   with 2026 early closes Nov 27 / Dec 24 / Jul 3.
12. **Alert-fatigue survey numbers 2024-2026** (PagerDuty 2025, incident.io
   2025, Catchpoint 2024, Splunk 2025 -- all second-hand via
   pingfatigue.com, flagged as such).

**Effect on the plan:** the recent window pushes AWAY from "add a
weekend term to the ratio" and TOWARD (i) keeping detection calendar-blind,
(ii) making the calendar govern *notification/urgency*, and (iii) adding a
positive heartbeat rather than only a negative age check.

**Recency scan -- cycle-2 addendum (rounds 21-23, 2026-08-17).** Three
further recency passes were run in the reconciliation session, mixing the
year-less canonical variant with 2025/2026-scoped queries. Result:
**one new finding (#40) and no supersession.** The 2026 material surfaced in
rounds 22-23 (OneUptime silences 2026-01-27 / 2026-02-09, Grafana silence
docs, `alertmanager#1057`, brtkwr 2025-12-10, plus the weekday-SLA hits) all
restate mechanisms already recorded from #38/#39/#36/#37/#10/#28 -- **no
2024-2026 source read in this window supersedes any canonical source in this
brief.** The one genuinely new item, #40, is undated and community-tier, so
it is recorded as corroboration rather than as a recency finding. **Explicit
negative for the window:** repeated searching found **no** 2024-2026 source
that documents a business-calendar-aware freshness monitor as a shipped
feature -- the count of vendors checked and found lacking is now 8 of 8
(§C1 finding 8), and #40 makes it a ninth independent negative on the
suppression side.

---

# PART C -- SYNTHESIS

## C1. Key findings (each cited)

1. **The literature's answer to "what instead of suppression" is
   unambiguous and tri-vendor: keep EVALUATING, change only the
   NOTIFICATION.** Grafana: a mute timing "suppresses notifications but does
   not interrupt alert evaluation" (#38). PagerDuty: outside support hours
   "incidents are created regardless ... but **how responders are notified**
   varies" (#23). Alertmanager: a muted route "will not send any
   notifications" yet "**otherwise acts normally**" (#11). CloudWatch alarm
   mute rules are the same shape. **Three independent implementations agree
   that the calendar belongs on the routing leg, never on the detector.**
2. **A pure age/interval ratio must misfire across a weekend, arithmetically.**
   `_band` (`cycle_health.py:101-109`) is `age_sec / interval_sec` with
   `interval_sec = 86400` for `paper_trades`/`signals_log`
   (`freshness_cron.py:139-141`). A Friday-15:00 write is 2.0x by
   **Sunday 15:00** and stays red until the Monday cycle -- so red is the
   CORRECT reading of the ratio and the ratio is the wrong question. Measured:
   30.0% of pages land Sat/Sun and **38.4% land on Monday** (A8).
3. **Therefore weekend suppression alone is insufficient AND dangerous.**
   Insufficient because Monday is the largest bucket (165 of 430) and a
   Sat/Sun mute does not touch it. Dangerous because a Friday-dead writer is
   indistinguishable from an idle weekend under an age-only detector -- the
   AWS "**no dogs barking**" anti-pattern (#34), and "Silencing instead of
   fixing ... Alert debt accumulates" (#39). Monte Carlo's shipping
   exclusion-window feature does exactly this and its docs carry **no warning
   at all** (#17).
4. **The correct primitive is an EXPECTED-NEXT-WRITE, not an age.**
   healthchecks.io separates *schedule* from *grace*: cron mode "ensures that
   jobs run **at the correct time** and not just at the correct time
   intervals" (#10). Airflow's own users asked for "a timedelta since **the
   last successful run**" (#22). Monte Carlo: "match the run schedule"
   (#18). pyfinagent already has this primitive for the cycle clock
   (`_CYCLE_COMPLETED_STALE_SEC`, `cycle_health.py:80`) but **not** for the
   table bands.
5. **A positive heartbeat beats a negative age check for the dead-writer
   case.** "A service that stops reporting metrics is often in worse shape
   than one reporting elevated error rates ... the error rate is not high, it
   is **nonexistent**" (#31); the prescribed fix for expected-idle periods is
   a heartbeat "**regardless of traffic**", not suppression. The Prometheus
   `Watchdog` `expr: vector(1)` + external absence-detector is the canonical
   form (#24).
6. **An HTTP GET must not page.** RFC 9110 §9.2.1: safe methods are
   "essentially **read-only**; i.e., the client does not request, and does not
   expect, any state change on the origin server" (#21). Azure states the
   remedy verbatim: "**if the health status is reported through a dashboard,
   you don't want every request to the dashboard to trigger a health check.
   Instead, periodically check the system health, and cache the status**"
   (#7). The nuance that matters: RFC 9110 *permits* server-side side effects
   but says "**the client cannot be held responsible**" -- so the design fault
   is not "a GET had an effect", it is "**a poller was made the trigger of a
   page**".
7. **The alarm is 4.4x over Google's own pager ceiling.** SRE Workbook: "We
   target a **maximum of two incidents per on-call shift**" and "**All alerts
   should be immediately actionable**" (#35). Measured 17.5 pages/day = ~8.75
   per 12h shift from this ONE alarm for ONE already-known condition (A8).
   Huawei's 4M-alert study names it **[A5] Repeating Alerts**, agreed
   significant by **94.4%** of on-call engineers (#6).
8. **No off-the-shelf trading-calendar-aware freshness monitor exists.**
   Confirmed twice by explicit search (rounds 16 and 20) and by reading dbt
   (#5), Datadog (#16), Monte Carlo (#18), Anomalo (#19), Great Expectations
   (#20), Tecton (#29), Databricks (#36), Sifflet (#37) in full: **8 of 8
   carry no business-day/holiday configuration.** Even a capital-markets
   staleness paper (#25) omits session awareness entirely. pyfinagent would
   be building this, not adopting it.
9. **Calendar libraries are community-maintained and patched in arrears.**
   exchange_calendars: "***All*** of the exchange calendars are maintained by
   user contributions" (#8). pandas_market_calendars shipped BSE 2025/26 and
   JPX 2026/27 fixes after the fact, and warns CME "may cause some holidays
   missing until this calendar is fully tested and vetted". NYSE itself
   publishes 3 years ahead (#32) -- so the *upstream truth* is knowable long
   in advance; the *library* is the weak link.
10. **Half-days are sessions, and observance rules are exchange-specific.**
    NYSE 2026 early closes: Nov 27, Dec 24, Jul 3, all 1:00 p.m. ET (#32).
    `is_session` returns **True** on a half-day, so a trading-day gate
    correctly does NOT skip them -- but a *volume/age* expectation
    calibrated on full days will be wrong. And "Because the holiday falls on
    Saturday, January 1, 2028, no New Year's Day holiday is observed" is a
    rule no `weekday() < 5` test can express.

## C2. Consensus vs debate

**Consensus (high confidence):**
- Calendar logic belongs on notification/urgency, not detection (#11, #23,
  #38, CloudWatch).
- Every page must be actionable; non-actionable pages destroy trust
  (#2, #3, #6, #34, #35).
- Edge-triggered / state-transition beats level-triggered for a
  persistently-true condition (#1 reset time, #4 state machine, #6 [A5],
  and pyfinagent's own `freshness_cron.py:146-151`).
- A dashboard read path should not perform the expensive check, let alone
  page (#7, #21, #34).

**Genuine debate / unresolved:**
- **Declarative calendar vs learned pattern.** Alertmanager/PagerDuty/Grafana
  say declare it; Databricks (#36) and Sifflet (#37) say learn it, and
  Databricks is actively *removing* manual thresholds. For pyfinagent the
  declarative route wins on evidence -- 6 monitored tables, one known
  schedule, and a 3-7 day learning window (#16) would be re-learning a
  calendar that NYSE already published to 2028.
- **Does suppression actually hide failures?** *(UPDATED by cycle-2 round 21
  -- the earlier text is preserved so the change in epistemic status is
  visible, not silently overwritten.)* Cycle 1 recorded this as
  **UNSUPPORTED** at #15 (the obvious vendor source does not argue it) and
  only **partially corroborated** at #39 ("Alert debt", "miss cascading
  failures") and #34 ("no dogs barking"), concluding that nobody in the
  sources read stated it and that Main should present it as a reasoned design
  constraint rather than a cited finding.
  **Round 21 found a source that does state the general claim -- #40** --
  "Incorrect suppression can mask outages and violate SLOs", the named
  failure mode "Over-suppression: Missing critical pages / Broad rule
  matches", and "postmortem shows a critical alert had been suppressed by
  mistake ... masked root cause and delayed recovery". **Three qualifications
  keep this from being a clean win:**
  (i) #40 is **community tier**, the lowest weight in the hierarchy -- it
  raises the claim from *unsupported* to *supported only at the weakest
  tier*, not to *established*;
  (ii) it states the **general** claim, and **still nobody states the sharp
  one** -- "a weekend mute hides a Friday-dead writer" remains uncited, and
  pyfinagent's own Monday-spike measurement (A8) remains its strongest
  support;
  (iii) it supplies a **different and stronger argument** that does not
  depend on the disputed one: **"Stale suppression -- Alerts never resume /
  Suppression state never expires."** Every source that endorses suppression
  pairs it with a **finite duration** and an expiry obligation (#38 silences
  are one-off with a fixed end; #39 "Set Conservative Durations"; #40 audit +
  expiry). **A recurring weekend/holiday mute is by construction a
  suppression that never expires** -- it re-arms itself every week, forever,
  with no human ever re-approving it. That is the strongest literature-backed
  objection to the weekend-mute design, and unlike the "hides a dead writer"
  claim it is directly cited. Main should lead with the never-expires
  argument.
- **Absolute age vs statistical inter-arrival.** dataintellect (#25) argues
  for a Poisson model at 1-in-a-billion; everyone else uses a multiplier
  (2x-3x) of the expected interval (#28, #31, Paradime). For a once-daily
  batch, N is far too small for the Poisson route.

## C3. Pitfalls (from the literature, each mapped to a concrete trap here)

| # | Pitfall | Source | Trap for 86.109 |
|---|---|---|---|
| P1 | The weekday-cron parser that silently defaults to daily | #28 (verbatim: `0 2 * * 1-5` "NOT properly handled ... would incorrectly default to daily") | If Main encodes the schedule as a cron string, a naive parser reproduces the bug it is fixing |
| P2 | Fail-open direction inverts when you negate the predicate | A3 + `news/fetcher.py:162` | `is_market_closed()` failing open to `True` **silences** the alarm; `is_trading_day()` failing open to `True` **keeps it loud**. Use the latter polarity |
| P3 | A second, drifting calendar notion in one module | A2 (`is_weekday_et` at `:262` is holiday-blind) | A holiday-aware `_band` beside a holiday-blind `is_weekday_et` disagree ~9-10 weekdays/yr |
| P4 | Suppression that also suppresses the training signal | #17 (Monte Carlo excludes from BOTH) | Do not let a calendar gate remove data the alarm needs to notice a real death |
| P5 | Exclusion/mute left on | #39, #15, **#40 (cycle-2)** | A permanent Sat/Sun mute never expires and nobody re-reads it. #40 names this as its own failure mode -- "**Stale suppression: Alerts never resume / Suppression state never expires**" -- and every source that endorses suppression pairs it with a finite duration (#38 one-off silences; #39 "Set Conservative Durations"). **A recurring calendar mute is a suppression that re-arms itself forever with no human re-approval**; that is the strongest *cited* objection to the weekend-mute design (§C2) |
| P6 | Tuning the threshold instead of the model | #12 §5.3, #14 | Raising `CRITICAL_RATIO` to 3.5 hides Monday too -- and hides a genuinely dead writer for 3.5 days |
| P7 | croniter without a tz-aware datetime | #30 ("**Be sure to init your croniter instance with a TZ aware datetime for this to work!**") | ET vs UTC; pyfinagent already anchors on `_NYSE_TZ` -- keep it |
| P8 | Half-day miscalibration | #32, "0.5 trading days" convention | Nov 27 / Dec 24 / Jul 3 2026 are sessions with ~40-50% of normal volume |
| P9 | Calendar lib patched in arrears | #8, #9, pmc changelog | Pin + test the specific dates you rely on, as `test_phase_50_4_calendar.py` already does |
| P10 | Fixing the emitter but leaving three ungated callers | A4, A7 | The 82.10 fix gated the CRON; the three HTTP sites still default `emit_alarm=True` |
| P11 | Suppressing at the wrong point in the pipeline | **#40 (cycle-2)**: "Source-level suppression: Suppress **before ingestion** -- Saves cost -- **Loses raw event context**" vs "Delivery suppression: Suppress notifications at routing/delivery stage -- ... **Leaves storage of raw alerts intact**" | Names the axis that decides `_band` vs `notify()`. Putting the calendar in `_band` (`cycle_health.py:101-109`) is **source-level** -- the red state is never computed, so the dashboard goes green and the evidence is gone. Putting it on the `notify(...)` call (`freshness_cron.py:162-175`) is **delivery-level** -- the state is still computed, still visible, only the page is withheld. The literature's preference (#38 "**suppresses notifications but does not interrupt alert evaluation**") is delivery-level, and it is the same conclusion §C4 already reaches -- now with the general principle behind it, not just the Grafana instance |
| P12 | **The severity whitelist does not work here, and that is the finding** | **#40 (cycle-2)**: "Suppression must never silence alerts tied to SLO-critical indicators unless explicit executive-approved maintenance windows exist"; fix = "Add **severity whitelist**" | The literature's standard carve-out is *"never suppress critical"*. **Re-verified live 2026-08-17: it is inapplicable to 86.109, because the routine false positive is ALREADY critical.** `freshness_cron.py:162-166` fires the freshness page with a literal `severity="P1"`, and `alerting.py:54` defines `_CRITICAL_SEVERITIES = frozenset({"P0", "P1", "critical", "CRITICAL"})`, so every one of the measured 991 Slack deliveries took the critical branch at `:83-93` -- bypassing the consecutive-occurrence threshold, with only the 1h repeat window applying. **The weekend noise and a genuinely dead writer are emitted at the same severity by the same call site**, so no severity-based rule can separate them. Two consequences for Main: (i) do not reach for a severity whitelist as the safety net -- it would either mute everything or nothing; (ii) if a calendar gate goes on `notify()`, the thing that must stay loud has to be distinguished by **what the condition is** (e.g. an expected-next-write miss, per §C4) and **not** by its severity label. Optionally re-grade routine staleness below P1 first, which would then make the standard carve-out usable |

## C4. Application to pyfinagent (external finding -> file:line)

**The single highest-leverage change, and it is one line-ish.** Per #7
(Azure), #21 (RFC 9110) and #34 (AWS), the dashboard read path should not
page **at all**. `freshness_cron.py` already owns the transition gate
(`:146-151`) and already runs every 6h (`:62`). The three HTTP sites --
`backend/api/paper_trading.py:498`, `backend/api/observability_api.py:36`
and `:55` -- simply need `emit_alarm=False`. That alone removes the polling
term: `CycleHealthStrip.tsx:105` polls `window.setInterval(tick, 30_000)`,
i.e. **2,880 evaluations/day per open browser tab**, which is the engine
behind the measured 17.5 pages/day and the 61-minute p75 inter-arrival
(= AlertDeduper's 1h P1 repeat window, `alerting.py:83-93`, saturated).
Note this requires **no calendar work at all** and is the change the
literature supports most strongly.

**Then, and only then, the calendar question.** Per #11/#23/#38, put the
calendar on the *notification* leg inside `freshness_cron.run_freshness_check`
-- NOT inside `_band` (`cycle_health.py:101-109`). Concretely: keep `_band`
calendar-blind so the dashboard keeps showing red (Grafana: "does not
interrupt alert evaluation"), and gate only the `notify(...)` call at
`freshness_cron.py:162-175`. Reuse the proven shape at
`backend/slack_bot/scheduler.py:365-375` (`_is_us_trading_day_now`,
ET-today, fail-open via `markets.is_trading_day`) -- **which does exist and
does satisfy the criterion-2 premise** (A1).

**Do not add a day term to `_band`.** It would be the fourth calendar notion
in `cycle_health.py` (A2) and would silence the dashboard, not just the pager.

**Close the Monday hole with an expected-next-write term, not a wider
ratio.** Per #10/#22/#28/#31: the right question on Monday 09:00 ET is not
"is `paper_trades` 66h old?" but "**has the cycle that was due at 10:00 ET
today run yet, and did the last one that was due actually run?**"
pyfinagent already computes exactly that for the cycle clock
(`cycle_health.py:264-297`, `success_age_sec` / `should_alarm_success`) and
already fires it through a transition gate
(`slack_bot/scheduler.py:837-838`). **The cheapest correct design is to let
the table bands defer to the cycle clock rather than invent a second
schedule model.**

**Fail-open polarity (P2).** Write the guard as
`if not is_trading_day(et_today, "US"): skip notify` so that a missing
`exchange_calendars` returns `True` -> **notify** -> loud. Never
`if is_market_closed(): skip`.

**Audit trail for what was suppressed (#39).** `freshness_cron.py:184-192`
already logs still-red at WARNING without paging -- extend the same idiom to
non-trading-day suppressions so a suppressed weekend leaves a durable trace
("Monitor what you suppress. And always, always document why").

## C5. Internal code inventory

| File | Anchors | Role | Status |
|---|---|---|---|
| `backend/services/cycle_health.py` | 754 lines; `_band` :101-109, `_fire_freshness_alarm` :123-158, `is_weekday_et` :262, `compute_freshness` :640-753, emit at :732-733 | The detector + the level-triggered pager | **The change site.** 3 calendar notions already, none holiday-aware |
| `backend/services/freshness_cron.py` | 268 lines; `_last_red_sources` :69, gate :146-151, notify :162-175, still-red log :184-192, `DEFAULT_INTERVAL_HOURS=6` :62 | The only gated caller | Healthy; the model to extend |
| `backend/backtest/markets.py` | `is_trading_day` :192-213, `get_trading_calendar` :168-189 | Trading-day predicate | Correct; **3 fail-open paths all return True** |
| `backend/slack_bot/scheduler.py` | `_is_us_trading_day_now` :365-375; used :565, :610; transition flags :101/:113/:121/:125, consumed :690-691/:751-752/:808-809/:837-838 | **The 51.3 digest application** + 4 transition gates | Proven in production; reuse it |
| `backend/services/observability/alerting.py` | `AlertDeduper` :63-107, P1 branch :83-93, `_CRITICAL_SEVERITIES` :54, phase-66 note :46-53 | Dedup | P1 = repeat-window only (1h default) |
| `backend/api/paper_trading.py` | import :26, route :477, call :498 | HTTP caller #1 | **Ungated** (`emit_alarm` defaults True) |
| `backend/api/observability_api.py` | :36, :55 (function-local imports) | HTTP callers #2, #3 | **Ungated** |
| `frontend/src/components/CycleHealthStrip.tsx` | poll :105 `setInterval(tick, 30_000)`; error latch :82 | The poller | 2,880 evaluations/day/tab |
| `frontend/src/lib/format.ts` | :243 | Comment: "`is_trading_day` + exchange_calendars owns the authoritative gate; this is a UI hint" | Doc only |
| `backend/news/fetcher.py` | :162 | Comment: deliberately does NOT use `is_trading_day` because "it fails OPEN" | Precedent for P2 |
| `backend/services/autonomous_loop.py` | import :663, call :674 `is_trading_day(local_date, mk)` | **Second production caller of `is_trading_day`** -- added by the cycle-2 re-derivation (A1b); also uses a function-local import | Not a change site, but it is the other consumer whose behaviour a calendar change must not perturb |
| `backend/tests/test_phase_50_4_calendar.py` | 13 refs; US/EU/KR incl. Seollal + Chuseok | Calendar tests | Green; the fixture model for date pinning |
| `backend/tests/test_phase_51_3_digest_guard.py` | :45-49 monkeypatches `backend.backtest.markets.is_trading_day` both ways | Digest-guard tests | The test idiom to copy |
| `backend/tests/test_phase_82_10_freshness_paging.py` | :182-219 scheduler->evaluator->`compute_freshness` path; :462-466 asserts `emit_alarm` default True + keyword-only | Paging tests | **:462 pins the current default -- changing the HTTP sites does not break it, but re-read it before editing** |
| `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | `StandardOutPath`/`StandardErrorPath` | Live log path = repo-root `backend.log` | Not under `handoff/logs/` |
| `backend.log` + `handoff/logs/backend.log.*.gz` | 1,149 matching lines | The measurement corpus | See A8 |

## C6. Audit-class coverage log (loop-until-dry, K=2)

| Round | Probe | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | SLO/calendar, edge-vs-level, health-check, calendar libs | 9 (#1-#9) | no |
| 2 | expected-next-write, 2026 batch alerting | 2 (#10-#11 area) | no |
| 3 | suppression risk, holiday-aware TSAD | 3 (#12-#15 area) | no |
| 4 | Alertmanager time_intervals, last-successful-run | 1 | no |
| 5 | Datadog/Monte Carlo schedule-awareness | 1 (#16) | no |
| 6 | Monte Carlo exclusion windows + freshness rules | 2 (#17-#18) | no |
| 7 | Anomalo, Great Expectations | 2 sources, **0 new findings** (replicated the same negative) | **dry** |
| 8 | Airflow SLA, HTTP safe-method semantics | 2 (#21-#22) | no |
| 9 | Watchdog/dead-man's-switch, PagerDuty support hours | 2 (#23-#24) | no |
| 10 | trading-domain staleness, business-hours error budgets | 1 (#25); 1 fetch FAILED 403 | no |
| 11 | market-data session states, heartbeat pattern | 2 (#26-#27) | no |
| 12 | cronjob missed schedules, feature-store freshness | 2 (#28-#29) | no |
| 13 | croniter, metric-absence policies | 2 (#30-#31) | no |
| 14 | half-days/early closes, fail-open direction | 1 (#32) + 1 correction (#33) | no |
| 15 | named read-path anti-pattern, pager-load benchmarks | 2 (#34-#35) | no |
| 16 | calendar-aware OSS, learned next-commit | 2 (#36-#37) | no |
| 17 | calendar-aware best practice, read-path postmortem | **0** -- all mechanisms already recorded; postmortem search returned nothing | **dry** |
| 18 | quant incident narratives, mute-interval risk | 2 (#38-#39) | no |
| 19 | trading-day gate pattern, market-closed state machine | **0** -- results off-topic (TradingView indicators, patents) | **DRY 1** |
| 20 | freshness-SLO survey, exchange_calendars + monitoring | **0** -- no survey exists; no library links the two (confirms finding #8) | **DRY 2** |

| 21 | **(cycle 2)** business-calendar-aware freshness SLO (year-less canonical); suppression-danger postmortems | **1 (#40)** -- closed the §C2 gap; added stale-suppression, source-vs-delivery, severity-whitelist | no -- **streak reset** |
| 22 | **(cycle 2)** silence-expiry / stale-silence hygiene; `exchange_calendars` 2026 accuracy; health-check-side-effect anti-pattern (year-less canonical) | **0** -- 15 URLs evaluated, all de-dup against #7/#21/#38/#39/#40; the calendar-accuracy query collided on "Exchange" and returned Microsoft Outlook results | **DRY 1** |
| 23 | **(cycle 2)** edge-vs-level triggering (year-less canonical); weekday-only pipeline SLA + expected arrival (recency) | **0** -- edge/level hits were all hardware/epoll tier (already represented); SLA hits restated "during business days" (#elementary-data), learned weekday patterns (#36/#37) and expected-vs-actual start times (#10/#28) | **DRY 2** |

**Cycle-2 note on the dry claim.** Cycle 1 ended dry at rounds 19-20. The
cycle-2 run did **not** inherit that flag -- it re-ran the critic, and round
21 **was not dry**, which retroactively shows the cycle-1 stopping point was
slightly early. The streak was rebuilt honestly: `dry_rounds = 2` now refers
to rounds **22 and 23**, two CONSECUTIVE dry rounds after the round-21 find.
Earlier isolated dry rounds (7, 17, 19-20) do not count toward the current
pair because a non-dry round (21) intervened. `dry_rounds = 2 >=
K_required = 2` -> **`coverage.dry = true`**, `rounds = 23`.

Both dry rounds are **falsifiable rather than asserted**: every URL they
evaluated is written down in §B2b with the specific reason it was de-dup and
not new. A dry round with no visible evaluation would be unfalsifiable, and
the temptation on a reconciliation run is exactly to declare dryness cheaply
in order to reach the gate.

## C7. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **40**
      (39 from cycle 1 + #40 from cycle-2 round 21)
- [x] 10+ unique URLs total -- **134** (40 read-in-full + 94 snippet-only),
      enumerated mechanically after the final edit, not estimated.
      *(Cycle-1 wrote `~104` here while its envelope claimed 120 and the file
      held 118 -- three disagreeing figures, which is what failed the gate.
      Every figure in the brief now derives from the same mechanical count;
      see §"Envelope reconciliation (cycle 2)".)*
- [x] Recency scan (last 2 years) performed + reported -- §B3, 12 findings
- [x] Full papers / pages read (not abstracts); arXiv chain honoured
      (ar5iv for the 2022 paper, `/html/` for the 2025 ones, **never** `/pdf/`)
- [x] file:line anchors for every internal claim -- Part A + C5

Soft checks:
- [x] Internal exploration covered every module named in the scope, plus the
      frontend poller and the launchd log path that the scope did not name
- [x] Contradictions / consensus noted -- §C2, including **two honest
      negatives** (#15, #33) and the "does suppression hide failures?" claim,
      which cycle-2 round 21 moved from *unsourced* to *sourced at community
      tier only* (#40) while the sharp form of it remains uncited -- §C2
      records the change rather than overwriting the original judgement
- [x] Claims cited per-claim
- [ ] **Gap:** `tradinghours.com/data` returned HTTP 403 (1 failed fetch);
      superseded by the NYSE primary source
- [ ] **Gap:** the 719 pre-JSON-format log lines carry no date, so the cadence
      statistics rest on 430 of 1,149 lines -- an **undercount**, disclosed
- [ ] **Gap:** the alert-fatigue survey figures (§B3 item 12) are second-hand
      via an aggregator and are cited as context, not measurement



