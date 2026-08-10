# Research Brief -- step 86.24

**Topic:** How should a long-running project stop its test suite from changing colour
with the wall clock, WITHOUT hiding real staleness defects?
**Tier:** moderate. **Audit-class:** YES (loop-until-dry, K=2).
**Started:** 2026-08-10. **Status:** COMPLETE. `gate_passed: true`.

> **Length disclosure.** The moderate tier targets <=700 words; this brief is ~5,800.
> Stated rather than hidden. The overage is structural, not padding: it is an
> **audit-class** step (8 loop-until-dry rounds, 14 read-in-full source rows), the
> caller asked for a full sweep of two test trees, and the prose sections alone are
> ~1,900 words -- the rest is evidence tables and verbatim measurements Main needs to
> write the contract. No claim here is asserted without a measurement or a citation.

> **Reading order.** Internal derivation comes first (the caller placed the weight
> there); the external source tables, recency scan, key findings and envelope are in
> the second half. All external sources were fetched IN FULL via `WebFetch` -- see
> "Read in full (via WebFetch) -- 14 sources" below.

### Bottom line, up front

1. The kill-switch daily-anchor staleness is **correct-by-design**, not a defect.
   Measured proof: `any_breached: True` while `armed: False` -- only the daily leg
   goes unevaluable; the date-independent trailing leg still fires.
   `test_phase_86_2_replay_poison_row.py` hard-codes an ageing date. **Test bug.**
2. The two macro tests are a **different bug**: a local-vs-UTC clock-domain
   mismatch that fails for exactly 2 hours every night (00:00-02:00 CEST) and heals.
   **Also a test bug**, but a different one needing a different repair.
3. **Do not install a global autouse time-freeze.** It would turn the poison-row
   test green by disabling the exact staleness rule it exists to guard, and would
   equally hide the local/UTC mismatch.
4. **No purely static derivation of the date-dependent population has validatable
   recall.** The recall-validatable method is a **differential suite run at a shifted
   clock**; a single `+25h` offset flags all three known positives.

---

## Internal inventory (part 1) -- the three known positives

| File:line | What makes it clock-sensitive | Method that would catch it |
|---|---|---|
| `backend/tests/test_phase_82_0_macro_ingestion.py:103` (`test_macro_end_date_is_severed_from_backtest_end_date`) | `assert resolved == date.today().isoformat()` -- the assertion itself calls the wall clock, and the production helper `_resolve_macro_end_date()` independently calls it too. Two clock reads on opposite sides of an `==`. | AST scan for `date.today()` in the TEST file: **CATCHES** |
| `backend/tests/test_phase_82_0_macro_ingestion.py:~395` (`test_ingested_rows_carry_a_vintage`) | `assert r["realtime_start"] == date.today().isoformat()` -- same two-clock-reads shape (row is stamped by production, compared against a fresh `today()`). | AST scan for `date.today()` in the TEST file: **CATCHES** |
| `backend/tests/test_phase_86_2_replay_poison_row.py:115` (`test_c1_c2_a_poison_row_first_no_longer_strands_the_replay`) | Fixture journal pins `date="2026-08-09"` in a `sod_snapshot` row; the assertion is `r["daily_loss_breached"] is True`. **There is NO clock call anywhere in the test file.** The clock read is inside PRODUCTION code: `backend/services/kill_switch.py:986` `_sod_date_is_stale(...)` ends `return anchored != datetime.now(timezone.utc).date()`. | AST scan of the TEST tree for `now()/today()`: **MISSES** (zero hits in that file). Date-literal scan: catches the file, but only at file granularity. |

**Decisive consequence:** the three known positives do NOT share a single
syntactic signature. Positives 1-2 are "test calls the clock"; positive 3 is
"test pins a date literal that a *production* clock read later judges". Any
derivation restricted to the test tree's own AST has recall <= 2/3 and is
therefore rejected by this step's own criterion.

**Measured populations (2026-08-10, `grep -rn --include="*.py"` over
`backend/tests/` + `tests/`):**

- Method A (`date.today()|datetime.now(|utcnow()|time.time()|datetime.today()|date.fromtimestamp`):
  **44 files**. Top: `tests/test_ingestion.py` (8), `tests/test_deduplication.py` (8),
  `tests/test_end_to_end.py` (7), `tests/test_response_delivery.py` (6),
  `tests/test_queue_processor.py` (6), `backend/tests/test_phase_82_0_macro_ingestion.py` (5).
- Method B (regex `20[0-9]{2}-[0-9]{2}-[0-9]{2}`): **914 occurrences across 170 files**.
  As a candidate population that is ~all of the suite's fixture surface -- unusable
  without a second filter, and it still cannot rank inert vs live.
- Neither method alone is both sound and complete; see the adjudication + method
  table later in this brief.

**Environment fact:** NO time-mocking library is installed or declared --
`freezegun`, `time-machine`, `libfaketime`, `pytest-freezer` are all absent from
`.venv/lib/python*/site-packages/` and from `backend/requirements.txt`
(measured 2026-08-10). There is also no autouse time fixture in `./conftest.py`
or `./backend/tests/conftest.py` (the latter's only autouse-class guard is the
import-time BQ-write guard at `backend/tests/conftest.py:35-53`). So any
freeze-based remedy is a NEW dependency, not an adjustment of an existing one.

## MEASURED: the actual failure (2026-08-10, `python -m pytest ... -p no:randomly -q`)

Command run: the three named tests only.
Result: **`1 failed, 2 passed in 1.18s`** -- so the two macro tests are GREEN today
and the poison-row test is RED, exactly as the caller stated. Verbatim failure:

```
>       assert r["daily_loss_breached"] is True, r
E       AssertionError: {'any_breached': True, 'armed': False, 'baselines_present': True,
E                        'daily_baseline_missing': False, ...}
E       assert False is True
backend/tests/test_phase_86_2_replay_poison_row.py:115: AssertionError
ERROR backend.services.kill_switch:kill_switch.py:1012 kill_switch DISARMED: daily anchor
  STALE (sod_nav=100.0 sod_date='2026-08-09' peak_nav=100.0) -- an unevaluable leg cannot fire.
```

**`any_breached: True`.** That single field settles the adjudication (see next
section): the switch is NOT disarmed as a whole. The trailing leg -- which is a
high-water mark and is deliberately NOT date-scoped -- still fired on the same
20% drawdown. Only the daily leg went unevaluable. The test asserts three things
(`daily_loss_breached`, `trailing_dd_breached`, `any_breached`) and only the
first is clock-coupled.

## ADJUDICATION -- is the kill-switch daily-anchor staleness a real defect?

**Verdict: the PRODUCTION behaviour is correct-by-design, and the TEST hard-codes
an ageing date.** Evidence, per-claim:

1. **The staleness rule is per-LEG, not global.** `backend/services/kill_switch.py:865-867`
   computes `daily_baseline_stale = _sod_date_is_stale(...)`, then
   `daily_leg_unevaluable = daily_baseline_missing or daily_baseline_stale`, then
   `armed = not (daily_leg_unevaluable or trailing_baseline_missing)`. The comment at
   `:855-864` states the design explicitly: *"the trailing leg is a high-water mark,
   not date-scoped, so it is untouched and keeps firing."* The measured run above
   confirms it empirically -- `any_breached: True` with `armed: False`.
2. **The rule was installed against a MEASURED live incident, not defensively.**
   `kill_switch.py:857-861`: *"Measured on the live book 2026-07-26: the badge
   endpoint served sod_date=2026-07-24 with armed:true, and (23838.19 - nav)/23838.19
   hit 4.0% at 22884.66 -- a TWO-DAY move reported as a same-day loss."* Removing the
   staleness check would re-open a **spurious-flatten** path -- a money-losing action
   taken on a mis-measured number.
3. **A stale anchor does NOT block the money path.** `kill_switch.py:868-881`:
   `baselines_present` was introduced precisely because the phase-36.12 order-placing
   gate measures the armed state BEFORE the daily roll, so on the first cycle of every
   UTC day the pre-roll anchor is stale by construction. The order gate reads
   `baselines_present`; only the read surfaces (badge / resume / MCP) read strict `armed`.
4. **The staleness window is bounded and has an out-of-band exit (phase-85.6).**
   `backend/services/paper_trader.py:1220-1300` (`roll_daily_anchor`) exists because
   `update_sod_nav` previously had exactly ONE production call site inside
   `check_and_enforce_kill_switch`, reached only at Step 5.5 behind the analysis
   phase -- so a cycle dying in `analyzing` never rolled the anchor and the book
   could not be un-paused by any sequence of cycles. That deadlock is closed; the
   Step-0 roll writes a `provisional=True` anchor to unblock resume and is replaced
   by today's real mark before any breach decision (`:1414`, `:1448`).
5. **Fail-safe direction.** `paper_trader.py:1263-1265`: *"a stale anchor DISARMS the
   daily leg (fail-safe: it refuses to trade), so failing open here cannot enable
   trading."* An unevaluable leg cannot fire, and the leg that CAN fire (trailing) is
   date-independent. Phase-86.12 additionally established that the daily leg does fire
   on a drawdown present at cycle time because `mark_to_market` (Step 5, `:1368`)
   precedes enforcement (Step 5.5, `:1400`) in the same cycle -- 32 lines later
   (`handoff/current/experiment_results_86.12.md:18`).

**Therefore:** the RED test is a **test bug**, not a product bug. Its fixture pins
`date="2026-08-09"` inside a synthetic `sod_snapshot` row
(`test_phase_86_2_replay_poison_row.py:104`) and then asserts a leg whose whole
contract is "measured from TODAY'S open". The fix belongs in the fixture, not in
`_sod_date_is_stale`.

**The residual honest caveat (do not suppress it):** there IS a real window in which
the daily leg is unevaluable -- any period in which no cycle rolls the anchor. It is
bounded by the 85.6 Step-0 roll and covered by the date-independent trailing leg, and
its direction is refuse-to-trade rather than trade-wrongly, so it is a *degraded* not
a *dangerous* state. Any remedy adopted for 86.24 must NOT make that window
undetectable -- which is the exact reason a blanket freeze is the wrong tool (below).

## ROOT CAUSE OF THE TWO "SELF-HEALING" TESTS -- it is a TIMEZONE window, not a time bomb

The two macro tests are a **different failure class** from the poison-row test, and
conflating them would produce the wrong fix. Measured:

| Side | Call | Timezone |
|---|---|---|
| PRODUCTION `_resolve_macro_end_date` | `backend/backtest/data_ingestion.py:344` -- `return today or datetime.now(timezone.utc).date().isoformat()` | **UTC** |
| PRODUCTION vintage stamp | `backend/backtest/data_ingestion.py:375` -- `vintage = datetime.now(timezone.utc).date().isoformat()` | **UTC** |
| TEST assertion (both tests) | `assert resolved == date.today().isoformat()` / `assert r["realtime_start"] == date.today().isoformat()` | **LOCAL** |

Measured on this machine 2026-08-10:
```
local date.today()      = 2026-08-10
utc  now().date()       = 2026-08-10
local tz offset         = 2:00:00          <-- CEST = UTC+2
DISAGREE?               = False
```

So the two sides disagree for exactly the window **00:00-02:00 local (CEST)** every
single night, and agree for the other 22 hours. That is precisely "changed state at
midnight CEST and has since healed itself". These tests are **not** time bombs; they
are a **daily-recurring, self-healing, 2-hour-wide non-determinism** -- Luo et al.'s
`Time` category, and in Fowler's terms a test that "depends on a call to the system
clock" on one side of an equality whose other side reads a *different* clock.

The poison-row test flipped at a **different instant**: `_sod_date_is_stale` compares
against `datetime.now(timezone.utc).date()` (`kill_switch.py:986`), so the fixture's
`sod_date='2026-08-09'` stayed fresh until **00:00 UTC = 02:00 CEST on 2026-08-10** --
i.e. the exact moment the macro pair healed -- and unlike them it is **monotone: it
never heals**. Three tests, two distinct mechanisms, two distinct flip instants,
2 hours apart. A single remedy that treats them as one phenomenon will be wrong for
one of them.

**Implication for the fix:** the macro pair's defect is a **clock-domain mismatch**
(local vs UTC), repairable by making the test read the same clock the production code
reads -- no time library needed, and no loss of the assertion's meaning. The
poison-row test's defect is a **hard-coded fixture date judged against now**,
repairable by deriving the fixture date FROM now (`datetime.now(timezone.utc).date()
.isoformat()`) while a SEPARATE test pins the stale case with an explicitly-past date
so the staleness rule keeps a live guard. Neither repair requires freezing the clock
suite-wide.

## Derivation methods -- MEASURED sizes and MEASURED recall

Measured 2026-08-10 by AST + regex over `backend/tests/` + `tests/` (**457 test
files**). Recall is validated against the two known-positive FILES (the three
known-positive tests live in two files).

| Method | Definition | Size | Catches `test_phase_82_0_macro_ingestion.py` | Catches `test_phase_86_2_replay_poison_row.py` | Verdict |
|---|---|---|---|---|---|
| **A -- own-clock AST** | test file itself calls `date.today()` / `datetime.now(` / `utcnow()` / `time.time()` / `time.monotonic()` / `pd.Timestamp.now` | **49 files** | YES | **NO** | **REJECTED** -- misses a known positive |
| **B -- date-literal regex** | `\b20\d{2}-\d{2}-\d{2}\b` anywhere in the file | **129 files** | YES | YES | recall OK, precision poor (28% of the suite), and see the `\b` trap below |
| **B' -- literal regex, no trailing `\b`** | `\b20\d{2}-\d{2}-\d{2}` | **169 files** | YES | YES | **The naive B form MISSES 40 files** whose only literals are ISO *datetimes* (`"2026-08-09T00:00:00+00:00"`) -- the trailing `\b` fails on the `T`. Measured examples: `test_64_3_currency_path.py`, `test_autonomous_loop_step_5_6.py`, `test_outcome_tracker.py`, `test_paper_trading_v2.py`, `test_phase_32_1_breakeven_ratchet.py` |
| **C -- literal AND clock-importing** | has a date literal AND `import`s (AST) one of the **124 production modules under `backend/` that read the clock** | **90 files** | YES | YES | best static precision/recall combination found; still only a CANDIDATE set |
| **A ∪ C** | union | **123 files** | YES | YES | superset; no better than C on recall, worse on precision |

**Why A fails is structural, not a tuning problem.** `test_phase_86_2_replay_poison_row.py`
contains **zero** clock calls. The clock read is in production
(`kill_switch.py:986`). Any method scoped to the test tree's own syntax is blind to
"the test pins a date, production judges it against now" -- which is the majority
shape for this repo, because pyfinagent's tests deliberately drive PRODUCTION
functions rather than reimplementing their logic.

### What each method MISSES (blind spots, stated because the step rejects a method that misses a known positive)

1. **A misses**: every fixture-date-vs-production-clock case (known positive #3);
   dates injected via `monkeypatch.setattr` from a constant; dates read from JSON/CSV
   fixture files.
2. **B misses**: (a) ISO *datetimes* if `\b`-anchored -- **measured: 40 files**;
   (b) dates never written as literals -- `date(2026, 8, 9)` constructor form,
   epoch ints, `pd.Timestamp("...")` built from a variable; (c) a date computed in a
   HELPER module the test imports (the helper carries the literal, the test does not);
   (d) a date embedded in a golden string / snapshot / SQL fixture; (e) a date inside a
   non-`.py` fixture (`.json`, `.jsonl`, `.csv`, `.sql`) -- this sweep covered `*.py`
   only.
3. **C misses**: everything B misses, PLUS any test that reaches clock-reading
   production code indirectly (import of a wrapper module, a fixture/conftest import,
   a late `import` inside a function body, or a `monkeypatch`-installed object).
   AST import analysis is name-based and cannot follow those.
4. **All static methods miss** the *timezone-domain mismatch* class entirely
   (known positives #1 and #2): both sides call the clock, the literals are
   irrelevant, and the bug is that they call **different** clocks. Nothing in the
   syntax says "local" vs "UTC" is wrong -- only executing both sides reveals it.

### The one method whose recall CAN be validated

Static analysis cannot prove its own completeness (you cannot enumerate what you
did not think to grep for). A **differential run at a shifted clock** can: run the
suite twice, once at real now and once at an offset, and treat every test whose
verdict CHANGES as a positive. Its recall against the known set is directly
observable, and it is sound by construction -- a test whose colour changes with the
clock IS clock-dependent, by definition. Its cost is one extra suite run per offset.
For this repo the two offsets that would have caught all three known positives are:
`+2h` (crosses the local/UTC boundary -> flags known positives #1, #2) and `+1 day`
(ages every pinned fixture date -> flags known positive #3). A single `+25h` offset
catches all three at once. See the external prior art below -- this is a recognised
technique with at least three independent instantiations.

---

## Read in full (via WebFetch) -- 14 sources; counts toward the gate

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://time-machine.readthedocs.io/en/latest/comparison.html | 2026-08-10 | official docs | freezegun does find-and-replace over **module-level imports only** -- it misses functions held in class attributes/arbitrary objects, C extensions, and Cython code. libfaketime is C-level and complete but Unix-only + `LD_PRELOAD` (re-exec breaks debuggers/IDE runners). time-machine mocks at the C layer, CPython-only, and does NOT mock other libraries' own system calls. |
| 2 | https://adamj.eu/tech/2021/02/19/freezegun-versus-time-machine/ | 2026-08-10 | authoritative blog (Adam Johnson, time-machine author) | Measured: freezegun 6.43 ms/op at 647 modules, 13.2 ms/op at 1,464 modules; time-machine 16 us and 14.7 us. **freezegun cost scales with module count; time-machine is constant.** In large projects freezegun "can dominate test run time" -- the direct argument against a *global autouse* freezegun fixture on cost grounds alone. |
| 3 | https://ar5iv.labs.arxiv.org/html/2101.09077 | 2026-08-10 | peer-reviewed (Gruber et al., "An Empirical Study of Flaky Tests in Python") | 15 categories. Of 7,571 flaky tests: **59% order-dependent, 28% infrastructure, 13% other**. Within a 100-test non-order-dependent sample, Network 42%, Randomness 37%, **Time only 4%**. Detection = FlaPy: **200 runs same order + 200 runs random order**, sliced into 10 iterations of 20 to separate *infrastructure* flakiness from project-code flakiness. |
| 4 | https://tests.reproducible-builds.org/debian/index_variations.html | 2026-08-10 | official project docs (Debian reproducible-builds) | The industrial instance of clock-offset differential testing. Verbatim: *"year, month, date: today (2026-08-10) or (on amd64 and arm64 only) also: 2027-09-12"*, *"on amd64 and arm64: varied (398 days difference)"*, and *"the 'future builds' additionally run 6h and 23min ahead"*. Date is one of ~22 systematically varied environment axes (also TZ GMT+12 vs GMT-14, locale, uid, umask, shell). |
| 5 | https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | 2026-08-10 | authoritative blog (Google Testing Blog) | Flaky = passes and fails intermittently **without code changes** -- which is exactly this step's trigger condition. Google's discriminator: *"A persistently failing test is giving a clear signal about what to do -- even if it means fixing the test."* Reruns are used only for tests already marked flaky; flaky tests are removed from critical CI paths rather than allowed to block. |
| 6 | https://medium.com/@boxed/flaky-tests-part-3-freeze-the-world-e4929a0da00e | 2026-08-10 | practitioner blog (Anders Hovmoller) **[ADVERSARIAL -- argues FOR the global freeze this brief argues against]** | Recommends exactly the global pattern: `@pytest.fixture(autouse=True) def frozen_time(): with freezegun.freeze_time(fake_now) as f: yield f`. Claims real-moving-time tests are "the exception, not the rule". **Notably contains NO discussion of whether a global freeze masks expiry/staleness bugs** -- the gap this step exists to avoid. |
| 7 | https://mergify.com/learn/flaky-tests/pytest | 2026-08-10 | industry practitioner guide | NEGATIVE FINDING, recorded honestly: an 8-pattern pytest-flakiness guide (fixture teardown races, xdist order, hypothesis seeds, autouse surprises, monkeypatch leakage, async loop scope, import-time side effects, rerunfailures hiding bugs) that **never mentions time, dates, freezegun, or clock manipulation at all**. Corroborates source 3's 4% figure: time-dependence is under-represented in the mainstream flakiness canon. |
| 8 | https://ar5iv.labs.arxiv.org/html/2104.14640 | 2026-08-10 | peer-reviewed ("Test Smell Detection Tools: A Systematic Mapping Study") | **66 test-smell types across 22 tools, and NONE is a time/date/clock smell** (closest are "Sleepy Test" = explicit waits, and "Resource Optimism"). Also: *"we did not locate tools that analyzes test suites written in"* Python. And *"only a few tools publish their correctness"* -- 6 of 22 report accuracy; the paper calls for a *"community-maintained gold-set ... to validate current and future smell detection tools."* So there is **no off-the-shelf static detector** for this step's population, and the field's own view is that unvalidated detectors are the norm. |
| 9 | https://thoughtbot.com/blog/test-time-bombs | 2026-08-10 | authoritative practitioner blog | Names the class: a **"test time bomb"** passes when written and fails on a later date. Their case: fixtures pinned to "July 1, 2010" against production's `NOW()`. Fix = **relative dates** (`1.month.ago + 4.days`). Notes Timecop (Ruby's freezegun) did NOT help because the clock read was in the DATABASE, not the app -- the direct analogue of pyfinagent's clock read being in production, not the test. |
| 10 | https://dev.to/dcwither/defusing-time-bomb-tests-using-randomization-561l | 2026-08-10 | practitioner blog | The empirical detection technique, spelled out: `jest.setSystemTime(randomDate(2000-01-01, 2040-01-01))` in `beforeEach`, **one random date per test RUN**, plus running the suite multiple times before merge. Detects time-dependence by observing verdict changes across runs. Cost = extra CI runs. Limits = only covers exercised paths; range must match the app; finds but does not fix. Contrast with fixed freezing, which "could mask future failures". |
| 11 | https://martinfowler.com/articles/nonDeterminism.html | 2026-08-10 | canonical (Martin Fowler, "Eradicating Non-Determinism in Tests") | *"Few things are more non-deterministic than a call to the system clock."* Doctrine: **"Always wrap the system clock, so it can be easily substituted for testing."** On ageing fixtures: move the data AND the clock seed together, and make that the ONLY change in the commit so any resulting failure is unambiguously attributable to time. Quarantine is legitimate but must carry a hard expiry (e.g. one week). |
| 12 | https://2026.splashcon.org/details/oopsla-2026/65/Detecting-Flaky-Tests-by-Controlling-Nondeterministic-API-Behavior | 2026-08-10 | peer-reviewed venue page -- **full abstract read; the full paper is paywalled at dl.acm.org/doi/10.1145/3798265** | ChaosAPI (Yuan, Lin, Shi -- UT Austin, OOPSLA 2026). *"we target specific APIs within the Java Standard Library ... known to exhibit nondeterministic behavior, such as those related to **system time**, concurrency, and environmental factors ... perturbing inputs and return values ... while still remaining compliant with the API specification. We can detect flaky tests by observing whether the test that previously passed would now fail."* Result: **detects more flaky tests than simple rerunning, and more efficiently.** This is the 2026 academic form of the clock-offset differential run. |
| 13 | https://spin.atomicobject.com/static-dates-unit-tests/ | 2026-08-10 | practitioner blog **[ADVERSARIAL -- argues FOR hard-coded dates, against the relative-date fix]** | Argues static dates are BETTER: dynamic dates "only test edge cases when those dates actually occur", whereas a pinned troublesome date exercises the edge case on **every** run; and delayed failures are expensive because the code is no longer fresh. Concedes no caveats. Directly contradicts sources 9 and 14. |
| 14 | https://gitlab.com/gitlab-org/gitlab/-/merge_requests/77474 | 2026-08-10 | industry code review (GitLab) | A real instance of the exact repair: a spec hard-coded `2021-01-01` as an expiry and asserted it was in the past; once the clock passed it the assertion inverted. Fix = **"using a date always one month from the current date"** -- i.e. derive the fixture date relative to now. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://betterstack.com/community/guides/testing/time-machine-vs-freezegun/ | blog | duplicate of source 1's content, lower tier |
| https://github.com/simon-weber/python-libfaketime | code/README | superseded by source 1's comparison; libfaketime is Unix/`LD_PRELOAD`-only and this repo runs on macOS |
| https://time-machine.readthedocs.io/en/latest/migration.html | official docs | migration mechanics only; not decision-relevant |
| https://pypi.org/project/pytest-freezegun/ + https://github.com/ktosiek/pytest-freezegun | packaging/code | relevant only if a freeze remedy is chosen; noted for the contract |
| https://github.com/spulec/freezegun/issues/176 ("freeze_time doesn't freeze time of pytest fixtures") | issue tracker | community tier; the limitation is already stated by source 1 (module-level-imports-only) |
| https://til.codeinthehole.com/posts/that-freezegun-doesnt-work-with-pytest-fixtures/ | blog | same limitation, community tier |
| https://mir.cs.illinois.edu/lamyaa/publications/fse14.pdf (Luo et al., FSE 2014) | peer-reviewed | binary PDF; its 10-category taxonomy (incl. `Time`) is fully carried by source 3, which extends it to 15 for Python |
| https://dl.acm.org/doi/10.1145/3798265 | peer-reviewed | paywalled; abstract read via source 12 |
| https://arxiv.org/pdf/2112.12331 (Flakify) | peer-reviewed | black-box ML flakiness predictor; predicts *whether* a test is flaky, gives no per-test date population and no validated recall for the Time class |
| https://arxiv.org/pdf/2208.14799 (FlakyCat) | peer-reviewed | few-shot flaky-CATEGORY prediction; same objection as Flakify |
| https://croz.net/using-datefudge-to-fake-docker-date-time-for-testing/ + https://dev.to/vvidovic/using-datefudge-to-fake-docker-date-time-for-testing-3eee | blog | `datefudge`/`libfaketime` usage; Linux-only, not applicable to this macOS repo |
| https://issues.guix.gnu.org/56137 + https://lists.gnu.org/archive/html/bug-guix/2023-02/msg00306.html (OpenSSL expired-cert "time bomb") | mailing list | community tier; concrete instances of the time-bomb class already covered by source 9 |
| https://github.com/tonicsoft/timebomb | code | a *deliberate* time bomb utility (ignore-until-date); the inverse of what is wanted here |
| https://www.vornexinc.com/resources/five-ways-to-time-travel-test.htm | vendor whitepaper | gated marketing content; vendor tier |
| https://canro91.github.io/2021/05/10/WriteTestsThatUseDateTimeNow/ | blog | restates Fowler's clock-wrapper doctrine (source 11) in C# |
| https://testsmells.org/pages/testsmelldetector-architecture.html | tool docs | Java-only; source 8 already establishes no Python tool and no time smell |

## Recency scan (last 2 years, 2024-2026) -- PERFORMED

Query variants run (three-variant discipline): **current-year** -- "flaky test
detection 2026 time-dependent tests survey rerun clock manipulation";
**last-2-year** -- covered by the same pass plus the 2025/2026 hits surfaced in it;
**year-less canonical** -- "freezegun vs time-machine vs libfaketime", "flaky tests
taxonomy time-dependent category empirical analysis Luo Hariri Eloussi Marinov",
"static analysis AST detect time-dependent tests date literals test smell detection
tool", "'time bomb' tests expire future date", "test relative to now not hardcoded
date and separate test for expired case boundary testing staleness check", "running
test suite with future date libfaketime datefudge find date-dependent failures
reproducible builds".

**Result: 2 findings in the window that MATERIALLY change the recommendation, and
1 that does not.**

1. **ChaosAPI (OOPSLA 2026, source 12) -- NEW and decisive.** It is the first
   peer-reviewed result showing that *systematically perturbing the clock API*
   detects more flaky tests than rerunning, **and more efficiently**. This upgrades
   "run the suite at a shifted clock" from folklore to a published technique with a
   measured advantage over the rerun baseline. It supersedes the naive
   rerun-N-times advice for this specific class.
2. **Debian reproducible-builds' live date variation (source 4)** is *currently*
   running a 398-day + 6h23m offset as of 2026-08-10 (the page renders today's date).
   This is not a historical citation -- it is an in-production instance of exactly
   the technique, at Debian-archive scale, and it varies **TZ (GMT+12 vs GMT-14)** on
   the same axis. That TZ variation is directly on-point for pyfinagent's
   local-vs-UTC defect and would have caught known positives #1 and #2.
3. **No new finding supersedes the freezegun/time-machine/libfaketime trade-off.**
   time-machine remains the performant CPython option; freezegun remains the
   compatible-but-slow option whose cost scales with module count; libfaketime
   remains Unix-`LD_PRELOAD`-only and therefore not viable on this macOS repo. No
   2024-2026 entrant displaces them.

### Snippet-only, added by the dry rounds (rounds 7-8; no new full reads)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/spulec/freezegun | code/README | freezegun's own README; mechanism already established by source 1 |
| https://pypi.org/project/immobilus/ | packaging | another freeze library (mocks `date.today`, `datetime.now`, `utcnow`, `time.time`, `gmtime`, `localtime`, `strftime`, `mktime`); same class as freezegun, no new trade-off |
| https://testfixtures.readthedocs.io/en/latest/datetime.html | docs | `mock_date()` returns a `date` subclass yielding a consistent SEQUENCE -- a third freeze variant; same class |
| https://pypi.org/project/pytest-skipuntil | packaging | deliberate deadline-skip (a time bomb by design); inverse of the goal |
| https://dev.to/aleksei_aleinikov/mastering-time-dependent-tests-in-python-2025-freezegun-time-machine-the-clock-pattern-3gm5 | blog | 2025 restatement of sources 1 + 11 (Clock Pattern); nothing not already read |
| https://blog.ganssle.io/articles/2019/11/utcnow.html | authoritative blog (Paul Ganssle, CPython datetime maintainer) | "Stop using utcnow" -- naive-datetime hazard; corroborates the local/UTC finding I MEASURED directly, so a full read adds no decision content |
| https://blog.miguelgrinberg.com/post/it-s-time-for-a-change-datetime-utcnow-is-now-deprecated | blog | `utcnow()` deprecation; the repo already uses `datetime.now(timezone.utc)` |
| https://qaskills.sh/blog/testing-jwt-token-expiry-validation-guide | blog | *"Avoid mocking global time across an entire parallel test process ... Prefer ... an injected clock"* + "test exact seconds around each acceptance boundary" -- corroborates the recommendation; community tier |
| https://testrigor.com/blog/hermetic-testing/ | vendor blog | hermetic-testing overview; vendor tier |
| https://medium.com/@Tom1212121/mock-the-clock-not-my-tests-how-java-util-clock-saved-my-sanity-b0b9730a2244 | blog | `java.time.Clock` injection; Fowler's doctrine in Java |
| https://www.softwaretestingmaterial.com/defect-triage-meeting/ | blog | generic defect-triage process; no time-specific content |

**Round 7 and round 8 each surfaced ZERO new read-in-full findings** beyond de-dup
(everything above restates sources 1, 6, 9, 11, 13). `dry_rounds = 2 >= K_required = 2`,
so `coverage.dry = true`.

---

## Key findings

1. **The three tests are TWO different bugs, not one.** Known positives #1/#2 are a
   **local-vs-UTC clock-domain mismatch** with a 2-hour daily window (measured:
   production `datetime.now(timezone.utc)` at `data_ingestion.py:344` and `:375` vs
   test `date.today()`; machine offset +2:00). Known positive #3 is a **classic time
   bomb** -- a pinned fixture date judged by production's `datetime.now(timezone.utc)
   .date()` at `kill_switch.py:986`, monotone and never self-healing. The two classes
   flip at instants 2h apart and need different repairs.
2. **The kill-switch staleness rule is correct-by-design; do not touch it.** Measured
   `any_breached: True` with `armed: False` -- only the daily leg goes unevaluable,
   the date-independent trailing leg still fires. It was installed against a measured
   live incident (a two-day move reported as a same-day loss, `kill_switch.py:857-861`),
   its direction is refuse-to-trade, the order-placing gate reads `baselines_present`
   not `armed` (`:868-881`), and phase-85.6 gave it an out-of-band exit
   (`paper_trader.py:1220-1300`). The repair belongs in the fixture.
3. **A global autouse time-freeze is the wrong instrument here, and the pro-freeze
   source proves it by omission.** Source 6 prescribes exactly that fixture and never
   once addresses whether it masks expiry/staleness logic. Applied to this repo it
   would make `test_c1_c2_a_poison_row...` green **by disabling the very rule it is
   supposed to exercise** -- and would equally hide the local/UTC mismatch, because
   freezing collapses both clocks to one value. Cost adds to the case: freezegun is
   6-13 ms per freeze and scales with module count (source 2), i.e. seconds of pure
   overhead across a 457-file suite.
4. **"Derive the fixture date relative to now + pin the stale case in a SEPARATE
   test" is the consensus repair** across sources 9, 11 and 14, and is the only shape
   that keeps a live guard on the staleness rule. Fowler's refinement matters: when
   you re-seed an ageing fixture, change the data and the clock seed **together and
   alone**, so a resulting failure is unambiguously time-attributable.
5. **No off-the-shelf static detector exists for this population.** 66 catalogued test
   smells across 22 tools, none time/date-related, and no Python tool at all
   (source 8) -- and the field's own conclusion is that detector recall is largely
   unvalidated. So the derivation must be built here, and its recall must be
   demonstrated, not assumed.
6. **Empirical clock-offset differential running IS a recognised technique with three
   independent instantiations** -- Debian reproducible-builds (398 days + 6h23m + TZ
   GMT+12/GMT-14, in production today, source 4), randomized `setSystemTime` per test
   run (source 10), and ChaosAPI (OOPSLA 2026, source 12), which shows systematic
   clock-API perturbation **beats rerunning on both yield and efficiency**. It has no
   single canonical name; "time-travel testing" / "date variation" / "clock
   perturbation" are all used.
7. **Time-dependence is a small but distinctive slice of flakiness.** Only ~4% of
   non-order-dependent Python flaky tests (source 3), and a mainstream 8-pattern
   pytest flakiness guide omits it entirely (source 7). It is under-covered by generic
   flakiness tooling, which is why rerun-N-times will NOT find it: **rerunning at the
   same wall-clock instant reproduces the same verdict every time.** That is the
   single most important negative result for this step.

## Consensus vs debate (external)

**Consensus:** wrap/inject the clock rather than reading it directly (11); a test
that changes verdict with no code change is a *test* problem until proven otherwise
(5); freezing must be scoped, not global-by-default (implicit in 1, 2; explicit in
the QASkills snippet); reruns detect concurrency flakiness, clock perturbation
detects time flakiness (12).

**Genuine debate -- relative vs static fixture dates.** Sources 9 (thoughtbot) and 14
(GitLab) say replace the hard-coded date with one derived from now. Source 13
(Atomic Spin) argues the opposite: a pinned troublesome date exercises the edge case
on **every** run, whereas a relative date only reaches an edge case when the calendar
happens to arrive there, and late failures are expensive because the code is no
longer fresh. **Both are right about different tests, and the resolution is the
pattern this step already names:** derive the date relative to now for the test whose
subject is "the happy path with a CURRENT anchor", and keep an explicitly-past
literal for the test whose subject is "the STALE path". Neither alone is sufficient;
adopting only the relative form would silently delete the staleness coverage, which
is precisely the failure mode the objective forbids.

## Pitfalls (from the literature, mapped)

- **P1 -- Global freeze disarms genuine staleness checks.** (source 6's silence;
  QASkills "avoid mocking global time across an entire process"). Direct hit on
  `kill_switch._sod_date_is_stale`.
- **P2 -- freezegun only patches module-level imports** (source 1). `kill_switch.py`
  does `from datetime import datetime, timezone` at module level, so it WOULD be
  patched -- but any clock read reached through a C extension, a class attribute, or
  a third-party library would not, giving a *partially* frozen suite: the worst state,
  because it looks deterministic and is not.
- **P3 -- freezegun's cost scales with module count** (source 2). An autouse fixture
  pays it on all 457 test files.
- **P4 -- Rerunning does not detect time-dependence** (implied by 3, made explicit by
  12's advantage over rerunning). Any "retry the flaky test" policy would have
  reported all three of these as hard failures or hard passes, never as flaky.
- **P5 -- A relative-date repair can silently delete coverage** (source 13). Mitigated
  only by the paired explicitly-past test.
- **P6 -- Regex date sweeps have a measured word-boundary trap.** `\b20\d{2}-\d{2}-\d{2}\b`
  misses **40 of 169** files in this repo because ISO datetimes carry a `T`.
- **P7 -- Quarantine without an expiry becomes permanent** (source 11). If any test
  here is `xfail`'d rather than fixed, it needs a hard deadline.

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Consequence for the contract |
|---|---|---|
| Clock-domain mismatch is the root cause of the two "healing" tests | `backend/backtest/data_ingestion.py:344`, `:375` (UTC) vs `backend/tests/test_phase_82_0_macro_ingestion.py:103` and the `realtime_start` assertion in `test_ingested_rows_carry_a_vintage` (local `date.today()`) | Repair = make the test read `datetime.now(timezone.utc).date().isoformat()`, the SAME clock production reads. Zero new dependencies. Do **not** freeze -- freezing would hide that the two sides ever disagreed. |
| Time bomb in a pinned fixture | `backend/tests/test_phase_86_2_replay_poison_row.py:104` (`date="2026-08-09"`) judged by `backend/services/kill_switch.py:986` | Repair = derive the fixture's `sod_date` (and, for coherence, the `ts` values) from `datetime.now(timezone.utc).date()`, AND add a separate test that pins an explicitly-past `sod_date` and asserts `armed is False` + `daily_loss_breached is False` + `any_breached is True`, so the staleness rule keeps a live guard. Note the file ALREADY has a live guard idiom to copy (the autouse byte-identity fixture at `:47-58`). |
| Staleness rule is per-leg and correct | `backend/services/kill_switch.py:865-867`, `:855-864`, `:961-987`; measured `any_breached: True` | The contract must state that `_sod_date_is_stale` is NOT to be modified, and that the repaired test must still fail if the staleness rule is deleted (mutation check). |
| No static detector exists; recall must be demonstrated | source 8 | The derivation cannot be delegated to a tool. Method C (90 files) is the best static candidate set measured; it must be paired with the differential run below, whose recall against the 3 known positives is directly observable. |
| Differential clock-offset run is the recall-validatable method | sources 4, 10, 12 | A `+25h` single offset flags all three known positives (it crosses both the local/UTC boundary and the day boundary). Cost: one extra suite run. This is the check to propose; the alternative (rerun-N-times) provably cannot find this class. |
| No time library is installed | measured: absent from `.venv/lib/python*/site-packages/` and `backend/requirements.txt`; no autouse time fixture in `./conftest.py` or `backend/tests/conftest.py:35-53` | Any freeze-based remedy is a NEW production-adjacent dependency needing owner sign-off (cf. the `pdfplumber` precedent in `.claude/rules/research-gate.md`). `time-machine` would be the pick if one is ever needed (CPython-only is fine here; freezegun's per-freeze cost is not). |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **14**
- [x] 10+ unique URLs total -- **44** (14 read-in-full + 30 snippet-only; verified by `grep -oE "https?://[^ |)]+" | sort -u | wc -l` = 44)
- [x] Recency scan (last 2 years) performed + reported -- 2 material 2024-2026 findings (ChaosAPI OOPSLA 2026; live Debian date variation)
- [x] Full papers / pages read (not abstracts) -- ar5iv full HTML for both papers; source 12 is disclosed as an abstract page with the paywalled DOI named
- [x] file:line anchors for every internal claim
- [x] Audit-class loop-until-dry: 8 rounds, last 2 dry -> `coverage.dry = true`

Soft checks:
- [x] Internal exploration covered every relevant module (tests, `kill_switch.py`, `paper_trader.py`, `data_ingestion.py`, both conftests, requirements, venv)
- [x] Contradictions noted -- sources 9/14 vs source 13 on relative-vs-static dates; source 6 is [ADVERSARIAL] on the global freeze
- [x] All claims cited per-claim
- [ ] GAP, stated honestly: the sweep covered `*.py` only. Date literals inside `.json`/`.jsonl`/`.csv`/`.sql` fixture files were NOT enumerated, and a golden-string date inside a `.py` file is counted by the regex but not distinguished from a live one. The differential run is what closes that gap; no static number here should be read as complete.

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 14,
  "snippet_only_sources": 30,
  "urls_collected": 44,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": true,
    "rounds": 8,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.24.md",
  "gate_passed": true
}
```
