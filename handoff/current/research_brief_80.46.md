# Research Brief -- phase-80.46 (flaky CI gate: subprocess timeout under CPU contention)

Tier: **moderate** (caller-specified). NOT audit-class.
Started: 2026-07-26. Researcher: Layer-3 harness researcher.

## Question

A test run reported `249 passed, 1 skipped, 2060 deselected, 1 warning, 1 ERROR
in 95.89s` (normal ~16s). Five subsequent runs clean; the ERROR was never
identified. Main's UNTESTED hypothesis: `backend/tests/test_phase_75_ci_gates.py`
shells out to `pytest --collect-only` with `timeout=60`, and a
`subprocess.TimeoutExpired` surfaces in pytest as an **ERROR** (not FAILURE),
matching the observed shape.

Sub-questions:
1. Does an uncaught exception in a test body land in pytest's ERROR bucket or
   the FAILURE bucket? (load-bearing for the hypothesis)
2. Flaky-test taxonomy: where does "fixed timeout under resource contention"
   rank, and what is the recommended remedy?
3. Timeout design for subprocesses in tests: bigger constant / scaled / retry /
   eliminate?
4. In-process alternatives to shelling out to pytest (`pytest.main`, Collector
   API); is re-entrant `pytest.main()` inside a running session safe?
5. Recency scan 2025-2026.

## Status log (write-first)

- [x] skeleton written
- [ ] internal audit
- [ ] external sources
- [ ] recency scan
- [ ] recommendation
- [ ] envelope

## Queries run

(filled in below)

## HEADLINE (written early, write-first): the hypothesis is REFUTED

Two independent authoritative sources say an exception raised **inside a test
function body** is a **FAILURE**, and only setup / teardown / collection
exceptions are **ERRORS**. `subprocess.TimeoutExpired` at
`test_phase_75_ci_gates.py:120` is raised in the test BODY. Therefore a
`--collect-only` subprocess timeout would have printed `1 failed`, not
`1 error`. The observed line had **zero** `failed`. Details + the surviving
alternative hypotheses are in "Key findings" and "Recommendation".

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://github.com/pytest-dev/pytest/discussions/7950 | 2026-07-27 | official-project discussion (maintainer answer) | WebFetch (full) | "pytest considers a *failure* any assertion error or exception raised inside a test function; *errors* happen when an assertion error or exception is raised during setup/teardown/collection." junitxml uses the same mapping. |
| https://docs.pytest.org/en/stable/how-to/usage.html | 2026-07-27 | official doc | WebFetch (full) | `pytest.main()` "will not raise SystemExit but return the exit code instead"; BUT "Calling pytest.main() will result in importing your tests and any modules that they import. Due to the caching mechanism of python's import system, making subsequent calls to pytest.main() from the same process will not reflect changes to those files between the calls. For this reason, making multiple calls to pytest.main() from the same process ... is not recommended." |

| https://github.com/pytest-dev/pytest-timeout (README.rst, raw) | 2026-07-27 | official plugin doc | WebFetch (full, raw.githubusercontent) | "your test suite should aim to be fast, with timeouts being a last resort, not an expected failure mode"; signal method uses `pytest.fail()` to interrupt -> reported as a **failure**; thread method `os._exit()`s the whole process -> "no teardown, JUnit XML output etc."; cautions that timeouts are problematic while debugging / under resource constraints. |
| https://arxiv.org/html/2602.03556 (SAP HANA, 2026) | 2026-07-27 | peer-reviewed preprint | WebFetch (full HTML) | 559 issue reports: **Concurrency 130 (23%)**, **Timeout 87 (16%)** -- "Tests that exceed time limits, e.g., **due to system load or slow hardware**, causing flaky failures"; Oracle Brittleness 57 (10%), Configuration 55 (10%), Async Wait 52 (9%). System tests 18% timeout-flaky vs native unit tests 7%; cause = system tests had individually-calibrated timeouts "close to the test's average execution time" while unit tests had a single global 1h timeout rarely approached. "there seems to be no one-size-fits-all solution readily available". |
| https://ar5iv.labs.arxiv.org/html/2310.12132 ("The Effects of Computational Resources on Flaky Tests") | 2026-07-27 | peer-reviewed preprint | WebFetch (full, ar5iv fallback after /html/ 404) | RAFT = "a flaky test that has a statistically different failure rate when resources are constrained compared to an unconstrained test execution". **283 / 608 (46.6%)** of flaky tests across 52 projects are RAFTs. **CPU availability is the single most influential resource** (Java 82 / JS 28 induced failures, far above memory/disk/network). One test failed 2x at full capacity but **80x at 0.1 CPU**. RAFTs rise sharply below 1 core / 1 GiB. |
| https://arxiv.org/html/2504.16777 ("Systemic Flakiness", 2025) | 2026-07-27 | peer-reviewed preprint | WebFetch (full HTML) | "flaky tests often exist in clusters, with co-occurring failures that share the same root causes"; **75% of 810 flaky tests belonged to clusters**, mean cluster size 13.5 spanning 2.9 test classes. Predominant systemic causes = networking + external dependencies. Recommendation: fix the SHARED root cause, do not debug tests individually. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://mir.cs.illinois.edu/lamyaa/publications/fse14.pdf (Luo et al., FSE 2014, "An Empirical Analysis of Flaky Tests") | peer-reviewed (canonical, year-less query hit) | The mir.cs.illinois.edu/winglam mirror 404'd; the lamyaa mirror is a binary PDF and the pdfplumber chain was out of the moderate-tier call budget. Superseded for THIS question by the 2026 SAP HANA replication (read in full), which uses the same taxonomy shape (Async Wait / Concurrency / Test-Order / Timeout) on a 559-report corpus. |
| https://dl.acm.org/doi/10.1145/2635868.2635920 | peer-reviewed (ACM canonical record) | Paywalled; metadata only (201 commits, 51 projects, FSE 2014). |
| https://arxiv.org/pdf/2402.05223 ("Taming Timeout Flakiness", SAP HANA) | peer-reviewed preprint | `arxiv.org/html/2402.05223v1` 404'd and the ar5iv render is a "Conversion to HTML had a Fatal error" stub. Same group's 2026 follow-up (2602.03556) WAS read in full. |
| https://arxiv.org/pdf/2112.04919 ("A Qualitative Study on the Sources, Impacts, and Mitigation Strategies of Flaky Tests") | peer-reviewed preprint | Budget; corroborates the taxonomy already covered. |
| https://arxiv.org/pdf/2401.15788 ("230,439 Test Failures Later: Flaky Failure Classifiers") | peer-reviewed preprint | Budget; about post-hoc classification, not timeout design. |
| https://arxiv.org/pdf/2602.19098 ("A Systematic Evaluation of Environmental Flakiness in JavaScript Tests", 2026) | peer-reviewed preprint | Budget; JS-specific, recency-scan evidence only. |
| https://arxiv.org/pdf/2605.11482 ("NeuroFlake: Neuro-Symbolic LLM Framework for Flaky Test Classification", 2026) | peer-reviewed preprint | Budget; recency-scan evidence only. |
| https://arxiv.org/pdf/2502.02760 ("A Preliminary Study of Fixed Flaky Tests in Rust Projects", 2025) | peer-reviewed preprint | Budget; language-specific. |
| https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | industry (canonical) | FETCH ATTEMPTED AND FAILED: the fetched page returned only the 25-comment section + nav chrome, no post body. Comments alone are community-tier, so this is recorded as snippet-only and does NOT count toward the gate. |
| https://docs.pytest.org/en/7.1.x/how-to/failures.html | official doc | Snippet only; it is about `--pdb`/`-x` handling, not the error-vs-failure taxonomy. |
| https://docs.pytest.org/en/stable/example/reportingdemo.html | official doc | Snippet only; demo output, superseded by discussion #7950's explicit rule. |
| https://experts.illinois.edu/en/publications/an-empirical-analysis-of-flaky-tests | institutional record | Metadata only. |

## Queries run (three-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| year-less canonical | `pytest difference between "error" and "failure" in test summary exception raised in test body` | the load-bearing semantics question |
| year-less canonical | `"An Empirical Analysis of Flaky Tests" Luo Hariri Eloussi Marinov taxonomy async wait 45% concurrency test order dependency` | founding-paper prior art |
| current-year (2026) | `flaky tests 2026 empirical study root causes resource contention timeout CI mitigation` | frontier |
| last-2-year (2025) | `flaky tests survey 2025 arxiv taxonomy timeout machine load infrastructure flakiness` | recency window |

## Recency scan (2024-2026)

**Performed.** Result: **4 new findings** that materially complement (and in one
case supersede) the canonical FSE-2014 taxonomy.

1. **arXiv:2602.03556 (2026, SAP HANA)** -- a modern 559-report replication of
   the Luo-style taxonomy. It splits out **Timeout as its own first-class root
   cause at 16%** (Luo 2014 folded most of this under Async Wait), and defines it
   explicitly as failures "due to system load or slow hardware". Its most
   directly transferable finding: the test class whose timeouts were calibrated
   **close to the average execution time** had 18% timeout flakiness vs 7% for
   the class governed by one loose global timeout. That is an empirical argument
   against tight per-call constants.
2. **arXiv:2310.12132 (v2, in the window)** -- quantifies the exact mechanism in
   Main's hypothesis: 46.6% of flaky tests change failure rate under resource
   constraint, and **CPU is the dominant resource**. Confirms "CPU contention
   makes tests flaky" is mainstream, measured, and not folklore.
3. **arXiv:2504.16777 (2025, Systemic Flakiness)** -- 75% of flaky tests fail in
   clusters sharing one root cause. Relevant negative evidence here: a
   contention-induced flake would be expected to take **several** tests down at
   once, not exactly one. The observed run had exactly one non-passing item.
4. **2026 JS/LLM work (arXiv:2602.19098, arXiv:2605.11482)** -- environmental
   flakiness and LLM-based flake classification are active 2026 topics; neither
   changes the recommendation here.

No 2024-2026 source contradicts pytest's error-vs-failure rule, which has been
stable since at least pytest 5.x.

## Key findings

1. **An exception in a test BODY is a FAILURE; only setup/teardown/collection
   exceptions are ERRORS.** "pytest considers a *failure* any assertion error or
   exception raised inside a test function; *errors* happen when an assertion
   error or exception is raised during setup/teardown/collection." (pytest-dev
   maintainer, https://github.com/pytest-dev/pytest/discussions/7950, accessed
   2026-07-27). `subprocess.run(..., timeout=60)` at
   `backend/tests/test_phase_75_ci_gates.py:120-124` sits in a plain test body
   with no fixture wrapper, so `TimeoutExpired` propagates from the CALL phase
   -> pytest records `failed`, not `error`.
2. **The observed summary line has zero `failed`.** `249 passed, 1 skipped, 2060
   deselected, 1 warning, 1 ERROR` -- if the collection-count gate had timed
   out, the line would have read `... 1 failed ...` (and, if it were the only
   problem, no `error` at all). The shape does not match the hypothesis.
3. **A collection error is also excluded.** `pytest.ini` (read in full, 10
   lines) sets only `markers`; there is **no `addopts`, no
   `--continue-on-collection-errors`, no `filterwarnings`**. Without
   `--continue-on-collection-errors`, a collection error aborts with
   `!!! Interrupted: N error during collection !!!` and runs **zero** tests --
   incompatible with `249 passed`.
4. **Therefore the surviving explanation is a fixture SETUP or TEARDOWN error on
   exactly one test.** Both produce `error` with no `failed`. They are
   distinguishable from the numbers alone: a **setup** error means that test did
   NOT run (so the passed-count is one LOWER than a clean run), while a
   **teardown** error means it ran and passed (so the passed-count is
   UNCHANGED). Main can settle this with arithmetic, no re-run required: if a
   clean run of the same selection reports 250 passed -> setup error; if it
   reports 249 passed -> teardown error.
5. **CPU contention is a real, measured flakiness cause -- just not via this
   code path.** 46.6% of flaky tests are resource-affected, and CPU is the
   dominant resource (arXiv:2310.12132). So the contention half of Main's
   reasoning is well-supported by the literature; only the pytest-reporting half
   is wrong. Something in the suite IS load-sensitive; the ERROR bucket says it
   is in a fixture, not in this test body.
6. **Tight, hand-set timeouts are the empirically worse design.** SAP HANA's
   system tests, whose timeouts were calibrated "close to the test's average
   execution time", had 18% timeout-flakiness vs 7% for tests under one loose
   global timeout (arXiv:2602.03556). pytest-timeout's own README frames
   timeouts as "a last resort, not an expected failure mode".
7. **`pytest.main()` is officially discouraged for repeated in-process runs.**
   "Calling pytest.main() will result in importing your tests and any modules
   that they import. Due to the caching mechanism of python's import system,
   making subsequent calls to pytest.main() from the same process will not
   reflect changes to those files between the calls. For this reason, making
   multiple calls to pytest.main() from the same process ... is not recommended."
   (https://docs.pytest.org/en/stable/how-to/usage.html, accessed 2026-07-27).
   Note the stated reason is **import caching / staleness**, not a hard
   re-entrancy crash -- and it is exactly the failure mode that would bite here,
   since the outer session has already imported every module under
   `backend/tests/`.
8. **Clustered-failure evidence argues against contention too.** 75% of flaky
   failures co-occur in clusters averaging 13.5 tests (arXiv:2504.16777). A
   machine-wide CPU squeeze severe enough to blow a 9.5x headroom margin would
   be expected to disturb more than one item.

## Internal code inventory

### A. Every `subprocess.run` in `backend/tests/test_phase_75_ci_gates.py`

| Line | Command | timeout | Fact it establishes | Obtainable in-process? |
|---|---|---|---|---|
| :120-124 | `sys.executable -m pytest backend/tests/ -q -m "not requires_live" --collect-only` | **60** | pytest's OWN marker selection deselects exactly `EXPECTED_REQUIRES_LIVE_DESELECTED = 16` (:113), plus arithmetic `total-collected==deselected` and `collected>0` | Partly. `pytest.main([...], plugins=[counter])` could count items, but it is re-entrant collection inside a live session and the official docs discourage repeated in-process `pytest.main()`. A static AST scan for `@pytest.mark.requires_live` gets a WEAKER fact (see Recommendation rank 4). |
| :164-169 | `sys.executable -m pytest backend/tests/test_phase_23_2_14_no_reentrant_locks.py -q -m "not requires_live" --collect-only` | **30** | the 5 lock-guard tests are SELECTED (not accidentally excluded) by the marker filter | Same as above; single-file so far cheaper. |
| :188-192 | `sys.executable scripts/qa/coverage_tier_check.py --coverage-json <missing>` | **15** | M4: missing coverage json -> exit 2, never silent exit 0 | YES, trivially -- the script is importable Python; call its `main()`/entry with `SystemExit` capture. The subprocess is only buying "exit code as the CLI produces it". |
| :225-229 | `... coverage_tier_check.py --doc <mutated> --coverage-json <synthetic>` | **15** | M5: a bar above measured coverage exits 1 and names `paper_trader.py` on stderr | YES (same); stderr assertion would become a capsys assertion. |
| :256-260 | `... coverage_tier_check.py --coverage-json <real-ish>` | **15** | companion: real measurements exit 0 (proves M5 tests the bar, not a broken comparison) | YES (same). |

Measured headroom (this machine, idle, 2026-07-27): the :120 collection costs
**6.99 s** wall (`2294/2310 tests collected (16 deselected) in 6.99s`) against a
60 s budget -> **8.6x**. That is by far the tightest ratio of the five; the three
`coverage_tier_check.py` calls are interpreter-start-bound (~0.2-0.5 s) against
15 s -> ~30-75x. So :120 is the only one with a plausible contention exposure,
which is consistent with Main having singled it out.

### B. The invariant that must survive any fix

`EXPECTED_REQUIRES_LIVE_DESELECTED = 16` at :113 (phase-80.44 removed the
`1563/1579` totals pin; the file's comment at :129-141 records three prior
re-baselines and the reasoning). Any fix MUST still fail when a
`@pytest.mark.requires_live` decorator is added or removed. Also preserved must
be the two derived assertions at :150 and :154.

Cross-check performed: `EXPECTED_REQUIRES_LIVE_DESELECTED = 16` matches the live
measurement (16 deselected) as of 2026-07-27 -- the gate is currently
authoritative, not stale.

### C. Repo-wide pattern search

- **`pytest` invoked as a subprocess: ONLY this file.** `grep -rn '"pytest"'
  backend/tests scripts` returns `test_phase_75_ci_gates.py:121`,
  `test_phase_75_ci_gates.py:165`, and a version string in
  `backend/tests/test_phase_75_deps.py:91`. There is no other in-suite
  pytest-in-pytest.
- **No retry/timeout helper exists.** `pytest-timeout`, `pytest-rerunfailures`,
  `flaky`, `pytest-xdist` are all absent from `backend/requirements.txt`.
- **`pytest.ini` (10 lines, read in full)** registers only the `requires_live`
  marker. **No `addopts`, no `filterwarnings`, no
  `--continue-on-collection-errors`.**
- **`backend/tests/conftest.py` (22 lines, read in full)** only does
  `os.environ.setdefault("PYFINAGENT_TEST_NO_BQ", "1")` at import time. It
  defines **no fixtures**, so it is not the ERROR source.
- 53 occurrences of `timeout=` across `backend/tests/*.py` -- fixed constants
  are the house style, so any policy chosen here has 50+ other sites it could
  later be applied to. Not in scope for 80.46.

### D. The leading ALTERNATIVE hypothesis (found during the internal sweep)

`backend/tests/test_phase_76_9_2_max_bridge.py:134-158` -- the **`live_bridge`
fixture**, function-scoped (`@pytest.fixture()` at :134), consumed by **four**
tests (:160, :165, :181, :192), carrying **no** `requires_live` marker (so it
runs in the ordinary suite):

- **SETUP** spawns a real `subprocess.Popen` of the bridge script (:142-143),
  then polls health with `for _ in range(50): urlopen(base+"/health",
  timeout=1) ... except: time.sleep(0.1)`. Worst case is **50 x (1 s socket
  timeout + 0.1 s sleep) = ~55 s** before the `else:` branch calls
  `pytest.fail("bridge never became healthy")` (:151-153).
- `pytest.fail()` raised **inside a fixture** is an exception during SETUP ->
  pytest records it as an **ERROR**, not a failure. This matches the observed
  bucket exactly, where `TimeoutExpired` in a test body does not.
- The magnitude matches too: 95.89 s - ~16 s = **~80 s** of excess, and this one
  fixture can burn ~55 s of it on its own under exactly the condition Main
  measured (three competing pytest subprocesses + vitest).
- **TEARDOWN** (:155-158) is also exposed: `proc.kill()` then
  `upstream.shutdown()` on a `ThreadingHTTPServer`. `shutdown()` blocks until
  the `serve_forever` loop acknowledges, with **no timeout** -- a starved
  daemon thread can stall it arbitrarily. A teardown exception is likewise an
  **ERROR**, and a teardown error leaves the test counted as **passed**.
- `_free_port()` (:128) is a classic bind-then-release race: two concurrent
  suites can be handed the same port.

This is the only `ThreadingHTTPServer` / `serve_forever` in the whole test tree
(`grep -rln` -> one file), i.e. the only place a real server is booted in-process.

### E. Arithmetic reconstruction of the mystery run (measured, not asserted)

Measured today: `2294/2310 tests collected (16 deselected)`. Observed mystery
line: `249 passed, 1 skipped, 2060 deselected`. `249 + 1 + 2060 = 2310`.

- The run therefore used a `-k`-style selection over the whole `backend/tests/`
  tree (a `-m "not requires_live"` run deselects **16**, not 2060), selecting
  **250** items.
- pytest's `deselected` is fixed at selection time; a **setup**-errored test is
  still SELECTED but is counted in `error`, not `passed`. So:
  - setup error => selected = 249 passed + 1 skipped + 1 error = **251**, i.e.
    the tree must have held **2311** items at that moment;
  - teardown error => selected = 249 + 1 = **250** and the errored test is ALSO
    inside the 249 passed, i.e. the tree held **2310** -- today's number.
- Caveat, stated plainly: `backend/tests/test_phase_80_40_perf_metrics_drawdown.py`
  is currently **untracked** and step 80.44 edited this area today, so the total
  may legitimately have been 2311 then. The +-1 is suggestive, **not** proof.

**Decisive checks Main can run for near-zero cost (no full-suite run):**
1. Re-run the SAME `-k` selection and compare the passed count: 250 passed =>
   the mystery item was a **teardown** error; 249 passed => **setup** error.
2. Add `-ra` (or scroll back): the `short test summary info` block prints
   `ERROR <nodeid> - at setup of ...` / `- at teardown of ...` and NAMES the
   test. That single line ends the investigation.
3. Check whether the `-k` expression even selected
   `test_phase_75_ci_gates.py::test_backend_not_requires_live_collection_count_is_stable`.
   If it did not, the original hypothesis is refuted a third time, independently.

## Application to pyfinagent + RECOMMENDATION

### Verdict on the hypothesis: **WRONG** (stated plainly, as requested)

The load-bearing claim -- "`subprocess.TimeoutExpired` surfaces in pytest as an
ERROR rather than a FAILURE" -- is **false**. pytest's rule is phase-based, not
exception-type-based: body -> failure, setup/teardown/collection -> error
(discussion #7950, corroborated by pytest-timeout's own README, where a fired
timeout uses `pytest.fail()` and is reported as a failure). The observed line
had **zero** `failed`. Three independent facts refute the hypothesis:
(1) the phase rule; (2) the absent `failed` count; (3) `pytest.ini` has no
`--continue-on-collection-errors`, so a collection error would have run zero
tests, not 249.

The *contention* half of Main's reasoning is well-supported (46.6% of flaky
tests are resource-affected, CPU dominant -- arXiv:2310.12132). The evidence
just points the contention at a **fixture**, and
`test_phase_76_9_2_max_bridge.py::live_bridge` is a near-perfect fit on both
bucket (setup/teardown -> ERROR) and magnitude (~55 s poll ceiling vs ~80 s
excess).

### Ranked candidate fixes for the `test_phase_75_ci_gates.py` timeouts

Note the re-motivation: these timeouts are **not** the cause of the 80.46
incident, but they remain a latent flake class, and the failure mode is *worse*
than what was observed -- a fired timeout reddens the lane as a FAILURE.

**Rank 1 (recommend): loose timeout derived from a measured baseline, >=20x, not
a tuned one.** Raise :120's 60 s to ~300 s (43x today's measured 6.99 s) and
:164's 30 s to ~120 s; leave the three 15 s script calls alone (already ~30-75x).
Evidence: SAP HANA measured **18% timeout-flakiness for tests whose timeouts
were calibrated "close to the test's average execution time" vs 7% for tests
under one loose global timeout** (arXiv:2602.03556). pytest-timeout frames
timeouts as "a last resort, not an expected failure mode". One-line diff, zero
semantic change, the `EXPECTED_REQUIRES_LIVE_DESELECTED = 16` invariant is
untouched.
**Strongest counter-argument to my own top pick:** it does not eliminate the
flake class, it only pushes its probability down -- and it *lengthens* the
worst case, so a genuinely wedged child now stalls CI for 5 minutes instead of
1. If someone later breaks collection into an infinite loop, the 300 s wait is
pure dead time on every run until it is noticed. I still prefer it because the
timeout here is a *hang-catcher of last resort*, not an assertion: nothing about
the gate's meaning depends on collection finishing in under a minute, so the
tight bound buys no signal and costs false reds. A defensible compromise is
300 s **plus** a comment recording the measured 6.99 s baseline and the date, so
the next reader can see the headroom ratio rather than re-deriving it.

**Rank 2: in-process collection via `pytest.main([...], plugins=[counter])` with
a `pytest_collection_modifyitems` hook.** Removes the timeout entirely -- the
strongest fix *if* it were safe. It is not clearly safe: the official docs say
"making multiple calls to `pytest.main()` from the same process ... is not
recommended" (import caching), and here the call would be *re-entrant inside an
already-running session* whose plugin manager, capture, and `cacheprovider` are
live. A CI gate should not be the first user of a documented-discouraged API.
Only worth revisiting if the timeout class actually starts firing.

**Rank 3: retry-once-on-`TimeoutExpired` with a longer second budget.**
Effective and cheap in practice; Google's documented policy is to rerun only
tests already marked flaky. Costs more code than rank 1 and risks masking a real
hang. Acceptable as an addition to rank 1, not a substitute.

**Rank 4 (do NOT do): replace the subprocess with a static AST/grep scan for
`@pytest.mark.requires_live`.** It is fast and timeout-free but changes the
fact: it would assert "16 decorators exist in source" rather than "pytest's own
selection deselects 16", missing `pytestmark`-applied and conftest-applied
markers and any breakage of the `-m` expression itself. It is also exactly the
source-scan-as-guard antipattern this project has already flagged.

**Separately (bigger money, outside 80.46 as written): the actual incident.**
If the `-ra` / re-run check confirms the `live_bridge` fixture, the fix is
(a) bound the health poll by wall clock rather than iteration count and raise
the ceiling, (b) give `upstream.shutdown()` a watchdog or run it in a joined
thread with a timeout, and (c) make `_free_port()` collision-tolerant. Per
`feedback_queue_discovered_defects_in_masterplan`, that belongs in its OWN
research-gated masterplan step, written for an executor with no memory of this
discovery -- not folded into 80.46 silently.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 2 official
      pytest docs/discussions, 1 official plugin README, 3 peer-reviewed
      preprints)
- [x] 10+ unique URLs total (6 read-in-full + 12 snippet-only = 18)
- [x] Recency scan (2024-2026) performed + reported (4 findings)
- [x] Full pages read, not abstracts, for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the named module in full + repo-wide pattern
      grep + pytest config + conftest
- [x] Contradictions noted (the hypothesis is refuted; the contention mechanism
      it assumed is nonetheless real and cited)
- [x] All claims cited per-claim
- [ ] GAP: Luo et al. FSE 2014 (the year-less canonical) was NOT read in full --
      both mirrors failed (404 / binary PDF) within the moderate-tier budget. Its
      taxonomy is covered by the 2026 SAP HANA replication, which was read in
      full. Flagged, not papered over.
- [ ] GAP: the Google Testing Blog post body did not render (comments only), so
      the rerun-policy claim in Rank 3 rests on the comment thread + secondary
      coverage, not the primary post.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Main's hypothesis is REFUTED. pytest buckets by PHASE, not exception type: an exception in a test body is a FAILURE; only setup/teardown/collection exceptions are ERRORS (pytest-dev discussion #7950). subprocess.TimeoutExpired at test_phase_75_ci_gates.py:120 is in a test body, so it would have printed '1 failed' -- the observed line had zero failed. A collection error is also excluded: pytest.ini has no --continue-on-collection-errors, so it would have run zero tests, not 249. The surviving explanation is a fixture setup/teardown error, and the internal sweep found a near-perfect fit: test_phase_76_9_2_max_bridge.py:134-158 live_bridge boots a real subprocess and polls health up to 50x(1s+0.1s) = ~55s before pytest.fail() INSIDE the fixture (=ERROR), with an unbounded ThreadingHTTPServer.shutdown() in teardown. Measured collection is 6.99s vs the 60s budget (8.6x). Recommendation: raise the two pytest-subprocess timeouts to a loose >=20x bound (SAP HANA 2026: tight-calibrated timeouts 18% flaky vs 7% for loose global); reject in-process pytest.main (officially discouraged, re-entrant) and reject a source-scan replacement (weaker fact).",
  "brief_path": "handoff/current/research_brief_80.46.md",
  "gate_passed": true
}
```
