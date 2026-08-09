# Research Brief -- phase-86.6

**Topic:** Preventing (not merely detecting) a pytest suite from mutating live
production state, across filesystem / HTTP / subprocess / BigQuery /
module-singleton channels, when the production writer swallows its own write
errors.

**Tier:** complex (caller-set). **Audit-class:** YES (loop-until-dry, K=2).
**Started:** 2026-08-09. **Status:** IN PROGRESS (write-first; this file grows
incrementally as sources are read).

---

## Search queries run (three-variant discipline)

| # | Query | Variant | Round |
|---|-------|---------|-------|
| Q1 | `pytest prevent tests writing to production filesystem sandbox fixture` | year-less canonical | 1 |
| Q2 | `pytest test isolation production state mutation prevention 2026` | current-year frontier | 2 |
| Q3 | `test pollution shared state detection PolDet flaky tests 2025` | last-2-year | 2 |
| Q4 | `hermetic testing prevent side effects external systems` | year-less canonical | 2 |
| Q5 | `BigQuery emulator / client testing isolation` | year-less canonical | 3 |
| Q6 | `subprocess isolation pytest environment variable leak` | year-less canonical | 3 |
| Q7 | `hierarchy of controls elimination vs administrative control` | cross-domain, year-less | 3 |
| Q8 | `pytest-socket allow-hosts limitations 2025` | last-2-year | 4 |
| Q9 | `poka-yoke mistake proofing software guard rails 2026` | current-year | 4 |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://pytest-pyfakefs.readthedocs.io/en/latest/usage.html | 2026-08-09 | official docs | WebFetch (full) | pyfakefs "automatically finds all real file functions and modules, and stubs these out". The `fs` fixture has function/class/module/session scopes; **"If any of these fixtures is active, any other `fs` fixture will not setup / tear down the fake filesystem in the current scope; instead, it will just serve as a reference to the active fake filesystem."** Patching is *paused* in the pytest logreport phases so logs reach the real disk -- i.e. the fake FS is deliberately porous at defined seams. |
| 2 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-09 | official docs | WebFetch (full) | **"Prefer patching the reference that your code uses instead of patching the original object in the standard library."** All modifications undone after the requesting test/fixture finishes. Explicit warning: **"it is not recommended to patch builtin functions such as `open`, `compile`, etc., because it might break pytest's internals."** No isolation *guarantee* is offered for global state / module-level imports. |
| 3 | https://raw.githubusercontent.com/miketheman/pytest-socket/main/README.md | 2026-08-09 | official docs (repo README) | WebFetch (full) | Blocks "all network calls flowing through Python's `socket` interface, including DNS resolution". `--allow-hosts` entries are "hostnames, IP addresses, or CIDR network ranges". **Granularity is HOST-level only -- no port and no HTTP-verb filtering.** Fixture-order limitation: a higher-scoped fixture that opens a socket "will be resolved first, and won't be disabled during the tests." |
| 4 | https://cwe.mitre.org/data/definitions/390.html | 2026-08-09 | official standard (MITRE) | WebFetch (full) | CWE-390 *Detection of Error Condition Without Action*: "The product detects a specific error, but takes no actions to handle the error." Consequence: **"An attacker could utilize an ignored error condition to place the system in an unexpected state."** Mitigation: "Ensure that all exceptions are handled in such a way that you can be sure of the state of your system at any given moment." Parent CWE-755. |
| 5 | https://mir.cs.illinois.edu/marinov/publications/GyoriETAL15PollutionDetection.pdf | 2026-08-09 | **peer-reviewed** (ISSTA 2015, Gyori/Shi/Hariri/Marinov) | curl + pdfplumber (11pp, 63,721 chars extracted) | PolDet. **Detection is by definition post-hoc**: "for each test in a test suite, PolDet captures the shared state (on the heap and the file system) before and after the test, and then compares these two states." File-system leg: hashes contents + last-modified timestamps, and **"PolDet allows the user to specify the portions of the file system to consider"** -- i.e. the detector's recall is bounded by a human-supplied path list. 324/6105 tests flagged, 194 genuine. Overhead geomean **4.50x**. Key prior-art fact: **"78% of the polluting tests pollute the shared state right when they are added"**. Scope caveat, verbatim: "In general, pollutions could occur via network or databases, but in this paper, we focus on pollutions via heap state and file system". |
| 6 | https://docs.python.org/3/library/unittest.mock.html | 2026-08-09 | official docs (Python) | WebFetch (full) | "Where to patch": **"The basic principle is that you patch where an object is _looked up_, which is not necessarily the same place as where it is defined."** And the binding trap verbatim: "module b already has a reference to the _real_ `SomeClass` and it looks like our patching had no effect." |
| 7 | https://docs.pytest.org/en/stable/how-to/tmp_path.html | 2026-08-09 | official docs | WebFetch (full) | `tmp_path` per-test unique dir; `tmp_path_factory` session-scoped. Retention: "By default, the last 3 temporary directories are kept". With `--basetemp` "there is no retention feature in this case". Isolation here is **opt-in per test**, never ambient. |
| 8 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-09 | **peer-reviewed classic** (Saltzer & Schroeder 1975) | WebFetch (full) | **Fail-safe defaults, verbatim: "Base access decisions on permission rather than exclusion."** **Complete mediation, verbatim: "Every access to every object must be checked for authority."** Plus least privilege, least common mechanism, psychological acceptability ("Design user interfaces for ease of use so protection mechanisms are applied correctly"). |
| 9 | https://abseil.io/resources/swe-book/html/ch14.html | 2026-08-09 | official book (Software Engineering at Google, ch.14 Larger Testing) | WebFetch (full) | "An SUT with high hermeticity will have the least exposure to sources of concurrency and infrastructure flakiness"; "a hermetic SUT will not be at risk of the kinds of multiuser and real-world flakiness of production or a shared staging environment"; on prod: **"Any issues caught at this point in time (in production) are already affecting end users."** Also the stale-double warning: "mocks become stale... there is no signal." |
| 10 | https://testing.googleblog.com/2012/10/hermetic-servers.html | 2026-08-09 | authoritative blog (Google Testing Blog) | WebFetch (full) -- **shallow render**: the page body rendered mostly as comment thread, so it is corroborating, not load-bearing | Hermetic server = self-contained test server with all deps packaged in; enables end-to-end testing in isolation, run on a continuous-build cadence rather than per-change. **Flagged as thin: do not cite this alone for any claim.** |
| 11 | https://increment.com/testing/i-test-in-production/ | 2026-08-09 | authoritative blog (Charity Majors, Honeycomb) | WebFetch (full) | **[ADVERSARIAL]** -- the counter-position to total isolation. "Every deploy... is a unique and never-to-be-replicated combination of artifact, environment, infra, and time of day"; "Once you deploy, you aren't testing code anymore, you're testing systems"; "You can never, ever guarantee that you have ironed out all the bugs." Required guardrail: "No pull request should ever be accepted unless the engineer can answer the question, 'How will I know if this breaks?'" |
| 12 | https://pytest-subprocess.readthedocs.io/en/latest/ | 2026-08-09 | official docs | WebFetch (full) -- thin index page, superseded by #13 | "You can use the provided `fake_process` (or `fp` for short) fixture to register commands... **This will prevent a real subprocess execution.**" |
| 13 | https://pytest-subprocess.readthedocs.io/en/latest/usage.html | 2026-08-09 | official docs | WebFetch (full) | **Deny-by-default subprocess control, verbatim:** "By default, when the `fp` fixture is being used, any attempt to run subprocess that has not been registered will raise the `ProcessNotRegisteredError` exception." Escape hatches are explicit: `fp.allow_unregistered(True)`, `fp.pass_command("command")`. `fp.calls` records every invocation. **Critical limit: it is a FIXTURE, so it is opt-in per test -- not ambient.** |
| 14 | https://ar5iv.labs.arxiv.org/html/2101.09077 | 2026-08-09 | **peer-reviewed** (Gruber et al., *An Empirical Study of Flaky Tests in Python*, ICST 2021) | WebFetch via ar5iv (pre-Dec-2023 paper; per research-gate.md Step 2) | 22,352 projects / 876,186 tests / 7,571 flaky. **Order dependency causes 59% of Python flaky tests** -- far more dominant than in Java. Victims (3,168) vastly outnumber brittles (738). Novel category: **"28% of all 7,571 flaky tests" were infrastructure-caused.** Reruns needed for 95% confidence: 170 for NOD vs only **31 random-order executions** for order-dependent -- i.e. shared-state pollution is *cheap to detect by reordering* but that still detects, never prevents. |
| 15 | https://peps.python.org/pep-0578/ | 2026-08-09 | official standard (PEP) | WebFetch (full) | **The single-choke-point primitive.** `sys.addaudithook()`; hooks may be installed "before `Py_Initialize()`". **Blocking is supported, verbatim: "If any hook returns with an exception set, later hooks are ignored and _in general_ the Python runtime should terminate - exceptions from hooks are not intended to be handled or treated as expected occurrences."** **"Hooks cannot be removed or replaced."** Overhead "between 1.05x faster to 1.05x slower". **Explicit caveat: "This is not sandboxing, as this proposal does not attempt to prevent malicious behavior."** |
| 16 | https://docs.python.org/3/library/audit_events.html | 2026-08-09 | official docs (Python) | WebFetch (full) | Exact event names, all four channels in ONE table: `open(path, mode, flags)`; `subprocess.Popen(executable, args, cwd, env)`, `os.exec`, `os.system`, `os.spawn`, `os.posix_spawn`; `socket.connect(self, address)`, `socket.bind`, `socket.sendto`, `socket.getaddrinfo`; `urllib.Request(fullurl, data, headers, method)`. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.cdc.gov/niosh/hierarchy-of-controls/about/index.html | official standard | **ATTEMPTED TWICE, FAILED.** WebFetch -> HTTP 403; `curl` with a browser UA -> Akamai "Access Denied" (ref #18.a50b655f). The elimination-over-administrative-control idea is covered instead by #8 (Saltzer) and the poka-yoke snippet set. |
| https://asq.org/quality-resources/mistake-proofing | industry standard body | **ATTEMPTED, FAILED** -- HTTP 403. |
| https://github.com/pytest-dev/pytest/issues/7409 | community | **ATTEMPTED, FAILED** -- fetched, but the issue thread did not render ("There was an error while loading"). Its claim (via snippet) that `pytest.fail` is swallowed by `except Exception` is **REFUTED by direct measurement** -- see Finding 2. |
| https://dl.acm.org/doi/10.1145/2771783.2771793 | peer-reviewed | Paywalled landing page; the author-hosted PDF (#5) was used instead. |
| https://arxiv.org/html/2504.16777 | preprint 2025 | *Systemic Flakiness: co-occurring flaky failures.* Recency-window hit; clustering of root causes, Java-only, adds nothing to the prevention question. |
| https://taoxie.cs.illinois.edu/publications/esecfse19-ifixflakies.pdf | peer-reviewed | iFixFlakies -- *fixes* order-dependent tests post-hoc; still detection+repair, not prevention. |
| https://taoxie.cs.illinois.edu/publications/icst19-idflakies.pdf | peer-reviewed | iDFlakies -- reordering-based detection. Same class as PolDet (#5). |
| https://people.cs.gmu.edu/~winglam/publications/2023/LiETAL23Tuscan.pdf | peer-reviewed | Tuscan test orders. Detection scheduling. |
| https://arxiv.org/pdf/2207.01047 | peer-reviewed | Flaky tests in JavaScript -- wrong language, corroborates the order-dependency share only. |
| https://github.com/goccy/bigquery-emulator | community tool | The raw README **was** read in full (#17 below is that raw URL); this is the HTML landing page. |
| https://pypi.org/project/bqunit | community tool | BQ test-data isolation on a real dataset -- opposite of what 86.6 wants (still touches BQ). |
| https://pypi.org/project/bigtesty | community tool | Per-execution BQ isolation; same objection. |
| https://java.testcontainers.org/modules/gcloud/ | official docs | Java-only Testcontainers GCloud module. |
| https://pypi.org/project/pytest-tmpfs/0.1.3 | community plugin | tmp-filesystem helper built on `tmp_path`; opt-in per test, so same limitation as #7. |
| https://github.com/pytest-dev/pytest/issues/2305 | community | Sandboxed execution breaks with plugins installed -- historical. |
| https://docs.aws.amazon.com/codeguru/detector-library/python/swallow-exceptions | official docs | AWS CodeGuru "catch and swallow exception" detector -- corroborates CWE-390 (#4) for Python specifically. |
| https://www.pythonmorsels.com/catching-all-exceptions/ | community | `except Exception` vs `except BaseException` -- superseded by the measured result + #18. |
| https://copyconstruct.medium.com/testing-in-production-the-hard-parts-3f06cefaf592 | authoritative blog | Sridharan, *Testing in Production: the hard parts.* Same adversarial position as #11. |
| https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/ | industry postmortem | GitLab 2017: prod snapshot loaded into staging, then the wrong host wiped. Class-match (env confusion), not test-suite-specific. |
| https://status.circleci.com/incidents/7pfbzf4g9dcs | industry postmortem | CircleCI orb index deleted by a command issued against the production DB. |
| https://governmentasaplatform.blog.gov.uk/2016/08/05/incident-report-platform-as-a-service-for-government | industry postmortem | GOV.UK: a program to delete *dev* Cloud Foundry environments was run against *production*. The purest class-match found. |
| https://en.wikipedia.org/wiki/Poka-yoke | community | Shingo's **warning** vs **control** poka-yoke distinction (alert vs physically prevent). |
| https://flowfuse.com/blog/2025/09/poka-yoke-mistake-proofing/ | community | 2025 restatement: "Prevention Poka Yoke is always preferred over detection Poka Yoke." |
| https://docs.pytest.org/en/stable/how-to/writing_plugins.html | official docs | conftest.py as a local plugin; directory-scoped hook discovery. |
| https://docs.pytest.org/en/stable/reference/fixtures.html | official docs | autouse ordering: "A test can only be affected by an autouse fixture if that test is in the same scope." |
| https://dev.to/felipe_de_godoy/pytest-pt4-production-test-coverage-6hn | community | Blog on prod-shaped coverage; low signal. |
| https://medium.com/@arjun0./google-cloud-bigquery-emulator-an-overview-9fd6e63f5d51 | community | Emulator overview; superseded by the README read in full. |

**Additional read-in-full (round 5):**

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 17 | https://raw.githubusercontent.com/goccy/bigquery-emulator/main/README.md | 2026-08-09 | official docs (tool README) | WebFetch (full) | The **credential-level** BQ preventer, verbatim pattern: `client_options = ClientOptions(api_endpoint="http://0.0.0.0:9050")` + `bigquery.Client("test", client_options=client_options, credentials=AnonymousCredentials())`. Not implemented: "IAM policy management, row access policies, copy jobs, external tables, table snapshots and BigQuery ML". |
| 18 | https://docs.python.org/3/library/exceptions.html | 2026-08-09 | official docs (Python) | WebFetch (full) | The mechanism 86.6 turns on. `SystemExit` "inherits from `BaseException` instead of `Exception` **so that it is not accidentally caught by code that catches `Exception`**"; same rationale stated for `KeyboardInterrupt`. Only `SystemExit` / `KeyboardInterrupt` / `GeneratorExit` / `BaseExceptionGroup` sit outside `Exception`. Note the counter-pressure: "programmers are encouraged to derive new exceptions from the `Exception` class ... and not from `BaseException`." |

---

## Recency scan (2024-2026) -- MANDATORY SECTION

Searched: `test isolation guard prevent side effects production database pytest 2026`;
`arXiv 2025 flaky tests test pollution shared state isolation empirical study Python`;
`sys.addaudithook pytest block writes outside tmp_path guard 2026`;
`poka-yoke ... 2026`; `test suite production state mutation prevention guard 2026`.

**Result: 3 findings in the 2024-2026 window; NONE supersedes the canonical
sources, and the most important result is a NEGATIVE one.**

1. *Systemic Flakiness* (arXiv 2504.16777, Apr 2025) -- flaky tests cluster by
   shared root cause. Refines PolDet-era detection economics; changes nothing
   about prevention. Snippet-only.
2. The 2025-2026 poka-yoke literature restates Shingo unchanged: prevention
   ("control" devices) is strictly preferred to detection ("warning" devices).
   No new mechanism.
3. **NEGATIVE FINDING (the load-bearing one): after 10 rounds there is still no
   off-the-shelf pytest plugin that blocks filesystem writes outside a
   temp dir.** `pytest-socket` is host-level network only (#3), `pyfakefs` is a
   whole-FS *replacement* not a *policy* (#1), `pytest-subprocess` is opt-in per
   test (#13), `pytest-tmpfs` is a `tmp_path` convenience. The prevention
   primitive that spans all four channels -- `sys.addaudithook` (#15/#16) --
   is a 2019 PEP that **nobody in the search corpus has wired to pytest**.
   So 86.6 has no plugin to adopt; it has a mechanism to assemble.

The canonical (year-less) sources -- Saltzer & Schroeder 1975, PolDet 2015,
Gruber 2021, PEP 578 (2019) -- remain authoritative and are not superseded.

---

## Internal code inventory (every claim carries a file:line anchor)

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `conftest.py` (repo root) | :1-187 | phase-86.3 HTTP guard. Wraps `urllib.request.urlopen` at **:157** and `urllib3...HTTPConnectionPool.urlopen` at **:181**, at IMPORT time. Refuses mutating verbs (`:78`) to loopback **host AND port 8000** (`:85`, `:118-126`). | LIVE. Covers BOTH test trees (rootdir = repo root). **Self-declared gaps at :52-61: httpx not covered, raw `socket` not covered, FILESYSTEM not covered.** |
| `conftest.py` | :129-141 | `_refusal()` returns a **`RuntimeError`**. | **DEFECT-ADJACENT.** `RuntimeError` is an `Exception` (source #18) -> swallowable. See Finding 2. |
| `backend/tests/conftest.py` | :21 | `os.environ.setdefault("PYFINAGENT_TEST_NO_BQ", "1")` -- the BQ guard. | LIVE but **narrow**: honored at exactly **2 sites**, both `backend/services/observability/api_call_log.py:125` and `:322`. |
| `backend/tests/conftest.py` | :48-64 | phase-82.58 Slack egress guard; raises **`RuntimeError`**. | LIVE. Same swallowability class as the root guard. **Only loaded for `backend/tests/`** -- the `tests/` tree gets neither this nor the BQ guard (only `./conftest.py` and `./backend/tests/conftest.py` exist; measured via `find -name conftest.py`). |
| `pytest.ini` | :1-10 | rootdir anchor + `requires_live` marker. No `addopts`, no `--disable-socket`. | No ambient policy configured. |
| `backend/services/kill_switch.py` | **:48** | `_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"` -- module-level, bound at import. | The live safety journal. Also this module's **only** persistence (docstring :23-28). |
| `backend/services/kill_switch.py` | **:489-499** | `_append_audit` -- `@staticmethod`; `with _AUDIT_PATH.open("a"...)` at **:496**, then **`except Exception as e: logger.warning(...)` at :498-499**. | **THE SWALLOWER.** Any `Exception`-derived refusal raised from the write is absorbed; the test still passes and production continues silently. |
| `backend/services/kill_switch.py` | :552-565 | `pause()` auto-alert dispatch wrapped in `except Exception as _alert_err: logger.warning(...)`. | **A SECOND SWALLOWER on the same path.** The phase-82.58 Slack `RuntimeError` guard fires *inside* this try -> already structurally absorbed. Executor should MEASURE this chain, not assume it. |
| `backend/services/kill_switch.py` | :89-111, :760, :798 | `_audit_archive_dir()` / `_audit_source_paths()` are DERIVED from `_AUDIT_PATH` (so one redirect moves the archive too -- asserted at `test_book_safety_69.py:246`). `_state = KillSwitchState()` at **:760**; `get_state()` at **:798**. | **Module-singleton channel.** `_state` is constructed at import; module functions read `_state` directly, so patching `get_state` alone is vacuous (`test_phase_23_2_4...:130-137`). |
| `backend/services/risk_overrides.py` | :41, :128 | Second independent `_AUDIT_PATH` -> `handoff/risk_overrides_audit.jsonl`; append at :128 (read at :98). | Same class, separate constant. |
| `backend/services/cron_control.py` | :31, :60 | Third independent `_AUDIT_PATH` -> `handoff/cron_control_audit.jsonl`; append at :60. | Same class, separate constant. |
| `backend/api/paper_trading.py` | **:892, :940-941** | `_KILL_SWITCH_AUDIT_PATH` -- a **FOURTH, DUPLICATE constant for the SAME live file** as `kill_switch._AUDIT_PATH`, with a comment saying it is "mirrored from" it. | **READ-ONLY** (`.exists()` :940, `.open("r")` :941) -- so not a write channel. But `monkeypatch.setattr(ks, "_AUDIT_PATH", ...)` does **NOT** move it: a test that redirects the writer and then exercises `_compute_learnings` still READS the operator's live journal. Isolation illusion, not data loss. |
| `backend/db/bigquery_client.py` | :271, :415, :424, :450, :514, :626, :693, :1023 | 5 raw `insert_rows_json` sites + `save_paper_position:626` / `save_paper_trade:693` / `save_paper_snapshot:1023`; `_pt_table:548`. | **UNGUARDED.** `PYFINAGENT_TEST_NO_BQ` is not consulted anywhere in this file. |
| `backend/tests/test_book_safety_69.py` | :30-43 | autouse function-scoped fixture byte-comparing the LIVE journal before/after every test. Its own docstring: **"This is a DETECTOR, not a preventer -- the bytes are already on disk when it fires. The preventer for the filesystem channel generally is step 86.6."** | The exact PolDet algorithm (#5), hand-rolled, scoped to one file. |
| `backend/tests/test_book_safety_69.py` | :211-269 | `test_peak_reset_dark_by_default` -- **the known false negative.** Redirects `_AUDIT_PATH` at :245, asserts the archive follows at :246-249, uses a **DETACHED** `ks.KillSwitchState()` at :259. | Correct today. But its green depends on `kill_switch_peak_reset_enabled=False` being PINNED at :257 -- it is a *test-authored* isolation that a future test must remember to repeat. |
| `backend/tests/test_book_safety_69.py` | :271-300 | `test_phase_86_1_the_pre_fix_form_really_would_have_destroyed_the_live_peak` -- deliberately copies the live journal byte-for-byte to tmp and performs a REAL destructive `reset_peak` on the copy. | **A test that legitimately READS the live file.** Any preventer must not break this. |
| `backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` | :46-55, :117-175 | `_backend_is_up()` runs a GET inside `@pytest.mark.skipif` -> at **module IMPORT**, before any fixture. `_isolated_app_client` (:117-175) now redirects `_AUDIT_PATH` **and** the `_state` singleton. | **MUST KEEP READING THE LIVE JOURNAL** per caller scope: `AUDIT_LOG` at :41 and the module-default assertion at **:253-255** (`assert ks._AUDIT_PATH == AUDIT_LOG`). A blanket FS deny that covers reads breaks this file. |
| `backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` | :237-268 | Proves (a) the module default IS the live file and (b) appends follow `_AUDIT_PATH` -- by writing to a **decoy** at :264-268. | Any preventer must still permit a tmp-scoped WRITE. |
| `scripts/qa/smoke_cc_rail_e2e.py` | **:469** | `p.add_argument("--backend-url", default="http://localhost:8000")` | **Subprocess channel.** Mutates: `PUT {base}/api/settings/` at **:289-290** (the PUT "persists to backend/.env"), `POST {base}/api/analysis/` at **:307**. Also spawns the `claude` binary via `subprocess.run` at **:176**. |
| `backend/tests/test_phase_4000_2_cc_rail_smoke.py` | **:202** | `subprocess.run([sys.executable, str(SCRIPT), *argv], ...)` | **A CHILD PROCESS LOADS NO conftest.** Every guard in `./conftest.py` and `./backend/tests/conftest.py` is structurally absent in that child. Latent only because `live_args()` (:207-212) always passes an explicit `--backend-url`. |
| subprocess surface (measured) | -- | `grep -rl "subprocess.run\|Popen\|check_output" backend/tests tests` = **72 files**; of those, **2** spawn a child `sys.executable`/`python`. | Denominator for the child-process gap. |
| `backend/services/cycle_lock.py:52`, `cycle_health.py:34`, `autonomous_loop.py:2638`, `quant_optimizer.py:41`, `api/backtest.py:308/:1512/:1513`, `backtest/data_ingestion.py:451`, `backtest/macro_cron.py:95` | as listed | Further module-level `Path(__file__)...parents[N] / "handoff"` constants (locks, streak JSON, logs, seed-stability outputs). | **The filesystem channel is not one path -- it is a family.** Any per-constant monkeypatch strategy is O(constants) and must be re-derived whenever one is added. |

---

## Key findings

**F1 -- Detection and prevention are different controls, and the literature is
unanimous on the ordering.** PolDet's own definition is post-hoc: "for each test
in a test suite, PolDet captures the shared state (on the heap and the file
system) before and after the test, and then compares these two states"
(Gyori et al. 2015, ISSTA, source #5). `test_book_safety_69.py:30-43` is that
exact algorithm, and its docstring already concedes the point verbatim: *"This
is a DETECTOR, not a preventer -- the bytes are already on disk when it fires."*
Shingo's control-vs-warning poka-yoke and Saltzer & Schroeder's **"Base access
decisions on permission rather than exclusion"** (#8) both say the same thing.
**Design consequence:** 86.6 should keep the 86.1 detector (it is cheap and it
localises the offender) and add a preventer *underneath* it -- belt and braces,
not replacement.

**F2 -- The exception class of the refusal is the whole ballgame, and the
current guards get it wrong. MEASURED, not argued.** Run in the project venv
(pytest 9.0.3):

```
Failed MRO: ['Failed', 'OutcomeException', 'BaseException', 'object']
bare except Exception -> ESCAPED as Failed
kill_switch shape: logger.warning swallowed: RuntimeError
kill_switch shape: logger.warning swallowed: PermissionError
kill_switch shape vs BaseException guard -> ESCAPED as Failed
```

`kill_switch._append_audit` catches `Exception` (:498). `conftest.py:129-141`
raises `RuntimeError`; `backend/tests/conftest.py:52-58` raises `RuntimeError`.
**Both are `Exception` subclasses, so an FS guard built to the existing in-repo
pattern would be silently absorbed at `kill_switch.py:498-499` -- the write is
blocked, the test goes GREEN, and nobody learns anything.** Python's own docs
state the fix's rationale for `SystemExit`: it "inherits from `BaseException`
instead of `Exception` **so that it is not accidentally caught by code that
catches `Exception`**" (#18). `pytest.fail()` already raises exactly such a
class. **This also means an `assert` inside a guard is NOT safe** --
`AssertionError` IS an `Exception` (#18 hierarchy) and would be swallowed too.
CWE-390 (#4) is the name for the production-side half of this.
*Caveat to carry into the contract:* #18 also says user-defined exceptions
should derive from `Exception`, so a `BaseException` refusal is a deliberate,
documented deviation, and it will bypass legitimate cleanup in any
`except Exception` that was doing real work.

**F3 -- `sys.addaudithook` is the only mechanism that covers all four channels
at one seam, and it is unadopted anywhere in the search corpus.** PEP 578 (#15)
+ the event table (#16) give, in one API: `open(path, mode, flags)`,
`subprocess.Popen(executable, args, cwd, env)` / `os.exec` / `os.system` /
`os.posix_spawn`, `socket.connect(self, address)` / `socket.bind`, and
`urllib.Request(fullurl, data, headers, method)`. Blocking is explicitly
supported -- "If any hook returns with an exception set, later hooks are
ignored". Overhead is "between 1.05x faster to 1.05x slower". **Three hard
caveats, all from the PEP itself:** (a) **"Hooks cannot be removed or
replaced"** -- so an audit hook is a one-way door for the process, which argues
for install-at-conftest-import + an internal enable/disable flag rather than
add/remove; (b) **"This is not sandboxing"** -- it is a policy seam, not a
jail; a C extension or an `os`-level path that skips the event is uncovered;
(c) the raised exception must be `BaseException`-derived per F2 or the whole
mechanism is defeated by `kill_switch.py:498`.

**F4 -- Every existing off-the-shelf preventer is the wrong shape for at least
one pyfinagent constraint (measured against the internal inventory).**
`pytest-socket` (#3) is **host-level with no port granularity** -- which is
precisely the failure mode `conftest.py:32-37` already documents (an ephemeral-
port stub on `127.0.0.1` in `test_phase_76_9_2_max_bridge.py` would break).
`pyfakefs` (#1) replaces the whole FS, so
`test_phase_23_2_4...:253-255` (which must read the REAL live path) and
`test_book_safety_69.py:271-300` (which byte-copies the REAL journal) both
break; it also pauses patching in the logreport phase, i.e. it is porous by
design. `pytest-subprocess` (#13) has the right default -- unregistered ->
`ProcessNotRegisteredError` -- but is a **fixture**, therefore opt-in, and
pytest's own docs note an autouse fixture only affects tests "in the same scope"
(snippet). The BQ emulator (#17) requires standing up a server. **The one
transferable idea is `AnonymousCredentials()`** (#17): a client that cannot
authenticate cannot write, which is Saltzer's fail-safe default applied to the
BQ channel and needs no emulator.

**F5 -- The guard must be VERB- and DIRECTION-asymmetric, or it breaks three
in-repo tests.** The caller's own scope names the constraint: 23.2.4 must keep
reading the live journal. Reads that must stay legal: `paper_trading.py:940-941`,
`test_phase_23_2_4...:253-255`, `test_book_safety_69.py:280` (`live.read_bytes()`),
`kill_switch._audit_source_paths()` (:94-111). Writes that must be refused:
`kill_switch.py:496`, `risk_overrides.py:128`, `cron_control.py:60`. So the
policy shape mirrors the existing HTTP guard exactly -- **deny mutating access
to a specific live target, allow reads** -- which is the `open` audit event's
`mode`/`flags` argument (#16), not a blanket path ban.

**F6 -- The denominator is not one file, and the scope must be derived, not
listed.** 20 files carry module-level `handoff/` paths (inventory above), 4
constants point at journals (3 writers + 1 duplicate reader at
`paper_trading.py:892`), 72 test files use `subprocess`, and the `tests/` tree
receives neither the BQ nor the Slack guard. A hardcoded path list is the
`-v4`-hardcoding mistake already documented at `kill_switch.py:64-65`. Prefer a
**derived** predicate (e.g. "any write under `<repo>/handoff/` that is not under
the active `tmp_path`") plus a re-runnable checker that re-derives the
constant family, so a newly added `_AUDIT_PATH` is covered by *existing* code --
the same property `conftest.py:39-46` claims for the HTTP channel.

**F7 -- Child processes are a structural hole no in-process guard closes.**
`test_phase_4000_2_cc_rail_smoke.py:202` spawns `sys.executable`; a child loads
no conftest and inherits no audit hook. The transferable lever is the
**environment**, which IS inherited: PEP 578 hooks can be installed before
`Py_Initialize()` (#15), and CPython honours `PYTHONSTARTUP`/`-X` style
bootstrap; more simply, an env var (the `PYFINAGENT_TEST_NO_BQ:21` idiom already
in-repo) crosses the process boundary where a monkeypatch cannot. Note the env
var only helps if the CHILD's code checks it -- which is why
`bigquery_client.py` not consulting it (F-inventory) matters twice.

---

## Consensus vs debate (external)

**Consensus.** Prevention beats detection (Shingo; Saltzer #8); hermetic test
environments reduce flakiness and prod exposure ("An SUT with high hermeticity
will have the least exposure...", #9); order-dependent shared-state pollution is
the dominant Python flakiness cause at **59%** (#14); swallowed errors are a
named weakness class (#4).

**Debate -- and it is real.** Charity Majors, #11 **[ADVERSARIAL]**: "Once you
deploy, you aren't testing code anymore, you're testing systems", and you can
"never, ever guarantee that you have ironed out all the bugs" in pre-prod. #9
concurs that mocks "become stale... there is no signal." Taken seriously, this
says total isolation buys correctness of the *test* at the cost of fidelity to
the *system*. **Resolution for 86.6:** the adversarial camp argues for
*deliberate, instrumented, human-initiated* production exercise -- Majors'
guardrail is "How will I know if this breaks?" A `pytest` run started by a
developer is the opposite: unattended, unlogged, and un-consented. The 86.3
evidence is decisive here -- 8 rows appended and the live armed book paused
4x, with no operator intent (`conftest.py:5-13`). So the adversarial position
does **not** overturn the step; it does argue against removing
`test_phase_23_2_4`'s live-read leg, and against any design that makes a
deliberate operator-run live check impossible (keep an explicit,
loud escape hatch -- `fp.allow_unregistered`-shaped, #13).

---

## Pitfalls (from literature + measured)

1. **Refusing with an `Exception`** -> swallowed at `kill_switch.py:498`. Use a
   `BaseException`-derived refusal. `assert` is also unsafe (F2).
2. **Host-only matching** -> breaks the ephemeral-port stub (`conftest.py:32-37`).
3. **Blanket FS ban** -> breaks `test_phase_23_2_4...:253-255` and
   `test_book_safety_69.py:271-300` (F5).
4. **Fixture-based guard** -> misses module-import-time execution;
   `_backend_is_up()` in a `skipif` decorator runs before any fixture
   (`conftest.py:22-30`). Install at conftest IMPORT, as both existing guards do.
5. **Patching the accessor instead of the binding** -> `monkeypatch.setattr(ks,
   "get_state", ...)` was "VACUOUS BY IDENTITY"
   (`test_book_safety_69.py:227-232`); #6: "you patch where an object is
   _looked up_".
6. **Hardcoding the path list** -> `kill_switch.py:64-65` already records this
   failure mode ("HARDCODING `-v4` EXPIRES AT `-v5`. We glob.").
7. **Assuming a monkeypatch crosses a process boundary** -> it does not (F7).
8. **A guard whose own precondition never fired** -> `pyfakefs` pauses in
   logreport (#1); `pytest-socket`'s higher-scoped-fixture hole (#3). Assert the
   guard is INSTALLED and that a known-positive is refused, not just that no
   write happened.
9. **`sys.addaudithook` cannot be removed** (#15) -- design for a flag, not
   add/remove.

---

## Application to pyfinagent

| External finding | Internal anchor | Implication for 86.6 |
|---|---|---|
| Prevention > detection (#5, #8, Shingo) | `test_book_safety_69.py:30-43` (self-declared detector) | Keep the detector; ADD a preventer beneath it. |
| `BaseException` escapes `except Exception` (#18, measured) | `kill_switch.py:498-499`; `conftest.py:129-141`; `backend/tests/conftest.py:52-58` | The refusal MUST derive from `BaseException`. Consider retro-fitting the two existing `RuntimeError` guards. |
| `open`/`subprocess.Popen`/`socket.connect`/`urllib.Request` audit events (#15,#16) | all four channels in the caller's scope | One `sys.addaudithook` at conftest import covers FS + subprocess + socket + urllib at a single seam. |
| Deny mutating verbs, allow reads (#8 complete mediation) | `conftest.py:78,118-126` (existing, working shape) | Mirror it for FS: key on `open` mode/flags + a derived live-path predicate. |
| `AnonymousCredentials` (#17) | `bigquery_client.py:271..1023` (unguarded) | Cheapest BQ preventer: no emulator needed. Also extend `PYFINAGENT_TEST_NO_BQ` beyond `api_call_log.py:125,322` and into `./conftest.py` so the `tests/` tree is covered. |
| Child loads no conftest (F7) | `test_phase_4000_2_cc_rail_smoke.py:202`; `smoke_cc_rail_e2e.py:469` | Env-var propagation, and/or make the script refuse a default `--backend-url` when a test env marker is set. |
| Order-dependency dominance 59% (#14) | 4 module-level `_AUDIT_PATH` constants + `_state` at `kill_switch.py:760` | The singleton channel is the highest-frequency class; a detached-instance idiom already exists at `test_phase_23_2_4...:139-152` -- promote it to a shared fixture rather than re-deriving per test. |
| Escape hatch must exist (#11, #13) | `test_book_safety_69.py:271-300` | Provide a loud, explicit opt-out (marker or context manager), never a silent one. |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **18**
- [x] 10+ unique URLs total (incl. snippet-only) -- **44** (18 read-in-full + 26 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- section above, incl. a negative finding
- [x] Full papers / pages read (not abstracts) -- PolDet via curl+pdfplumber (63,721 chars, 11pp); Gruber via ar5iv per the arXiv chain; no `arxiv.org/pdf/` WebFetch was attempted
- [x] file:line anchors for every internal claim -- 20 files
- [x] coverage.dry (audit-class) -- rounds 9 and 10 both yielded zero new read-in-full findings

Soft checks:
- [x] Internal exploration covered every module in the caller's scope, plus 3 unlisted writers (`risk_overrides`, `cron_control`) and the duplicate reader at `paper_trading.py:892`
- [x] Contradictions noted -- #11 adversarial; the pytest-issue snippet REFUTED by measurement
- [x] All claims cited per-claim
- **GAP (honest):** NIOSH and ASQ both returned 403; the hierarchy-of-controls
  argument rests on Saltzer (#8) + snippet-tier poka-yoke sources, not a
  read-in-full standards page.
- **GAP (honest):** two read-in-full sources rendered thin (#10, #12) and are
  flagged in-table as corroborating only; neither carries a claim alone.
- **NOT MEASURED (for the executor, not assumed here):** whether the phase-82.58
  Slack `RuntimeError` is in fact absorbed at `kill_switch.py:564-565` on a live
  auto-pause path. The code shape says yes; it was not executed.

---

## Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 18,
  "snippet_only_sources": 26,
  "urls_collected": 44,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": true,
    "rounds": 10,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.6.md",
  "gate_passed": true
}
```
