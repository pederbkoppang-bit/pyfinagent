# Research Brief — phase-86.3: stop the pytest suite mutating the LIVE kill switch

**Tier:** moderate (caller-specified). Not audit-class.
**Date:** 2026-08-09
**Deliverable for:** masterplan step 86.3 (inherits two criteria from 36.21)

## Question

How do mature Python/pytest codebases prevent a test suite from reaching a real
network host and mutating production state — and specifically, how to build a
SESSION-SCOPED, opt-out-proof guard that protects a NEW test module by default,
while preserving the original value of
`backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` (proving
pause/resume does not deadlock — the re-entrant-lock bug fixed in 0ed72940).

## Hard safety constraint honoured

No `pytest` invocation on `backend/tests/`, no execution of the offending
module, no POST to localhost:8000, no HTTP request of any kind issued by this
session. Every internal claim below comes from `Read` / `grep` / `sed` /
`git log` on the working tree.

---

## HEADLINE FINDING (read before anything else)

**The exact interception seam the fix needs ALREADY EXISTS, already sits in the
offending module's call path, and is already installed at the right lifecycle
point.** `backend/tests/conftest.py:45-61` monkeypatches
`urllib.request.urlopen` at conftest **import** time (not in a fixture) — but
its predicate is a single-host DENYLIST, so `http://localhost:8000/...` sails
straight through:

```python
# backend/tests/conftest.py:47-61
_REAL_URLOPEN = urllib.request.urlopen

def _no_slack_egress(req, *args, **kwargs):
    url = getattr(req, "full_url", None) or str(req)
    if "slack.com" in url:
        raise RuntimeError(
            "phase-82.58 test guard: refusing a live Slack POST from the test "
            f"suite (url={url!r}). ..."
        )
    return _REAL_URLOPEN(req, *args, **kwargs)

urllib.request.urlopen = _no_slack_egress
```

The offending module resolves `urlopen` through a **module-attribute lookup at
call time** (`import urllib.request` inside each helper body, then
`urllib.request.urlopen(...)` — lines 48-51, 59-61, 67-77), so the conftest
patch IS in its path. The 82.58 author wrote the limitation down verbatim:
*"Scoped to slack.com ONLY -- this is not a general network jail, and tests that
legitimately reach other hosts are unaffected."* 86.3 is exactly the request to
widen that predicate from "one bad host" to "mutating verbs against the live
backend", and to relocate it where a new module cannot dodge it.

---

## Read in full (8; gate floor is 5 — counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://raw.githubusercontent.com/miketheman/pytest-socket/main/README.md | 2026-08-09 | official docs (plugin README) | WebFetch (raw markdown) | "Disables all network calls flowing through Python's `socket` interface, **including DNS resolution**." Options: `--disable-socket`, `--allow-hosts=127.0.0.1,…` (IPs + CIDR), `--allow-unix-socket`, `--force-enable-socket` ("takes precedence over `--disable-socket`"). Per-test opt-back-in: `@pytest.mark.enable_socket`, the `socket_enabled` fixture, `@pytest.mark.allow_hosts([...])`. Programmatic install: `disable_socket()` inside `pytest_runtest_setup()` in `conftest.py`. **Documented scope caveat:** "If you create another fixture that creates a socket usage that has a 'higher' instantiation order, such as at the module/class/session, then the higher order fixture will be resolved first, and won't be disabled during the tests." |
| https://docs.pytest.org/en/stable/how-to/fixtures.html | 2026-08-09 | official docs | WebFetch | Autouse: "'Autouse' fixtures are a convenient way to make all tests automatically **request** them" — "both tests are affected by it, even though neither test **requested** it." conftest: "puts the fixture function into a separate `conftest.py` file so that tests from multiple test modules in the directory can access the fixture function"; available to that directory **and its subdirectories** with no import. Scopes function/class/module/package/session; "Higher-scoped fixtures are instantiated first"; teardown "in the _reverse order_". `yield` = setup-before / teardown-after. |
| https://pytest-test-categories.readthedocs.io/en/latest/architecture/adr-001-network-isolation.html | 2026-08-09 | official docs (ADR) | WebFetch | Chooses socket patching: "**Socket-level patching is the most reliable interception point**", applied before DNS resolution "preventing bypass attempts". Tiered policy: small = "all network blocked", medium = "localhost-only", large = permitted; allowlist schema explicitly carries "localhost", "127.0.0.1", "::1". **On opt-out-proofing:** "This plugin intentionally provides **no per-test override markers** (e.g., `@pytest.mark.allow_network`). This is a deliberate architectural decision, not a missing feature." / "Small tests must be hermetic. Period. No escape hatches." Rejected alternatives: import-hook blocking ("too inflexible"), context-manager wrapping (doesn't integrate with pytest), network namespaces ("platform-specific and heavy-weight"), proxy blocking ("too complex"). |
| https://blog.jerrycodes.com/no-http-requests/ | 2026-08-09 | authoritative blog | WebFetch | The canonical library-level recipe — an **autouse conftest fixture**, not socket blocking: `@pytest.fixture(autouse=True) def no_http_requests(monkeypatch): … monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen_mock)`. Rationale: "The fixture relies on the fact that all HTTP requests eventually go through `urllib3.connectionpool.HTTPConnectionPool.urlopen`." Opt-back-in via `@responses.activate`, or "the fixture can be extended to allow specific hosts via an `allowed_hosts` set". **Load-bearing for us: `urllib3` is the `requests`/`httpx` layer — stdlib `urllib.request` does NOT go through it, so this exact recipe would MISS the offending module.** |
| https://lundberg.github.io/respx/ | 2026-08-09 | official docs | WebFetch | "a simple, _yet powerful_, utility for mocking out the **HTTPX**, _and **HTTP Core**_, libraries." Requires "HTTPX 0.25+". No urllib/`http.client` interception is offered or mentioned. Confirms respx is the wrong layer for this defect. |
| https://fastapi.tiangolo.com/advanced/async-tests/ | 2026-08-09 | official docs | WebFetch | "The `TestClient` is based on [HTTPX] … we can use it directly to test the API." Async recipe: `async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac: response = await ac.get("/")`. "The `ASGITransport` pattern ensures requests go directly to your FastAPI application **without requiring a live server**." This is the mechanism that lets the deadlock assertion survive with no :8000. |
| https://pytest-test-categories.readthedocs.io/en/latest/examples/network-isolation.html | 2026-08-09 | official docs | WebFetch | Concrete shape of a tiered policy: markers declared in `[tool.pytest.ini_options]`, `test_categories_enforcement = "strict"` (modes strict / warn), tests needing a local server are marked `@pytest.mark.medium  # Medium tests can access localhost`, and a violation raises `HermeticityViolationError: Network access attempted / Attempted connection to: api.example.com:443`. Notably provides **no host:PORT granularity** — the allowlist is host-level. |
| https://www.sonarsource.com/resources/library/audit-logging/ | 2026-08-09 | industry/vendor engineering | WebFetch | Audit vs application logs: application logs are "often rotated or pruned"; audit logs "**Must be immutable and tamper‑evident**" and exist for "Security, compliance, accountability". "**Separating your audit logs from your general application logs is a critical first step.**" Integrity controls: "Write‑Once, Read‑Many (WORM) storage to prevent modification after write"; "Hashing and digital signatures (e.g., hash chains) to create tamper‑evident trails"; "Non-repudiation is the assurance that someone cannot deny the validity of something." **Explicitly does NOT address separating test activity from the production trail** — see "Gap in the literature" below. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://monkey.work/blog/2025-12-23-block-network-access-in-tests/ | blog (2025-12) | **Attempted, HTTP 404.** Recorded as an attempt, not a read. |
| https://github.com/miketheman/pytest-socket | repo | README read via `raw.githubusercontent.com` instead |
| https://github.com/miketheman/pytest-socket/discussions/355 | community | "Using `disable_socket` (in conftest) then overriding `allow_hosts` in a specific test" — the opt-back-in pattern; community tier |
| https://github.com/miketheman/pytest-socket/issues/13 | community | "Unexpected interaction between `--disable-socket` and `--allow-hosts`" — evidence the two flags compose non-obviously |
| https://pypi.org/project/pytest-socket/ , /0.5.0/ , /0.3.2/ | registry | version metadata only |
| https://github.com/best-doctor/pytest_network | repo | alternative "disable network on socket level" plugin; same host-granularity ceiling |
| https://github.com/mbachry/pytest-socket | repo | fork |
| https://www.tonylykke.com/posts/2018/07/31/disabling-the-internet-for-pytest/ | blog (2018) | **year-less canonical prior art** — the founding "disable the internet for pytest" write-up |
| https://tomwojcik.com/posts/2021-08-18/globally-block-internet-connection-in-django/ | blog (2021) | "Two ways to block internet connection in Python tests" — httpretty `allow_net_connect=False` + monkeypatch |
| https://pypi.org/project/pytest-recording/ (0.3.4, 0.3.6) | registry/plugin | `pytest.mark.block_network` + `--block-network`; VCR.py-based |
| https://pytest-django.readthedocs.io/en/latest/database.html | official docs | pytest-django's `django_db` opt-IN model — the inverse of what 36.21 demands (opt-in ⇒ dodgeable by omission) |
| https://laravel-news.com/prevent-stray-requests | industry (cross-domain) | Laravel `preventStrayRequests()` — same "fail loudly on an unfaked outbound call" pattern in PHP |
| https://mswjs.io/docs/best-practices/avoid-request-assertions/ | official docs (cross-domain) | MSW; JS ecosystem analogue |
| https://dev.to/sirech/fail-a-test-in-jest-if-an-unexpected-network-request-happens-mmj | community (cross-domain) | Jest equivalent |
| https://qaskills.sh/blog/pytest-fixtures-conftest-complete-guide-2026 | blog (2026) | recency-scan hit; conftest/fixture guide, nothing new beyond the pytest docs |
| https://dev.to/aleksei_aleinikov/pytest-in-2025-the-only-testing-guide-youll-ever-need-lb8 | community (2025) | recency-scan hit |
| https://oneuptime.com/blog/post/2026-02-08-how-to-create-an-isolated-docker-network-for-testing/view | blog (2026-02) | Docker `--internal` network isolation — infrastructure-tier alternative; rejected as heavyweight for a single-Mac local deployment |
| https://realpython.com/pytest-python-testing/ | tutorial | `disable_network_calls()` + `autouse=True` in conftest as the "global safety net" idiom |
| https://news.lavx.hu/article/audit-logging-deep-dive-engineering-tamper-proof-trails-for-security-and-compliance | blog | audit-trail tamper-evidence; corroborates Sonar |
| https://visuresolutions.com/alm-guide/software-development-process-for-safety-critical-systems/ | industry | safety-critical SDLC overview |
| https://ldra.com/testing-auto-generated-code-for-safety-critical-systems/ | industry | test-evidence auditability in certified systems |

**URLs collected: 30 (8 read in full, 22 snippet-only/attempted).** Floor is 10.

### Search-query variants run (three-variant discipline)

1. **Current-year (2026):** `pytest block network access conftest 2026 safety production database`
2. **Last-2-year (2025):** `pytest-socket 2025 release network isolation tests best practice`
3. **Year-less canonical (×3):** `pytest-socket disable_socket socket_allow_hosts session autouse fixture`; `prevent tests from making real HTTP requests to production accidentally test isolation`; `test code writing to production audit log safety-critical system integrity`

The year-less pass is what surfaced the 2018 Tony Lykke and 2021 Tom Wojcik prior
art and the `urllib3`-layer recipe — none of which appeared in the year-locked
passes.

## Recency scan (2024-2026)

Performed. **Result: no 2024-2026 finding supersedes the canonical mechanisms;
one 2025-2026 source materially strengthens the opt-out-proofing argument, and
one confirms an important negative.**

- **New and load-bearing:** `pytest-test-categories` ADR-001 + its
  network-isolation examples page (readthedocs, current) is the most recent
  articulation of the *opt-out-proof* property 36.21 demands, and it states the
  design position explicitly — "no per-test override markers … a deliberate
  architectural decision, not a missing feature" / "No escape hatches." That is
  newer and sharper than the 2018-2021 prior art, which all assumed a per-test
  opt-back-in.
- **Confirmed negative (important):** the 2024-2026 window produced **no
  mechanism with HTTP-verb granularity**. Every tool found — pytest-socket,
  pytest_network, pytest-recording, the ADR plugin, httpretty, responses, respx,
  msw, Laravel `preventStrayRequests` — gates on *host* (or on *library*), never
  on *method*. This is decisive for 86.3, because the required policy is
  "GET localhost:8000 yes, POST localhost:8000 no" (see Option B below).
- **No change to pytest itself** in the relevant area: autouse/conftest/scope
  semantics in the current stable docs are unchanged from the behaviour the
  repo's existing guards already rely on.
- Adjacent-domain 2026 material (Docker `--internal` test networks, MSW
  best-practices) converges on the same principle — isolate by default, make the
  escape explicit — but adds no Python-specific mechanism.

### Gap in the literature (a finding, not a failure)

The safety-critical / audit-logging literature (Sonar, LavX, LDRA, Visure) is
unanimous that an audit trail "must be immutable and tamper-evident" and that
separation is "a critical first step" — but **every source frames the threat as
a malicious or buggy *production* actor. None of the eight sources read, and
none of the snippet-only industry sources, addresses the *test suite itself* as
a writer to the production audit journal.** The nearest transferable rule is
Sonar's environment/stream separation principle applied one level out: the test
environment must not share the production journal's file handle at all. The
pyfinagent-local precedent (`test_phase_36_12_…:40-57`) is, on this evidence,
ahead of the published guidance rather than behind it.

## Key findings

1. **Socket-level is the most reliable interception point, but it is
   host-granular.** ADR-001: "Socket-level patching is the most reliable
   interception point", and it fires "before DNS resolution, preventing bypass
   attempts". pytest-socket confirms: it disables "all network calls flowing
   through Python's `socket` interface, including DNS resolution"
   (https://raw.githubusercontent.com/miketheman/pytest-socket/main/README.md).
   Neither offers method/verb granularity — allowlists are `--allow-hosts` /
   `["localhost","127.0.0.1","::1"]`.

2. **Library-level guards are layer-specific, and the popular recipes target the
   WRONG layer for this defect.** The canonical autouse recipe patches
   `urllib3.connectionpool.HTTPConnectionPool.urlopen` because "all HTTP requests
   eventually go through" it (blog.jerrycodes.com) — true for `requests` and
   `httpx`, false for stdlib `urllib.request`, which goes down `http.client`
   directly. `respx` is "for mocking out the HTTPX, and HTTP Core, libraries"
   only (lundberg.github.io/respx). `responses` / `requests-mock` patch the
   `requests` transport adapter. **The only interception layers that catch
   `urllib.request.urlopen` are (a) the socket layer, or (b) patching
   `urllib.request.urlopen` itself** — which is precisely what
   `backend/tests/conftest.py:61` already does.

3. **`conftest.py` + autouse is the documented opt-out-proof primitive.** Autouse
   fixtures make "all tests automatically **request** them" — "both tests are
   affected by it, even though neither test **requested** it" — and a conftest
   fixture is visible to "tests from multiple test modules in the directory" and
   its subdirectories with no import (docs.pytest.org). A new module inherits it
   by existing in the tree; there is nothing to forget.

4. **Autouse fixtures do not cover collection time; import-time patches do.**
   pytest-socket's own README records the sibling hazard: a higher-scoped fixture
   "will be resolved first, and won't be disabled during the tests". The
   pyfinagent conftest already documents the same reasoning for choosing import
   time over a fixture: *"Setting the guard at conftest IMPORT time (not in a
   fixture) means it is active before test collection imports any module"*
   (`backend/tests/conftest.py:11-13`). **This matters concretely here:**
   `test_phase_23_2_4_…:112` and `:165` evaluate `_backend_is_up()` inside a
   `@pytest.mark.skipif(...)` decorator, i.e. at module import during
   collection — before any fixture of any scope can run.

5. **The original value survives without a server.** FastAPI's own docs:
   `TestClient` "is based on HTTPX" and the `ASGITransport(app=app)` pattern
   "ensures requests go directly to your FastAPI application without requiring a
   live server". A deadlock is an in-process property of the lock; it reproduces
   in-process. This is already idiomatic in this repo (see inventory).

6. **Opt-out-proof means no escape hatch, by design.** ADR-001 is explicit that
   omitting per-test override markers is "a deliberate architectural decision,
   not a missing feature" — "Small tests must be hermetic. Period." That is the
   published argument for 36.21's inherited criterion.

7. **Audit journals are held to a higher bar than logs, and the literature has a
   blind spot.** "Must be immutable and tamper-evident"; "Separating your audit
   logs from your general application logs is a critical first step"
   (sonarsource.com). No source read frames the test suite as a writer — see
   "Gap in the literature".

## Consensus vs debate (external)

- **Consensus:** block by default at the *lowest practical layer*; make the
  legitimate exception explicit and narrow; fail loudly rather than silently
  no-op; put the switch in `conftest.py` so coverage is structural. Every source
  from 2018 through 2026 agrees.
- **Genuine debate:** *where* the escape hatch lives. pytest-socket, httpretty
  and pytest-recording all ship per-test opt-back-in markers
  (`@pytest.mark.enable_socket`, `@pytest.mark.allow_hosts`,
  `--force-enable-socket`); ADR-001 takes the opposite position and ships none.
  36.21's inherited criterion sides with ADR-001 for the *default*, which argues
  for making the pyfinagent exception a deliberate, greppable, narrow constant
  rather than a marker any module can sprinkle on itself.
- **Second debate:** socket layer vs library layer. ADR-001 and pytest-socket say
  socket ("most reliable"); the blog tradition says library (cheaper, no plugin,
  keeps higher-level mocking usable). For *this* defect the library layer is only
  viable if the library chosen is `urllib.request` itself.

## Pitfalls (from literature + measured here)

- **Wrong-layer guard** — patching `urllib3` (the most-cited recipe) would leave
  this exact defect wide open. Verify the layer against the module's actual call.
- **Host-granular allowlist is a false floor** — `--allow-hosts=127.0.0.1` is
  required by the GET probes *and* re-permits the mutating POST. Same string,
  both effects.
- **Blocking `127.0.0.1` wholesale breaks legitimate tests** —
  `test_phase_76_9_2_max_bridge.py` binds its own `ThreadingHTTPServer` on
  `127.0.0.1:<ephemeral>` and POSTs to it (measured, see inventory). A host-level
  rule is too coarse; the rule must be host **and port**.
- **A blocked socket FAILS a test, it does not skip it** — pytest-socket raises
  `SocketBlockedError` from inside the test body, turning the live-cycle test
  RED. The 86.3 criteria forbid that outcome for
  `test_phase_23_2_4_audit_log_clean_transitions`.
- **Fixture-scope ordering** — the README caveat: a module/session fixture that
  itself opens a socket resolves before the function-scoped `socket_enabled`.
- **Detector ≠ preventer** — a byte-compare at teardown (36.12's pattern) fires
  *after* the live file has already been mutated. It converts a silent
  corruption into a loud one; it does not stop it.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/tests/conftest.py` | 62 (whole file) | **The only conftest.py in either Python test tree.** Two import-time guards: `os.environ.setdefault("PYFINAGENT_TEST_NO_BQ","1")` (:21, phase-61.2) and the `urllib.request.urlopen` slack.com denylist (:45-61, phase-82.58) | **THE FIX SEAM.** Denylist of one host; `localhost:8000` passes. Lives under `backend/tests/` only |
| `backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` | 200 | The defect. `_backend_is_up()` :46-54, `_get_paused_state()` :57-61, `_post_state_transition()` :64-85; 4 tests, two gated by `@pytest.mark.skipif(not _backend_is_up(), …)` at :112 and :165 | **MUTATING.** POST pause :125, POST resume :137, POST pause :149, conditional 4th restore POST resume :158 |
| `pytest.ini` (repo root) | 10 | `[pytest] markers = requires_live: …` and nothing else | **No `addopts`, no `testpaths`, no plugin list.** Its presence at the repo root is what makes rootdir = repo root, so a root `conftest.py` would be collected for BOTH trees |
| `tests/` (repo root) | — | Second Python test tree: `tests/services/`, `tests/api/`, `tests/slack_bot/`, `tests/scheduler/`, … + 100+ `verify_phase_*.py`. 32 files reference `localhost:8000`/`127.0.0.1:8000` | **`find tests -name conftest.py` ⇒ 0.** A `backend/tests/conftest.py` guard has ZERO reach here |
| `backend/tests/test_phase_36_12_kill_switch_trading_path_block.py` | :40-57 (`_live_audit_file_is_write_protected`), :60-89 (`captured_alerts`) | Prior art #1 | see analysis below |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | :115+ (`ks_tmp_audit`) | Prior art #2 | see analysis below |
| `backend/tests/test_phase_76_9_2_max_bridge.py` | :130 `s.bind(("127.0.0.1", 0))`, :137-147, :161-265 | **The ephemeral-port prior art the caller asked about — and a live-host module the caller's list MISSED.** Spawns its own `ThreadingHTTPServer` on a free port and POSTs to it | Legitimate. **A host-level `127.0.0.1` block breaks this module** |
| `backend/tests/test_phase_80_2_error_response_contract.py` | :212-214, :233, :256 | `fastapi.testclient.TestClient(app, raise_server_exceptions=False)` + `starlette.testclient.TestClient` | **In-repo model for criterion 4** — no live server, no network |
| `backend/tests/auth_helper.py`, `test_paper_trading_v2.py`, `test_phase_80_40_perf_metrics_drawdown.py`, `test_phase_82_10_freshness_paging.py`, `backend/tests/api/test_sovereign.py`, `tests/api/test_cron_dashboard.py`, `tests/api/test_observability.py` | — | Already use `TestClient`/`ASGITransport` | The in-process pattern is established repo idiom, not a novelty |
| `backend/requirements.txt` | :60-61 | `pytest==9.0.3`, `pytest-cov==7.1.0` | **`pytest-socket` is NOT installed and NOT in any requirements file** (grep across `*.txt`/`*.toml`/`*.cfg`/`*.ini`/`*.py` ⇒ 0 hits; `.venv/…/pytest_socket*` ⇒ no matches). Adding it means a new prod-requirements row |
| repo root | — | `pyproject.toml`, `setup.cfg`, `tox.ini` | **None exist.** `pytest.ini` is the only pytest config surface |

### Prior-art fixture #1 — `_live_audit_file_is_write_protected` (36.12:40-57)

```python
@pytest.fixture(autouse=True)
def _live_audit_file_is_write_protected():
    live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (...)
```

**Protects:** any byte change to the live audit journal caused by a test *in that
one module*, and it fails loudly — deliberately, because "`_append_audit` swallows
write errors, so a raise inside the code under test could not surface -- the byte
comparison is the only reliable detector" (docstring :44-48).

**Does NOT protect:** (a) any other module — it is module-local, autouse only
within its own file, which is exactly 36.21's complaint; (b) it is a **detector,
not a preventer** — the live file is already mutated when the assert fires, and
the fixture restores nothing; (c) function-scoped, so a write during collection
or during a module/session fixture is outside its window; (d) non-file live
state — the actual paused flag, BQ rows, Slack posts, the scheduler; (e) it is
blind to *which* line wrote, only that the bytes differ.

The docstring already names the gap 86.3 must close: *"The autouse fixture below
is ported verbatim in intent from `test_phase_36_7_…` -- without it, that brace
would exist only in the 36.7 module"* — i.e. the pattern has been **copy-pasted
between modules twice**, which is the definition of a per-module opt-in.

### Prior-art fixture #2 — `ks_tmp_audit` (36.7)

```python
@pytest.fixture
def ks_tmp_audit(tmp_path, monkeypatch):
    """Point kill_switch's audit I/O at a tmp tree."""
```

**Protects:** it is a true **preventer** — monkeypatches the kill-switch module's
`_AUDIT_PATH` (and siblings) into `tmp_path`, so the write never reaches the live
file.

**Does NOT protect:** it is **not autouse and not session-scoped** — a test only
gets it by naming it in its signature. A new test (or a new module) that forgets
to request it writes to the live journal with no complaint. That is precisely the
"dodge by omission" hole 36.21's first inherited criterion forbids. It also only
covers the *direct Python* write path (`_append_audit`); it does nothing about an
HTTP POST to a *separate live backend process*, which is how 23.2.4 causes the
damage — the live backend's own `_AUDIT_PATH` is unaffected by an in-test
monkeypatch.

**Combined gap:** #1 detects file damage after the fact, per-module. #2 prevents
file damage, per-test-opt-in, in-process only. **Neither can stop an HTTP POST to
a separate live server, and neither is inherited by a new module.**

### Live-host classification (the caller's list, corrected)

**Rule used** (stated, per `feedback_measure_dont_assert_claims`): a module
"reaches a live host" iff it contains an **executed call site** to a real network
client — enumerated as `grep -rn "urlopen(" backend/tests/*.py` plus
`requests.*` / `httpx.*` over a real transport — whose target host:port is
resolved at runtime. A URL that appears only in a docstring, as a string argument
to a pure function, or as the subject of AST/regex analysis does **not** count.
Then **MUTATING** iff any such call site uses a non-idempotent verb
(POST/PUT/DELETE/PATCH); **READ-ONLY** if all are GET/HEAD.

| Module (caller's list + corrections) | Reaches live host? | Target | Verbs | Class |
|---|---|---|---|---|
| `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | YES | `localhost:8000` | GET `/api/health` :51, GET `/api/paper-trading/kill-switch` :60, **POST `/api/paper-trading/pause` ×2, POST `/api/paper-trading/resume` ×1-2** :77 | **MUTATING — the defect** |
| `test_phase_23_2_9_ticker_meta_latency.py` | YES | `localhost:8000` | GET `/api/health` :35, GET `/api/paper-trading/ticker-meta` :108/:126/:133 | READ-ONLY |
| `test_phase_23_2_13_governance_watcher.py` | YES | `localhost:8000` | GET `/api/health` :37, :148 | READ-ONLY |
| `test_phase_23_2_7_red_line_nav_match.py` | YES | `localhost:8000` | GET :55, :68 | READ-ONLY |
| `test_phase_76_9_2_max_bridge.py` **(MISSING from the caller's list)** | YES | `127.0.0.1:<ephemeral>` — its own `ThreadingHTTPServer`, port from `s.bind(("127.0.0.1", 0))` :130 | GET `/health` :147/:161, **POST `/v1/messages` :173, :187, :265** | **Mutating verbs, benign target.** Self-spawned stub, never production. **Constrains the fix: the rule must be port-aware** |
| `test_phase_36_7_kill_switch_rotation_rearm.py` | **NO** | — | `curl -s http://localhost:8000/api/paper-trading/kill-switch` appears at :10 **inside the module docstring** | caller's list FALSE POSITIVE |
| `test_phase_75_17_verification_paths.py` | **NO** | — | :303 `cmd = "curl -s http://localhost:8000/openapi.json \| jq ."` is a **string argument to `fp_reason(...)`**, a pure path-resolver under test; never executed | caller's list FALSE POSITIVE |
| `test_phase_75_deploy_surface.py` | **NO** | — | :236-249 AST-inspects `requests.get(...)` call nodes in *another file's source*; :419-425 asserts a CORS regex against the literal `"http://localhost:3000"` | caller's list FALSE POSITIVE |
| `test_phase_80_2_error_response_contract.py` | **NO** | — | in-process `TestClient(app)` :212-214 / `RawTestClient(wrapped)` :256 | caller's list FALSE POSITIVE — **and the model to copy** |

**Net:** the caller's list of 8 contains **4 false positives** and **omits 1**
live-host module. Of the 5 modules that genuinely reach a host, exactly **one
(`23_2_4`) mutates production**; one (`76_9_2`) mutates only its own stub; three
are read-only GETs against `:8000`.

**Scope consequence:** the fix does not need to touch the three read-only GET
modules at all, and it **must not** break `76_9_2`. The blast radius of a
correctly-scoped fix is one module plus a conftest.

### Two collection-time facts the fix must respect

1. **`_backend_is_up()` runs during collection, not during a test.** It is the
   argument to `@pytest.mark.skipif(...)` at :112 and :165, evaluated when the
   module is imported. No fixture of any scope — session included — runs early
   enough to intercept it. Only the conftest's *import-time* patch is ahead of it.
   (The repo's own conftest asserts this ordering at :11-13; I did not execute
   pytest to re-derive it.)
2. **`_backend_is_up()` catches only `(urllib.error.URLError, OSError,
   TimeoutError)`** (:53). A guard that raises `RuntimeError` — the shape the
   existing 82.58 guard uses — on the **GET probe** path would escape the `except`,
   propagate out of the decorator expression, and turn the whole module into a
   **collection error**. `test_phase_23_2_4_audit_log_clean_transitions` would
   then not run at all, breaking the second inherited 36.21 criterion. **The
   guard must therefore let GETs through and block only the mutating verbs**, or
   raise a type inside that tuple. Blocking GETs is not required by the defect
   anyway: the measured harm is 8 rows appended and 4 pauses, all from POSTs.

### What `test_phase_23_2_4_audit_log_clean_transitions` actually depends on

(:166-199, the test whose trigger allowlist must stay byte-unchanged.) It needs:
(a) the module to collect; (b) `_backend_is_up()` truthy, else it skips; (c)
`handoff/kill_switch_audit.jsonl` to exist with ≥3 parseable rows; (d) each of
the last 10 rows to carry `ts`+`event`, with `event ∈ {"pause","resume",
"sod_snapshot","peak_update","cleanup"}` (:187) and, on pause/resume rows,
`trigger ∈ {"manual","auto","test"}` (:195). It does **not** depend on the live
cycle above it having run — the live file already satisfies (c)/(d). So a fix
that stops the POSTs but keeps collection and the GET probe working leaves this
test green with both allowlists untouched.

---

## Application to pyfinagent

| External finding | pyfinagent anchor |
|---|---|
| Only the socket layer or `urllib.request` itself catches `urlopen` (finding 2) | `backend/tests/conftest.py:47-61` already patches `urllib.request.urlopen`; `backend/tests/test_phase_23_2_4_…:51,60,77` call it by module attribute ⇒ the patch is in-path |
| conftest + autouse = protected-by-existence (finding 3) | No `conftest.py` at repo root; `find tests -name conftest.py` ⇒ 0; `pytest.ini` at root fixes rootdir there |
| Autouse cannot cover collection; import-time can (finding 4) | `test_phase_23_2_4_…:112,:165` `skipif` fires at import; `conftest.py:11-13` documents the same reasoning |
| Host-granular allowlists are too coarse (finding 1, pitfall 3) | `test_phase_76_9_2_max_bridge.py:130` binds `127.0.0.1:0` and POSTs to it |
| No published tool has verb granularity (recency scan) | The required policy is GET-yes/POST-no on one host:port ⇒ must be hand-rolled |
| In-process ASGI keeps the assertion (finding 5) | `test_phase_80_2_error_response_contract.py:212-256`, `backend/tests/auth_helper.py` already do it |
| Audit journals need separation (finding 7) | `handoff/kill_switch_audit.jsonl`; `test_phase_36_12_…:40-57` already enforces it for one module |

---

## Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **8**
- [x] 10+ unique URLs total (incl. snippet-only) — **30**
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (both test trees, both
      requirements files, every pytest config surface, all 9 classified modules)
- [x] Contradictions / consensus noted (socket-vs-library; escape-hatch debate)
- [x] All claims cited per-claim
- [ ] **Gap disclosed:** the audit-logging literature does not address test-suite
      writes to a production journal (see "Gap in the literature"). Not a
      shortfall in sourcing — a genuine absence, reported as a finding.

---

## Recommended fix direction

Three concrete options. **I recommend A + C together, with B as an optional
later hardening layer.**

### Option A — widen the existing conftest guard, and move it to a ROOT `conftest.py`

Turn `_no_slack_egress` into a general egress policy: resolve the request's
**method** (`req.get_method()` when it is a `urllib.request.Request`, else
`"GET"` for a bare URL string) and its **host:port**; refuse when the verb is
non-idempotent AND the target is the live backend (`localhost:8000` /
`127.0.0.1:8000` / `[::1]:8000`). Keep the slack.com rule. Install it at **import
time** in a NEW repo-root `conftest.py` so both `backend/tests/` and `tests/`
inherit it (rootdir is the repo root because `pytest.ini` lives there), and have
`backend/tests/conftest.py` stop duplicating it.

- **Pros:** reuses a seam *proven* to sit in the offending module's call path;
  import-time, so it is live before the `skipif` at :112; the only mechanism that
  can express verb granularity; port-aware, so `76_9_2` survives; covers the
  `tests/` tree that has no conftest today; zero new dependency; GET probes keep
  working, so the module still collects and
  `test_phase_23_2_4_audit_log_clean_transitions` passes with both allowlists
  byte-unchanged; a new module is protected by *existing*, not by opting in.
- **Cons:** covers `urllib.request` only — a future module using
  `requests`/`httpx`/raw `socket` would bypass it. Mitigate by adding the
  `urllib3.connectionpool.HTTPConnectionPool.urlopen` patch from
  blog.jerrycodes.com alongside it (same file, same import-time install), which
  closes `requests` + `httpx` for free.
- **Mutation-resistance requirement** (per `feedback_mutation_test_guards_and_fixtures`):
  ship a test that proves the guard FIRES — assert that a POST to
  `http://localhost:8000/api/paper-trading/pause` from inside the suite raises,
  and that a GET to the same host and a POST to `127.0.0.1:<other port>` do not.
  Without that, the guard is unfalsifiable.

### Option B — adopt `pytest-socket` (`addopts = --disable-socket --allow-hosts=127.0.0.1`)

- **Pros:** socket-level — ADR-001's "most reliable interception point"; catches
  `requests`, `httpx`, raw sockets and DNS uniformly; a new module is covered
  with no action; standard, maintained plugin.
- **Cons — decisive as a *primary* mechanism:** (1) **host-granular only** — the
  `--allow-hosts=127.0.0.1` needed by the GET probes and by `76_9_2` is the same
  string that re-permits the POST to `:8000`; there is no way to express
  "GET yes, POST no". (2) Omitting the allowlist blocks `76_9_2`'s legitimate
  ephemeral-port servers and turns the 23.2.4 live tests **RED** — a blocked
  socket raises inside the test body, it does not skip. (3) pytest-socket installs
  in `pytest_runtest_setup`, i.e. **after** collection, so the collection-time
  `_backend_is_up()` probe at :112 still fires for real regardless. (4) New row in
  `backend/requirements.txt` (a prod file — the only test deps today are `pytest`
  + `pytest-cov`).
- **Verdict:** genuinely valuable as **defence in depth** once A + C are in — it
  closes the non-urllib clients A cannot see. Not sufficient alone, and it cannot
  satisfy the two inherited 36.21 criteria on its own.

### Option C — rewrite the pause/resume cycle onto in-process `TestClient`

The test's real subject is a **re-entrant-lock deadlock inside the app process**,
which reproduces perfectly in-process. Drive pause → resume → pause through
`TestClient(app)` (or `AsyncClient(transport=ASGITransport(app=app))`) against a
`KillSwitchState` whose `_AUDIT_PATH` is redirected to `tmp_path` using
`ks_tmp_audit`'s exact technique, asserting each transition returns inside the
5.0s budget and that the `snapshot()` re-entry does not hang.

- **Pros:** removes the *cause* — the module stops needing a live server, a live
  book, or the live journal. The assertion gets **stronger**: it runs
  unconditionally instead of being silently skipped whenever :8000 is down, so
  the regression lock is actually armed in CI. Satisfies "not deleted, not
  blanket-skipped". Already idiomatic here
  (`test_phase_80_2_error_response_contract.py:212-256`).
- **Cons:** fixes the instance, not the class — the next module can repeat the
  mistake. Must be paired with A. Also loses the "production-shape evidence"
  the original docstring valued; keep that by leaving the read-only GET probe
  (`_backend_is_up` + `_get_paused_state`) as an optional live *observation*,
  which the Option-A policy permits.

### Why A + C is the single best answer

- **86.3's five criteria** are met by the pair: C removes the mutating POSTs at
  source; A prevents any future module from re-introducing them; the guard's own
  mutation test proves A is armed; the read-only GET path is untouched, so
  nothing that was green goes red; and one module + one new root conftest is the
  entire blast radius (derived above, not assumed).
- **36.21 inherited criterion 1 — session-scoped, un-dodgeable by omission:**
  satisfied by an *import-time* install in a **repo-root** `conftest.py`. This is
  strictly stronger than session-scoped autouse, because it also covers
  collection time (where this module's `skipif` lives) — and it is the only
  placement that reaches the `tests/` tree, which today has no conftest at all.
  Per ADR-001, ship it **without a per-test override marker**; the single narrow
  exception (a named constant listing the allowed host:port pairs) stays greppable
  in one file.
- **36.21 inherited criterion 2 — `test_phase_23_2_4_audit_log_clean_transitions`
  still passes, trigger allowlist byte-unchanged:** satisfied because the guard
  permits GETs, so the module still collects, `_backend_is_up()` still returns
  truthy against a running backend, and the test's dependencies (a) - (d) above
  are all preserved. Lines :187 and :195 are never touched. **This is exactly why
  Option B alone fails** — `--disable-socket` errors or reddens that test.

**Additional recommendation (cheap, high value):** promote 36.12's
`_live_audit_file_is_write_protected` byte-compare into the same root
`conftest.py` as a **function-scoped autouse** fixture. The urlopen guard watches
the HTTP door; the byte-compare watches the *file* door, catching in-process
`_append_audit` writes that no network guard can see. Function-scoped (not
session-scoped) so a violation names the offending test instead of failing the
whole run; the journal is small, so the per-test read is negligible. That closes
both entrances to the live journal with one file.

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 22,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The interception seam already exists: backend/tests/conftest.py:47-61 patches urllib.request.urlopen at IMPORT time but denies only slack.com, so localhost:8000 passes. The offending module resolves urlopen by module attribute, so the patch is in-path. Externally (8 sources read in full), only the socket layer or urllib.request itself catches urllib.request.urlopen -- urllib3/respx/responses are the wrong layer -- and NO published tool (pytest-socket, pytest_network, pytest-recording, ADR-001) offers HTTP-VERB granularity, only host granularity. That is decisive: the required policy is GET-localhost:8000-yes / POST-no. Two collection-time facts constrain the fix: the skipif at :112 evaluates _backend_is_up() at import (before any fixture), and it catches only (URLError, OSError, TimeoutError), so a RuntimeError on the GET path would error collection and break the criterion that test_phase_23_2_4_audit_log_clean_transitions still passes. The caller's 8-module live-host list has 4 false positives (36_7, 75_17, 75_deploy_surface, 80_2 -- docstrings, string args, AST subjects, in-process TestClient) and omits test_phase_76_9_2_max_bridge.py, which POSTs to its own ephemeral-port 127.0.0.1 server -- so a host-level block breaks it. Exactly ONE module mutates production. Recommended: (A) widen the guard to a verb+host:port policy installed at import time in a NEW repo-root conftest.py (rootdir is the repo root via pytest.ini; the tests/ tree has ZERO conftest today), plus (C) rewrite the pause/resume cycle onto in-process TestClient with ks_tmp_audit-style path redirection. pytest-socket is defence-in-depth only.",
  "brief_path": "handoff/current/research_brief_86.3.md",
  "gate_passed": true
}
```
