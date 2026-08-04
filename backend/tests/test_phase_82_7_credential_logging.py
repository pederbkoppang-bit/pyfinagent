"""phase-82.7 -- a credential in a URL must never reach a log record.

THE DEFECT. `httpx` logs "HTTP Request: GET <full-url>" at INFO. Eight modules build
outbound URLs carrying a credential, so every one of those requests wrote the key into
the log. A `SecretRedactionFilter` has existed since phase-60.4 and its regex was
correct -- but it was attached ONLY inside `backend.main.setup_logging`, whose sole
non-test call site is the FastAPI lifespan (`backend/main.py:151`). The 54
`logging.basicConfig()` bootstraps across backend/, scripts/ and functions/ each
installed a root handler with no filter, so every script/CLI path ran unprotected. That
is how the key reached an operator-visible console during the 2026-08-03 macro backfill.

So the bug was never "no redaction" -- it was REACHABILITY. These tests are written to
fail on the reachability bug specifically: several of them would still pass against the
old code if they merely called `redact_secrets()` directly, which is exactly the trap
the research brief warned about.

NO REAL CREDENTIAL APPEARS IN THIS FILE. Every value below is a synthetic placeholder.
"""
from __future__ import annotations

import importlib
import logging

import httpx
import pytest

from backend.services.observability.log_redaction import (
    SecretRedactionFilter,
    install_secret_redaction,
    redact_secrets,
)

# Obviously fake. Long enough to clear the filter's 8-char minimum.
FAKE_KEY = "SYNTHETIC0000NOTAREALKEY0000000A"

# The eight leaking sites found by the 82.7 audit-class sweep (loop-until-dry, 16 query
# shapes, 3 dry rounds). module -> (credential param, provider).
LEAK_SITES = [
    # module, credential param, provider, TRANSPORT the module actually uses.
    # Q/A cycle-1 [BLOCK]: the first version drove httpx for all eight, but three
    # use `requests` -- so for those it exercised a transport the module never
    # touches and passed regardless of whether the real channel leaked. It did
    # leak. Vacuity shape #5: a fixture that cannot represent the failure.
    ("backend.backtest.data_ingestion", "api_key", "FRED", "httpx"),
    ("backend.tools.fred_data", "api_key", "FRED", "httpx"),
    ("backend.services.fx_rates", "api_key", "FRED", "requests"),
    ("backend.econ_calendar.sources.fred_releases", "api_key", "FRED", "requests"),
    ("backend.tools.alphavantage", "apikey", "AlphaVantage", "httpx"),
    ("backend.tools.social_sentiment", "apikey", "AlphaVantage", "httpx"),
    ("backend.news.sources.finnhub", "token", "Finnhub", "httpx"),
    ("backend.econ_calendar.sources.finnhub_earnings", "token", "Finnhub", "requests"),
]

# The loggers each transport ACTUALLY emits the request URL through. These are
# DESCENDANTS: `urllib3.connectionpool`, not `urllib3`. The 82.7 cycle-1 fix
# attached to the parents and leaked, contradicting log_redaction.py's own
# docstring rule that logger-level filters do not reach descendant records.
TRANSPORT_LOGGERS = {
    "httpx": ["httpx", "httpcore.http11"],
    "requests": ["urllib3.connectionpool", "urllib3.util.retry"],
}



class _Capture(logging.Handler):
    """A handler that records what it EMITS, i.e. post-filter."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:
            self.lines.append("<format-error>")


@pytest.fixture
def capture(monkeypatch):
    """A root handler configured the way a SCRIPT's basicConfig would leave it.

    Deliberately NOT the FastAPI handler: the defect only ever appeared on paths that
    never ran setup_logging(), so the fixture must reproduce that shape or it cannot
    represent the failure.
    """
    root = logging.getLogger()
    h = _Capture()
    h.setLevel(logging.DEBUG)
    root.addHandler(h)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    yield h
    root.removeHandler(h)
    root.setLevel(old_level)


# ── criterion 1: a FRED fetch at INFO leaks nothing ──────────────────

def test_a_real_httpx_request_at_INFO_does_not_log_the_key(capture):
    """END TO END through the actual leak channel.

    Uses httpx's MockTransport so a REAL httpx request is issued (no network), which
    makes httpx emit its real "HTTP Request: GET <url>" INFO record through its real
    logger. Nothing here calls redact_secrets(); if the filter is not REACHABLE from
    this logger, the key appears and the test fails.
    """
    install_secret_redaction()
    logging.getLogger("httpx").setLevel(logging.INFO)

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=GDP&api_key={FAKE_KEY}&file_type=json"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=transport) as client:
        client.get(url)

    blob = "\n".join(capture.lines)
    assert FAKE_KEY not in blob, f"the credential reached a log record:\n{blob}"
    assert "api_key=***REDACTED***" in blob, (
        f"expected a redaction marker, so we know the URL WAS logged and scrubbed "
        f"rather than simply never emitted:\n{blob}"
    )


def test_the_url_is_actually_logged_so_the_test_above_cannot_pass_vacuously(capture):
    """Guard on the guard. If httpx stopped logging request URLs entirely, the test
    above would pass for the wrong reason forever."""
    install_secret_redaction()
    logging.getLogger("httpx").setLevel(logging.INFO)
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with httpx.Client(transport=transport) as client:
        client.get("https://api.stlouisfed.org/fred/series/observations?series_id=GDP")
    blob = "\n".join(capture.lines)
    assert "api.stlouisfed.org" in blob, (
        "httpx no longer logs request URLs; the redaction tests are now vacuous"
    )


# ── criterion 3: the whole fixed set, not just FRED ──────────────────

@pytest.mark.parametrize("module,param,provider,transport", LEAK_SITES,
                         ids=[m.rsplit(".", 1)[-1] for m, _, _, _ in LEAK_SITES])
def test_every_swept_site_is_covered_on_its_own_transport(module, param, provider,
                                                          transport, capture):
    """Coverage on the channel the module ACTUALLY uses.

    Emits through the descendant loggers of that module's real transport --
    `urllib3.connectionpool` for requests, `httpcore.http11` for httpx -- which is
    where the credential-bearing URL is printed. The previous version drove httpx
    for every site and so could not observe the failure for the three
    requests-based modules.
    """
    importlib.import_module(module)
    url = f"https://{provider}.invalid/query?symbol=AAPL&{param}={FAKE_KEY}"

    for logger_name in TRANSPORT_LOGGERS[transport]:
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.DEBUG)
        lg.info("Starting new HTTPS connection: %s", url)

    blob = "\n".join(capture.lines)
    assert FAKE_KEY not in blob, (
        f"{provider} ({module}) leaked {param} through its {transport} transport"
    )
    assert blob.count("***REDACTED***") >= len(TRANSPORT_LOGGERS[transport])


@pytest.mark.parametrize("logger_name", [
    "urllib3.connectionpool", "urllib3.util.retry", "httpcore.http11",
    "httpx", "urllib3", "httpcore", "requests",
    "backend.tools.fred_data", "backend.services.fx_rates",
    "a.brand.new.logger.nobody.enumerated",
])
def test_descendant_loggers_are_covered_not_just_the_named_parents(logger_name, capture):
    """Descendant loggers must be covered, not just the named parents.

    `_EMITTING_LIBRARY_LOGGERS` names parents; the emitters are descendants
    (`urllib3.connectionpool`, `httpcore.http11`). A logger-level filter does NOT
    reach descendant records -- log_redaction.py's own docstring says so, and the
    cycle-1 implementation violated it. The last id is deliberately a logger no
    enumeration could contain.

    THIS TEST DOES NOT PIN THE CYCLE-1 BLOCKER, despite what its first docstring
    claimed. Cycle-2 Q/A [WARN-1] measured that it SURVIVES a mutant restoring the
    cycle-1 design, and I reproduced that: all 10 parameters pass with the
    addHandler hook removed. The reason is fixture ordering -- `capture` attaches
    its handler to root BEFORE the test body calls install, so the plain handler
    leg (which existed in cycle 1, and in phase-60.4 before it) scrubs the record
    and the test cannot tell the two designs apart.

    The blocker IS pinned, by test_install_survives_a_later_basicConfig,
    test_a_handler_added_after_install_is_filtered and
    test_the_addhandler_hook_wraps_exactly_once -- 11 kills under that mutant, none
    of them this one. Making this test observe the leg it names requires attaching
    the capture handler AFTER install; queued as 82.33 rather than changed after a
    PASS verdict, so the shipped tree is the graded tree.
    """
    install_secret_redaction()
    lg = logging.getLogger(logger_name)
    lg.setLevel(logging.DEBUG)
    lg.info("GET https://api.stlouisfed.org/fred?api_key=%s", FAKE_KEY)
    blob = "\n".join(capture.lines)
    assert FAKE_KEY not in blob, f"{logger_name} leaked the credential"
    assert "***REDACTED***" in blob


def test_a_backend_module_logger_re_emitting_a_transport_error_does_not_leak(capture):
    """The 9th shape the Q/A found: modules catch a transport exception and log it.

    `str(requests.exceptions.RequestException)` reconstructs the credential-bearing
    URL, so `logger.warning("fetch failed: %s", exc)` re-emits the key through a
    `backend.*` logger -- a channel neither the URL-construction sweep nor the
    library-logger filter addressed.
    """
    install_secret_redaction()
    lg = logging.getLogger("backend.tools.fred_data")
    lg.setLevel(logging.DEBUG)
    exc = RuntimeError(
        f"HTTPSConnectionPool: /fred/series?api_key={FAKE_KEY} (Caused by timeout)"
    )
    lg.warning("FRED fetch failed: %s", exc)
    blob = "\n".join(capture.lines)
    assert blob, "nothing logged; the test would pass vacuously"
    assert FAKE_KEY not in blob, f"the key was re-emitted by a backend logger:\n{blob}"


# ── the reachability bug specifically ────────────────────────────────

def test_removing_the_filter_from_the_logger_makes_the_key_leak(capture):
    """THE MUTATION THIS SUITE EXISTS FOR.

    The research brief warned that a test which calls `redact_secrets()` directly
    re-pins the function that already worked while leaving the REACHABILITY bug -- the
    actual defect -- untested. This test proves the suite is sensitive to reachability:
    strip the filter and the same request leaks.
    """
    # Strip the filter EVERYWHERE, not just from the httpx logger. Filters
    # rewrite the LogRecord IN PLACE and handlers share one record object, so a
    # root handler that still carries the filter scrubs the record before this
    # test's capture handler ever formats it -- the first version of this test
    # removed it from the httpx logger alone and consequently could not observe
    # the leak it claimed to prove.
    hx = logging.getLogger("httpx")
    hx.setLevel(logging.INFO)
    stripped: list[tuple] = []
    for name in ("httpx", "httpcore", "urllib3", "requests"):
        lg = logging.getLogger(name)
        for f in [f for f in lg.filters if isinstance(f, SecretRedactionFilter)]:
            lg.removeFilter(f); stripped.append((lg, f))
    for h in list(logging.getLogger().handlers):
        for f in [f for f in h.filters if isinstance(f, SecretRedactionFilter)]:
            h.removeFilter(f); stripped.append((h, f))
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?api_key={FAKE_KEY}"
        transport = httpx.MockTransport(lambda request: httpx.Response(200))
        with httpx.Client(transport=transport) as client:
            client.get(url)
        blob = "\n".join(capture.lines)
        assert FAKE_KEY in blob, (
            "with the filter removed the key should appear -- if it does not, these "
            "tests are not actually exercising the leak channel and prove nothing"
        )
    finally:
        for owner, f in stripped:
            owner.addFilter(f)


def test_install_is_idempotent():
    """It is called at import by eight modules plus main.py. If it stacked a filter per
    call, a long-running process would accumulate hundreds."""
    hx = logging.getLogger("httpx")
    install_secret_redaction()
    before = sum(isinstance(f, SecretRedactionFilter) for f in hx.filters)
    for _ in range(5):
        install_secret_redaction()
    after = sum(isinstance(f, SecretRedactionFilter) for f in hx.filters)
    assert before == after == 1, f"filter count went {before} -> {after}"


def test_the_addhandler_hook_wraps_exactly_once():
    """A surviving mutant found in my own matrix, now closed.

    MU8 (drop the `_HOOK_INSTALLED` early-return) left the whole suite GREEN:
    `test_install_is_idempotent` counts filters on the httpx logger, which cannot
    see `logging.Logger.addHandler` being re-wrapped. install() is called by nine
    modules plus main.py, so an un-guarded hook nests ten wrappers deep and every
    addHandler call in the process pays for all of them.

    Identity is the observable property: after repeated installs the patched
    function must be the SAME object, not a fresh wrapper around the previous one.
    """
    import backend.services.observability.log_redaction as lr

    install_secret_redaction()
    first = logging.Logger.addHandler
    for _ in range(5):
        install_secret_redaction()
    assert logging.Logger.addHandler is first, (
        "addHandler was re-wrapped; the hook is not idempotent and nests one "
        "wrapper per install() call"
    )
    assert lr._HOOK_INSTALLED is True


def test_install_survives_a_later_basicConfig():
    """The losing script ORDER: install (root has no handlers) -> basicConfig.

    Q/A cycle-1 [BLOCK]: the first version of this test was TRUE BY CONSTRUCTION
    (vacuity shape #4). It called `logging.basicConfig()` while the `capture`
    fixture had already attached a root handler -- and basicConfig is a documented
    no-op when root has handlers, measured before=1 after=1. The unfiltered handler
    whose survival it claimed to prove was never created.

    This version clears root first so basicConfig genuinely creates one, and does
    NOT use the fixture. That order -- import a leak site, THEN basicConfig -- is
    the order in the checked-in `scripts/migrations/extend_historical_data.py`.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        assert not root.handlers, "precondition: root must be empty for basicConfig to act"
        install_secret_redaction()

        logging.basicConfig(level=logging.INFO)      # NOW it really adds a handler
        assert root.handlers, "basicConfig did not add a handler; the test is vacuous"

        sink = _Capture()
        root.addHandler(sink)
        root.setLevel(logging.DEBUG)
        logging.getLogger("urllib3.connectionpool").info(
            "https://api.stlouisfed.org/fred?api_key=%s", FAKE_KEY)

        blob = "\n".join(sink.lines)
        assert blob, "nothing logged"
        assert FAKE_KEY not in blob, f"a handler created AFTER install leaked:\n{blob}"
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


def test_a_handler_added_after_install_is_filtered():
    """The mechanism behind the test above, asserted directly: the addHandler hook
    means a handler attached at any later point carries the filter."""
    install_secret_redaction()
    lg = logging.getLogger("test.late.handler")
    h = _Capture()
    lg.addHandler(h)
    try:
        assert any(isinstance(f, SecretRedactionFilter) for f in h.filters), (
            "a handler added after install carries no filter -- redaction would "
            "depend on call ORDER, which is the script shape that leaked"
        )
    finally:
        lg.removeHandler(h)


@pytest.mark.parametrize("module,param,provider,transport", LEAK_SITES,
                         ids=[m.rsplit(".", 1)[-1] for m, _, _, _ in LEAK_SITES])
def test_every_swept_site_installs_redaction_at_import(module, param, provider, transport):
    """Coverage by construction: importing any leak-capable module must leave the
    process protected, so a script that imports only that module is safe."""
    import backend.services.observability.log_redaction as lr

    lr._HOOK_INSTALLED = False                     # force the hook to re-arm
    mod = importlib.import_module(module)
    importlib.reload(mod)
    assert lr._HOOK_INSTALLED, (
        f"{module} can build a {provider} URL with {param} but importing it does "
        "not arm redaction -- a script importing only this module leaks"
    )


# ── gap G3: the exception channel ────────────────────────────────────

def test_a_raised_http_error_does_not_leak_the_key(capture):
    """G3. `record.getMessage()` covers the format string only. An
    `httpx.HTTPStatusError` repr carries the full request URL and reaches the handler
    via exc_info/exc_text -- so a call that logged nothing sensitive itself would leak
    the moment it FAILED, which is precisely when it gets logged at ERROR."""
    install_secret_redaction()
    log = logging.getLogger("backend.test.g3")
    install_secret_redaction(log)

    url = f"https://api.stlouisfed.org/fred/series?api_key={FAKE_KEY}"
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    try:
        with httpx.Client(transport=transport) as client:
            client.get(url).raise_for_status()
    except httpx.HTTPStatusError:
        log.exception("macro fetch failed")

    blob = "\n".join(capture.lines)
    assert blob, "nothing was logged; the test would pass vacuously"
    assert FAKE_KEY not in blob, f"the key escaped through the exception channel:\n{blob}"


def test_exception_type_is_preserved_when_its_message_is_scrubbed():
    """Scrubbing must not make tracebacks undiagnosable."""
    rec = logging.LogRecord("t", logging.ERROR, __file__, 1, "boom", (), None)
    exc = httpx.HTTPStatusError(
        f"Server error for url https://x.invalid?api_key={FAKE_KEY}",
        request=httpx.Request("GET", "https://x.invalid"),
        response=httpx.Response(500, request=httpx.Request("GET", "https://x.invalid")),
    )
    rec.exc_info = (type(exc), exc, None)
    rec.exc_text = f"Traceback...\nhttpx.HTTPStatusError: url ...?api_key={FAKE_KEY}"
    SecretRedactionFilter().filter(rec)

    # The EXCEPTION OBJECT is left untouched -- reconstructing it would fail for
    # any type with required kwargs (HTTPStatusError needs request=/response=),
    # and a reconstruction that raises would be swallowed by the filter's
    # fail-open except, silently leaving the credential in place. What the
    # handler actually renders is `exc_text`, and that is what gets scrubbed.
    assert rec.exc_info[0] is httpx.HTTPStatusError, "the type must stay diagnosable"
    assert FAKE_KEY not in rec.exc_text
    assert "api_key=***REDACTED***" in rec.exc_text


# ── the regex itself ─────────────────────────────────────────────────

@pytest.mark.parametrize("param", ["api_key", "apikey", "api-key", "token",
                                   "access_token", "auth_token", "secret", "client_secret"])
def test_regex_covers_every_credential_param_name_used_in_this_repo(param):
    assert FAKE_KEY not in redact_secrets(f"GET https://x.invalid/q?{param}={FAKE_KEY}&z=1")


def test_regex_leaves_innocuous_short_values_alone():
    """Discriminating: a filter that redacted everything would satisfy every test
    above while destroying the logs."""
    assert redact_secrets("GET https://x.invalid/q?key=1&limit=50") == (
        "GET https://x.invalid/q?key=1&limit=50"
    )
