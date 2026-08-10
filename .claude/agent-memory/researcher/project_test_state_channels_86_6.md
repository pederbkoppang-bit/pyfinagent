---
name: test-state-channels-86-6
description: 86.6 -- a guard that raises an Exception is SWALLOWED by the production catch-all (measured); sys.addaudithook is the only single seam covering FS+subprocess+socket+urllib; 4 constants point at ONE live journal
metadata:
  type: project
---

Phase-86.6 research (2026-08-09): preventing a pytest suite from mutating live
production state across filesystem / HTTP / subprocess / BigQuery /
module-singleton channels.

**The exception class of a refusal is the whole ballgame -- MEASURED, pytest
9.0.3 in the project venv.** `_pytest.outcomes.Failed` MRO is
`Failed -> OutcomeException -> BaseException` -- it does NOT derive from
`Exception`, so `pytest.fail()` escapes a bare `except Exception`. But BOTH
in-repo guards raise `RuntimeError` (`conftest.py:129-141`,
`backend/tests/conftest.py:52-58`), and `kill_switch._append_audit` catches
`Exception` at `:498-499` and only `logger.warning`s. So a filesystem guard
built to the existing in-repo pattern would be **silently absorbed: write
blocked, test GREEN, nobody learns anything.** `assert` is unsafe for the same
reason -- `AssertionError` IS an `Exception`. Python's own docs give the
rationale: `SystemExit` "inherits from BaseException instead of Exception so
that it is not accidentally caught by code that catches Exception". Counter-
pressure to disclose: the same docs say user-defined exceptions SHOULD derive
from `Exception`, so a BaseException refusal is a deliberate deviation and will
bypass legitimate `except Exception` cleanup.

**`sys.addaudithook` (PEP 578) is the only mechanism covering all four channels
at ONE seam** -- events `open(path, mode, flags)`, `subprocess.Popen`,
`socket.connect`, `urllib.Request`. Blocking by raising IS supported; overhead
1.05x either way. Three hard caveats from the PEP: hooks **cannot be removed or
replaced** (design a flag, not add/remove); "this is not sandboxing"; and the
raised exception must be BaseException-derived or the seam is defeated. After 10
search rounds: **no off-the-shelf pytest plugin blocks FS writes outside a temp
dir**, and nobody in the corpus has wired PEP 578 to pytest.

**Four module-level constants point at the SAME live kill-switch journal**, and
`monkeypatch.setattr(ks, "_AUDIT_PATH", ...)` moves only one:
`kill_switch.py:48` (writer, the one everyone patches), plus
`api/paper_trading.py:892` -- a DUPLICATE constant, READ-ONLY at `:940-941`, so
it is an isolation *illusion*, not data loss. Two more independent writers in
the same class: `risk_overrides.py:41/:128`, `cron_control.py:31/:60`. 20 files
carry module-level `handoff/` paths -- the FS channel is a FAMILY, not a path.

**Off-the-shelf preventers are each the wrong shape here:** `pytest-socket` is
HOST-level with no port granularity (breaks the ephemeral-port stub the root
conftest already documents); `pyfakefs` replaces the whole FS so it breaks the
two tests that must read the REAL journal, and it pauses patching in logreport;
`pytest-subprocess` has the right default (`ProcessNotRegisteredError`) but is a
FIXTURE, so opt-in. The one transferable idea is BQ `AnonymousCredentials()` --
a client that cannot authenticate cannot write, no emulator needed.

**A child process loads NO conftest** -- `test_phase_4000_2_cc_rail_smoke.py:202`
spawns `sys.executable`; every guard is structurally absent there. 72 test files
use `subprocess`; 2 spawn a child python. Env vars cross the boundary,
monkeypatch does not -- but only if the CHILD checks them, and
`bigquery_client.py` never consults `PYFINAGENT_TEST_NO_BQ` (honored at exactly
2 sites, both `observability/api_call_log.py:125,:322`). That guard also lives in
`backend/tests/conftest.py:21`, so the `tests/` tree gets neither it nor the
Slack guard.

**Why:** 86.3 measured 8 rows appended to the live journal and the operator's
armed book paused 4x by a test run. The 86.1 fixture at
`test_book_safety_69.py:30-43` already concedes in its own docstring: "This is a
DETECTOR, not a preventer -- the bytes are already on disk when it fires."

**How to apply:** when designing any test-isolation guard in this repo, (1) check
what catches it before choosing the exception class, (2) derive the protected
path family rather than listing constants (`kill_switch.py:64-65` records the
hardcoding failure mode), (3) allow READS -- `test_phase_23_2_4...:253-255` must
keep asserting the module default IS the live file, (4) install at conftest
IMPORT, not in a fixture, because `_backend_is_up()` runs inside a `skipif`
decorator at module import. Related: [[test-suite-live-egress-86-3]],
[[flag-accident-landmine-86-1]].
