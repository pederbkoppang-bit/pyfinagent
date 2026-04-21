# Contract -- Cycle 90 / phase-4.9 step 4.9.2

Step: 4.9.2 Startup loader with no hot-reload

## Research-gate upheld (5th cycle)

Researcher (12 URLs: Python signal/hashlib/sys docs, FastAPI
lifespan + settings patterns, Uvicorn SIGHUP discussion, file
integrity monitor pattern) + Explore (FastAPI main.py lifespan
ContextManager at line 108-180, /api/health endpoint at 251-270,
no pre-existing signal handlers, no watchdog dep, audit script
that ALREADY calls load() and digest).

## Hypothesis

Ship `backend/governance/limits_loader.py` with:
- `load_once()`: calls cached `load()` from 4.9.0, then (once
  per process) installs `signal.signal(SIGHUP, SIG_IGN)`,
  captures boot digest, spawns a daemon polling thread that
  `os._exit(2)` if the digest changes.
- `get_digest()`: returns the boot-time digest (64-char hex).
- Env escape: `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1` skips
  the watcher thread. Required because the audit scripts load
  the module in a test context and we don't want them dying.

FastAPI wiring:
- `backend/main.py` lifespan setup calls `load_once()`.
- `/api/health` includes `limits_digest` in the response body.

## Scope

Files created:

1. **NEW** `backend/governance/limits_loader.py` with:
   - `load_once()` idempotent (module-level `_initialized` gate).
   - SIGHUP ignored (main thread only; one-shot).
   - 10-second polling thread (`daemon=True`) comparing
     `hashlib.file_digest` against the boot value; mismatch ->
     `os._exit(2)`.
   - `get_digest()` returns the boot digest string.
   - `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER` env skip for tests.

2. **MODIFY** `backend/main.py`:
   - Call `load_once()` early in lifespan setup.
   - Extend `/api/health` to include `limits_digest`.

3. **NEW** `scripts/audit/limits_loader_audit.py`:
   Six teeth:
   (a) imports `load_once` + `get_digest` without error.
   (b) `load_once()` called twice returns the same frozen object
       and digest is a 64-char lowercase hex.
   (c) SIGHUP ignored after `load_once()` (reads
       `signal.getsignal(SIGHUP)` and compares to `SIG_IGN`).
   (d) spawning the watcher thread is gated by the env var.
   (e) health endpoint in main.py references `limits_digest`.
   (f) `os._exit` present in the watcher code path (not
       `sys.exit`, which is catchable).

## Immutable success criteria

1. load_once_pattern: `load_once()` + `get_digest()` callable;
   digest is 64-char hex.
2. sighup_ignored: post-load the SIGHUP handler == SIG_IGN.
3. mutation_kills_process: watcher code uses `os._exit` on
   digest mismatch (verified statically -- we do NOT actually
   mutate the file in the audit).
4. digest_exposed_to_healthcheck: main.py health endpoint
   returns `limits_digest`.

## Verification (immutable, from masterplan)

    python -c "from backend.governance.limits_loader import load_once, get_digest; load_once(); d=get_digest(); assert len(d) == 64"

Plus: `python scripts/audit/limits_loader_audit.py --check`.

## Anti-rubber-stamp

qa must:
- Verify SIGHUP handler is REALLY installed (call
  `signal.getsignal(signal.SIGHUP)` after `load_once()`).
- Verify `os._exit` is the termination call (not `sys.exit`).
- Verify watcher thread is `daemon=True` so clean shutdown works.
- Verify the env-var disable path is narrow (only used in tests;
  production has it unset).
- Verify health endpoint actually includes the digest, not just
  logs it.

## References

- Researcher cycle-90 findings (12 URLs).
- backend/governance/limits_schema.py (4.9.0 load + digest).
- FastAPI lifespan docs.
- Python 3.11+ `hashlib.file_digest` API.
