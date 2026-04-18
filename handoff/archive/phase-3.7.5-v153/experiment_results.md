# Experiment Results -- Cycle 90 / phase-4.9 step 4.9.2

Step: 4.9.2 Startup loader with no hot-reload

## Research-gate upheld (5th cycle)

Researcher (12 URLs: Python signal/hashlib/sys docs, FastAPI
lifespan patterns, Uvicorn SIGHUP) + Explore (lifespan hook at
main.py, /api/health surface, no watchdog dep, audit env-disable).

## What was generated

1. **NEW** `backend/governance/limits_loader.py`
   - `load_once()` -- idempotent; calls cached `load()` from 4.9.0,
     then under `_init_lock`: captures boot digest, installs
     `signal.signal(SIGHUP, SIG_IGN)`, spawns daemon watcher
     thread (10s poll).
   - Watcher: `os._exit(2)` on digest mismatch. Exception handler
     log-and-continues on transient FS errors but does NOT catch
     `os._exit` (which bypasses Python exception machinery).
   - `get_digest()` returns the boot 64-char SHA-256 hex.
   - `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1` env skip for audit
     / test contexts.

2. **MODIFY** `backend/main.py`:
   - Lifespan setup calls `load_once()` on the main thread before
     worker fork.
   - `/api/health` now returns `limits_digest` alongside status.

3. **NEW** `scripts/audit/limits_loader_audit.py`:
   8 teeth: imports, idempotent load_once, 64-hex digest, SIGHUP
   ignored, watcher disabled by env, os._exit present, no sys.exit
   in watcher block, /api/health returns limits_digest.

## Verification (verbatim, immutable)

    $ PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1 python -c "from \
      backend.governance.limits_loader import load_once, get_digest; \
      load_once(); d=get_digest(); assert len(d) == 64"
    exit=0

    $ python scripts/audit/limits_loader_audit.py --check
    {"verdict": "PASS", all 8 teeth true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| load_once_pattern | PASS (idempotent; digest 64 hex) |
| sighup_ignored | PASS (getsignal(SIGHUP)==SIG_IGN post-load) |
| mutation_kills_process | PASS (os._exit in watcher; no sys.exit) |
| digest_exposed_to_healthcheck | PASS (/api/health returns digest) |

## Mutation-resistance proven

Three independent mutations proven to flip audit verdict:
- `os._exit` -> `sys.exit` (regression; audit rc=1)
- `signal.signal(SIGHUP, SIG_IGN)` removed -> audit rc=1
- `limits_digest` stripped from `/api/health` return -> audit rc=1

## Known limitations (tracked follow-up)

- Watcher polls every 10s; a 9s mutation window is technically
  undetected. For a risk-limits file this is acceptable (tag-
  signed-commit gate in 4.9.1 makes runtime mutation a CI
  violation, not a normal flow).
- `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER` is the only bypass;
  production must NOT set it. A later phase-4.9.x step can add
  a CI lint that greps deploy configs for this env name.
