"""phase-4.9 step 4.9.2 Startup loader with no hot-reload.

Wraps the pydantic-frozen `load()` from 4.9.0 with the extra
operational guarantees required by an immutable risk-limits core:

1. **SIGHUP is ignored**. Conventional reload signal. We opt out
   once at load_once() so no operator or supervisor can hot-
   rotate the limits at runtime -- a full process restart is the
   only path.

2. **File watcher kills the process on digest change**. The
   pydantic `frozen=True` model protects against in-process
   mutation but cannot see edits to the underlying YAML file.
   A daemon polling thread hashes the file every 10s; mismatch
   -> `os._exit(2)`. `os._exit` bypasses atexit/finally so the
   kill is immediate and unconditional.

3. **Digest is surfaced to healthcheck**. `get_digest()` returns
   the boot-time SHA-256 of the YAML bytes. Operators can curl
   the health endpoint on every live instance + `sort -u` the
   digests to confirm a consistent governance state.

Env escape for test contexts:
    PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1
skips the watcher thread so the immutable_limits + limits_loader
audits don't SIGKILL themselves when they temporarily mutate
limits.yaml during mutation-resistance tests.
"""
from __future__ import annotations

import hashlib
import logging
import os
import signal
import threading
import time
from pathlib import Path

from .limits_schema import LIMITS_FILE, RiskLimits, load

logger = logging.getLogger(__name__)


_WATCH_INTERVAL_SECONDS = 10
_EXIT_CODE_ON_MUTATION = 2

_initialized = False
_boot_digest: str | None = None
_watcher_thread: threading.Thread | None = None
_init_lock = threading.Lock()


def _file_digest(path: Path) -> str:
    """Chunked SHA-256 of the given file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _watcher_loop(path: Path, boot_digest: str) -> None:
    """Poll the file digest; `os._exit` on any change."""
    while True:
        try:
            time.sleep(_WATCH_INTERVAL_SECONDS)
            current = _file_digest(path)
            if current != boot_digest:
                logger.critical(
                    "IMMUTABLE LIMITS MUTATED at runtime: "
                    "boot_digest=%s current=%s path=%s -- KILLING PROCESS",
                    boot_digest, current, path,
                )
                # os._exit bypasses atexit + finally so SystemExit
                # cannot be caught upstream. This is the point.
                os._exit(_EXIT_CODE_ON_MUTATION)
        except Exception:
            # Defensive: the watcher must never kill itself on a
            # transient filesystem error. The process keeps running
            # but we'll notice on the NEXT tick if the file is
            # really mutated.
            logger.exception("governance watcher tick failed")
            continue


def load_once() -> RiskLimits:
    """Boot-time load: cached limits + SIGHUP-ignore + watcher.

    Idempotent across repeated calls; the underlying `load()` is
    already `lru_cache`d (4.9.0). The side effects (signal handler,
    watcher thread) are guarded by `_initialized` so they run
    exactly once per process.
    """
    global _initialized, _boot_digest, _watcher_thread
    limits = load()
    with _init_lock:
        if _initialized:
            return limits
        _boot_digest = _file_digest(Path(LIMITS_FILE))
        # SIGHUP ignore. signal.signal must run on the main thread;
        # load_once() is called from FastAPI lifespan (main thread)
        # before any worker forks, so this is the safe call site.
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except (ValueError, OSError) as e:
            # Worker threads / non-main contexts (e.g., audits) may
            # raise here. We log and continue so the caller still
            # gets the limits object and the digest.
            logger.warning(
                "could not install SIGHUP ignore (non-main thread?): %s", e
            )
        if os.environ.get("PYFINAGENT_DISABLE_GOVERNANCE_WATCHER") != "1":
            _watcher_thread = threading.Thread(
                target=_watcher_loop,
                args=(Path(LIMITS_FILE), _boot_digest),
                daemon=True,
                name="governance-limits-watcher",
            )
            _watcher_thread.start()
            logger.info(
                "governance watcher started (interval=%ds, digest=%s...)",
                _WATCH_INTERVAL_SECONDS, _boot_digest[:12],
            )
        else:
            logger.info(
                "governance watcher DISABLED via "
                "PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1"
            )
        _initialized = True
    return limits


def get_digest() -> str:
    """Return the 64-char SHA-256 digest captured at boot.

    Raises RuntimeError if `load_once()` has not been called. That
    is intentional: no code should query the digest before boot.
    """
    if _boot_digest is None:
        raise RuntimeError(
            "get_digest() called before load_once(); boot has not run"
        )
    return _boot_digest


def is_initialized() -> bool:
    """Test-helper. Returns True if load_once() has run."""
    return _initialized


__all__ = [
    "get_digest",
    "is_initialized",
    "load_once",
]
