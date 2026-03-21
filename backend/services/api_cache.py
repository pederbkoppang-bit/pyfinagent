"""
API Response Cache — thread-safe in-memory TTL cache for BigQuery-backed endpoints.

Provides a module-level singleton cache with per-endpoint configurable TTLs.
Write endpoints explicitly invalidate relevant cache keys on mutation.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0


class APICache:
    """Thread-safe in-memory TTL cache."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._total_gets = 0
        self._total_hits = 0

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            self._total_gets += 1
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            entry.hits += 1
            self._total_hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store a value with TTL expiration."""
        now = time.monotonic()
        with self._lock:
            self._store[key] = CacheEntry(
                value=value,
                expires_at=now + ttl_seconds,
                created_at=now,
            )

    def invalidate(self, pattern: str) -> int:
        """Invalidate all keys matching a glob pattern (e.g., 'reports:*').

        Converts glob '*' → regex '.*' for matching. Returns count of evicted entries.
        """
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
        evicted = 0
        with self._lock:
            keys_to_delete = [k for k in self._store if regex.match(k)]
            for k in keys_to_delete:
                del self._store[k]
                evicted += 1
        if evicted:
            logger.debug("Cache invalidated %d keys matching '%s'", evicted, pattern)
        return evicted

    def stats(self) -> dict:
        """Cache statistics: entry count, hit rate, total gets/hits."""
        with self._lock:
            now = time.monotonic()
            active = {k: v for k, v in self._store.items() if now <= v.expires_at}
            expired = len(self._store) - len(active)
            # Evict expired entries while we're at it
            if expired:
                self._store = active

            hit_rate = (self._total_hits / self._total_gets * 100) if self._total_gets > 0 else 0
            return {
                "entries": len(self._store),
                "total_gets": self._total_gets,
                "total_hits": self._total_hits,
                "hit_rate_pct": round(hit_rate, 1),
            }

    def clear(self) -> int:
        """Flush all entries. Returns count of evicted entries."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._total_gets = 0
            self._total_hits = 0
        logger.info("Cache cleared (%d entries evicted)", count)
        return count


# ── Module-level singleton ───────────────────────────────────────

_api_cache = APICache()


def get_api_cache() -> APICache:
    return _api_cache


# ── Default TTL registry (tunable by optimizer) ─────────────────

ENDPOINT_TTLS: dict[str, float] = {
    # Reports
    "reports:list": 30.0,
    "reports:cost-summary": 60.0,
    "reports:cost-history": 120.0,
    "reports:ticker": 60.0,
    # Paper trading
    "paper:status": 15.0,
    "paper:portfolio": 15.0,
    "paper:trades": 60.0,
    "paper:snapshots": 120.0,
    "paper:performance": 60.0,
    # Settings
    "settings:full": 300.0,
    "settings:models": 3600.0,
    # Backtest optimizer
    "backtest:experiments": 10.0,
    "backtest:best": 30.0,
}
