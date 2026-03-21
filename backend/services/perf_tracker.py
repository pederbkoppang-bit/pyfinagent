"""
Performance Tracker — thread-safe per-endpoint latency recording.

Collects request timing data from the middleware and provides
p50/p95/p99 percentile summaries, slow endpoint detection,
and TSV export for the autoresearch optimizer.
"""

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LatencyEntry:
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: float
    cache_hit: bool = False


class PerfTracker:
    """Thread-safe latency recorder with percentile summaries."""

    def __init__(self, max_entries: int = 10_000):
        self._entries: list[LatencyEntry] = []
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def record(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> None:
        entry = LatencyEntry(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            timestamp=time.time(),
            cache_hit=cache_hit,
        )
        with self._lock:
            self._entries.append(entry)
            # Evict oldest entries if over limit
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

    def summarize(self, window_seconds: float = 300) -> dict:
        """Latency summary for the last `window_seconds`."""
        cutoff = time.time() - window_seconds
        with self._lock:
            recent = [e for e in self._entries if e.timestamp >= cutoff]

        if not recent:
            return {
                "window_seconds": window_seconds,
                "total_requests": 0,
                "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "cache_hit_rate_pct": 0,
                "per_endpoint": {},
            }

        latencies = [e.latency_ms for e in recent]
        cache_hits = sum(1 for e in recent if e.cache_hit)
        hit_rate = cache_hits / len(recent) * 100

        # Per-endpoint breakdown
        by_endpoint: dict[str, list[float]] = {}
        for e in recent:
            by_endpoint.setdefault(e.endpoint, []).append(e.latency_ms)

        per_endpoint = {}
        for ep, lats in by_endpoint.items():
            lats_sorted = sorted(lats)
            per_endpoint[ep] = {
                "count": len(lats),
                "p50_ms": round(self._percentile(lats_sorted, 50), 1),
                "p95_ms": round(self._percentile(lats_sorted, 95), 1),
            }

        latencies_sorted = sorted(latencies)
        return {
            "window_seconds": window_seconds,
            "total_requests": len(recent),
            "p50_ms": round(self._percentile(latencies_sorted, 50), 1),
            "p95_ms": round(self._percentile(latencies_sorted, 95), 1),
            "p99_ms": round(self._percentile(latencies_sorted, 99), 1),
            "cache_hit_rate_pct": round(hit_rate, 1),
            "per_endpoint": per_endpoint,
        }

    def get_slow_endpoints(self, threshold_ms: float = 1000) -> list[dict]:
        """Endpoints with p95 latency above threshold."""
        summary = self.summarize()
        slow = []
        for ep, data in summary.get("per_endpoint", {}).items():
            if data["p95_ms"] > threshold_ms:
                slow.append({"endpoint": ep, **data})
        return sorted(slow, key=lambda x: x["p95_ms"], reverse=True)

    def export_tsv(self, path: str | Path) -> int:
        """Dump raw entries to TSV. Returns row count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            entries = list(self._entries)
        with open(path, "w") as f:
            f.write("timestamp\tendpoint\tmethod\tstatus_code\tlatency_ms\tcache_hit\n")
            for e in entries:
                f.write(f"{e.timestamp:.3f}\t{e.endpoint}\t{e.method}\t{e.status_code}\t{e.latency_ms:.1f}\t{e.cache_hit}\n")
        return len(entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _percentile(sorted_data: list[float], p: float) -> float:
        if not sorted_data:
            return 0
        k = (len(sorted_data) - 1) * (p / 100)
        floor = int(k)
        ceil = min(floor + 1, len(sorted_data) - 1)
        if floor == ceil:
            return sorted_data[floor]
        d = k - floor
        return sorted_data[floor] + d * (sorted_data[ceil] - sorted_data[floor])


# ── Module-level singleton ───────────────────────────────────────

_perf_tracker = PerfTracker()


def get_perf_tracker() -> PerfTracker:
    return _perf_tracker
