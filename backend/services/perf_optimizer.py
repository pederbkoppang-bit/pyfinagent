"""
Performance Optimizer — autoresearch-style loop that tunes API cache TTLs.

Follows the same propose → apply → measure → keep/discard pattern
as `skill_optimizer.py`, but optimizes a single metric: p95 latency.
"""

import asyncio
import logging
import random
import time
from pathlib import Path

from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.services.perf_tracker import get_perf_tracker

logger = logging.getLogger(__name__)

_TSV_PATH = Path(__file__).parent / "experiments" / "perf_results.tsv"
_TSV_HEADER = "timestamp\tendpoint\tttl_before\tttl_after\tp95_before\tp95_after\thit_rate\tstatus\n"

# Tunable bounds
_MIN_TTL = 5.0
_MAX_TTL = 3600.0
_MEASURE_WINDOW = 60  # seconds to collect data before measuring
_MIN_IMPROVEMENT_PCT = 5.0  # require 5% p95 improvement to keep
_CONSECUTIVE_DISCARD_LIMIT = 5


class PerfOptimizer:
    """Autoresearch loop for tuning cache TTL parameters."""

    _running = False  # class-level flag for graceful stop

    async def run_loop(self, state: dict) -> None:
        """Main optimization loop. Updates `state` dict in place."""
        PerfOptimizer._running = True
        state["status"] = "running"
        state["iterations"] = 0
        state["kept"] = 0
        state["discarded"] = 0
        consecutive_discards = 0

        # Ensure TSV file exists
        _TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _TSV_PATH.exists():
            _TSV_PATH.write_text(_TSV_HEADER)

        # Baseline measurement
        await asyncio.sleep(_MEASURE_WINDOW)
        baseline = get_perf_tracker().summarize(window_seconds=_MEASURE_WINDOW)
        baseline_p95 = baseline.get("p95_ms", 0)
        state["best_p95_ms"] = baseline_p95
        logger.info("Perf optimizer baseline: p95=%.1fms", baseline_p95)

        while PerfOptimizer._running:
            state["iterations"] += 1

            # 1. Propose: pick a random endpoint TTL to modify
            endpoint_key = random.choice(list(ENDPOINT_TTLS.keys()))
            old_ttl = ENDPOINT_TTLS[endpoint_key]

            if consecutive_discards >= _CONSECUTIVE_DISCARD_LIMIT:
                # Think harder: try doubling TTL for maximum caching
                new_ttl = min(old_ttl * 2.0, _MAX_TTL)
                consecutive_discards = 0
                logger.info("Perf optimizer: think_harder -- doubling TTL for %s", endpoint_key)
            else:
                # Random ±20% perturbation
                factor = random.uniform(0.8, 1.2)
                new_ttl = max(_MIN_TTL, min(old_ttl * factor, _MAX_TTL))

            # 2. Apply
            ENDPOINT_TTLS[endpoint_key] = new_ttl
            get_api_cache().invalidate(endpoint_key.replace(":", ":*").rsplit(":", 1)[0] + ":*")

            # 3. Measure
            await asyncio.sleep(_MEASURE_WINDOW)
            if not PerfOptimizer._running:
                ENDPOINT_TTLS[endpoint_key] = old_ttl
                break

            result = get_perf_tracker().summarize(window_seconds=_MEASURE_WINDOW)
            new_p95 = result.get("p95_ms", 0)
            hit_rate = result.get("cache_hit_rate_pct", 0)
            prev_p95 = state.get("best_p95_ms", baseline_p95)

            # 4. Decide
            improved = False
            if prev_p95 > 0 and new_p95 > 0:
                improvement_pct = (prev_p95 - new_p95) / prev_p95 * 100
                improved = improvement_pct >= _MIN_IMPROVEMENT_PCT

            if improved:
                status = "keep"
                state["kept"] += 1
                state["best_p95_ms"] = new_p95
                consecutive_discards = 0
                logger.info(
                    "Perf optimizer: KEEP %s TTL %.0f->%.0f (p95 %.0f->%.0fms)",
                    endpoint_key, old_ttl, new_ttl, prev_p95, new_p95,
                )
            else:
                status = "discard"
                ENDPOINT_TTLS[endpoint_key] = old_ttl  # revert
                state["discarded"] += 1
                consecutive_discards += 1
                logger.info(
                    "Perf optimizer: DISCARD %s TTL %.0f->%.0f (p95 %.0f->%.0fms)",
                    endpoint_key, old_ttl, new_ttl, prev_p95, new_p95,
                )

            # 5. Log to TSV
            _log_experiment(endpoint_key, old_ttl, new_ttl, prev_p95, new_p95, hit_rate, status)

        state["status"] = "stopped"
        logger.info("Perf optimizer stopped after %d iterations", state["iterations"])


def _log_experiment(
    endpoint: str, ttl_before: float, ttl_after: float,
    p95_before: float, p95_after: float, hit_rate: float, status: str,
) -> None:
    """Append one row to perf_results.tsv."""
    _TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _TSV_PATH.exists():
        _TSV_PATH.write_text(_TSV_HEADER, encoding="utf-8")
    with open(_TSV_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{endpoint}\t"
            f"{ttl_before:.0f}\t{ttl_after:.0f}\t"
            f"{p95_before:.0f}\t{p95_after:.0f}\t"
            f"{hit_rate:.1f}\t{status}\n"
        )
