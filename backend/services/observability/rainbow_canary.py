"""phase-12.3 Rainbow canary SLO diff.

Lightweight helpers for comparing live-traffic latency between two colors
during a Rainbow canary rollout. No external deps (stdlib only); no
Prometheus / OTEL coupling; integrates with the existing
`backend/services/observability/api_call_log.py` buffer via a caller-
supplied partition function.

Design decisions (per phase-12.3 research brief):
- Threshold ratio `green_p95 / blue_p95 > 1.2 = regression` instead of
  Welch's t-test or Mann-Whitney U. Deterministic, readable, matches
  Flagger / Argo Rollouts practitioner convention at MVP sample sizes.
- No new `color` column on `api_call_log`. Operators tag canary traffic
  via `source="pyfinagent-green"` (or any caller-chosen scheme) and pass
  filter predicates to `canary_snapshot_from_buffer`.
- Fail-open on empty / too-few samples: return a neutral SLODiff with
  `regression=False` so a missing-data window never auto-triggers a
  rollback.

For real BQ-backed reads in production, use
`mcp_execute_sql_readonly` against
`pyfinagent_data.api_call_log WHERE ts > TIMESTAMP_SUB(..., INTERVAL 5 MINUTE)`.
This module's logic is BQ-agnostic: feed it two lists of latencies and
it returns the diff.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


DEFAULT_THRESHOLD = 1.2  # green_p95 / blue_p95 ratio that flips regression=True
DEFAULT_MIN_SAMPLES = 10  # per-color floor; below this, fail-open neutral


@dataclass
class SLODiff:
    """Result of a single canary snapshot comparison."""

    blue_p95: float
    green_p95: float
    ratio: float  # green_p95 / blue_p95 (0.0 when fail-open)
    regression: bool
    threshold: float
    blue_samples: int
    green_samples: int
    reason: str  # "ok" | "insufficient_samples" | "empty"


def percentile(values: Sequence[float], p: float) -> float:
    """Stdlib-only percentile. `p` in [0, 100].

    Uses linear interpolation between ordered positions (the same method
    `numpy.percentile(..., interpolation="linear")` produces). Empty input
    returns 0.0.
    """
    if not values:
        return 0.0
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"percentile p must be in [0, 100], got {p!r}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    # index in [0, n-1] at percentile p
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def compute_slo_diff(
    blue_latencies: Sequence[float],
    green_latencies: Sequence[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> SLODiff:
    """Compare two latency distributions; return a SLODiff.

    Fail-open rules:
      - Either side empty -> regression=False, reason="empty".
      - Either side below min_samples -> regression=False, reason="insufficient_samples".
      - Otherwise -> ratio = green_p95 / blue_p95; regression = ratio > threshold.

    The threshold is one-sided (we only alarm when GREEN is slower). If
    green is faster than blue (ratio < 1.0), we surface it as "ok" — that's
    a signal to promote, not to roll back.
    """
    n_blue = len(blue_latencies)
    n_green = len(green_latencies)

    if n_blue == 0 or n_green == 0:
        return SLODiff(
            blue_p95=0.0,
            green_p95=0.0,
            ratio=0.0,
            regression=False,
            threshold=threshold,
            blue_samples=n_blue,
            green_samples=n_green,
            reason="empty",
        )

    if n_blue < min_samples or n_green < min_samples:
        return SLODiff(
            blue_p95=percentile(blue_latencies, 95),
            green_p95=percentile(green_latencies, 95),
            ratio=0.0,
            regression=False,
            threshold=threshold,
            blue_samples=n_blue,
            green_samples=n_green,
            reason="insufficient_samples",
        )

    blue_p95 = percentile(blue_latencies, 95)
    green_p95 = percentile(green_latencies, 95)
    # Guard against zero blue_p95 (all samples were 0ms).
    ratio = (green_p95 / blue_p95) if blue_p95 > 0 else float("inf") if green_p95 > 0 else 0.0
    regression = ratio > threshold
    return SLODiff(
        blue_p95=blue_p95,
        green_p95=green_p95,
        ratio=ratio,
        regression=regression,
        threshold=threshold,
        blue_samples=n_blue,
        green_samples=n_green,
        reason="ok",
    )


def canary_snapshot_from_buffer(
    is_blue: Callable[[dict[str, Any]], bool],
    is_green: Callable[[dict[str, Any]], bool],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> SLODiff:
    """Partition the in-process api_call_log buffer by caller-supplied color predicates.

    Read-only. Does NOT flush the buffer. Since api_call_log has no `color`
    column, the caller owns the partition logic — typically a string match on
    `source` like `is_blue = lambda r: r["source"] == "pyfinagent-blue"`.

    Latency dimension: `row["latency_ms"]`. Rows with missing / non-numeric
    latency are silently skipped (fail-open).

    This is MVP-only: production reads query BQ `pyfinagent_data.api_call_log`
    directly via BQ MCP. See module docstring.
    """
    # Local import keeps this module loadable even when the observability
    # package isn't fully initialized in a narrow test context.
    from backend.services.observability.api_call_log import _buffer  # type: ignore[attr-defined]

    blue_lat: list[float] = []
    green_lat: list[float] = []
    # _buffer is a list of _ApiCallRow dataclass instances; use asdict-ish reads.
    for row in list(_buffer):
        try:
            d = row.__dict__ if hasattr(row, "__dict__") else dict(row)  # type: ignore[arg-type]
        except Exception:
            continue
        lat = d.get("latency_ms")
        if not isinstance(lat, (int, float)):
            continue
        if is_blue(d):
            blue_lat.append(float(lat))
        elif is_green(d):
            green_lat.append(float(lat))

    return compute_slo_diff(
        blue_lat, green_lat, threshold=threshold, min_samples=min_samples
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_SAMPLES",
    "SLODiff",
    "percentile",
    "compute_slo_diff",
    "canary_snapshot_from_buffer",
]
