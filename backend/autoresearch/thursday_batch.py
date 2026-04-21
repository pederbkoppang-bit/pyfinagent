"""phase-10.3 Thursday batch trigger routine.

Consumes exactly 1 sprint slot per week, samples >=100 candidates from the
15,000-combo space via Sobol QMC, persists a deterministic batch_id to the
weekly ledger.

Pure library: no APScheduler, no HTTP side effects. The ledger's idempotent
per-week upsert IS the slot counter; a second invocation for the same
week_iso is a no-op that returns already_fired=True.

Batch-id derivation: uuid5(NAMESPACE_DNS, f"thu_batch_{week_iso}_1") -- same
input always produces same id (AWS Powertools idempotency pattern).
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import uuid
from pathlib import Path
from typing import Any

from backend.autoresearch import weekly_ledger

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_SPACE = Path(__file__).parent / "candidate_space.yaml"
_DEFAULT_CALENDAR = Path(__file__).parent / "sprint_calendar.yaml"
_MIN_CANDIDATES = 100


def trigger_thursday_batch(
    week_iso: str,
    *,
    n_candidates: int = 128,
    ledger_path: Path | None = None,
    candidate_space_path: Path | None = None,
    calendar_path: Path | None = None,
) -> dict[str, Any]:
    """Consume the Thursday slot for `week_iso`.

    Returns `{batch_id, week_iso, candidates_kicked, slot_num, already_fired}`.
    Idempotent: second call for same `week_iso` returns `already_fired=True`
    without touching the ledger or re-sampling.
    """
    if n_candidates < _MIN_CANDIDATES:
        raise ValueError(
            f"n_candidates={n_candidates} below floor of {_MIN_CANDIDATES}"
        )

    lpath = Path(ledger_path) if ledger_path is not None else weekly_ledger.LEDGER_PATH
    batch_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"thu_batch_{week_iso}_1"))

    # Idempotency guard: if a row already exists for this week with a non-empty
    # thu_batch_id, return already_fired without re-writing.
    for row in weekly_ledger.read_rows(path=lpath):
        if row.get("week_iso") == week_iso and row.get("thu_batch_id"):
            logger.info(
                "thursday_batch: week %s already fired batch_id=%s; skipping",
                week_iso,
                row["thu_batch_id"],
            )
            return {
                "batch_id": row["thu_batch_id"],
                "week_iso": week_iso,
                "candidates_kicked": int(row.get("thu_candidates_kicked", "0") or "0"),
                "slot_num": 1,
                "already_fired": True,
            }

    # Enumerate the candidate space and sample via Sobol QMC.
    cspace = Path(candidate_space_path) if candidate_space_path else _DEFAULT_CANDIDATE_SPACE
    candidates = _sample_candidates(cspace, week_iso, n_candidates)

    ok = weekly_ledger.append_row(
        week_iso=week_iso,
        thu_batch_id=batch_id,
        thu_candidates_kicked=len(candidates),
        notes="kicked_off",
        path=lpath,
    )
    if not ok:
        logger.warning("thursday_batch: ledger append failed for week %s", week_iso)

    return {
        "batch_id": batch_id,
        "week_iso": week_iso,
        "candidates_kicked": len(candidates),
        "slot_num": 1,
        "already_fired": False,
    }


def _sample_candidates(candidate_space_path: Path, week_iso: str, n: int) -> list[tuple]:
    """Sobol-sample `n` candidate tuples from the Cartesian product of the
    7-dim space declared in candidate_space.yaml. Seed derived from week_iso
    for reproducibility (same week -> same sample)."""
    import yaml
    spec = yaml.safe_load(Path(candidate_space_path).read_text(encoding="utf-8"))

    dims = [
        spec["params"]["learning_rate"],
        spec["params"]["max_depth"],
        spec["params"]["n_estimators"],
        spec["params"]["rolling_window"],
        spec["prompts"],
        spec["features"],
        spec["model_archs"],
    ]
    # Enumerate full Cartesian product (5*4*3*2*5*5*5 = 15,000 in the shipping
    # config). Safe at this scale -- list fits comfortably in memory.
    enumerated = list(itertools.product(*dims))
    total = len(enumerated)
    if total == 0:
        return []

    seed = int(hashlib.md5(week_iso.encode("utf-8")).hexdigest(), 16) % (2**32)

    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=1, scramble=True, seed=seed)
        raw = sampler.random(n=n).flatten()
        # Map uniform [0, 1) -> [0, total); dedupe preserving order.
        indices: list[int] = []
        seen: set[int] = set()
        for u in raw:
            idx = int(u * total) % total
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        # If Sobol produced duplicates (possible when n approaches total's
        # density), fill with deterministic stride until we reach n.
        if len(indices) < n:
            stride = max(1, total // n)
            i = 0
            while len(indices) < n and i < total:
                if i not in seen:
                    seen.add(i)
                    indices.append(i)
                i += stride
    except Exception as exc:
        logger.warning("thursday_batch: Sobol fail-open to stride-sample: %r", exc)
        stride = max(1, total // n)
        indices = list(range(0, total, stride))[:n]

    return [enumerated[i] for i in indices[:n]]


__all__ = ["trigger_thursday_batch"]
