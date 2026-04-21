"""phase-10.4 Friday promotion gate routine.

Reads Thursday's batch off the weekly ledger, evaluates scored candidates via
the phase-8.5.5 PromotionGate (DSR + PBO), ranks passers by DSR desc / PBO
asc, promotes top-N (default 1, max 3) at a 5% starting allocation, and
persists `fri_promoted_ids` / `fri_rejected_ids` back to the ledger row.

Pure library: no APScheduler, no live-capital side effects. The ledger's
idempotent upsert per week_iso IS the slot counter.

Fail-closed: if the Thursday row is missing or `thu_batch_id` is empty,
returns `error="no_thursday_batch_on_ledger"` with empty lists -- never
silently promotes nothing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.autoresearch import weekly_ledger
from backend.autoresearch.gate import PromotionGate

logger = logging.getLogger(__name__)

_STARTING_ALLOCATION_PCT = 0.05
_DEFAULT_MAX_N = 3


def run_friday_promotion(
    week_iso: str,
    *,
    candidates: list[dict[str, Any]],
    top_n: int = 1,
    max_n: int = _DEFAULT_MAX_N,
    starting_allocation_pct: float = _STARTING_ALLOCATION_PCT,
    gate: PromotionGate | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Run the Friday promotion gate for `week_iso`.

    Returns `{promoted_ids, rejected_ids, allocations, already_fired, error}`.
    - `promoted_ids`: list of `trial_id` strings promoted this fire
    - `rejected_ids`: list of `trial_id` strings that failed the gate or were
      below the top-N cut
    - `allocations`: list of floats (same length as promoted_ids) each equal
      to `starting_allocation_pct` (uniform for v1; risk-parity is a future
      carry-forward)
    - `already_fired`: True iff a prior fire for this `week_iso` already
      populated `fri_promoted_ids`
    - `error`: non-empty string if the fire was aborted (fail-closed path);
      None otherwise
    """
    effective_n = max(0, min(int(top_n), int(max_n)))
    lpath = Path(ledger_path) if ledger_path is not None else weekly_ledger.LEDGER_PATH
    g = gate or PromotionGate()

    rows = weekly_ledger.read_rows(path=lpath)
    row = next((r for r in rows if r.get("week_iso") == week_iso), None)

    # Fail-closed: no Thursday batch on the ledger for this week.
    if row is None or not row.get("thu_batch_id"):
        logger.info(
            "friday_promotion: no Thursday batch on ledger for %s; fail-closed",
            week_iso,
        )
        return {
            "promoted_ids": [],
            "rejected_ids": [],
            "allocations": [],
            "already_fired": False,
            "error": "no_thursday_batch_on_ledger",
        }

    # Idempotency: already fired this week?
    prior_promoted = row.get("fri_promoted_ids", "") or ""
    prior_rejected = row.get("fri_rejected_ids", "") or ""
    if prior_promoted and prior_promoted != "[]":
        parsed = _parse_id_list(prior_promoted)
        logger.info(
            "friday_promotion: week %s already fired; %d promoted",
            week_iso,
            len(parsed),
        )
        return {
            "promoted_ids": parsed,
            "rejected_ids": _parse_id_list(prior_rejected),
            "allocations": [starting_allocation_pct] * len(parsed),
            "already_fired": True,
            "error": None,
        }

    # Evaluate candidates through the DSR/PBO gate.
    promoted_objs: list[dict[str, Any]] = []
    rejected_objs: list[dict[str, Any]] = []
    for c in candidates:
        verdict = g.evaluate(c)
        if verdict.get("promoted"):
            promoted_objs.append(c)
        else:
            rejected_objs.append(c)

    # Rank passers: DSR desc, then PBO asc (lower overfit = better tie-break).
    promoted_objs.sort(
        key=lambda c: (-_safe_float(c.get("dsr"), 0.0), _safe_float(c.get("pbo"), 1.0))
    )
    top = promoted_objs[:effective_n]
    below_cut = promoted_objs[effective_n:]

    promoted_ids = [str(c.get("trial_id")) for c in top]
    # below-cut passers count as "rejected" from this week's promotion (gate-pass
    # but lost the top-N bake-off); union with hard-gate-rejects for the ledger.
    rejected_ids = [str(c.get("trial_id")) for c in (below_cut + rejected_objs)]
    allocations = [starting_allocation_pct] * len(promoted_ids)

    # Preserve prior notes (e.g. Thursday's "kicked_off").
    prior_notes = (row.get("notes") or "").strip()
    alloc_marker = f"starting_alloc={starting_allocation_pct}"
    new_notes = (prior_notes + "; " + alloc_marker).strip("; ") if prior_notes else alloc_marker

    ok = weekly_ledger.append_row(
        week_iso=week_iso,
        thu_batch_id=row.get("thu_batch_id", ""),
        thu_candidates_kicked=row.get("thu_candidates_kicked", "0"),
        fri_promoted_ids=promoted_ids,
        fri_rejected_ids=rejected_ids,
        cost_usd=row.get("cost_usd", "0.0"),
        sortino_monthly=row.get("sortino_monthly", "0.0"),
        notes=new_notes,
        path=lpath,
    )
    if not ok:
        logger.warning("friday_promotion: ledger append failed for week %s", week_iso)
        return {
            "promoted_ids": promoted_ids,
            "rejected_ids": rejected_ids,
            "allocations": allocations,
            "already_fired": False,
            "error": "ledger_write_failed",
        }

    return {
        "promoted_ids": promoted_ids,
        "rejected_ids": rejected_ids,
        "allocations": allocations,
        "already_fired": False,
        "error": None,
    }


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _parse_id_list(s: str) -> list[str]:
    """Parse `"[id1,id2,id3]"` back to `["id1", "id2", "id3"]`. Fail-open to []."""
    if not s:
        return []
    inner = s.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    if not inner.strip():
        return []
    return [p.strip() for p in inner.split(",") if p.strip()]


__all__ = ["run_friday_promotion"]
