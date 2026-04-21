"""phase-10.2 Weekly ledger writer.

Appends one row per week to `backend/autoresearch/weekly_ledger.tsv`, keyed on
`week_iso` (e.g. "2026-W17"). If a row for the same week already exists, it
is overwritten in place -- making `append_row` idempotent per week. Raw TSV
layout + fail-open errors match phase-8.5.4 results.tsv + phase-8.5.9
seed-from-postmortem conventions.

ASCII-only. No imports beyond stdlib.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).parent / "weekly_ledger.tsv"

COLUMNS: tuple[str, ...] = (
    "week_iso",
    "thu_batch_id",
    "thu_candidates_kicked",
    "fri_promoted_ids",
    "fri_rejected_ids",
    "cost_usd",
    "sortino_monthly",
    "notes",
)


def _format_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def _parse_row(line: str) -> dict[str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(COLUMNS):
        return None
    return dict(zip(COLUMNS, parts))


def read_rows(*, path: Path = LEDGER_PATH) -> list[dict[str, str]]:
    """Return the data rows (header excluded). Fail-open: returns [] on any error."""
    p = Path(path)
    try:
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []
        # Skip header line (index 0).
        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            r = _parse_row(line)
            if r is not None:
                rows.append(r)
        return rows
    except Exception as exc:
        logger.warning("weekly_ledger: read fail-open: %r", exc)
        return []


def append_row(
    week_iso: str,
    thu_batch_id: str = "",
    thu_candidates_kicked: int | str = 0,
    fri_promoted_ids: list[str] | str | None = None,
    fri_rejected_ids: list[str] | str | None = None,
    cost_usd: float | str = 0.0,
    sortino_monthly: float | str = 0.0,
    notes: str = "",
    *,
    path: Path = LEDGER_PATH,
) -> bool:
    """Append or in-place update a row keyed on week_iso. Fail-open.

    Returns True on success, False on any IO / parse error.
    """
    p = Path(path)
    try:
        rows = read_rows(path=p)
        new_row = {
            "week_iso": str(week_iso),
            "thu_batch_id": str(thu_batch_id),
            "thu_candidates_kicked": str(thu_candidates_kicked),
            "fri_promoted_ids": _format_cell(fri_promoted_ids),
            "fri_rejected_ids": _format_cell(fri_rejected_ids),
            "cost_usd": str(cost_usd),
            "sortino_monthly": str(sortino_monthly),
            "notes": str(notes),
        }
        # Idempotent update: replace if same week_iso.
        for i, r in enumerate(rows):
            if r.get("week_iso") == new_row["week_iso"]:
                rows[i] = new_row
                break
        else:
            rows.append(new_row)
        header = "\t".join(COLUMNS) + "\n"
        body = "\n".join("\t".join(r[c] for c in COLUMNS) for r in rows) + ("\n" if rows else "")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(header + body, encoding="utf-8")
        return True
    except Exception as exc:
        logger.warning("weekly_ledger: append fail-open: %r", exc)
        return False


__all__ = ["LEDGER_PATH", "COLUMNS", "append_row", "read_rows"]
