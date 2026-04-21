"""phase-10.7 Rollback kill-switch wiring.

Auto-demotes a challenger when its realized drawdown exceeds `DD_TRIGGER`.
No human approval -- this is the safety bright line. Promotion is HITL
(phase-10.6); demotion is automatic (this file).

Writes three sinks on breach:
  1. JSONL audit   `handoff/demotion_audit.jsonl`         append-only
  2. State JSON    `handoff/logs/monthly_approval_state.json` upsert
  3. Weekly ledger `notes` column (optional, when week_iso is provided)

Pure library. Fail-open on individual sink failure; returns truthful
`demoted` flag. ASCII-only.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.autoresearch import weekly_ledger
from backend.autoresearch.promoter import DD_TRIGGER

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path("handoff/logs/monthly_approval_state.json")
_DEFAULT_AUDIT_PATH = Path("handoff/demotion_audit.jsonl")


def auto_demote_on_dd_breach(
    *,
    challenger_id: str,
    challenger_current_dd: float,
    dd_threshold: float = DD_TRIGGER,
    state_path: Path | None = None,
    audit_path: Path | None = None,
    ledger_path: Path | None = None,
    week_iso: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Auto-demote a challenger on DD breach. No HITL.

    Returns `{demoted, decision, challenger_id, dd, threshold, ts}`.
    `decision` is one of `"auto_demoted"`, `"already_demoted"`, `"no_breach"`.
    """
    spath = Path(state_path) if state_path is not None else _DEFAULT_STATE_PATH
    apath = Path(audit_path) if audit_path is not None else _DEFAULT_AUDIT_PATH
    now_dt = now or datetime.now(timezone.utc)
    ts_iso = now_dt.isoformat()
    dd = float(challenger_current_dd)
    threshold = float(dd_threshold)

    result: dict[str, Any] = {
        "demoted": False,
        "decision": "no_breach",
        "challenger_id": challenger_id,
        "dd": dd,
        "threshold": threshold,
        "ts": ts_iso,
    }

    if abs(dd) <= threshold:
        return result

    # Idempotency: if state already shows this challenger as demoted, no-op.
    state = _load_state(spath)
    demotions = state.get("demotions", {}) if isinstance(state.get("demotions"), dict) else {}
    prior = demotions.get(challenger_id)
    if isinstance(prior, dict) and prior.get("status") == "auto_demoted":
        result["demoted"] = True
        result["decision"] = "already_demoted"
        return result

    # Sink 1: JSONL audit append (append-only; never overwrites).
    try:
        _append_audit(apath, {
            "ts": ts_iso,
            "event": "auto_demoted",
            "challenger_id": challenger_id,
            "dd": dd,
            "threshold": threshold,
            "decision": "auto_demoted",
        })
    except Exception as exc:
        logger.warning("rollback: audit append fail-open: %r", exc)

    # Sink 2: state upsert.
    try:
        demotions[challenger_id] = {
            "status": "auto_demoted",
            "demoted_at_iso": ts_iso,
            "dd": dd,
            "threshold": threshold,
        }
        state["demotions"] = demotions
        _save_state(spath, state)
    except Exception as exc:
        logger.warning("rollback: state upsert fail-open: %r", exc)

    # Sink 3: weekly ledger notes (optional; only when week_iso supplied).
    if week_iso is not None:
        try:
            _append_ledger_notes(
                week_iso=week_iso,
                challenger_id=challenger_id,
                dd=dd,
                ledger_path=ledger_path,
            )
        except Exception as exc:
            logger.warning("rollback: ledger notes fail-open: %r", exc)

    result["demoted"] = True
    result["decision"] = "auto_demoted"
    return result


def _append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _load_state(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("rollback: state load fail-open: %r", exc)
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")


def _append_ledger_notes(
    *,
    week_iso: str,
    challenger_id: str,
    dd: float,
    ledger_path: Path | None,
) -> None:
    lpath = Path(ledger_path) if ledger_path is not None else weekly_ledger.LEDGER_PATH
    rows = weekly_ledger.read_rows(path=lpath)
    row = next((r for r in rows if r.get("week_iso") == week_iso), None)
    prior_notes = (row.get("notes") if row else "") or ""
    marker = f"auto_demoted:{challenger_id}:dd={dd:.4f}"
    new_notes = (prior_notes + "; " + marker).strip("; ") if prior_notes else marker
    weekly_ledger.append_row(
        week_iso=week_iso,
        thu_batch_id=(row.get("thu_batch_id", "") if row else ""),
        thu_candidates_kicked=(row.get("thu_candidates_kicked", "0") if row else "0"),
        fri_promoted_ids=(row.get("fri_promoted_ids", "") if row else ""),
        fri_rejected_ids=(row.get("fri_rejected_ids", "") if row else ""),
        cost_usd=(row.get("cost_usd", "0.0") if row else "0.0"),
        sortino_monthly=(row.get("sortino_monthly", "0.0") if row else "0.0"),
        notes=new_notes,
        path=lpath,
    )


__all__ = ["auto_demote_on_dd_breach", "DD_TRIGGER"]
