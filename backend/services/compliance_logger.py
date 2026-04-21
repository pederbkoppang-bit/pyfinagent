"""phase-4.8 step 4.8.9 FINRA GenAI compliance logger.

Every paper-trading trade writes one JSONL rationale record keyed
by `trade_id` to WORM storage. Record shape satisfies SEC 17a-4(f)(2)
audit-trail requirements + the HITL logging convention from FINRA
Reg Notice 24-09 and the FINRA 2026 Annual Oversight Report.

Storage backends:
- Production: GCS Bucket with a LOCKED 6-year retention policy.
  Activated by setting env `COMPLIANCE_WORM_BUCKET=<name>`. Per
  researcher findings, 6 years is the conservative choice: SEC
  17a-4 tiers trade-order records in the 6y bucket (17a-3(a)(1)),
  and GCS Bucket Lock cannot be SHORTENED once locked.
- Local dev fallback: append-only directory
  `handoff/rationale_worm/` -- honestly labeled "NOT WORM" because
  the filesystem does not enforce immutability. Used only when
  `COMPLIANCE_WORM_BUCKET` is unset.

The `fetch_rationale(trade_id)` function reads the record back,
which exercises the SEC 17a-4(f)(2)(iv) requirement that original
records be reconstructible.
"""
from __future__ import annotations

import json

from backend.utils import json_io
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Retention constants.
# SEC 17a-4 requires 6 years for 17a-3(a)(1) trade-order records.
# The masterplan's "3y" is this project's internal minimum floor.
RETENTION_YEARS_POLICY = 6          # production bucket-lock target
RETENTION_YEARS_MINIMUM = 3         # masterplan criterion floor
RETENTION_SECONDS_POLICY = RETENTION_YEARS_POLICY * 365 * 24 * 60 * 60

_REPO = Path(__file__).resolve().parents[2]
_LOCAL_WORM_DIR = _REPO / "handoff" / "rationale_worm"


@dataclass
class RationaleRecord:
    """SEC 17a-4(f)(2)-compatible trade rationale record."""
    trade_id: str
    created_at: str
    system_id: str
    agent_trace: list[dict]          # [{agent, model, version, role}, ...]
    input_signals: dict              # signals at decision time
    output_recommendation: str       # "BUY" / "SELL" / "HOLD"
    confidence: float                # 0..1
    # HITL fields -- REQUIRED per 17a-4(f)(2) + FINRA 24-09.
    approver_id: str                 # email or system-user id
    approved_at: str                 # ISO 8601 UTC
    decision: str                    # "approve" | "reject" | "modify"
    reason_code: str                 # free-text or enum
    # Original before HITL modification (for 17a-4(f)(2)(iv)).
    original_recommendation: str | None = None
    # Optional extras.
    extras: dict = field(default_factory=dict)

    def validate(self) -> None:
        for req in ("trade_id", "approver_id", "approved_at",
                     "decision", "reason_code"):
            v = getattr(self, req)
            if not isinstance(v, str) or not v:
                raise ValueError(
                    f"RationaleRecord.{req} is required and must be "
                    f"a non-empty string; got {v!r}"
                )
        if self.decision not in ("approve", "reject", "modify"):
            raise ValueError(
                f"decision must be approve/reject/modify, got {self.decision!r}"
            )


def _backend() -> str:
    """Return 'gcs' or 'local' based on env configuration."""
    if os.getenv("COMPLIANCE_WORM_BUCKET"):
        return "gcs"
    return "local"


def _local_path(trade_id: str) -> Path:
    _LOCAL_WORM_DIR.mkdir(parents=True, exist_ok=True)
    safe = trade_id.replace("/", "_")
    return _LOCAL_WORM_DIR / f"{safe}.json"


def _write_local(record: RationaleRecord) -> str:
    path = _local_path(record.trade_id)
    # Append-only semantic: refuse to overwrite an existing record.
    # This approximates WORM on filesystems that don't lock.
    if path.exists():
        raise FileExistsError(
            f"rationale for trade_id={record.trade_id} already exists "
            f"at {path} -- WORM local-dev semantic refuses overwrite"
        )
    path.write_text(json.dumps(asdict(record), indent=2) + "\n",
                     encoding="utf-8")
    return str(path)


def _read_local(trade_id: str) -> dict:
    path = _local_path(trade_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no rationale for trade_id={trade_id} at {path}"
        )
    return json_io.load_json_file(path)


def _write_gcs(record: RationaleRecord) -> str:
    bucket_name = os.environ["COMPLIANCE_WORM_BUCKET"]
    from google.cloud import storage    # type: ignore
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"rationales/{record.trade_id}.json"
    blob = bucket.blob(blob_name)
    if blob.exists():
        raise FileExistsError(
            f"rationale blob gs://{bucket_name}/{blob_name} already exists "
            "-- WORM refuses overwrite"
        )
    blob.upload_from_string(
        json.dumps(asdict(record), indent=2),
        content_type="application/json",
    )
    return f"gs://{bucket_name}/{blob_name}"


def _read_gcs(trade_id: str) -> dict:
    bucket_name = os.environ["COMPLIANCE_WORM_BUCKET"]
    from google.cloud import storage    # type: ignore
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"rationales/{trade_id}.json")
    if not blob.exists():
        raise FileNotFoundError(
            f"no rationale blob for trade_id={trade_id}"
        )
    return json_io.loads(blob.download_as_text())


def write_rationale(
    *,
    trade_id: str,
    system_id: str,
    agent_trace: list[dict],
    input_signals: dict,
    output_recommendation: str,
    confidence: float,
    approver_id: str,
    decision: str,
    reason_code: str,
    original_recommendation: str | None = None,
    created_at: str | None = None,
    approved_at: str | None = None,
    extras: dict | None = None,
) -> tuple[RationaleRecord, str]:
    """Write a rationale + return (record, location). Raises if any
    HITL field is missing. Raises if the trade_id already exists
    (WORM refuses overwrite)."""
    now = datetime.now(timezone.utc).isoformat()
    record = RationaleRecord(
        trade_id=trade_id,
        created_at=created_at or now,
        system_id=system_id,
        agent_trace=list(agent_trace),
        input_signals=dict(input_signals),
        output_recommendation=output_recommendation,
        confidence=float(confidence),
        approver_id=approver_id,
        approved_at=approved_at or now,
        decision=decision,
        reason_code=reason_code,
        original_recommendation=original_recommendation,
        extras=dict(extras or {}),
    )
    record.validate()
    if _backend() == "gcs":
        loc = _write_gcs(record)
    else:
        loc = _write_local(record)
    logger.info("compliance: wrote rationale for trade_id=%s to %s",
                 trade_id, loc)
    return record, loc


def fetch_rationale(trade_id: str) -> dict:
    """Read the rationale back; used by the audit + any SEC 17a-4(f)
    reconstruction request."""
    if _backend() == "gcs":
        return _read_gcs(trade_id)
    return _read_local(trade_id)


def retention_policy() -> dict:
    """Describe the retention policy in a stable, auditable shape."""
    return {
        "backend": _backend(),
        "bucket": os.getenv("COMPLIANCE_WORM_BUCKET"),
        "retention_years_policy": RETENTION_YEARS_POLICY,
        "retention_years_minimum": RETENTION_YEARS_MINIMUM,
        "retention_seconds_policy": RETENTION_SECONDS_POLICY,
        "citation": (
            "SEC 17a-4 requires 6 years for trade-order records "
            "(17a-3(a)(1)); masterplan criterion 'worm_retention_3y' "
            "is this project's internal floor, satisfied by the 6y policy."
        ),
    }


__all__ = [
    "RETENTION_YEARS_POLICY",
    "RETENTION_YEARS_MINIMUM",
    "RETENTION_SECONDS_POLICY",
    "RationaleRecord",
    "fetch_rationale",
    "retention_policy",
    "write_rationale",
]
