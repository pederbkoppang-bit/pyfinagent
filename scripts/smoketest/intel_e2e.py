"""phase-6.5.9 end-to-end smoketest for the Path-D intel pipeline.

Composes the 4 shipped phase-6.5 modules in 5 stages:

  S1 load_registry   -- YAML fixture -> list[SourceRow] (kill-switch filtered)
  S2 scan_sources    -- BaseScanner(dry_run=True) per active source
                        Each distinct source_type must emit >= 1 candidate.
  S3 score_novelty   -- _stub_embed injected; novelty_score(cand, []) -> 1.0
                        Every candidate scored in [0, 1].
  S4 enqueue_patches -- prompt_patch_queue.enqueue_patch with captive _insert.
  S5 digest_and_audit -- JSON summary appended to handoff/audit/intel_e2e.jsonl.

No live network, no live BQ, no live Voyage/Gemini. The `--fixtures` flag is
structural parity with other smoketests; this script's body is inherently
fixture-only.

Usage:
    python scripts/smoketest/intel_e2e.py --fixtures
    python scripts/smoketest/intel_e2e.py            # same behavior

Exit code 0 when `overall_ok`; 1 only on fatal exception. ASCII-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.intel import novelty_client as nc  # noqa: E402
from backend.intel import prompt_patch_queue as ppq  # noqa: E402
from backend.intel.scanner import BaseScanner, DocumentCandidate  # noqa: E402
from backend.intel.source_registry import SourceRow, load_from_yaml  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "backend" / "tests" / "fixtures" / "intel_sources.yaml"
_AUDIT_DIR = _REPO_ROOT / "handoff" / "audit"
_AUDIT_PATH = _AUDIT_DIR / "intel_e2e.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s1_load_registry() -> dict[str, Any]:
    try:
        rows = load_from_yaml(_FIXTURE_PATH)
        active = [r for r in rows if not r.kill_switch]
        return {
            "name": "load_registry",
            "ok": len(active) >= 1,
            "total_sources": len(rows),
            "active_count": len(active),
            "active_ids": [r.source_id for r in active],
            "active_sources": active,
        }
    except Exception as exc:
        return {
            "name": "load_registry",
            "ok": False,
            "error": repr(exc),
            "active_sources": [],
        }


def s2_scan_sources(active_sources: list[SourceRow]) -> dict[str, Any]:
    try:
        by_family: dict[str, list[DocumentCandidate]] = {}
        for src in active_sources:
            cands = BaseScanner(src).scan(dry_run=True)
            by_family.setdefault(src.source_type, []).extend(cands)
        per_family_ok = all(len(v) >= 1 for v in by_family.values()) and bool(by_family)
        flat = [c for v in by_family.values() for c in v]
        return {
            "name": "scan_sources",
            "ok": per_family_ok,
            "families": list(by_family.keys()),
            "per_family_counts": {k: len(v) for k, v in by_family.items()},
            "total_candidates": len(flat),
            "candidates": flat,
        }
    except Exception as exc:
        return {
            "name": "scan_sources",
            "ok": False,
            "error": repr(exc),
            "candidates": [],
        }


def s3_score_novelty(candidates: list[DocumentCandidate]) -> dict[str, Any]:
    try:
        scored: list[dict[str, Any]] = []
        for cand in candidates:
            text = cand.get("raw_text") or cand.get("title") or ""
            score, nn = nc.novelty_score(text, [], embedder=nc._stub_embed)
            scored.append(
                {
                    "doc_id": cand.get("doc_id"),
                    "source_id": cand.get("source_id"),
                    "novelty_score": score,
                    "nearest_neighbor_index": nn,
                }
            )
        all_valid = all(0.0 <= r["novelty_score"] <= 1.0 for r in scored)
        return {
            "name": "score_novelty",
            "ok": bool(scored) and all_valid,
            "scored_count": len(scored),
            "scorer": "stub:sha256_tiled_1024",
            "scores": scored,
        }
    except Exception as exc:
        return {"name": "score_novelty", "ok": False, "error": repr(exc), "scores": []}


def s4_enqueue_patches(
    candidates: list[DocumentCandidate], scores: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        fake_store: list[dict[str, Any]] = []

        def _captive_insert(rows, *, project=None, dataset=None):
            before = len(fake_store)
            for r in rows:
                pid = r["patch_id"]
                prior = [x for x in fake_store if x["patch_id"] == pid]
                latest = (
                    sorted(prior, key=lambda x: x["created_at"])[-1]["status"]
                    if prior
                    else None
                )
                if latest == "pending":
                    continue
                fake_store.append(r)
            return len(fake_store) - before

        original_insert = ppq._insert
        ppq._insert = _captive_insert  # type: ignore[assignment]
        try:
            pids: list[str] = []
            for cand, s in zip(candidates, scores):
                pid = ppq.enqueue_patch(
                    "strategy_hint",
                    f"Novelty {s['novelty_score']:.3f} on doc {cand.get('doc_id')}",
                    chunk_id=None,
                    rationale=f"source={cand.get('source_id')}",
                )
                pids.append(pid)
        finally:
            ppq._insert = original_insert  # type: ignore[assignment]

        unique_pids = set(pids)
        return {
            "name": "enqueue_patches",
            "ok": all(isinstance(p, str) and len(p) == 16 for p in pids) and bool(pids),
            "enqueued_count": len(pids),
            "unique_pid_count": len(unique_pids),
            "sample_patch_ids": sorted(list(unique_pids))[:3],
        }
    except Exception as exc:
        return {
            "name": "enqueue_patches",
            "ok": False,
            "error": repr(exc),
            "enqueued_count": 0,
        }


def _write_audit(summary: dict[str, Any]) -> bool:
    try:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with _AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        return True
    except Exception:
        return False


def s5_digest_and_audit(
    stages: list[dict[str, Any]], overall_ok: bool
) -> dict[str, Any]:
    # Build the Path-D "digest" (JSON summary; replaces phase-6.5.8 Slack digest
    # per masterplan.json::phase-6.5.8.superseded_by = "6.5.9").
    summary = {
        "ts": _now_iso(),
        "overall_ok": overall_ok,
        "stages": [
            {k: v for k, v in st.items() if k not in ("candidates", "scores", "active_sources")}
            for st in stages
        ],
    }
    wrote = _write_audit(summary)
    return {
        "name": "digest_and_audit",
        "ok": bool(wrote),
        "audit_path": str(_AUDIT_PATH),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true", help="use test fixtures (default behavior)")
    ap.parse_args(argv or [])

    stages: list[dict[str, Any]] = []

    # S1
    s1 = s1_load_registry()
    stages.append(s1)
    active = s1.get("active_sources") or []

    # S2
    s2 = s2_scan_sources(active) if s1["ok"] else {"name": "scan_sources", "ok": False, "skipped": True, "candidates": []}
    stages.append(s2)
    candidates = s2.get("candidates") or []

    # S3
    s3 = s3_score_novelty(candidates) if s2["ok"] else {"name": "score_novelty", "ok": False, "skipped": True, "scores": []}
    stages.append(s3)
    scores = s3.get("scores") or []

    # S4
    s4 = s4_enqueue_patches(candidates, scores) if s3["ok"] else {"name": "enqueue_patches", "ok": False, "skipped": True}
    stages.append(s4)

    pre_overall_ok = all(st["ok"] for st in stages)

    # S5 always attempts to record; audit line is useful even on partial failure.
    s5 = s5_digest_and_audit(stages, pre_overall_ok)
    stages.append(s5)

    overall_ok = all(st["ok"] for st in stages)

    # Print the JSON summary so operators can eyeball it in CI logs.
    print(json.dumps(s5["summary"], indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
