"""phase-6.5.1 tests for the intel-schema migration.

Asserts the DDL constants in `scripts/migrations/phase_6_5_intel_schema.py`
declare every required column for every intel table, preserve
idempotency (`IF NOT EXISTS`) and partition+cluster discipline, and
that `main(dry_run=True)` returns 0 without importing google-cloud-bigquery.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.migrations import phase_6_5_intel_schema as mig


_INTEL_SOURCES_FIELDS = {
    "source_id", "source_name", "source_type", "kill_switch",
    "rate_limit_per_day", "last_scanned_at", "created_at", "updated_at",
    "metadata",
}
_INTEL_DOCUMENTS_FIELDS = {
    "doc_id", "source_id", "source_type", "doc_type", "published_at",
    "ingested_at", "title", "authors", "url", "canonical_url",
    "content_hash", "raw_text", "language", "raw_payload",
}
_INTEL_CHUNKS_FIELDS = {
    "chunk_id", "doc_id", "chunk_index", "chunk_text", "embedding",
    "embedding_model", "tokens", "ingested_at",
}
_INTEL_NOVELTY_SCORES_FIELDS = {
    "chunk_id", "scorer_model", "scorer_version", "scored_at",
    "novelty_score", "nearest_neighbor_chunk_id",
    "nearest_neighbor_distance", "latency_ms", "cost_usd",
}
_INTEL_PROMPT_PATCHES_FIELDS = {
    "patch_id", "chunk_id", "patch_type", "patch_text", "rationale",
    "status", "created_at", "reviewed_at", "reviewed_by", "applied_at",
    "metadata",
}


_EXPECTED = {
    "intel_sources": (mig.DDL_INTEL_SOURCES, _INTEL_SOURCES_FIELDS),
    "intel_documents": (mig.DDL_INTEL_DOCUMENTS, _INTEL_DOCUMENTS_FIELDS),
    "intel_chunks": (mig.DDL_INTEL_CHUNKS, _INTEL_CHUNKS_FIELDS),
    "intel_novelty_scores": (mig.DDL_INTEL_NOVELTY_SCORES, _INTEL_NOVELTY_SCORES_FIELDS),
    "intel_prompt_patches": (mig.DDL_INTEL_PROMPT_PATCHES, _INTEL_PROMPT_PATCHES_FIELDS),
}


def _extract_columns(ddl: str) -> set[str]:
    """Return the set of column names declared between the column-list parens.

    The column list is the first `(...)` after `CREATE TABLE IF NOT EXISTS <name>`.
    Later parens (e.g. `OPTIONS (description = "...")`) are ignored.
    """
    start = ddl.find("(")
    assert start != -1, ddl
    # The column-list closes at the first `)` that is followed (after whitespace)
    # by PARTITION / CLUSTER / OPTIONS / end-of-DDL — none of which are inside
    # the column list itself.
    depth = 0
    end = -1
    for i in range(start, len(ddl)):
        c = ddl[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end != -1, ddl
    body = ddl[start + 1:end]

    cols: set[str] = set()
    line_pat = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+")
    for raw in body.split("\n"):
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        m = line_pat.match(line)
        if not m:
            continue
        name = m.group(1)
        if name.upper() in {"PARTITION", "CLUSTER", "OPTIONS"}:
            continue
        cols.add(name)
    return cols


def test_all_tables_declared_in_ddls_constant():
    declared = {t for t, _ in mig.DDLS}
    assert declared == set(_EXPECTED), declared


def test_each_ddl_has_expected_columns():
    for table, (ddl, expected) in _EXPECTED.items():
        got = _extract_columns(ddl)
        missing = expected - got
        extra = got - expected
        assert not missing, f"{table} missing columns: {missing}"
        assert not extra, f"{table} unexpected columns: {extra}"


def test_each_ddl_is_idempotent():
    for table, (ddl, _) in _EXPECTED.items():
        assert "CREATE TABLE IF NOT EXISTS" in ddl, f"{table} not idempotent"


def test_each_ddl_has_partition_and_cluster():
    for table, (ddl, _) in _EXPECTED.items():
        assert "PARTITION BY DATE(" in ddl, f"{table} missing DATE partition"
        assert "CLUSTER BY" in ddl, f"{table} missing CLUSTER BY"


def test_dry_run_returns_zero_without_bq_import(monkeypatch):
    bq_before = sys.modules.get("google.cloud.bigquery")
    try:
        sys.modules.pop("google.cloud.bigquery", None)
        rc = mig.main(dry_run=True)
        assert rc == 0
        assert "google.cloud.bigquery" not in sys.modules, (
            "dry-run must not import google-cloud-bigquery"
        )
    finally:
        if bq_before is not None:
            sys.modules["google.cloud.bigquery"] = bq_before


def test_embedding_column_is_array_float64():
    assert "embedding ARRAY<FLOAT64>" in mig.DDL_INTEL_CHUNKS


def test_jsonl_metadata_columns_are_json_type():
    # intel_sources, intel_documents, intel_prompt_patches carry JSON metadata
    for table in ("intel_sources", "intel_documents", "intel_prompt_patches"):
        ddl = _EXPECTED[table][0]
        assert "metadata JSON" in ddl or "raw_payload JSON" in ddl, table


def test_no_unicode_in_ddls():
    # ASCII-only discipline per .claude/rules/security.md
    for table, (ddl, _) in _EXPECTED.items():
        ddl.encode("ascii")  # raises UnicodeEncodeError if non-ASCII
