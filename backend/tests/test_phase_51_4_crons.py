"""phase-51.4: cron repairs.

Bug A -- autoresearch graceful preflight: skip cleanly (return a message) when the
configured EMBEDDING backend module is absent, instead of crashing nightly.
Bug B -- weekly_data_integrity: construct BigQueryClient(settings) + use the real
google client to return a populated {table_id: row_count} dict (was {}).

$0, no network: importlib.find_spec + the BQ client are monkeypatched.
"""
import importlib.util
import pathlib

import pytest

# run_memo.py lives under scripts/ (not an importable package) -> load by path.
_RM_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "autoresearch" / "run_memo.py"
_spec = importlib.util.spec_from_file_location("run_memo_under_test", _RM_PATH)
run_memo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_memo)


# ---- Bug A: embedding preflight ----

def test_preflight_skips_when_backend_absent(monkeypatch):
    monkeypatch.setenv("EMBEDDING", "huggingface:BAAI/bge-small-en-v1.5")
    monkeypatch.setattr("importlib.util.find_spec", lambda m: None)
    msg = run_memo._embedding_preflight()
    assert msg is not None
    assert "langchain_huggingface" in msg
    assert "pip install" in msg and "sentence-transformers" in msg


def test_preflight_proceeds_when_backend_present(monkeypatch):
    monkeypatch.setenv("EMBEDDING", "huggingface:BAAI/bge-small-en-v1.5")
    monkeypatch.setattr("importlib.util.find_spec", lambda m: object())  # truthy = installed
    assert run_memo._embedding_preflight() is None


def test_preflight_proceeds_for_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING", "somethingelse:x")
    assert run_memo._embedding_preflight() is None  # no known module to guard


# ---- Bug B: weekly_data_integrity __TABLES__ row counts ----

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def result(self, **kw):
        return self._rows


class _FakeGoogleClient:
    def query(self, sql):
        assert "__TABLES__" in sql
        return _FakeResult([
            {"table_id": "signals", "row_count": 100},
            {"table_id": "prices", "row_count": 50},
        ])


class _FakeBQ:
    last_args = None

    def __init__(self, *args, **kwargs):
        _FakeBQ.last_args = args
        self.client = _FakeGoogleClient()


def test_weekly_data_integrity_returns_populated_dict(monkeypatch):
    monkeypatch.setattr("backend.db.bigquery_client.BigQueryClient", _FakeBQ)
    from backend.slack_bot.jobs import weekly_data_integrity as wdi
    out = wdi._default_fetch_counts()
    assert out == {"signals": 100, "prices": 50}          # populated, not {} (the bug)
    assert _FakeBQ.last_args and len(_FakeBQ.last_args) >= 1  # constructed WITH settings (not no-args)
