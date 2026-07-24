"""phase-76.9 (P1, operator bug B1) -- both nightly launchd jobs failing.

CAUSE 1 (autoresearch): arxiv HTTP 429 hits gpt-researcher's unguarded
retrievers[0]-only planning path (upstream issue #1282) -> run_memo.py's
broad except -> ERROR memo + rc=1 -> run_nightly.sh pages. Fix: (a) move
arxiv off retrievers[0], (b) classify network-class exceptions and write
a WARN memo + rc=0 instead, keeping the ERROR/rc=1 path for real faults.

CAUSE 2 (ablation): the launchd job's raw `. backend/.env` dies on a
malformed backend/.env line (unbalanced-quote orphan fragment). Fix
already shipped in phase-75.11 (scripts/ops/run_ablation.sh); this step
proves it against a fixture that mirrors the real file's shape.

See handoff/current/contract.md (step 76.9) and
handoff/autoresearch/root_cause.md for the full narrative.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import stat
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# run_memo.py lives under scripts/ (not an importable package) -> load by
# path, matching the pattern in test_phase_51_4_crons.py.
_RM_PATH = REPO / "scripts" / "autoresearch" / "run_memo.py"
_spec = importlib.util.spec_from_file_location("run_memo_76_9_under_test", _RM_PATH)
run_memo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_memo)


# ─────────────────────────────────────────────────────────────────────
# CAUSE 1 -- network-weather classification + retriever order
# ─────────────────────────────────────────────────────────────────────


def _stub_args(topic: str = "test topic for phase 76.9") -> Namespace:
    return Namespace(topic=topic, topic_index=None)


def test_t_429_warn_exit0(monkeypatch, tmp_path):
    """A real arxiv.HTTPError, 429-shaped, must be tolerated: rc=0, a
    WARN memo written, NO ERROR memo. Mutation-killable: revert the
    _is_network_weather fall-through and this goes red (rc=1 + ERROR file).
    """
    import arxiv

    # Signature verified against .venv/lib/python3.14/site-packages/arxiv/
    # __init__.py:820 -- HTTPError(url: str, retry: int, status: int).
    err = arxiv.HTTPError(
        "https://export.arxiv.org/api/query?search_query=test", 3, 429,
    )

    async def _raise_429(topic: str) -> str:
        raise err

    monkeypatch.setattr(run_memo, "run_research", _raise_429)
    monkeypatch.setattr(run_memo, "MEMO_DIR", tmp_path)

    rc = asyncio.run(run_memo._main_async(_stub_args()))

    assert rc == 0

    warn_files = list(tmp_path.glob("*-WARN-topic*.md"))
    error_files = list(tmp_path.glob("*-ERROR-*"))
    assert len(warn_files) == 1, f"expected exactly one WARN memo, found {warn_files}"
    assert error_files == [], f"expected no ERROR memo, found {error_files}"

    body = warn_files[0].read_text(encoding="utf-8")
    assert "HTTPError" in body
    assert "-ERROR-" not in warn_files[0].name


def test_t_real_fault_exit1(monkeypatch, tmp_path):
    """A genuine (non-network) fault must still page: rc=1, ERROR memo
    written, no WARN memo. Locks the 75.11 paging seam.
    """
    async def _raise_boom(topic: str) -> str:
        raise ValueError("boom")

    monkeypatch.setattr(run_memo, "run_research", _raise_boom)
    monkeypatch.setattr(run_memo, "MEMO_DIR", tmp_path)

    rc = asyncio.run(run_memo._main_async(_stub_args()))

    assert rc == 1

    error_files = list(tmp_path.glob("*-ERROR-*"))
    warn_files = list(tmp_path.glob("*-WARN-*"))
    assert len(error_files) == 1, f"expected exactly one ERROR memo, found {error_files}"
    assert warn_files == [], f"expected no WARN memo, found {warn_files}"

    body = error_files[0].read_text(encoding="utf-8")
    assert "ValueError" in body
    assert "boom" in body


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out"),
        Exception("HTTP 503 Service Unavailable"),
        Exception("rate limit exceeded"),
    ],
)
def test_t_network_weather_classifier_matches_generic_network_errors(exc):
    """Direct unit coverage of _is_network_weather for the non-arxiv
    branches (connection/timeout classes + 429/503/rate-limit message
    tokens), independent of the end-to-end _main_async harness above.
    """
    assert run_memo._is_network_weather(exc) is True


def test_t_network_weather_classifier_rejects_real_faults():
    assert run_memo._is_network_weather(ValueError("boom")) is False
    assert run_memo._is_network_weather(KeyError("ANTHROPIC_API_KEY")) is False


def test_t_retriever_order():
    """RETRIEVER default must have semantic_scholar first and arxiv NOT
    first -- arxiv in retrievers[0] is the crash-prone PLANNING slot
    (gpt-researcher upstream issue #1282; researcher.py:62 uses
    retrievers[0] unguarded).

    env_defaults is a local dict built inside main() (which has
    sys.exit/argument-parsing/env-mutation side effects unsuitable for a
    unit test), so this is a source-level assertion on the literal
    RETRIEVER default string -- acceptable per contract Plan item 2 for
    the ordering check specifically. The behavioral network-tolerance
    path IS exercised end-to-end above (t_429_warn_exit0 /
    t_real_fault_exit1), so this test's narrower scope is limited to
    order, not behavior.
    """
    text = _RM_PATH.read_text(encoding="utf-8")
    assert '"RETRIEVER": "semantic_scholar,arxiv,duckduckgo"' in text

    retriever_list = "semantic_scholar,arxiv,duckduckgo".split(",")
    assert retriever_list[0] == "semantic_scholar"
    assert retriever_list[0] != "arxiv"
    assert "arxiv" in retriever_list  # still present -- deprioritized, not dropped

    # Old (broken) order must be gone.
    assert '"RETRIEVER": "arxiv,semantic_scholar,duckduckgo"' not in text


# ─────────────────────────────────────────────────────────────────────
# CAUSE 2 -- ablation fixture proof
# ─────────────────────────────────────────────────────────────────────


def _make_ablation_fixture(root: Path) -> None:
    """Build a throwaway repo tree that mirrors the shape run_ablation.sh
    expects: backend/.env (with a valid line, a comment, and the L80/L81-
    shaped orphan unbalanced-quote fragment), a .venv/bin stub that puts
    a `python` shim on PATH (the real system has no bare `python`, only
    `python3`), and a stub scripts/ablation/run_ablation.py.
    """
    (root / "backend").mkdir(parents=True, exist_ok=True)
    (root / "backend" / ".env").write_text(
        "VALID_KEY=hello\n"
        "# phase-61.1 (2026-06-12): operator tokens \"60.2 FLAG: ON\" / "
        "\"60.3 FLAG: ON\" / \"57.1 FLAG:\n"
        "  ON\"\n"
        "ANOTHER_KEY=world\n",
        encoding="utf-8",
    )

    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    python_shim = venv_bin / "python"
    python_shim.write_text(
        "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8",
    )
    python_shim.chmod(python_shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    activate = venv_bin / "activate"
    activate.write_text(
        f'#!/bin/sh\nexport PATH="{venv_bin}:$PATH"\n', encoding="utf-8",
    )
    activate.chmod(activate.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    ablation_dir = root / "scripts" / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "run_ablation.py").write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('stub ablation run -- phase-76.9 fixture')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )

    (root / "handoff" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "handoff" / "away_ops").mkdir(parents=True, exist_ok=True)


def test_t_ablation_fixture_survives_bad_env(tmp_path):
    fixture = tmp_path / "fixture_repo"
    _make_ablation_fixture(fixture)

    # --- Guard: the fixture must actually reproduce the raw-source EOF
    # failure (feedback_mutation_test_guards_and_fixtures -- a fixture
    # that can't represent the failure doesn't count). ---
    raw_source = subprocess.run(
        ["bash", "-c", f'. "{fixture / "backend" / ".env"}"'],
        capture_output=True, text=True, timeout=10,
    )
    assert raw_source.returncode != 0, (
        "fixture backend/.env did NOT reproduce the raw-source failure -- "
        "fixture is vacuous"
    )
    assert "unexpected EOF" in raw_source.stderr

    # --- Real assertion: the REAL scripts/ops/run_ablation.sh, pointed at
    # the fixture via SRE_OPS_REPO, must survive the malformed .env. ---
    run_ablation_sh = REPO / "scripts" / "ops" / "run_ablation.sh"
    env = dict(os.environ)
    env["SRE_OPS_REPO"] = str(fixture)

    result = subprocess.run(
        ["bash", str(run_ablation_sh)],
        capture_output=True, text=True, timeout=30, env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"run_ablation.sh failed against the fixture: rc={result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "unexpected EOF" not in combined

    log_text = (fixture / "handoff" / "logs" / "ablation.log").read_text(encoding="utf-8")
    assert "START ablation" in log_text
    assert "END ablation OK" in log_text
