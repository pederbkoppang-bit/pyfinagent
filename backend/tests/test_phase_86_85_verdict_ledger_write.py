"""phase-86.85 -- tests for the verdict-ledger WRITER.

**Why this file exists, stated plainly.** `contract_86.85.md` §6.3 promised
`backend/tests/test_phase_86_85_verdict_ledger_write.py`. The first implementation
shipped the checks as a `--self-test` subcommand inside the writer instead, and did
NOT disclose the substitution -- the cycle-2 Q/A flagged that as a scope-honesty
violation, correctly. This file discharges the promise.

It is NOT a duplicate of `--self-test`. The two have different jobs:

* `--self-test` is the **mutation-matrix target**: one process, one exit code, so
  `mutation_matrix_86_85.py` can score a cell KILLED/SURVIVED. It must stay
  self-contained and dependency-free.
* This file is the **regression suite**: it runs under pytest with the rest of the
  backend tests, so a future edit that breaks the ledger is caught by the normal
  suite rather than only by someone remembering to run the matrix.

Every test here asserts a property that a *mutation cell* also covers, so the two
cannot silently diverge -- the cell names are given in each docstring.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER = REPO_ROOT / "scripts" / "qa" / "verdict_ledger_write.py"


def _load():
    spec = importlib.util.spec_from_file_location("verdict_ledger_write", WRITER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vlw = _load()


@pytest.fixture()
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger.jsonl"


# --------------------------------------------------------------------------
# dedup key  (matrix M1, M9)
# --------------------------------------------------------------------------

def test_duplicate_step_and_run_id_is_refused(ledger: Path):
    """M1: a re-transcription of the SAME run must not append a second row."""
    vlw.append_row(vlw.build_row("1.1", "PASS", run_id="wf_a", cycle="1"), ledger)
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.append_row(vlw.build_row("1.1", "FAIL", run_id="wf_a", cycle="1"), ledger)
    assert exc.value.code == vlw.EXIT_DUPLICATE
    assert len(vlw.read_rows(ledger)) == 1, "the refused row must not have appended"


def test_retry_with_a_new_run_id_is_a_new_row(ledger: Path):
    """A retried spawn is a genuinely new attempt, not a duplicate."""
    vlw.append_row(vlw.build_row("1.1", "CONDITIONAL", run_id="wf_a"), ledger)
    vlw.append_row(vlw.build_row("1.1", "CONDITIONAL", run_id="wf_b"), ledger)
    assert len(vlw.read_rows(ledger)) == 2


def test_dedup_key_includes_step_id(ledger: Path):
    """M9: the same run_id under a DIFFERENT step must not collide.

    Dropping step_id makes the key global, so a legitimate row for a second step
    is refused and LOST -- an under-count, which fails OPEN.
    """
    vlw.append_row(vlw.build_row("1.1", "PASS", run_id="wf_same"), ledger)
    vlw.append_row(vlw.build_row("2.2", "PASS", run_id="wf_same"), ledger)
    assert len(vlw.read_rows(ledger)) == 2


def test_row_with_no_run_id_or_cycle_is_refused():
    """M3: an unkeyed row could never be deduplicated, so it is not written."""
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.build_row("1.1", "PASS")
    assert exc.value.code == vlw.EXIT_INVALID


# --------------------------------------------------------------------------
# vocabulary + input validation  (matrix M2, M10)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["PASS", "CONDITIONAL", "FAIL", "NO_VERDICT"])
def test_every_valid_verdict_round_trips(ledger: Path, verdict: str):
    vlw.append_row(vlw.build_row("3.1", verdict, run_id=f"wf_{verdict}"), ledger)
    assert vlw.emit_sequence("3.1", ledger) == [verdict]


@pytest.mark.parametrize("bad", ["MOSTLY_FINE", "pass!", "", "COND", "OK"])
def test_unknown_verdict_is_rejected_not_coerced(bad: str):
    """M2: coercion is how a non-PASS would silently become something else."""
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.build_row("3.1", bad, run_id="wf_x")
    assert exc.value.code == vlw.EXIT_INVALID


def test_empty_step_id_is_rejected():
    """M10: an unattributable row must not be written."""
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.build_row("   ", "PASS", run_id="wf_x")
    assert exc.value.code == vlw.EXIT_INVALID


# --------------------------------------------------------------------------
# ordering -- the load-bearing contract  (matrix M6)
# --------------------------------------------------------------------------

def test_sequence_is_oldest_to_newest_with_an_order_sensitive_fixture(ledger: Path):
    """M6: reversal must be observable.

    The fixture MUST use distinct verdicts. A previous revision asserted against
    ['CONDITIONAL']*3 -- a palindrome -- and a mutant returning out[::-1] survived
    it. Reversing [PASS,C,C] to [C,C,PASS] takes enforceEscalation from
    n=2/auto_fail=true to n=0/auto_fail=false, silently disarming the escalation.
    """
    for i, v in enumerate(("PASS", "CONDITIONAL", "FAIL"), start=1):
        vlw.append_row(vlw.build_row("4.1", v, run_id=f"wf_{i}"), ledger)
    seq = vlw.emit_sequence("4.1", ledger)
    assert seq != list(reversed(seq)), "fixture must not be palindromic (anti-vacuity)"
    assert seq == ["PASS", "CONDITIONAL", "FAIL"]


def test_sequence_filters_by_step(ledger: Path):
    vlw.append_row(vlw.build_row("4.1", "PASS", run_id="wf_1"), ledger)
    vlw.append_row(vlw.build_row("4.2", "FAIL", run_id="wf_2"), ledger)
    assert vlw.emit_sequence("4.1", ledger) == ["PASS"]


# --------------------------------------------------------------------------
# fail-LOUD -- silence is the one thing this writer may never do  (M4, M7, M8)
# --------------------------------------------------------------------------

def test_corrupt_ledger_line_is_loud(tmp_path: Path):
    """M4: a corrupt file must never read as 'no verdicts'."""
    bad = tmp_path / "corrupt.jsonl"
    bad.write_text('{"step_id": "5.1"\n')
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.read_rows(bad)
    assert exc.value.code == vlw.EXIT_IO


def test_out_of_vocabulary_token_is_loud_not_filtered(tmp_path: Path):
    """M7: a filtered sequence is SHORTER than the truth, so it fails OPEN.

    It would also bypass the consumer's own fail-closed 'unparseable' branch.
    """
    p = tmp_path / "oov.jsonl"
    p.write_text(
        '{"step_id":"6.1","verdict":"CONDITIONAL","run_id":"wf_1"}\n'
        '{"step_id":"6.1","verdict":"COND","run_id":"wf_2"}\n'
    )
    with pytest.raises(vlw.LedgerError) as exc:
        vlw.emit_sequence("6.1", p)
    assert exc.value.code == vlw.EXIT_IO


def test_io_failure_is_loud_never_a_silent_success(tmp_path: Path):
    """M8: with this guard reverted the writer prints the row, exits 0 and
    writes NOTHING -- manufacturing the exact absent-row state the reader is
    built to refuse."""
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        with pytest.raises(vlw.LedgerError) as exc:
            vlw.append_row(vlw.build_row("7.1", "PASS", run_id="wf_io"), ro / "l.jsonl")
        assert exc.value.code == vlw.EXIT_IO
        assert not (ro / "l.jsonl").exists()
    finally:
        ro.chmod(0o700)


# --------------------------------------------------------------------------
# event time vs write time  (matrix M5)
# --------------------------------------------------------------------------

def test_event_time_and_write_time_are_separate_fields():
    """M5: collapsing them lets a backfill masquerade as live history."""
    row = vlw.build_row("8.1", "PASS", run_id="wf_1", event_date="2026-01-02")
    assert row["date"] == "2026-01-02"
    assert row["recorded_at"].startswith("20")
    assert row["date"] != row["recorded_at"]


# --------------------------------------------------------------------------
# verdict semantics are UNCHANGED (criterion 7)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["CONDITIONAL", "FAIL", "NO_VERDICT"])
def test_a_non_pass_never_becomes_a_pass(ledger: Path, verdict: str):
    """The writer records; it never transforms."""
    vlw.append_row(vlw.build_row("9.1", verdict, run_id=f"wf_{verdict}"), ledger)
    assert vlw.emit_sequence("9.1", ledger) == [verdict]
    assert "PASS" not in vlw.emit_sequence("9.1", ledger) or verdict == "PASS"


# --------------------------------------------------------------------------
# cross-process persistence (criterion 3) -- a real second interpreter
# --------------------------------------------------------------------------

def test_written_in_one_process_and_read_back_in_another(ledger: Path):
    """The Layer-3 per-step loop runs across sessions, so in-memory is not enough."""
    w = subprocess.run(
        [sys.executable, str(WRITER), "--ledger", str(ledger), "--step", "10.1",
         "--verdict", "CONDITIONAL", "--run-id", "wf_x", "--cycle", "1"],
        capture_output=True, text=True,
    )
    assert w.returncode == 0, w.stderr
    r = subprocess.run(
        [sys.executable, str(WRITER), "--ledger", str(ledger),
         "--emit-sequence", "--step", "10.1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["CONDITIONAL"]


def test_self_test_subcommand_is_green():
    """The mutation matrix scores cells off this exit code, so it must stay green."""
    r = subprocess.run([sys.executable, str(WRITER), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELF-TEST PASSED" in r.stdout
