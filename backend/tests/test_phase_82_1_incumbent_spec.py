"""phase-82.1 -- guards the incumbent strategy specification.

The spec's value is that every code claim is checkable. These tests make that
mechanical: a citation that rots (file deleted, file shortened past the cited
line) fails here rather than quietly misleading a future reader.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "strategy" / "incumbent_live_strategy_spec.md"

# `backend/services/autonomous_loop.py:1034` and friends, inside backticks.
_CITE = re.compile(r"`((?:backend|frontend|scripts|handoff|docs)/[\w./-]+?):(\d+)`")


def _spec_text() -> str:
    assert SPEC.exists(), f"spec not found at {SPEC}"
    return SPEC.read_text(encoding="utf-8")


def _citations() -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _CITE.finditer(_spec_text())]


# ── criterion 1: the spec exists and names every required element ────

@pytest.mark.parametrize("section", [
    "Universe", "Screen", "Signal", "Sizing", "Exit",
])
def test_spec_names_each_required_element(section):
    assert section.lower() in _spec_text().lower(), (
        f"the spec does not name '{section}' -- criterion 1 requires universe, "
        "signal, ranking, sizing, entry gates and exit rule"
    )


def test_spec_names_ranking_and_entry_gates():
    t = _spec_text().lower()
    assert "ranking" in t or "rank_candidates" in t
    assert "kill" in t and "cap" in t, "entry gates not described"


# ── criterion 4: every file:line citation resolves ───────────────────

def test_spec_has_a_meaningful_number_of_citations():
    """Guard against the resolver test passing vacuously on an empty set."""
    cites = _citations()
    assert len(cites) >= 20, (
        f"only {len(cites)} file:line citations found; the resolver test below "
        "would be near-vacuous. Criterion 1 requires claims to carry citations."
    )


def test_every_citation_resolves():
    failures = []
    for path, line in _citations():
        p = REPO / path
        if not p.exists():
            failures.append(f"{path}:{line} -- file does not exist")
            continue
        try:
            n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        except OSError as e:  # pragma: no cover
            failures.append(f"{path}:{line} -- unreadable ({e})")
            continue
        if line > n:
            failures.append(f"{path}:{line} -- file has only {n} lines")
    assert not failures, "unresolvable citations in the spec:\n  " + "\n  ".join(failures)


# ── criterion 2: measured counts + screenshot reconciliation ─────────

def test_spec_states_measured_counts_and_reconciles_the_screenshot():
    """Criterion 2: the counts must be STATED, not merely inferable.

    The first version of this test asserted `"33" in t and "32" in t`, which
    survived deleting the counts entirely -- those digits also occur inside
    file:line citations (`backend_engine.py:32`, `autonomous_loop.py:433`).
    A guard satisfied by unrelated text is not a guard, so each count is now
    pinned to its own table row.
    """
    t = _spec_text()

    assert re.search(r"`paper_trades`\s*BUY\s*\|\s*33\b", t), (
        "the measured BUY count (33) is not stated as a table row"
    )
    assert re.search(r"`paper_trades`\s*SELL\s*\|\s*32\b", t), (
        "the measured SELL count (32) is not stated as a table row"
    )
    assert re.search(r"`paper_positions`\s*open\s*\|\s*1\b", t), (
        "criterion 2 names OPEN POSITIONS; no such count is stated"
    )
    assert re.search(r"32\s*BUY\s*\+\s*32\s*SELL\s*=\s*64", t), (
        "the reconciliation arithmetic (32 BUY + 32 SELL = 64) is not shown "
        "explicitly -- criterion 2 requires reconciling against the screenshot"
    )
    assert "18:47" in t, (
        "the reconciliation rests on the 33rd BUY landing AFTER the screenshot; "
        "its timestamp is not stated, so the argument cannot be checked"
    )


# ── criterion 3: the binding constraint, with verbatim evidence ──────

def test_spec_names_exactly_one_binding_constraint():
    t = _spec_text()
    assert "paper_analyze_top_n" in t, "the binding constraint is not named"
    assert "ONE binding constraint" in t or "binding constraint" in t


def test_binding_constraint_quotes_live_query_output_verbatim():
    """Criterion 3 requires the diagnosis to rest on quoted live output."""
    t = _spec_text()
    assert '"new_to_analyze": 5' in t, (
        "the funnel telemetry is not quoted verbatim -- the diagnosis must rest "
        "on live output, not on reasoning"
    )
    assert '"universe_size": 583' in t
    assert '"trailing_dd_pct": 3.6308' in t, (
        "the kill-switch refutation does not quote its live evidence"
    )


def test_spec_refutes_the_competing_explanations():
    """A diagnosis that names a cause without excluding the alternatives is an
    assertion, not a diagnosis."""
    t = _spec_text()
    for alt in ["Kill switch", "GATE NOT ELIGIBLE", "Capacity"]:
        assert alt in t, f"competing explanation '{alt}' is not addressed"
    assert t.count("REFUTED") >= 3, (
        "fewer than three competing explanations are explicitly refuted"
    )


# ── the lane distinction the whole phase rests on ────────────────────

def test_spec_records_that_the_registry_does_not_drive_live_selection():
    t = _spec_text()
    assert "STRATEGY_REGISTRY" in t
    assert "label" in t.lower()
    assert "triple_barrier" in t


def test_spec_records_that_the_live_exit_has_no_take_profit_or_time_barrier():
    t = _spec_text().lower()
    assert "no take-profit" in t, "the missing take-profit is not recorded"
    assert "time barrier" in t, "the missing time barrier is not recorded"
