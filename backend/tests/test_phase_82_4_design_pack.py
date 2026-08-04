"""phase-82.4 -- guards the design pack.

The pack's value is that every claim is checkable: citations that resolve,
numbers that reproduce from the run artifacts, and caveats that are actually
present rather than assumed.
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "docs" / "strategy" / "phase82_design_pack.md"
RESULTS = REPO / "backend" / "backtest" / "experiments" / "results"
MASTERPLAN = REPO / ".claude" / "masterplan.json"

# Q/A cycle-3: the first version required a `backend|frontend|scripts|handoff|
# docs/` PREFIX, so it saw 12 citations while 13 existed -- and the one it could
# not see (a bare `backtest_engine.py:665`) was the only one that did not
# resolve. A guard whose population is chosen by the author cannot certify
# "every". This matches ANY token ending in `.py`/`.md`/`.json`/`.tsv` + :NNN,
# prefixed or not, so a bare filename is caught and fails loudly.
_CITE = re.compile(r"`([\w./-]+\.(?:py|md|json|tsv|js|ts|tsx|sh)):(\d+)`")


def _pack() -> str:
    assert PACK.exists(), f"design pack not found at {PACK}"
    return PACK.read_text(encoding="utf-8")


# ── criterion 1: a section per strategy + a ranked recommendation ────

@pytest.mark.parametrize("strategy", [
    "triple_barrier", "stretch_regime", "qarp", "reversion_sigma",
])
def test_pack_covers_every_strategy(strategy):
    assert strategy in _pack(), f"{strategy} absent from the design pack"


def test_pack_states_its_ranking_criteria():
    """A recommendation without a stated rule is an opinion."""
    t = _pack()
    for token in ["DSR", "PBO", "turnover", "Pareto", "lexicographic"]:
        assert token.lower() in t.lower(), f"ranking criterion '{token}' not stated"
    assert "PRE-REGISTERED" in t, "the pack does not record that the rule predates the numbers"


def test_pack_contains_a_recommendation_section():
    assert re.search(r"##\s*6\.\s*Ranked recommendation", _pack())


def test_pack_forbids_a_weighted_composite():
    """The rule exists to keep the DSR-vs-PBO conflict visible."""
    t = _pack()
    assert "composite" in t.lower()
    assert re.search(r"[Nn]o weighted composite", t)


# ── criterion 2: every file:line citation resolves ───────────────────

def test_pack_has_a_meaningful_number_of_citations():
    """Guards the resolver below from passing vacuously."""
    n = len(_CITE.findall(_pack()))
    assert n >= 8, f"only {n} file:line citations; the resolver test would be near-vacuous"


def test_citation_regex_recall_matches_an_independent_count():
    """Q/A cycle-3 recall test: the guard must SEE every citation in the pack,
    not a subset it defined. Counted independently of _CITE -- any backticked
    token containing a dot-extension followed by :NNN."""
    text = _pack()
    independent = set(re.findall(r"`([^`\s]+\.[a-z]{1,4}:\d+)`", text))
    seen = {f"{p}:{n}" for p, n in _CITE.findall(text)}
    missed = independent - seen
    assert not missed, (
        f"the citation regex cannot see {sorted(missed)} -- its population is "
        "narrower than the pack's, so it cannot certify 'every citation resolves'"
    )


def test_every_citation_resolves():
    failures = []
    for path, line in _CITE.findall(_pack()):
        p = REPO / path
        if not p.exists():
            failures.append(f"{path}:{line} -- file does not exist")
            continue
        n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        if int(line) > n:
            failures.append(f"{path}:{line} -- file has only {n} lines")
    assert not failures, "unresolvable citations:\n  " + "\n  ".join(failures)


# ── criterion 3: the endogeneity caveat ──────────────────────────────

def test_pack_records_the_endogeneity_caveat():
    """Holding period is an OUTCOME: a trade stopped out on day 1 has a short
    hold by construction, so 'held longer => won more' is tautological and
    cannot support an action."""
    t = _pack()
    low = t.lower()
    assert "tautolog" in low, "the tautology is not named"
    assert "by construction" in low, "the 'by construction' mechanism is not stated"
    assert "holding period" in low
    assert re.search(r"outcome,? not a treatment", low), (
        "the pack does not state that holding period is an outcome rather than "
        "a treatment variable"
    )
    # Q/A cycle-2: a stub retaining only the grepped tokens SURVIVED this test.
    # Pin the CAUSAL sentence -- the thing that makes the caveat mean something
    # -- and a minimum length, so substance cannot be stripped while keywords
    # remain.
    # Whitespace-insensitive: the pack hard-wraps prose, so the phrase spans a
    # newline. Matching the raw string would fail on formatting, not substance.
    flat = re.sub(r"\s+", " ", low)
    assert "the stop caused the short hold" in flat, (
        "the caveat names the tautology but never states the causal direction; "
        "a keyword-only stub would satisfy the assertions above"
    )
    # Bounded by the NEXT numbered caveat, found by pattern rather than by a
    # hardcoded number -- inserting a caveat above renumbers the rest, and a
    # literal "9. **PBO" boundary breaks on a renumber rather than on substance.
    body = t[t.index("HOLDING PERIOD IS AN OUTCOME"):]
    nxt = re.search(r"\n\d+\. \*\*", body[1:])
    body = body[:nxt.start() + 1] if nxt else body
    assert len(body) > 900, (
        f"caveat 8 is only {len(body)} chars -- too short to carry the "
        "mechanism, the live-book counter-example and the actionable counterpart"
    )


def test_pack_gives_the_non_tautological_counterpart():
    """Naming the trap is half the job; the pack must also say what the same
    data DOES support -- stop placement, not hold length."""
    t = _pack()
    assert "0.5pp" in t or "worst point" in t, "the stop-placement finding is absent"
    assert "sigma" in t.lower() or "σ" in t


# ── criterion 4: one queued step per recommended action ──────────────

def test_every_recommended_action_has_a_queued_masterplan_step():
    t = _pack()
    m = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    phase = [p for p in m["phases"] if p["id"] == "phase-82"][0]
    by_id = {s["id"]: s for s in phase["steps"]}

    cited = sorted(set(re.findall(r"\*\*(82\.\d+)\*\*", t)) |
                   set(re.findall(r"\|\s*\*\*(82\.\d+)\*\*\s*\|", t)))
    assert cited, "the recommendation cites no masterplan steps"

    for sid in cited:
        assert sid in by_id, f"pack cites {sid}, which is not in the masterplan"
        step = by_id[sid]
        crit = (step.get("verification") or {}).get("criteria") or []
        assert crit, f"{sid} carries no verification criteria"
        assert (step.get("verification") or {}).get("command"), f"{sid} has no verification command"


def test_the_recommendation_table_names_priorities():
    assert re.search(r"\|\s*\*\*?P0\*\*?\s*\|", _pack()) or "| P0 |" in _pack()


# ── the numbers must reproduce from the artifacts ────────────────────

def test_headline_numbers_reproduce_from_the_run_artifacts():
    """Transcribed, not retyped: every headline value in the pack must be
    derivable from the result JSON it claims to come from."""
    hits = sorted(RESULTS.glob("*_phase_82_3_full_sample_3strat.json"))
    assert hits, "pass A artifact missing"
    d = json.loads(hits[-1].read_text(encoding="utf-8"))
    t = _pack()

    missing = []
    for strat, v in d["per_strategy"].items():
        runs = v["runs"]
        dsr = st.median([r["dsr"] for r in runs])
        ret = st.median([r.get("net_of_cost_return_pct") or 0 for r in runs])
        for label, val in (("DSR", f"{dsr:.4f}"), ("PBO", f"{v['pbo']:.4f}"),
                           ("return", f"{ret:.2f}")):
            if val not in t:
                missing.append(f"{strat} {label}={val}")
    assert not missing, "pack values that do not reproduce:\n  " + "\n  ".join(missing)


def test_pack_records_the_gate_outcome_as_zero_passers():
    t = _pack()
    assert "0 of 3" in t or "0/3" in t, "the gate outcome is not stated"
    assert "no winner" in t.lower(), "the pack does not state that no winner was declared"
