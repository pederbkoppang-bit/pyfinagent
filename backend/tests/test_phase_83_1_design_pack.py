"""phase-83.1: design pack + pre-registration criteria tests.

Conventions copied from test_phase_82_6_bridge_design.py: anti-stub size
check, concrete symbols not headings, and HTML comments are STRIPPED before
any content matching (the 82.6 Q/A found a stub whose only content was the
required tokens inside <!-- --> and it passed every test).

The ordering guard (C5/C6) uses strict `<` on st_mtime_ns: an artifact
strictly OLDER than the ranking file fails; equal mtimes PASS (git checkout
stamps a fresh clone identically). No pytest.skip escapes.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
PACK = _REPO / "handoff/current/research_brief_phase83.md"
RANKING = _REPO / "backend/backtest/experiments/preregistration_phase83_ranking.json"
_RESULTS_DIR = _REPO / "backend/backtest/experiments/results"

# The population rule is PRE-REGISTERED in the ranking file (naive '*83*'
# measured 71 false positives). Read the globs from the file so the tests
# and the registration cannot drift apart.
_CLOSED_VOCAB = {"survives_costs", "marginal", "fails_costs", "untestable_on_free_data"}


def _pack_text() -> str:
    """Pack text with HTML comments stripped (the 82.6 stub trap)."""
    text = PACK.read_text()
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _phase83_artifacts() -> list[pathlib.Path]:
    """Population = pre-registered globs UNION the content rule (top-level
    phase_tag == 'phase_83') -- the content rule is the binding backstop for
    the repo's canonical result_store naming, which carries no phase marker
    (ranking-file amendment 2026-08-07, Q/A finding)."""
    rule = json.loads(RANKING.read_text())["artifact_population_rule"]
    found: list[pathlib.Path] = []
    for g in rule["globs"]:
        found.extend(_REPO.glob(g))
    assert "content_rule" in rule, "content rule missing from the preregistration"
    for p in _RESULTS_DIR.glob("*.json"):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("phase_tag") == "phase_83":
            found.append(p)
    return sorted(set(found))


def _assert_ranking_predates_all_artifacts() -> None:
    r_ns = RANKING.stat().st_mtime_ns
    for art in _phase83_artifacts():
        assert not art.stat().st_mtime_ns < r_ns, (
            f"phase-83 backtest artifact {art.name} predates the pre-registration "
            f"({art.stat().st_mtime_ns} < {r_ns}) -- the ranking was not locked first"
        )


# ── C1: envelope ─────────────────────────────────────────────────────


def _envelope() -> dict:
    text = _pack_text()
    fences = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    assert fences, "no JSON fence found in the pack"
    return json.loads(fences[-1])  # the envelope ends the pack


def test_c1_pack_exists_and_envelope_parses():
    assert PACK.exists(), "design pack missing"
    assert PACK.stat().st_size > 2000, "pack is a stub (anti-stub floor, 82.6 convention)"
    env = _envelope()
    for key in ("tier", "external_sources_read_in_full", "snippet_only_sources",
                "urls_collected", "recency_scan_performed",
                "internal_files_inspected", "gate_passed"):
        assert key in env, f"envelope missing {key}"
    assert "coverage" in env and "dry" in env["coverage"], (
        "envelope must report coverage.dry (null is a legal, honest value)"
    )
    assert env["external_sources_read_in_full"] >= 5
    assert env["recency_scan_performed"] is True


# ── C2: gate thresholds read from source, compared at runtime ────────


def test_c2_recorded_thresholds_equal_promotion_gate_at_runtime():
    from backend.autoresearch.gate import PromotionGate

    gate = PromotionGate()
    text = _pack_text()

    def _recorded(name: str) -> float:
        m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
        assert m, f"pack does not record {name}"
        return float(m.group(1))

    assert _recorded("min_dsr") == float(gate.min_dsr)
    assert _recorded("max_pbo") == float(gate.max_pbo)
    assert _recorded("min_pbo_trials") == float(gate.min_pbo_trials)


# ── C3: negative evidence, >=3 failure modes with numbers ────────────


def test_c3_negative_evidence_section_with_numbers():
    text = _pack_text()
    m = re.search(r"## 5\. Negative evidence(.*?)\n## ", text, flags=re.DOTALL)
    assert m, "negative-evidence section missing"
    section = m.group(1)
    assert section.strip(), "negative-evidence section empty"
    # a failure mode = a numbered list entry; it counts only if it carries a
    # numeric figure (digits attached to %, bps, ratio, t=, or 'in N')
    entries = re.findall(r"(?m)^\d+\.\s+\*\*.*?(?=(?:^\d+\.\s+\*\*)|\Z)", section, flags=re.DOTALL)
    with_numbers = [
        e for e in entries
        if re.search(r"\d+(?:\.\d+)?\s*(?:%|pp|bps|/yr|/mo)|t=\d|\d+ (?:of|in) \d+", e)
    ]
    assert len(with_numbers) >= 3, (
        f"only {len(with_numbers)} failure modes carry a numeric figure"
    )


# ── C4: reference-case table, four rows, two non-empty columns ───────


def test_c4_reference_case_table_four_rows_no_empty_cells():
    text = _pack_text()
    m = re.search(r"## 6\. Reference-case table(.*?)\n## ", text, flags=re.DOTALL)
    assert m, "reference-case section missing"
    rows = [
        r for r in re.findall(r"(?m)^\|(.+)\|\s*$", m.group(1))
        if not re.match(r"\s*(?:Reference case|[-: ]+\|)", r)
    ]
    cases = {"COVID": False, "AI-datacenter": False, "Ukraine": False, "Iran-US": False}
    data_rows = []
    for r in rows:
        cells = [c.strip() for c in r.split("|")]
        if len(cells) >= 3 and any(k in cells[0] for k in cases):
            data_rows.append(cells)
            for k in cases:
                if k in cells[0]:
                    cases[k] = True
    assert len(data_rows) == 4, f"expected exactly 4 case rows, found {len(data_rows)}"
    assert all(cases.values()), f"cases missing from table: {[k for k, v in cases.items() if not v]}"
    for cells in data_rows:
        assert cells[1], f"empty free-data-source cell in row {cells[0]!r}"
        assert cells[2], f"empty cost-to-hold cell in row {cells[0]!r}"


# ── C5: pre-registration file, recorded hash, ordering ───────────────


def test_c5_ranking_file_hash_and_ordering():
    assert RANKING.exists(), "pre-registration ranking file missing"
    reg = json.loads(RANKING.read_text())
    assert reg["artifact_population_rule"]["globs"], "population rule missing"

    m = re.search(r"PREREGISTRATION_SHA256:\s*([0-9a-f]+)", _pack_text())
    assert m, "pack does not record the pre-registration SHA-256"
    recorded = m.group(1)
    assert len(recorded) == 64 and all(c in "0123456789abcdef" for c in recorded), (
        "recorded hash is not 64 lowercase hex chars"
    )
    actual = hashlib.sha256(RANKING.read_bytes()).hexdigest()
    assert recorded == actual, (
        f"recorded hash {recorded[:12]}... != actual {actual[:12]}... -- the "
        "ranking file changed after the pack recorded it (append-only policy "
        "requires recomputing the recorded hash in the same commit)"
    )
    _assert_ranking_predates_all_artifacts()


# ── C6: the ordering guard is mutation-tested ────────────────────────


def test_c6_backdated_phase83_artifact_fails_criterion_5():
    mut = _RESULTS_DIR / "20260101T000000Z_phase_83_1_MUTATION.json"
    try:
        mut.write_text('{"mutation": "backdated artifact for the 83.1 ordering guard"}')
        t = RANKING.stat().st_mtime_ns - 60_000_000_000  # 60s older
        os.utime(mut, ns=(t, t))
        # the glob must SEE the mutant, else an empty population makes C5 vacuous
        assert mut in _phase83_artifacts(), (
            "pre-registered globs did not match the mutant artifact -- the "
            "population rule is dead and criterion 5 is vacuously green"
        )
        with pytest.raises(AssertionError):
            _assert_ranking_predates_all_artifacts()
    finally:
        mut.unlink(missing_ok=True)


def test_c6b_backdated_canonical_named_artifact_also_fails_criterion_5():
    """The Q/A's escape shape: result_store.py:36 names artifacts
    f"{ts}_{safe_id}.json" with NO phase marker. The content rule
    (phase_tag == 'phase_83') must catch it -- recall over a member the
    author did NOT name to fit the globs."""
    mut = _RESULTS_DIR / "20260810T120000Z_a1b2c3d4.json"
    try:
        mut.write_text('{"phase_tag": "phase_83", "run_id": "a1b2c3d4"}')
        t = RANKING.stat().st_mtime_ns - 60_000_000_000
        os.utime(mut, ns=(t, t))
        assert mut in _phase83_artifacts(), (
            "content rule did not catch a canonical-named phase-83 artifact -- "
            "the ordering guard is blind to result_store output"
        )
        with pytest.raises(AssertionError):
            _assert_ranking_predates_all_artifacts()
        # negative control: the same canonical name WITHOUT the tag is not
        # phase-83 and must NOT enter the population
        mut.write_text('{"run_id": "a1b2c3d4"}')
        os.utime(mut, ns=(t, t))
        assert mut not in _phase83_artifacts(), (
            "an untagged artifact entered the phase-83 population -- the "
            "content rule is over-matching"
        )
    finally:
        mut.unlink(missing_ok=True)


# ── C7: closed-vocabulary cost classification, zero unclassified ─────


def test_c7_every_candidate_classified_from_closed_vocabulary():
    text = _pack_text()
    vocab_line = re.search(r"\*\*Closed vocabulary\*\*:\s*(.+)", text)
    assert vocab_line, "closed vocabulary not recorded in the pack"
    recorded_vocab = set(re.findall(r"`(\w+)`", vocab_line.group(1)))
    assert recorded_vocab == _CLOSED_VOCAB, (
        f"recorded vocabulary {recorded_vocab} != expected {_CLOSED_VOCAB}"
    )
    m = re.search(r"## 4\. Candidate designs(.*?)\n## ", text, flags=re.DOTALL)
    assert m, "candidate-design section missing"
    # EVERY table row in the section counts except the header and the
    # separator -- a candidate cannot escape the census by omitting its
    # ordinal (Q/A cycle-1 finding: the previous \d+-prefixed filter let an
    # unnumbered row with a blank classification pass).
    all_rows = re.findall(r"(?m)^\|(.+)\|\s*$", m.group(1))
    rows = [
        r for r in all_rows
        if "Candidate design" not in r  # header
        and not re.fullmatch(r"[\s\-:|]+", r)  # separator
    ]
    assert len(rows) >= 7, f"expected >=7 candidate rows, found {len(rows)}"
    unclassified = []
    for r in rows:
        cells = [c.strip() for c in r.split("|")]
        classification = cells[-1]
        if classification not in _CLOSED_VOCAB:
            unclassified.append((cells[1] if len(cells) > 1 else r, classification))
    assert len(unclassified) == 0, f"candidates without a valid classification: {unclassified}"
