"""phase-86.108 criterion 1 -- guard the census's population rule.

The census's value is entirely in its RULE, so the rule is what gets tested:
both formatter shapes are counted, the agent phrase is extracted without
formatter-prefix leakage, the double-logged Critic is visible as two lines
rather than silently folded, and no failure RATE is printed.

Every cell drives the real module. Fixtures are synthetic logs written to
tmp_path -- the real corpus is 40MB of rotated gzip and must not be a test
dependency.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qa" / "census_invalid_json_86_108.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_census86108", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_census86108"] = m
    spec.loader.exec_module(m)
    return m


# ── the two formatter shapes ──────────────────────────────────────────────

def test_both_formatter_shapes_are_parsed():
    """C-a: a `"module":`-keyed parser sees only the JSON share and looks
    complete. Measured on the real corpus that is 17.1% of the truth."""
    m = _mod()

    compact = "\x1b[33m19:11:09 W [debate]\x1b[0m Moderator returned invalid JSON, using raw text"
    msg, fmt = m.message_of(compact)
    assert fmt == "compact"
    assert msg == "Moderator returned invalid JSON, using raw text", msg

    js = json.dumps({"timestamp": "2026-08-17 19:11:09,000", "level": "WARNING",
                     "module": "debate",
                     "message": "Moderator returned invalid JSON, using raw text"})
    msg2, fmt2 = m.message_of(js)
    assert fmt2 == "json"
    assert msg2 == "Moderator returned invalid JSON, using raw text"

    # Both shapes must yield the SAME agent -- that is the point of C-a.
    assert m.agent_of(msg) == m.agent_of(msg2) == "Moderator"


# ── the agent-phrase extractor ────────────────────────────────────────────

def test_agent_phrase_does_not_leak_the_formatter_prefix():
    """The failure this replaces: an earlier draft took 'the last three
    capitalised words' and produced buckets like `W orchestrator Critic`
    alongside a bare `Critic` -- splitting one agent into two."""
    m = _mod()
    line = "\x1b[33m19:11:09 W [orchestrator]\x1b[0m Critic returned invalid JSON"
    msg, _ = m.message_of(line)
    assert m.agent_of(msg) == "Critic", m.agent_of(msg)
    for leak in ("W ", "[", "orchestrator", "19:11:09"):
        assert leak not in m.agent_of(msg)


def test_multiword_agents_are_not_collapsed():
    """C-b: there is no agent called 'Analyst'. Taking the last token merges
    three distinct analysts into one 926-line bucket."""
    m = _mod()
    names = ["Conservative Analyst", "Neutral Analyst", "Aggressive Analyst",
             "Devil's Advocate", "Risk Judge", "Synthesis-Final", "Critic-Retry"]
    for name in names:
        msg, _ = m.message_of(f"19:11:09 W [debate] {name} returned invalid JSON")
        assert m.agent_of(msg) == name, f"{name} -> {m.agent_of(msg)}"
    # And they must stay DISTINCT, which is the property the filing lost.
    assert len({m.agent_of(m.message_of(f"19:11:09 W [d] {n} returned invalid JSON")[0])
                for n in names}) == len(names)


def test_a_line_without_an_agent_phrase_is_reported_not_guessed():
    m = _mod()
    msg, _ = m.message_of("19:11:09 W [d] returned invalid JSON")
    assert m.agent_of(msg) == "(no agent phrase)"


# ── end-to-end over a synthetic corpus ────────────────────────────────────

def test_census_counts_lines_not_events_and_prints_no_rate(tmp_path, capsys, monkeypatch):
    """C-c and C-d in one run: the double-logged Critic must show as TWO
    lines, and no failure rate may be printed."""
    m = _mod()
    logs = tmp_path / "handoff" / "logs"
    logs.mkdir(parents=True)
    body = "\n".join([
        "19:11:09 W [debate] Critic returned invalid JSON",
        "19:11:09 W [debate] Critic returned invalid JSON, treating as PASS with draft.",
        "19:11:10 W [debate] Neutral Analyst returned invalid JSON, using raw text",
        json.dumps({"message": "Risk Judge returned invalid JSON", "module": "risk_debate"}),
        "19:11:12 I [debate] nothing interesting here",
    ]) + "\n"
    with gzip.open(logs / "backend.log.20260101T000000Z.gz", "wt", encoding="utf-8") as fh:
        fh.write(body)

    monkeypatch.setattr(m, "REPO", tmp_path)
    assert m.main(rotated_only=True) == 0
    out = capsys.readouterr().out

    assert "TOTAL matching LINES = 4" in out, out
    # C-c: the two Critic wordings are counted separately, so the double-log
    # is VISIBLE rather than silently folded into one.
    assert "'returned invalid JSON'" in out
    assert "treating as PASS with draft." in out
    # C-d: shares are labelled as shares; no failure rate is claimed.
    assert "NO FAILURE RATE IS PRINTED" in out
    assert "% of lines" in out
    # The rail impossibility is stated, not left for a reader to assume.
    assert "not derivable" in out


def test_the_glob_is_printed_so_the_population_is_auditable(tmp_path, capsys, monkeypatch):
    """A census whose corpus is not stated cannot be reproduced."""
    m = _mod()
    logs = tmp_path / "handoff" / "logs"
    logs.mkdir(parents=True)
    with gzip.open(logs / "backend.log.20260101T000000Z.gz", "wt", encoding="utf-8") as fh:
        fh.write("19:11:09 W [debate] Critic returned invalid JSON\n")
    monkeypatch.setattr(m, "REPO", tmp_path)
    m.main(rotated_only=True)
    out = capsys.readouterr().out
    assert "CORPUS (the glob, stated):" in out
    assert "backend.log.20260101T000000Z.gz" in out
    assert "MATCH RULE:" in out


def test_rotated_only_excludes_the_live_log(tmp_path, monkeypatch):
    """The research gate's corpus ended 2026-08-14 because it globbed only
    the rotated files; --rotated-only reproduces that boundary exactly, which
    is what makes the 2,859-vs-2,872 delta attributable rather than a drift."""
    m = _mod()
    (tmp_path / "handoff" / "logs").mkdir(parents=True)
    (tmp_path / "backend.log").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(m, "REPO", tmp_path)
    assert all(p.suffix == ".gz" for p in m.corpus(rotated_only=True))
    assert any(p.name == "backend.log" for p in m.corpus(rotated_only=False)), (
        "the live log must be INCLUDED by default -- excluding it silently is "
        "what made the gate's census stop at 2026-08-14"
    )
