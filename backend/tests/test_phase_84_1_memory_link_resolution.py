"""phase-84.1: memory-link auditor resolution tests.

All fixture corpora are synthetic under tmp_path (criterion 4: the auditor is
invoked via subprocess and asserted on .returncode). Criterion-3 fixtures are
built with ZERO dangling pointers and ZERO malformed frontmatter so both
readings of the criterion (with/without the founding dangling-pointer check)
agree on the expected exit code -- contract D5; the dangling check keeps its
own non-criterion regression test.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

AUDITOR = str(pathlib.Path(__file__).resolve().parents[2]
              / "scripts/housekeeping/audit_memory.py")

_FM = "---\nname: {name}\nmetadata:\n  type: feedback\n---\n"


def _mk(corpus: pathlib.Path, fname: str, name: str | None = None,
        body: str = "", raw: str | None = None):
    corpus.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (corpus / fname).write_text(raw)
        return
    nm = name or fname.removesuffix(".md")
    (corpus / fname).write_text(_FM.format(name=nm) + body + "\n")


def _index(corpus: pathlib.Path):
    files = sorted(p.name for p in corpus.glob("*.md") if p.name != "MEMORY.md")
    lines = [f"- [{f}]({f}) — x" for f in files]
    (corpus / "MEMORY.md").write_text("\n".join(lines) + "\n")


def _run(corpus: pathlib.Path, siblings: list[pathlib.Path] | None = None):
    cmd = [sys.executable, AUDITOR, "--dir", str(corpus)]
    for s in (siblings if siblings is not None else []):
        cmd += ["--sibling", str(s)]
    if siblings is None:
        cmd += ["--sibling", str(corpus / "_none")]  # isolate from real corpora
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout


# ── criterion 5: the six named cases ─────────────────────────────────
#
# C7 mutation coverage (measured 2026-08-07 on this 17-test suite):
# dropping the normalization fold so resolution is exact-filename-only
# turns these C5 cases RED -- test_c5_hyphen_underscore_same_directory,
# test_c5_case_folding_same_directory, test_c5_cross_corpus_match,
# test_c5_stem_resolution_without_frontmatter_name -- while
# test_c5_exact_same_directory_match stays GREEN (8 failed suite-wide,
# incl. non-C5 tests that also depend on normalization). Dropping the
# frontmatter-name alias lookup turns exactly one case RED:
# test_c5_frontmatter_name_alias.


def test_c5_exact_same_directory_match(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[feedback_beta]]")
    _mk(c, "feedback_beta.md")
    _index(c)
    rc, out = _run(c)
    assert rc == 0
    assert "UNRESOLVABLE" not in out and "NORMALIZED LINK" not in out
    assert "2 exact" not in out  # one exact link only
    assert "1 exact" in out


def test_c5_case_folding_same_directory(tmp_path):
    """Cycle-2 addition (Q/A B2): criterion 1's 'folds letter case' leg had no
    failing guard -- deleting .casefold() left the suite green (an equivalent
    mutant on today's all-lowercase corpora, but the criterion names the leg).
    A case-differing link must resolve; dropping the fold turns this red."""
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[Feedback-Beta-Thing]]")
    _mk(c, "feedback_beta_thing.md")
    _index(c)
    rc, out = _run(c)
    assert rc == 0, f"case-differing link failed to resolve:\n{out}"
    assert "NORMALIZED LINK" in out


def test_c5_hyphen_underscore_same_directory(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[feedback-beta-thing]]")
    _mk(c, "feedback_beta_thing.md")
    _index(c)
    rc, out = _run(c)
    assert rc == 0
    assert "NORMALIZED LINK: feedback_alpha.md -> [[feedback-beta-thing]]" in out
    assert "feedback_beta_thing.md" in out


def test_c5_leading_type_prefix_match(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[beta-thing]]")
    _mk(c, "feedback_beta_thing.md", name="beta-thing")
    _index(c)
    rc, out = _run(c)
    assert rc == 0
    assert "NORMALIZED LINK" in out


def test_c5_stem_resolution_without_frontmatter_name(tmp_path):
    """Cycle-3 addition (Q/A F2): every other fixture's frontmatter name
    normalizes to the same key as its stem, so the alias index key masked any
    stem-leg regression -- a mutant dropping prefix-stripping from the stem key
    alone survived all 16 tests while mis-classing 13 real researcher links.
    A target with NO `name:` line (valid: MALFORMED requires only `type:`)
    forces resolution through the prefix-stripped STEM key alone."""
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[beta-thing]]")
    _mk(c, "feedback_beta_thing.md",
        raw="---\nmetadata:\n  type: feedback\n---\nbody\n")
    _index(c)
    rc, out = _run(c)
    assert rc == 0, f"stem-only link failed to resolve:\n{out}"
    assert "NORMALIZED LINK" in out and "feedback_beta_thing.md" in out


def test_c5_cross_corpus_match(tmp_path):
    c, other = tmp_path / "c", tmp_path / "other"
    _mk(c, "feedback_alpha.md", body="see [[gamma-lesson]]")
    _index(c)
    _mk(other, "feedback_gamma_lesson.md")
    _index(other)
    rc, out = _run(c, siblings=[other])
    assert rc == 0
    assert "CROSS-CORPUS LINK: feedback_alpha.md -> [[gamma-lesson]]" in out
    assert str(other) in out, "the owning directory must be named in the line"


def test_c5_unresolvable_fails(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[does-not-exist-anywhere]]")
    _index(c)
    rc, out = _run(c)
    assert rc == 1
    assert "UNRESOLVABLE LINK: feedback_alpha.md -> [[does-not-exist-anywhere]]" in out


def test_c5_frontmatter_name_alias(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[totally-different-slug]]")
    _mk(c, "feedback_beta.md", name="totally-different-slug")
    _index(c)
    rc, out = _run(c)
    assert rc == 0
    assert "NORMALIZED LINK" in out and "feedback_beta.md" in out


# ── criterion 3: exit-code truth table, BOTH directions ──────────────


def test_c3_direction_a_findings_alone_pass(tmp_path):
    c, other = tmp_path / "c", tmp_path / "other"
    _mk(c, "feedback_alpha.md", body="[[feedback-beta-x]] and [[remote-thing]]")
    _mk(c, "feedback_beta_x.md")
    _index(c)
    _mk(other, "feedback_remote_thing.md")
    _index(other)
    rc, out = _run(c, siblings=[other])
    assert rc == 0, f"normalized+cross findings alone must PASS:\n{out}"
    assert "NORMALIZED LINK" in out and "CROSS-CORPUS LINK" in out


def test_c3_direction_b_unresolvable_alone_fails(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="[[nope-never]]")
    _index(c)
    rc, _ = _run(c)
    assert rc == 1


def test_c3_direction_b_no_pointer_alone_fails(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md")
    _mk(c, "feedback_orphan.md")
    (c / "MEMORY.md").write_text("- [feedback_alpha.md](feedback_alpha.md) — x\n")
    rc, out = _run(c)
    assert rc == 1
    assert "NO POINTER: feedback_orphan.md" in out


# ── D4: ambiguity is reported, never failed ──────────────────────────


def test_d4_ambiguous_link_reports_all_candidates_and_passes(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="see [[shared-key]]")
    _mk(c, "feedback_shared_key.md")
    _mk(c, "project_shared_key.md")  # same key after prefix strip
    _index(c)
    rc, out = _run(c)
    assert rc == 0, f"an existing-but-ambiguous target must not fail:\n{out}"
    assert "AMBIGUOUS LINK" in out
    assert "feedback_shared_key.md" in out and "project_shared_key.md" in out


# ── D5: the founding dangling-pointer check still fails ──────────────


def test_d5_dangling_pointer_still_exits_1(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md")
    (c / "MEMORY.md").write_text(
        "- [feedback_alpha.md](feedback_alpha.md) — x\n"
        "- [feedback_gone.md](feedback_gone.md) — x\n")
    rc, out = _run(c)
    assert rc == 1
    assert "DANGLING POINTER" in out


# ── frontmatter robustness: the yaml-crash shape must not crash ──────


def test_yaml_hostile_frontmatter_does_not_crash(tmp_path):
    c = tmp_path / "c"
    raw = ("---\nname: feedback_hostile\n"
           "description: measured X: and then Y: over Z — with unicode ≥\n"
           "metadata:\n  type: feedback\n---\nbody [[feedback-alpha]]\n")
    _mk(c, "feedback_hostile.md", raw=raw)
    _mk(c, "feedback_alpha.md")
    _index(c)
    rc, out = _run(c)
    assert rc == 0, f"yaml-hostile frontmatter crashed or failed the run:\n{out}"
    assert "NORMALIZED LINK" in out


# ── D6: code blocks are not links ────────────────────────────────────


def test_d6_code_spans_are_not_links(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md",
        body="attr `[[nodiscard]]` and\n```\n[[in-a-fence]]\n```\nreal: [[feedback-beta]]")
    _mk(c, "feedback_beta.md")
    _index(c)
    rc, out = _run(c)
    assert rc == 0, f"code-span pseudo-links failed the run:\n{out}"
    assert "nodiscard" not in out and "in-a-fence" not in out


# ── criterion 6: the three REAL corpora, structural assertions only ──


def test_c6_real_corpora_structural(tmp_path):
    main = pathlib.Path.home() / (
        ".claude/projects/-Users-ford--openclaw-workspace-pyfinagent/memory")
    repo = pathlib.Path(AUDITOR).parents[2]
    for corpus in (main, repo / ".claude/agent-memory/qa",
                   repo / ".claude/agent-memory/researcher"):
        p = subprocess.run([sys.executable, AUDITOR, "--dir", str(corpus)],
                           capture_output=True, text=True)
        # structural only: completes without raising, header present,
        # census line present; exit code and counts are NOT asserted
        # (the corpora are written by other agents and drift daily)
        assert p.returncode in (0, 1), f"{corpus}: crashed"
        assert "Traceback" not in p.stderr, f"{corpus}: raised:\n{p.stderr}"
        assert p.stdout.startswith("memory files: "), f"{corpus}: header missing"
        assert "links: " in p.stdout, f"{corpus}: census line missing"


# ── criterion 8: the auditor performs no writes ──────────────────────


def test_c8_auditor_is_read_only(tmp_path):
    c = tmp_path / "c"
    _mk(c, "feedback_alpha.md", body="[[feedback-beta]] [[nope]]")
    _mk(c, "feedback_beta.md")
    _index(c)
    before = {}
    for p in sorted(c.iterdir()):
        st = p.stat()
        before[p.name] = (p.read_bytes(), st.st_mtime_ns)
    rc, _ = _run(c)
    assert rc == 1  # [[nope]] is unresolvable -- a failing run must also not write
    after_names = sorted(p.name for p in c.iterdir())
    assert after_names == sorted(before), "files created or deleted"
    for p in sorted(c.iterdir()):
        st = p.stat()
        assert (p.read_bytes(), st.st_mtime_ns) == before[p.name], (
            f"{p.name}: bytes or mtime changed -- the auditor wrote"
        )
