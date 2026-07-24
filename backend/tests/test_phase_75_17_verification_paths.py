"""phase-75.17: absent-path verification family -- triage + repair.

Ten `status=done` masterplan steps carried a `verification.command` that
references a filesystem path never present on disk (nine phase-29
`go_live_drills` plan-name mismatches + one skill file pair retired by
phase-26.4). Their PASS was therefore unreproducible -- unrunnable
governance rot (see research_brief_75.17.md). This step repairs by
ANNOTATING (a `superseded_record` sibling), never by amending the
immutable `verification.command` / `success_criteria`.

This suite proves three things, each independently mutation-tested (the
matrix + verbatim results live in live_check_75.17.md):

  (a) The annotation did NOT amend anything -- byte-identity of
      command + success_criteria against the pre-75.17 baseline commit
      for every touched step.
  (b) Exactly one `superseded_record` per step, repo-wide (no
      double-annotation; a raw duplicate-key insertion inside one step's
      JSON object is caught even though `json.loads` would otherwise
      silently collapse it).
  (c) The committed sweep (`scripts/qa/sweep_absent_verification_paths.py`)
      is the thing that actually found these ten -- it returns the exact
      ten-step genuine set against the baseline masterplan snapshot and
      an EMPTY genuine set against the live (annotated) masterplan, it
      handles all four `verification` shapes without crashing (the
      list-shaped fixture is asserted BY SHAPE, not just by outcome --
      the phase-75.2.1 lesson: a fixture that cannot represent a shape
      can't prove the crash-guard), and its false-positive resolvers are
      unit-proven against the exact hard cases the research brief named.

All offline: no network, no Slack, no gcloud. `git show <baseline>` and
`git log --diff-filter` are local repo reads only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qa.sweep_absent_verification_paths import (  # noqa: E402
    classify,
    fp_reason,
    verif_commands,
)

# Pre-75.17 baseline commit (a fixed SHA, not "HEAD" -- HEAD moves as later
# steps land; the byte-identity claim is meaningful only against a pinned
# ancestor, mirroring test_phase_75_2_1_push_approval.py's BASELINE_COMMIT
# pattern).
BASELINE_COMMIT = "7739922d8ab9ed8398dbb97da1a9a8d6ed6894ae"

CLASS_I_STEPS = (
    "4.17.2", "4.17.3", "4.17.4", "4.17.5", "4.17.6",
    "4.17.7", "4.17.8", "4.17.11", "4.17.12",
)
CLASS_II_STEPS = ("4.14.26",)
TOUCHED_STEPS = CLASS_I_STEPS + CLASS_II_STEPS  # the 10 genuine defects
PRIOR_HOLDERS = ("4.14.4", "4.14.24", "4.17.9", "68.5")  # pre-existing holders; excluded here
ALL_HOLDERS = PRIOR_HOLDERS + TOUCHED_STEPS  # expect exactly 14

ON_DISK_EQUIVALENT = {
    "4.17.2": "scripts/go_live_drills/smoke_test_4_17_2.py",
    "4.17.3": "scripts/go_live_drills/smoke_test_4_17_3.py",
    "4.17.4": "scripts/go_live_drills/smoke_test_4_17_4.py",
    "4.17.5": "scripts/go_live_drills/smoke_test_4_17_5.py",
    "4.17.6": "scripts/go_live_drills/smoke_test_4_17_6.py",
    "4.17.7": "scripts/go_live_drills/smoke_test_4_17_7.py",
    "4.17.8": "scripts/go_live_drills/smoke_test_4_17_8.py",
    "4.17.11": "scripts/go_live_drills/smoke_test_4_17_11.py",
    "4.17.12": "scripts/go_live_drills/smoke_test_4_17_12.py",
}


def _masterplan_at(ref: str | None) -> dict:
    if ref is None:
        return json.loads((REPO_ROOT / ".claude/masterplan.json").read_text(encoding="utf-8"))
    text = subprocess.check_output(
        ["git", "show", f"{ref}:.claude/masterplan.json"], cwd=str(REPO_ROOT), text=True,
    )
    return json.loads(text)


def _steps_by_id(mp: dict) -> dict:
    return {s["id"]: s for ph in mp["phases"] for s in (ph.get("steps") or [])}


# ── (a) byte-identity: annotate, never amend ─────────────────────────

@pytest.mark.parametrize("step_id", TOUCHED_STEPS)
def test_verification_byte_identical_to_baseline(step_id):
    before = _steps_by_id(_masterplan_at(BASELINE_COMMIT))[step_id]["verification"]
    after = _steps_by_id(_masterplan_at(None))[step_id]["verification"]
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True), (
        f"{step_id}: verification block was AMENDED, not annotated"
    )


@pytest.mark.parametrize("step_id", TOUCHED_STEPS)
def test_step_stays_done_and_gained_a_superseded_record(step_id):
    step = _steps_by_id(_masterplan_at(None))[step_id]
    assert step["status"] == "done"
    rec = step.get("superseded_record")
    assert rec, f"{step_id}: no superseded_record"
    assert rec["criteria_amended"] is False


@pytest.mark.parametrize("step_id", CLASS_I_STEPS)
def test_class_i_record_names_the_never_existed_path_and_on_disk_equivalent(step_id):
    rec = _steps_by_id(_masterplan_at(None))[step_id]["superseded_record"]
    assert rec["retired_by_commit"] is None
    assert rec["already_broken_before_retirement"] is True
    assert rec["on_disk_equivalent"] == ON_DISK_EQUIVALENT[step_id]
    assert (REPO_ROOT / rec["on_disk_equivalent"]).exists(), (
        f"{step_id}: on_disk_equivalent {rec['on_disk_equivalent']} does not exist"
    )
    assert "scripts/go_live_drills/" in rec["on_disk_equivalent"]


def test_4_14_26_record_names_the_retiring_commit():
    rec = _steps_by_id(_masterplan_at(None))["4.14.26"]["superseded_record"]
    assert rec["retired_by_commit"] == "f7e24d0a"
    assert rec["retired_in_step"] == "26.4"
    assert rec["already_broken_before_retirement"] is False
    assert "neutral_analyst.md" in rec["reason"]
    assert "devils_advocate_agent.md" in rec["reason"]
    # the retirement claim is independently reproducible against git, not asserted
    for fname in ("neutral_analyst.md", "devils_advocate_agent.md"):
        path = f"backend/agents/skills/{fname}"
        assert not (REPO_ROOT / path).exists(), f"{path} should be absent (retired)"
        deleted = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "--all", "--diff-filter=D", "--oneline", "--", path],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        assert deleted, f"{path}: no deletion found in history"
        assert deleted.splitlines()[0].startswith("f7e24d0a")


@pytest.mark.parametrize("step_id", PRIOR_HOLDERS)
def test_already_annotated_steps_were_not_touched(step_id):
    """4.14.4 / 4.14.24 / 4.17.9 must be untouched by this step -- their
    superseded_record must be byte-identical to the pre-75.17 baseline."""
    before = _steps_by_id(_masterplan_at(BASELINE_COMMIT))[step_id]
    after = _steps_by_id(_masterplan_at(None))[step_id]
    assert json.dumps(before.get("superseded_record"), sort_keys=True) == \
        json.dumps(after.get("superseded_record"), sort_keys=True), \
        f"{step_id}: already-annotated step was re-touched"


# ── (b) exactly one superseded_record per step, repo-wide ────────────

def test_exactly_one_superseded_record_repo_wide():
    """Uses object_pairs_hook to catch a raw duplicate-key insertion inside
    a single step's JSON object -- json.loads would otherwise silently
    keep only the LAST occurrence and hide a double-annotation (M9)."""
    text = (REPO_ROOT / ".claude/masterplan.json").read_text(encoding="utf-8")
    duplicate_key_counts: list[int] = []

    def hook(pairs):
        seen: dict[str, list] = {}
        for k, v in pairs:
            seen.setdefault(k, []).append(v)
        for k, values in seen.items():
            if k == "superseded_record" and len(values) > 1:
                duplicate_key_counts.append(len(values))
        return {k: values[-1] for k, values in seen.items()}

    mp = json.loads(text, object_pairs_hook=hook)
    assert duplicate_key_counts == [], (
        f"duplicate superseded_record keys found within a single step object: {duplicate_key_counts}"
    )
    steps = _steps_by_id(mp)
    holders = sorted(sid for sid, s in steps.items() if "superseded_record" in s)
    assert holders == sorted(ALL_HOLDERS), (
        f"expected exactly {sorted(ALL_HOLDERS)}, got {holders}"
    )
    assert len(holders) == 14


def test_masterplan_diff_touches_only_the_ten_sibling_insertions():
    """git-diff purity proof: every removed line vs baseline is a
    trailing-comma artifact of appending a new last sibling key, paired
    1:1 with an identical added line plus the comma -- never a real
    content change."""
    diff = subprocess.check_output(
        ["git", "diff", BASELINE_COMMIT, "--", ".claude/masterplan.json"],
        cwd=str(REPO_ROOT), text=True,
    )
    removed = [l[1:] for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    for line in removed:
        assert (line + ",") in added, f"non-comma-artifact removal found: {line!r}"


# ── (c) the sweep classifier itself ───────────────────────────────────

def test_sweep_over_live_masterplan_is_clean():
    live = _masterplan_at(None)
    result = classify(live, REPO_ROOT)
    assert result["genuine"] == {}, f"unexpected genuine defects remain: {result['genuine']}"


def test_sweep_over_baseline_masterplan_finds_exactly_the_ten():
    baseline = _masterplan_at(BASELINE_COMMIT)
    result = classify(baseline, REPO_ROOT)
    assert set(result["genuine"].keys()) == set(TOUCHED_STEPS), (
        f"expected exactly {sorted(TOUCHED_STEPS)}, got {sorted(result['genuine'].keys())}"
    )
    # class assignment: never-existed for the 9 drills, retired for 4.14.26
    for step_id in CLASS_I_STEPS:
        classes = {row["class"] for row in result["genuine"][step_id]}
        assert classes == {"never-existed"}, f"{step_id}: {classes}"
    for row in result["genuine"]["4.14.26"]:
        assert row["class"] == "retired"
        assert row["retired_by_commit"].startswith("f7e24d0a")


def test_sweep_shape_census_matches_the_corrected_figures():
    live = _masterplan_at(None)
    census = classify(live, REPO_ROOT)["shape_census"]
    assert census == {"dict": 720, "str": 126, "list": 13, "none": 24}


# ── (d) all four verification shapes, exercised BY SHAPE (M7 target) ──

_FIXTURE_ABSENT_PATH = "scripts/qa/FIXTURE_DOES_NOT_EXIST_75_17.py"


def _build_all_shapes_fixture() -> dict:
    """One done step per shape, each naming the SAME absent path so the
    only variable under test is the verification SHAPE. The list-shaped
    entry is what a naive `.get('command')` crashes on -- this is the
    load-bearing fixture entry (M7 mutates exactly this one)."""
    cmd = f"python {_FIXTURE_ABSENT_PATH}"
    return {
        "phases": [{
            "steps": [
                {"id": "FIXTURE.dict001", "status": "done",
                 "verification": {"command": cmd, "success_criteria": ["x"]}},
                {"id": "FIXTURE.str001", "status": "done",
                 "verification": cmd},
                {"id": "FIXTURE.list001", "status": "done",
                 "verification": [cmd, "echo unrelated"]},
                {"id": "FIXTURE.none001", "status": "done",
                 "verification": None},
            ]
        }]
    }


def test_all_four_verification_shapes_handled_without_crash():
    fixture = _build_all_shapes_fixture()
    steps = _steps_by_id(fixture)

    # Shape assertions FIRST: prove the fixture actually exercises each
    # shape (the phase-75.2.1 lesson -- a fixture that silently degrades
    # to a different shape can't prove the crash-guard, even if the
    # aggregate outcome still looks right).
    assert isinstance(steps["FIXTURE.dict001"]["verification"], dict)
    assert isinstance(steps["FIXTURE.str001"]["verification"], str)
    assert isinstance(steps["FIXTURE.list001"]["verification"], list)
    assert steps["FIXTURE.none001"]["verification"] is None

    # verif_commands must not crash on any shape and must extract the
    # command out of all three non-None shapes.
    for sid in ("FIXTURE.dict001", "FIXTURE.str001", "FIXTURE.list001"):
        cmds = verif_commands(steps[sid]["verification"])
        assert cmds and any(_FIXTURE_ABSENT_PATH in c for c in cmds), sid
    assert verif_commands(steps["FIXTURE.none001"]["verification"]) == []

    result = classify(fixture, REPO_ROOT)
    assert "FIXTURE.dict001" in result["genuine"]
    assert "FIXTURE.str001" in result["genuine"]
    assert "FIXTURE.list001" in result["genuine"], (
        "list-shaped verification not detected as genuine -- the naive "
        "`.get('command')` crash-guard case is not being exercised"
    )
    assert "FIXTURE.none001" not in result["genuine"]


# ── (e) resolver non-flag proofs (M3-M6, M10 hard cases) ──────────────

def test_resolver_frontend_relative_path_not_flagged():
    """M3: `lib/icons.ts` only resolves under frontend/src/ -- must not
    be flagged as a genuine absent path."""
    cmd = "grep -q IconName lib/icons.ts"
    assert fp_reason("lib/icons.ts", cmd, REPO_ROOT) is not None
    assert (REPO_ROOT / "frontend/src/lib/icons.ts").exists()


def test_resolver_url_fragment_not_flagged():
    """M4: a route fragment like /openapi.json is not a filesystem path.
    A leading-slash token fails the well-formedness gate before reaching
    the url-route branch, so the FP class is 'malformed-token' rather
    than 'url-route' -- the outcome (excluded from genuine) is the
    contracted behavior, not the specific label."""
    cmd = "curl -s http://localhost:8000/openapi.json | jq ."
    assert fp_reason("/openapi.json", cmd, REPO_ROOT) is not None


def test_resolver_truncated_plist_not_flagged():
    """M5: a truncated launchd plist path must not be treated as a repo
    file (it isn't repo-relative; it's an absolute host path). Same
    leading-slash malformed-token short-circuit as the URL-fragment case
    above -- excluded either way."""
    cmd = "launchctl list | grep /Library/LaunchAgents/com.py"
    assert fp_reason("/Library/LaunchAgents/com.py", cmd, REPO_ROOT) is not None


def test_resolver_negative_assertion_not_flagged():
    """M6: `test ! -f X || ...` (4.14.19 class) PASSES because X is
    absent -- absence is the assertion, not a defect."""
    cmd = "test ! -f backend/mcp/__init__.py || python scripts/audit/mcp_inventory.py --check"
    assert fp_reason("backend/mcp/__init__.py", cmd, REPO_ROOT) == "absence-asserted"
    assert not (REPO_ROOT / "backend/mcp/__init__.py").exists()


def test_resolver_glob_prefix_truncation_reresolved():
    """M10 / the 7.12 hard case: the extractor truncates
    `alt_data_ic_*.tsv` at the glob metacharacter -> token
    `.../alt_data_ic_`; it must be re-globbed before being called absent."""
    real_dir = "backend/backtest/experiments/results"
    cmd = f"ls {real_dir}/alt_data_ic_*.tsv | head -n 1"
    token = f"{real_dir}/alt_data_ic_"
    assert fp_reason(token, cmd, REPO_ROOT) == "glob-prefix-matches"
    assert list((REPO_ROOT / real_dir).glob("alt_data_ic_*"))
