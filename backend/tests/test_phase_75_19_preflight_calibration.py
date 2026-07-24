"""phase-75.19: recalibrated masterplan preflight gate.

The pre-75.19 preflight was status-blind, annotation-blind, and
container-blind: on 2026-07-24 it reported "863 steps, 151 broken,
8 unparseable" across 222 BROKEN lines while the true genuine residue was
ZERO (all 14 genuinely-unrunnable done steps already carried
`superseded_record` annotations from 75.2.1/75.17). ~100% effective false
positives is why nobody acted on it (Sadowski et al. CACM 2018; see
research_brief_75.19.md).

This suite pins the recalibration behaviorally, each guard designed so a
concrete mutation flips it (qa.md section 4c; the matrix + verbatim results
live in live_check_75.19.md):

  (a) STATUS-AWARENESS -- one fixture per status class on the SAME absent
      path; only `done` reports (criterion 1).
  (b) TRANSIENT/NON-SOURCE exclusion BY CONSTRUCTION via the imported
      75.17 `fp_reason` classes -- handoff/ output, gitignored path, URL
      fragment, frontend-relative, truncated abs-host path -- each pinned
      as its own case, no observed-string allowlist (criterion 2).
  (c) POSITIVE genuine-defect fixture -- a done step naming an absent
      non-transient project path MUST be reported (the anti-vacuity leg:
      a gate that can only say CLEAN proves nothing).
  (d) ANNOTATION exclusion -- `superseded_record` steps are dispositioned.
  (e) CONTAINER semantics -- `subphases[]` scanned, archive containers
      excluded-but-counted.
  (f) LIST-shaped verification scanned BY SHAPE (75.2.1 lesson: a fixture
      that cannot represent the shape can't prove the guard).
  (g) SUMMARY/ROWS internal consistency incl. on the LIVE masterplan,
      where every reported id must resolve to a real step (criterion 3).
  (h) SHLEX-untokenizable commands are still scanned via regex extractors
      and bucketed, never reported broken solely for tokenization.
  (i) IMPORT leg status-gated like the path leg.

All offline: git_classify is stubbed for fixtures; the live-masterplan
tests do one `git ls-files` read.
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

from scripts.meta.preflight_verify_masterplan import (
    build_report,
    check_consistency,
    iter_steps,
)

# A project-shaped path that must NOT exist -- the positive fixture's
# subject. Guarded by an existence assert so the fixture cannot rot into
# vacuity (and so the criterion-6 FIXTURE MUTATION -- pointing this at a
# real file -- fails loudly).
ABSENT = "backend/tests/fixtures/absent_75_19_fixture_target.py"
EXISTING = "backend/main.py"

# Stub adjudicators for fixture runs: no subprocess, no repo git state.
FAKE_GIT_CLASSIFY = lambda path, root: ("never-existed", "")
NO_BASENAMES: set[str] = set()


def _report(mp: dict) -> dict:
    return build_report(
        mp, REPO_ROOT,
        git_classify_fn=FAKE_GIT_CLASSIFY,
        repo_basenames=NO_BASENAMES,
    )


def _mp(*steps: dict, container: str = "steps") -> dict:
    return {"phases": [{"id": "phase-T", "status": "in_progress",
                        container: list(steps)}]}


def _step(sid: str, status: str, cmd, **extra) -> dict:
    return {"id": sid, "status": status, "verification": cmd, **extra}


def test_fixture_target_is_absent_and_positive_control_exists():
    assert not (REPO_ROOT / ABSENT).exists(), (
        f"{ABSENT} must not exist -- the positive fixture depends on it")
    assert (REPO_ROOT / EXISTING).exists()


# -- (c) positive genuine defect: the gate CAN say broken ----------------

def test_done_step_with_absent_path_is_reported():
    rep = _report(_mp(_step("T.1", "done", f"pytest {ABSENT} -q")))
    assert "T.1" in rep["genuine"], "genuine absent path on a done step MUST report"
    entry = rep["genuine"]["T.1"][0]
    assert entry["kind"] == "path" and entry["ref"] == ABSENT
    assert rep["summary"]["genuine_lines"] == 1


def test_done_step_with_existing_path_is_clean():
    rep = _report(_mp(_step("T.1", "done", f"pytest {EXISTING} -q")))
    assert rep["genuine"] == {}


# -- (a) status-awareness (criterion 1) ----------------------------------

@pytest.mark.parametrize("status", ["pending", "deferred", "dropped",
                                    "superseded", "in_progress", "blocked"])
def test_non_done_step_with_absent_path_is_not_reported(status):
    rep = _report(_mp(_step("T.1", status, f"pytest {ABSENT} -q")))
    assert rep["genuine"] == {}, (
        f"status={status} step reported broken for a not-yet-created artifact")
    assert rep["summary"]["by_status"].get(status) == 1  # counted, not dropped


# -- (d) annotation-awareness --------------------------------------------

def test_superseded_record_step_is_dispositioned_not_reported():
    annotated = _step("T.1", "done", f"pytest {ABSENT} -q",
                      superseded_record={"superseded_at": "2026-07-24",
                                         "criteria_amended": False})
    rep = _report(_mp(annotated))
    assert rep["genuine"] == {}
    assert rep["summary"]["annotated_excluded"] == 1


# -- (b) transient / non-source exclusion by construction (criterion 2) --

@pytest.mark.parametrize("ref,label", [
    ("handoff/current/live_check_fixture_75_19.md", "handoff output"),
    ("handoff/logs/auto-push.log", "gitignored runtime log"),
    ("/openapi.json", "URL fragment"),
    ("lib/icons.ts", "frontend-relative path"),
    ("/Library/LaunchAgents/com.py", "truncated abs-host path"),
])
def test_transient_and_nonsource_refs_excluded_by_construction(ref, label):
    rep = _report(_mp(_step("T.1", "done", f"test -f {ref}")))
    assert rep["genuine"] == {}, f"{label} ({ref}) must be excluded by fp_reason class"


# -- (e) container semantics ---------------------------------------------

def test_subphases_steps_are_scanned():
    mp = _mp(_step("T.sub", "done", f"pytest {ABSENT} -q"), container="subphases")
    rep = _report(mp)
    assert "T.sub" in rep["genuine"], "subphases[] steps must be scanned (38.10 is done)"


@pytest.mark.parametrize("container", ["archived_legacy_steps",
                                       "archived_dropped_steps"])
def test_archive_containers_excluded_but_counted(container):
    mp = _mp(_step("T.arc", "done", f"pytest {ABSENT} -q"), container=container)
    rep = _report(mp)
    assert rep["genuine"] == {}, f"{container} entries are archive duplicates"
    assert rep["summary"]["archived_excluded"] == 1


# -- (f) list-shaped verification, asserted BY SHAPE ---------------------

def test_list_shaped_verification_is_scanned():
    ver = ["echo first", f"pytest {ABSENT} -q"]
    assert isinstance(ver, list)  # by-shape: this fixture IS the list case
    rep = _report(_mp(_step("T.1", "done", ver)))
    assert "T.1" in rep["genuine"], "list-shaped verification silently dropped"


def test_dict_shaped_verification_is_scanned():
    rep = _report(_mp(_step("T.1", "done",
                            {"command": f"pytest {ABSENT} -q",
                             "success_criteria": ["x"]})))
    assert "T.1" in rep["genuine"]


# -- (h) shlex-untokenizable: bucketed AND still scanned -----------------

def test_shlex_untokenizable_command_is_still_scanned_via_regex():
    cmd = f'pytest {ABSENT} -q "unclosed'
    rep = _report(_mp(_step("T.1", "done", cmd)))
    assert rep["summary"]["shlex_untokenizable"] == 1
    assert rep["buckets"]["shlex_untokenizable"][0][0] == "T.1"
    assert "T.1" in rep["genuine"], "untokenizable command must still be regex-scanned"


def test_shlex_untokenizable_alone_is_not_broken():
    cmd = f'pytest {EXISTING} -q "unclosed'
    rep = _report(_mp(_step("T.1", "done", cmd)))
    assert rep["genuine"] == {}, "tokenization failure alone must never report broken"
    assert rep["summary"]["shlex_untokenizable"] == 1


# -- (i) import leg, status-gated ----------------------------------------

IMPORT_CMD = 'python -c "from backend.nonexistent_75_19_mod import x"'


def test_unimportable_module_on_done_step_is_reported():
    rep = _report(_mp(_step("T.1", "done", IMPORT_CMD)))
    assert "T.1" in rep["genuine"]
    assert rep["genuine"]["T.1"][0]["kind"] == "import"


def test_unimportable_module_on_pending_step_is_not_reported():
    rep = _report(_mp(_step("T.1", "pending", IMPORT_CMD)))
    assert rep["genuine"] == {}


# -- per-step de-duplication (the old 8.4 double-emission) ---------------

def test_same_ref_twice_in_one_command_emits_one_line():
    rep = _report(_mp(_step("T.1", "done", f"test -f {ABSENT} && cat {ABSENT}")))
    assert rep["summary"]["genuine_lines"] == 1


# -- (g) summary/rows internal consistency (criterion 3) -----------------

def test_fixture_summary_agrees_with_rows():
    rep = _report(_mp(
        _step("T.1", "done", f"pytest {ABSENT} -q && {IMPORT_CMD}"),
        _step("T.2", "done", f"pytest {EXISTING} -q"),
        _step("T.3", "pending", f"pytest {ABSENT} -q"),
    ))
    assert check_consistency(rep) == []
    assert rep["summary"]["genuine_lines"] == 2  # path + import, both on T.1
    assert rep["summary"]["genuine_steps"] == 1
    assert rep["summary"]["live_steps"] == 3
    assert rep["summary"]["scanned_done_unannotated"] == 2


def test_consistency_checker_actually_detects_a_broken_summary():
    # The guard must be able to fail: corrupt a real report and require a
    # complaint (mutating THIS stub is part of the 4c matrix).
    rep = _report(_mp(_step("T.1", "done", f"pytest {ABSENT} -q")))
    rep["summary"]["genuine_lines"] += 1
    assert check_consistency(rep), "corrupted summary must be flagged"


# -- live masterplan: zero unresolved ids + consistency (criterion 3/4) --

def _live_masterplan() -> dict:
    return json.loads(
        (REPO_ROOT / ".claude/masterplan.json").read_text(encoding="utf-8"))


def test_live_report_ids_all_resolve_and_summary_consistent():
    mp = _live_masterplan()
    rep = build_report(mp, REPO_ROOT, git_classify_fn=FAKE_GIT_CLASSIFY)
    assert check_consistency(rep) == []
    live_ids = {str(s.get("id")) for kind, s in iter_steps(mp) if kind == "live"}
    unresolved = set(rep["genuine"]) - live_ids
    assert unresolved == set(), f"reported ids not in masterplan: {unresolved}"


def test_live_masterplan_is_currently_clean():
    # Time-anchored reality pin (mirrors test_phase_75_17 (c)): as of
    # 2026-07-24 every genuinely-unrunnable done step is annotated, so the
    # recalibrated gate must read CLEAN. If this fails later, either a real
    # defect appeared (fix or annotate it -- the gate is working) or the
    # calibration regressed (the rest of this suite localizes which).
    rep = build_report(_live_masterplan(), REPO_ROOT,
                       git_classify_fn=FAKE_GIT_CLASSIFY)
    assert rep["genuine"] == {}, (
        f"unexpected genuine residue: {json.dumps(rep['genuine'], indent=2)}")


def test_live_annotated_count_matches_the_75_17_census():
    # The 75.17 census pins 14 superseded_record HOLDERS repo-wide, but one
    # (68.5) is status=pending -- the status filter excludes it before the
    # annotation branch, so annotated_excluded (done-only) is 13. Measured
    # 2026-07-24; both counts asserted so drift in either direction flags.
    mp = _live_masterplan()
    rep = build_report(mp, REPO_ROOT, git_classify_fn=FAKE_GIT_CLASSIFY)
    holders = [(kind, s) for kind, s in iter_steps(mp)
               if "superseded_record" in s]
    assert len(holders) == 14, "superseded_record holder census drifted from 75.17's 14"
    done_holders = [s for kind, s in holders if s.get("status") == "done"]
    assert len(done_holders) == 13
    assert rep["summary"]["annotated_excluded"] == 13


# -- CLI contract (16.38's immutable consumer + exit codes) --------------

def _run_cli(mp: dict, tmp_path: Path, *flags: str) -> subprocess.CompletedProcess:
    p = tmp_path / "mp.json"
    p.write_text(json.dumps(mp), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/meta/preflight_verify_masterplan.py"),
         str(p), *flags],
        capture_output=True, text=True, timeout=120, check=False,
    )


def test_cli_exit_1_on_genuine_and_0_on_clean(tmp_path):
    broken = _run_cli(_mp(_step("T.1", "done", f"pytest {ABSENT} -q")), tmp_path)
    assert broken.returncode == 1
    assert "[GENUINE] step=T.1" in broken.stderr
    clean = _run_cli(_mp(_step("T.1", "done", f"pytest {EXISTING} -q")), tmp_path)
    assert clean.returncode == 0


def test_cli_quiet_flag_survives_16_38(tmp_path):
    r = _run_cli(_mp(_step("T.1", "done", f"pytest {EXISTING} -q")),
                 tmp_path, "--quiet")
    assert r.returncode == 0
    assert r.stdout == ""


def test_cli_exit_2_on_malformed_json(tmp_path):
    p = tmp_path / "mp.json"
    p.write_text("{not json", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/meta/preflight_verify_masterplan.py"),
         str(p)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert r.returncode == 2
