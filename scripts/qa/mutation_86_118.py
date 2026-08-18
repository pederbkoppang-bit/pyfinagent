#!/usr/bin/env python3
"""phase-86.118 criterion 7 -- mutation matrix for every guard this step added.

Built on `scripts/qa/guardlib.py`, which supplies the scoring discipline this
project has paid for repeatedly:

* the CONTROL is observed GREEN first, per target; a red control makes that
  target's cells UNSCORABLE rather than producing a row of kills;
* `pytest` exit 5 is "no tests collected" and must NEVER score as a kill, so a
  mutant has to exit **1**;
* the mutant must COLLECT THE SAME NUMBER OF TESTS as the control, so a mutation
  that breaks import (collecting zero) is UNSCORABLE;
* the NAMED test must be among the failures -- "something went red" is not
  evidence that THIS guard fired. Step 86.59 shipped a cell whose kill was on a
  path that published nothing, and this rule is what catches that;
* restore is verified by SHA-256, and SIGTERM/SIGINT/SIGHUP restore EVERY
  target (`try/finally` does not run on a signal).

Run: ``python scripts/qa/mutation_86_118.py``

HAZARD: this edits real files in place and restores them. Never wire it into a
hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardlib import Cell, MutationMatrix, Target, pytest_runner  # noqa: E402

# The 75_17 suite still carries ONE known-red test that this step deliberately
# did not repair (`masterplan_diff_touches_only_the_ten_sibling_insertions` --
# its correct pin is contested; see live_check §5). A red control makes EVERY
# cell on that target UNSCORABLE, so it is deselected by name and the exclusion
# is stated here rather than hidden behind a `-k` expression. Nothing else is
# excluded: the deselect names one test, so a second failure appearing in this
# file would still turn the control red and stop the matrix.
DESELECT_KNOWN_RED = [
    "--deselect",
    "backend/tests/test_phase_75_17_verification_paths.py"
    "::test_masterplan_diff_touches_only_the_ten_sibling_insertions",
]

T_5717 = REPO / "backend/tests/test_phase_75_17_verification_paths.py"
T_PROMPT = REPO / "backend/tests/test_phase_75_prompt_contracts.py"
T_SRE = REPO / "backend/tests/test_phase_75_sre_ops.py"
T_571 = REPO / "backend/tests/test_phase_57_1_reject_binding.py"
T_603 = REPO / "backend/tests/test_phase_60_3_data_integrity.py"
T_402 = REPO / "backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py"
SWEEP = REPO / "scripts/qa/sweep_absent_verification_paths.py"

# The sweep classifier is not a test file, so its control is the pair of suites
# that consume it -- the same two tests its defect had held red.
SWEEP_SUITE = [
    "backend/tests/test_phase_75_17_verification_paths.py",
    "backend/tests/test_phase_75_19_preflight_calibration.py",
]

TARGETS = [
    Target(T_5717, pytest_runner(str(T_5717.relative_to(REPO)), REPO,
                                 DESELECT_KNOWN_RED), name="75_17"),
    Target(T_PROMPT, pytest_runner(str(T_PROMPT.relative_to(REPO)), REPO), name="75_prompt"),
    Target(T_SRE, pytest_runner(str(T_SRE.relative_to(REPO)), REPO), name="75_sre_ops"),
    Target(T_571, pytest_runner(str(T_571.relative_to(REPO)), REPO), name="57_1"),
    Target(T_603, pytest_runner(str(T_603.relative_to(REPO)), REPO), name="60_3"),
    Target(T_402, pytest_runner(str(T_402.relative_to(REPO)), REPO), name="40_2"),
    Target(SWEEP, pytest_runner(SWEEP_SUITE, REPO,
                                DESELECT_KNOWN_RED), name="sweep_classifier"),
]

CELLS = [
    # ── the classifier repair (rows 9 + 11) ────────────────────────────────
    Cell(
        "M1 the `||`-alternative branch is removed -> a satisfied fallback arm "
        "is reported as a genuine absent path again",
        SWEEP,
        '    if "||" in cmd:',
        '    if False:  # MUTANT',
        "test_sweep_over_live_masterplan_is_clean",
    ),
    Cell(
        "M1b ... and the SIBLING test that the same one-line defect held red",
        SWEEP,
        "                    if o and o != t and _resolves_on_disk(repo_root, o):",
        "                    if False:  # MUTANT",
        "test_sweep_over_live_masterplan_is_clean",
    ),
    # ── the git-pin (row 10) ───────────────────────────────────────────────
    Cell(
        "M2 the shape census reads the LIVE masterplan again -> it rots with "
        "every masterplan commit",
        T_5717,
        "    pinned = _masterplan_at(BASELINE_COMMIT)",
        "    pinned = _masterplan_at(None)  # MUTANT",
        "test_sweep_shape_census_matches_the_corrected_figures",
    ),
    # ── the archive-aware resolvers (rows 12 + 13) ─────────────────────────
    Cell(
        "M3 the resolver stops searching the archive -> a consumed artifact "
        "reads as missing",
        T_PROMPT,
        '                  *sorted((root / "handoff" / "archive").rglob(name))]',
        "                  ]  # MUTANT",
        "test_operator_decision_note_exists_with_token",
    ),
    Cell(
        "M3b ... same defect in the sre-ops copy",
        T_SRE,
        '                  *sorted((REPO_ROOT / "handoff" / "archive").rglob(name))]',
        "                  ]  # MUTANT",
        "test_c1_runbook_and_operator_token_drafted",
    ),
    Cell(
        # RE-AIMED after the first run. The original cell mutated the `raise`,
        # which is UNREACHABLE while the artifact is found in the archive -- so
        # it changed nothing and SURVIVED. That was a mutation the current tree
        # makes equivalent, not a gap in the guard. This version makes the
        # resolver return empty text on EVERY input, which is the failure mode
        # worth guarding: silent empty content turns the token assertions
        # vacuous instead of failing.
        "M3c the resolver returns empty text instead of raising -> the token "
        "assertions would pass over nothing",
        T_PROMPT,
        "    for path in candidates:\n"
        "        if path.is_file():\n"
        "            return path.read_text(encoding=\"utf-8\")",
        '    return ""  # MUTANT',
        "test_operator_decision_note_exists_with_token",
    ),
    # ── shipped-default assertions (rows 3 + 6) ────────────────────────────
    Cell(
        "M4 the default assertion reads the DEPLOYMENT again -> a legitimate "
        "operator promotion in backend/.env reddens the test",
        T_603,
        '    assert Settings.model_fields["paper_data_integrity_enabled"].default is False',
        "    assert Settings().paper_data_integrity_enabled is False  # MUTANT",
        "test_60_3_flag_defaults_off",
    ),
    # ── the fixture pin (rows 3/4/5) ───────────────────────────────────────
    Cell(
        "M5 the fixture stops PINNING the flag -> the 'flag-OFF' tests inherit "
        "backend/.env and run flag-ON, which is the original defect",
        T_571,
        '        "paper_risk_judge_reject_binding": False,',
        "        # MUTANT: pin removed",
        "test_reject_binding_swap_path_off_emits_on_blocks",
    ),
    # ── the effort-policy pin (row 2) ──────────────────────────────────────
    Cell(
        "M6 the effort pin is reverted to the superseded value -> the guard "
        "contradicts CLAUDE.md instead of catching a clobber",
        T_402,
        '    assert parsed.get("effortLevel") == "max", (',
        '    assert parsed.get("effortLevel") == "xhigh", (  # MUTANT',
        "test_phase_40_2_settings_json_still_valid_json_after_edit",
    ),
    # ── the bootstrap oracle (row 14) ──────────────────────────────────────
    Cell(
        "M7 the oracle is disarmed entirely -> it can never reject anything, "
        "which is the vacuity the previous version shipped",
        T_SRE,
        "    hint_vars: set[str] = set()\n    offenders: list[str] = []",
        "    return []  # MUTANT\n    hint_vars: set[str] = set()\n    offenders: list[str] = []",
        "test_c6_no_launchctl_bootstrap_executed_in_ops_scripts",
    ),
    Cell(
        "M7b quoted-span stripping removed -> the oracle fires on a hint STRING "
        "again, which is exactly why this test was red",
        T_SRE,
        '        if "launchctl bootstrap" in _QUOTED.sub("", stripped):',
        '        if "launchctl bootstrap" in stripped:  # MUTANT',
        "test_c6_no_launchctl_bootstrap_executed_in_ops_scripts",
    ),
    Cell(
        "M7c the eval-of-a-hint-variable branch is removed -> the one hazard "
        "that no line-local check can see goes unguarded",
        T_SRE,
        '        if re.search(r"\\b(eval|bash\\s+-c|sh\\s+-c)\\b", stripped) and (',
        "        if False and (  # MUTANT",
        "test_c6_no_launchctl_bootstrap_executed_in_ops_scripts",
    ),
    Cell(
        "M7d the assignment branch is removed -> storing the verb in a variable "
        "reads as executing it, re-breaking the real ops script",
        T_SRE,
        "            hint_vars.add(assign.group(1))   # storing it is not running it\n"
        "            continue",
        "            hint_vars.add(assign.group(1))  # MUTANT\n"
        "            offenders.append(stripped)",
        "test_c6_no_launchctl_bootstrap_executed_in_ops_scripts",
    ),
]


# Declared UP FRONT with the measurement that justifies it, per the project's
# standing discipline: an equivalent mutation is stated, never quietly omitted.
EQUIVALENT_BY_DESIGN = [
    (
        "M4b substituting `assert s_off.paper_risk_judge_reject_binding is False` "
        "for the declared-default assertion in test_phase_57_1",
        "EQUIVALENT BY DESIGN, and the equivalence IS the finding. The fixture "
        "now PINS that flag to False, so an assertion against the fixture "
        "instance restates what the fixture just set and cannot fail for any "
        "input -- measured: the substitution ran GREEN. That is precisely why "
        "the assertion was re-aimed at the declared field default. The "
        "load-bearing mutation for the surviving assertion is a change to the "
        "DECLARED default in backend/config/settings.py, which was driven by "
        "hand and observed RED (1 failed, 6 passed) with a SHA-256-verified "
        "restore. It is deliberately NOT automated here: a concurrent peer "
        "session is editing settings.py for step 86.120, and a "
        "backup/mutate/restore cycle on a file another session is writing can "
        "silently revert its work.",
    ),
]


def main() -> int:
    return MutationMatrix(
        TARGETS, CELLS,
        equivalent_by_design=EQUIVALENT_BY_DESIGN,
        title="phase-86.118 -- MUTATION MATRIX (criterion 7)",
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
