#!/usr/bin/env python3
"""phase-86.59 criterion 7 -- mutation matrix for the rank-stability measurement.

A guard that cannot fail is not a guard.  Every invariant in
``rank_stability_86_59.py`` is reverted here one at a time, and the run must go
RED **naming that guard**.  Scoring discipline, learned the hard way on prior
steps:

* the CONTROL is observed GREEN first; if it is not, every cell is UNSCORABLE
  and the matrix reports that instead of a row of kills;
* a non-zero exit is NOT a kill on its own -- the named guard must appear in
  the output, otherwise the mutant merely crashed somewhere else and the guard
  was never reached;
* a mutant whose text does not apply (anchor not found) is UNSCORABLE, never a
  kill -- a typo would otherwise score as a success;
* restore is verified by SHA-256 against the pre-mutation bytes, so a partial
  restore cannot silently poison the next cell.

Run: ``python scripts/qa/mutation_86_59.py``
"""

from __future__ import annotations

import hashlib
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "scripts" / "qa" / "rank_stability_86_59.py"

# (cell, mode, old, new, guard that must fire)
CELLS: list[tuple[str, list[str], str, str, str]] = [
    (
        "M1 de-duplication removed -> the panel keeps duplicate (ticker,date) keys",
        ["--verify"],
        'df = df.drop_duplicates(subset=["ticker", "date"], keep="first")',
        "df = df  # MUTANT: dedup removed",
        "panel_keys_unique_after_dedup",
    ),
    (
        "M2 full ranking truncated -> it no longer covers the cross-section",
        ["--verify"],
        "        screen_data, full = replay_session(\n"
        "            df, tickers, sess, sector_lookup=sectors, top_n=10**9\n        )",
        "        screen_data, full = replay_session(\n"
        "            df, tickers, sess, sector_lookup=sectors, top_n=5\n        )",
        "full_rank_covers_cross_section",
    ),
    (
        "M3 slate taken from the WRONG offset -> disagrees with an independent top_n call",
        ["--verify"],
        "        ranked = full[:SCREEN_TOP_N]",
        "        ranked = full[1:SCREEN_TOP_N + 1]  # MUTANT",
        "slate_matches_an_independent_top_n_call",
    ),
    (
        "M4 trailing window off by one -> factor definitions shift",
        ["--verify"],
        "        sess = all_sessions[end_i - window + 1: end_i + 1]\n"
        '        _ok(f"window_len_{d}"',
        "        sess = all_sessions[end_i - window + 2: end_i + 1]\n"
        '        _ok(f"window_len_{d}"',
        "window_len_",
    ),
    (
        "M5 adjacency check dropped -> non-consecutive sessions would pass as consecutive",
        ["--verify"],
        "    replay_dates = all_sessions[window - 1:][-n_cycles:]\n"
        '    _ok("replay_dates_are_consecutive_sessions"',
        "    replay_dates = all_sessions[window - 1:][-n_cycles:][::2]\n"
        '    _ok("replay_dates_are_consecutive_sessions"',
        "replay_dates_are_consecutive_sessions",
    ),
    (
        "M6 pair count decoupled from the cycle count",
        ["--verify"],
        "    for i in range(1, len(results)):\n        rho = spearman",
        "    for i in range(2, len(results)):\n        rho = spearman",
        "pairs_is_cycles_minus_one",
    ),
    (
        "M7 correlation silently unavailable -> rho would be reported from nothing",
        ["--verify"],
        "    common = sorted(set(a) & set(b))\n    n = len(common)",
        "    common = sorted(set(a) & set(b))[:2]\n    n = len(common)",
        "every_pair_has_a_rho",
    ),
    (
        "M8 sector map dropped -> criterion 3 would measure the harness, not the book",
        ["--verify"],
        "def load_sectors() -> dict[str, str]:\n    if not SECTORS.exists():",
        "def load_sectors() -> dict[str, str]:\n    return {}\n    if not SECTORS.exists():",
        "sector_map_present",
    ),
    (
        "M9 sector coverage collapsed -> a mostly-UNKNOWN map would pass",
        ["--verify"],
        '    known = sum(1 for v in sectors.values() if v)\n'
        '    # CYCLE-4 FINDING 2.',
        '    known = 1  # MUTANT\n'
        '    # CYCLE-4 FINDING 2.',
        "sector_map_covers_the_panel_at_the_published_operating_point",
    ),
    (
        # CYCLE-4 FINDING 2. M9 collapses coverage to 1/513, which the OLD 50%
        # floor also caught. This cell is the one that discriminates: 401/513 =
        # 78.2% is the level the Q/A used to SWAP soft_diversity's and min_k's
        # turnover cost, and it is 28pp ABOVE the old floor -- so it SURVIVED
        # before this fix and must be KILLED after it. A cell that only
        # reproduces M9's collapse would prove nothing about the change.
        "M9b coverage degraded to the level that INVERTS the published ordering "
        "(78.2%, above the old 50% floor)",
        ["--verify"],
        '    known = sum(1 for v in sectors.values() if v)\n'
        '    # CYCLE-4 FINDING 2.',
        '    known = int(0.782 * len(tickers))  # MUTANT\n'
        '    # CYCLE-4 FINDING 2.',
        "sector_map_covers_the_panel_at_the_published_operating_point",
    ),
    (
        "M10 fidelity comparison emptied -> the replay becomes unfalsifiable",
        ["--verify"],
        "    observed = load_observed()\n    fid = []",
        "    observed = {}\n    fid = []",
        "fidelity_has_at_least_one_comparable_session",
    ),
    (
        "M11 dispersion skips cycles -> finding (a) computed on a partial window",
        ["--dispersion"],
        "        if None in (s1, s3, s6):\n            continue",
        "        if True:\n            continue",
        "dispersion_measured_on_every_cycle",
    ),
    (
        "M12 tie-explanation weakened -> a real reweighting would pass as a rounding artefact",
        ["--dispersion"],
        "    return moved <= 2 * max(ties, 0)",
        "    return moved >= 0  # MUTANT: accepts anything",
        "tie_rule_rejects_known_bad_inputs",
    ),
    (
        "M14 baseline arm POISONED with a flag -> every criterion-4 delta reads zero",
        ["--flags"],
        '    ("baseline",              {}),',
        '    ("baseline",              {"sector_neutral": True}),  # MUTANT',
        "baseline_arm_applies_no_flags",
    ),
    (
        # CYCLE-4 FINDING 1, half one: the CALL SITE drifts from the label.
        # This is the exact edit the Q/A executed (k=4) and which SURVIVED
        # before the fix, reporting +6.3pp under a row still labelled
        # `min_k_sectors=3` -- tying ASK-2 and inverting the ordering ASK-1's
        # promotion recommendation rests on.
        "M23 min_k call site passes a k the row label does not claim",
        ["--flags"],
        "        _k = MIN_K_SECTORS\n",
        "        _k = 4  # MUTANT\n",
        "min_k_arm_used_the_labelled_k",
    ),
    (
        # CYCLE-4 FINDING 1, half two: the LABEL drifts from the call site.
        # Mutating only one side would leave the other direction unfalsified --
        # a compound property licenses nothing about the clause it did not
        # exercise -- so both directions get their own cell.
        "M24 min_k row label claims a k the call site did not pass",
        ["--flags"],
        'MIN_K_ARM = f"min_k_sectors={MIN_K_SECTORS}"',
        'MIN_K_ARM = "min_k_sectors=9"  # MUTANT',
        "min_k_arm_used_the_labelled_k",
    ),
    (
        "M15 every arm made identical to baseline -> deltas degenerate silently",
        ["--flags"],
        "    return all(arms[k].get(\"distinct\") != base for k in others)",
        "    return any(arms[k].get(\"distinct\") != base for k in others)  # MUTANT",
        "predicates_reject_known_bad_inputs",
    ),
    (
        "M16 a suffixed non-US ticker reaches the panel -> the universe silently widens",
        ["--verify"],
        '    return not any("." in tk for tk in tickers)',
        "    return True  # MUTANT",
        "predicates_reject_known_bad_inputs",
    ),
    (
        "M17 window/cycle budget check dropped -> a short panel would run truncated",
        ["--verify"],
        "    return n_sessions >= window + n_cycles",
        "    return n_sessions >= 0  # MUTANT",
        "predicates_reject_known_bad_inputs",
    ),
    (
        "M18 dedup tripwire made vacuous -> a repaired-or-broken dedup goes unnoticed",
        ["--verify"],
        "    return dropped > 0",
        "    return dropped >= 0  # MUTANT",
        "predicates_reject_known_bad_inputs",
    ),
    (
        "M19 multidim arm silently skipped -> finding (a)'s rebuttal rests on nothing",
        ["--dispersion"],
        "        rank_identical.append({",
        "        if True: continue  # MUTANT\n        rank_identical.append({",
        "price_only_multidim_arm_ran",
    ),
    (
        "M20 a flag injected into the BASELINE reference at the replay seam "
        "(definition left byte-identical) -> every delta reads against a "
        "flagged baseline",
        ["--flags"],
        "        _sd, base = replay_session(df, tickers, sess, sector_lookup=sectors)",
        "        _sd, base = replay_session(df, tickers, sess, sector_lookup=sectors,\n"
        "                                   momentum_52wh_tilt=True,\n"
        "                                   momentum_52wh_tilt_k=0.2)  # MUTANT",
        "baseline_slate_matches_an_unflagged_direct_call",
    ),
    (
        "M22 a flag injected at the ARMS LOOP -> the baseline ROW every delta "
        "is subtracted from is poisoned while each arm looks unchanged",
        ["--flags"],
        "            _sd, ranked = replay_session(\n"
        "                df, tickers, sess, sector_lookup=sectors, **kw\n"
        "            )",
        "            _sd, ranked = replay_session(\n"
        "                df, tickers, sess, sector_lookup=sectors,\n"
        "                **({**kw, 'momentum_52wh_tilt': True,\n"
        "                   'momentum_52wh_tilt_k': 0.2} if not kw else kw)\n"
        "            )  # MUTANT",
        "baseline_ROW_matches_an_unflagged_direct_call",
    ),
    (
        "M21 predicate fixture emptied -> the four predicate-backed guards go blind",
        ["--verify"],
        "_PREDICATE_FIXTURE: list[tuple[str, bool]] = [",
        "_PREDICATE_FIXTURE: list[tuple[str, bool]] = []\n_UNUSED_FIXTURE = [  # MUTANT",
        "fixture_exercises_every_predicate_on_rejecting_inputs",
    ),
    (
        "M13 flag arm silently produces no turnover series",
        ["--flags"],
        "        turns = [one_sided_turnover(slates[i - 1], slates[i])\n"
        "                 for i in range(1, len(slates))]",
        "        turns = []",
        "flag_arms_all_ran",
    ),
]

# M12's mutant must be checked for NON-EQUIVALENCE: `len(moved) >= 0` is always
# true, so the guard can never fire -- which means the CONTROL must be the thing
# that proves the guard fires at all. That is what the paired negative below
# does: it forces a displacement count the real rule rejects.
NEGATIVE_CONTROL = (
    "M12b tie-explanation guard proven CAPABLE of firing (paired negative)",
    ["--dispersion"],
    "    return moved <= 2 * max(ties, 0)",
    "    return moved <= -1  # MUTANT: rule made unsatisfiable",
    "tie_rule_rejects_known_bad_inputs",
)


# --------------------------------------------------------------------------
# COVERAGE -- the structural fix for "a matrix licenses only its own cells"
# --------------------------------------------------------------------------

# A cell list proves things about the cells in it and nothing about the guards
# it forgot. The cycle-1 Q/A made exactly that finding: 19 guards existed, 13
# were named by a cell, and two of the six uncovered ones were UNKILLABLE. So
# coverage is now itself checked: every `_ok("name", ...)` in the target must be
# named by a cell or listed below with a reason. Adding a guard without a cell
# FAILS this matrix.
COVERED_TRANSITIVELY: dict[str, str] = {
    # `displacements_are_tie_explained` consumes `_tie_explained`, whose own
    # fixture guard (`tie_rule_rejects_known_bad_inputs`) is mutated by M12/M12b.
    "displacements_are_tie_explained":
        "covered via _tie_explained -- cells M12 and M12b",
    "panel_carries_no_non_us_symbols":
        "covered via the _us_only predicate fixture -- cell M16",
    "enough_sessions_for_window":
        "covered via the _enough_sessions predicate fixture -- cell M17",
    "dedup_actually_fired_on_this_panel":
        "covered via the _dedup_fired predicate fixture -- cell M18",
    "flag_arms_are_distinguishable_from_baseline":
        "covered via the _arms_distinguishable predicate fixture -- cell M15",
    "baseline_slate_matches_an_unflagged_direct_call":
        "the min_k reference; cell M20 injects at that call site",
    # These are per-cycle f-string guards; the cells target the literal prefix.
    "window_len_": "cell M4 (f-string guard, matched by prefix)",
    "full_rank_covers_cross_section_": "cell M2 (f-string guard)",
    "slate_matches_an_independent_top_n_call_": "cell M3 (f-string guard)",
}


def guard_names_in_target() -> list[str]:
    """Every `_ok(...)` invariant name in the target, via AST -- not regex.

    A regex over source text would match this file's own prose about guard
    names; the AST sees only real calls.
    """
    import ast

    tree = ast.parse(TARGET.read_text())
    names = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_ok" and node.args):
            continue
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            names.append(a.value)
        elif isinstance(a, ast.JoinedStr):
            # f"window_len_{d}" -> the literal prefix
            lead = "".join(v.value for v in a.values
                           if isinstance(v, ast.Constant) and isinstance(v.value, str))
            names.append(lead)
    return sorted(set(names))


def check_coverage() -> tuple[bool, list[str]]:
    named = {guard for *_, guard in [*CELLS, NEGATIVE_CONTROL]}
    uncovered = []
    for g in guard_names_in_target():
        if any(g.startswith(n) or n.startswith(g) for n in named):
            continue
        if g in COVERED_TRANSITIVELY:
            continue
        uncovered.append(g)
    return (not uncovered), uncovered


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Cycle counts per mode. The cycle-2 Q/A found that kill/survive can be
# CYCLE-COUNT DEPENDENT (a w=0.15 baseline poison survived at 4 cycles and was
# killed at 20), so running every cell at 4 made the matrix a WEAKER oracle than
# the published run. Criterion-4 cells now run at the published 20.
MODE_CYCLES: dict[str, str] = {
    "--flags": "20",
    "--dispersion": "4",
    "--verify": "4",
}


def run(mode: list[str]) -> tuple[int, str]:
    cycles = MODE_CYCLES.get(mode[0], "4")
    r = subprocess.run(
        [sys.executable, str(TARGET), *mode, "--cycles", cycles],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return r.returncode, (r.stdout + r.stderr)


def main() -> int:
    original = TARGET.read_bytes()
    base_sha = hashlib.sha256(original).hexdigest()

    # POISON CHECK. This script writes mutations to a real file on disk, so an
    # interrupted run can STRAND one there. That is not hypothetical: a 2-minute
    # command timeout SIGTERMed this matrix mid-cell on 2026-08-18 and left
    # `return moved >= 0  # MUTANT` in the target, which then failed the very
    # fixture it was added to protect. Refuse to start from a poisoned baseline
    # rather than mutate a mutant and score the result.
    if b"MUTANT" in original:
        print("REFUSING TO RUN: the target already contains a stranded MUTANT "
              "marker.\nRestore it (git checkout / manual revert) before "
              "scoring anything -- a matrix run from a poisoned baseline is "
              "not a measurement.")
        return 2

    # try/finally does NOT run on SIGTERM, so the restore is also wired to the
    # signals a timeout actually sends.
    def _restore_and_die(signum, _frame):
        TARGET.write_bytes(original)
        print(f"\nsignal {signum} -- target restored, matrix aborted")
        raise SystemExit(130)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _restore_and_die)
        except (ValueError, OSError):
            pass

    print("=" * 78)
    print("phase-86.59 -- MUTATION MATRIX (criterion 7)")
    print("=" * 78)
    print(f"target : {TARGET.relative_to(REPO)}")
    print(f"sha256 : {base_sha[:16]}...")
    print()

    # ---- CONTROL FIRST. No cell is scorable until this is green. ----------
    control_fail = []
    for mode in (["--verify"], ["--dispersion"], ["--flags"]):
        rc, out = run(mode)
        state = "GREEN" if rc == 0 else "RED"
        print(f"control {' '.join(mode):<14} -> rc={rc} {state}")
        if rc != 0:
            control_fail.append((mode, out[-800:]))
    print()
    if control_fail:
        print("CONTROL IS NOT GREEN -- every cell below is UNSCORABLE.")
        for mode, out in control_fail:
            print(f"\n--- {' '.join(mode)} ---\n{out}")
        return 1

    # ---- COVERAGE GATE. Criterion 7 says EVERY new guard, not "these N". ---
    covered, uncovered = check_coverage()
    all_guards = guard_names_in_target()
    print(f"coverage: {len(all_guards)} guards in target, "
          f"{len(all_guards) - len(uncovered)} covered by a cell or an explicit "
          f"transitive entry")
    if not covered:
        print("COVERAGE GAP -- these guards have no mutation cell:")
        for g in uncovered:
            print(f"  - {g}")
        print("Add a cell, or record it in COVERED_TRANSITIVELY with a reason. "
              "A matrix licenses only the cells it contains.")
        return 1
    print()

    results = []
    for name, mode, old, new, guard in [*CELLS, NEGATIVE_CONTROL]:
        text = original.decode()
        if text.count(old) != 1:
            results.append((name, "UNSCORABLE",
                            f"anchor appears {text.count(old)}x, expected exactly 1"))
            print(f"[UNSCORABLE] {name}\n             anchor not unique")
            continue
        TARGET.write_text(text.replace(old, new, 1))
        try:
            rc, out = run(mode)
        finally:
            TARGET.write_bytes(original)
            assert sha(TARGET) == base_sha, "RESTORE FAILED -- tree is dirty"

        if rc == 0:
            verdict, detail = "SURVIVED", "run stayed green under the mutation"
        elif guard in out:
            verdict, detail = "KILLED", f"guard `{guard}` fired"
        else:
            verdict, detail = "UNSCORABLE", (
                f"rc={rc} but `{guard}` never appeared -- the mutant crashed "
                f"elsewhere and the guard was never reached")
        results.append((name, verdict, detail))
        print(f"[{verdict}] {name}\n           {detail}")

    print()
    print("-" * 78)
    killed = sum(1 for _, v, _ in results if v == "KILLED")
    survived = [n for n, v, _ in results if v == "SURVIVED"]
    unscorable = [n for n, v, _ in results if v == "UNSCORABLE"]
    print(f"KILLED {killed} / {len(results)}   "
          f"SURVIVED {len(survived)}   UNSCORABLE {len(unscorable)}")
    for n in survived:
        print(f"  SURVIVED: {n}")
    for n in unscorable:
        print(f"  UNSCORABLE: {n}")
    print(f"restore verified: sha256 unchanged ({sha(TARGET)[:16]}...)")
    return 0 if not survived and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
