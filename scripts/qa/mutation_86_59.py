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
        '    _ok("sector_map_covers_most_of_the_panel"',
        '    known = 1\n'
        '    _ok("sector_map_covers_most_of_the_panel"',
        "sector_map_covers_most_of_the_panel",
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


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(mode: list[str]) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, str(TARGET), *mode, "--cycles", "4"],
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
