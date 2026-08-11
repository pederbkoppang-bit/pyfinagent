---
name: claims-correctness-86-34
description: Step 86.34 measurements -- a TZ shift can never be hour-independent (|offset|/24 of the day only); .venv.py313.bak defeats every exact-match filter (95.5% of the scanned .py population); a "verbatim" digest block went stale from a legitimate edit
metadata:
  type: project
---

Three findings from step 86.34 (2026-08-10), all MEASURED, all recurring classes
rather than one-off instances.

## 1. A `TZ=` shift is structurally incapable of being hour-independent

For a zone with fixed offset `o`, `local_date != utc_date` holds on **exactly
`|o|` of the 24 UTC hours**. Measured across all 24 hours:

```
Pacific/Midway   -11:00  deltas [-1, 0]   green 11/24
Pacific/Kiritimati +14:00 deltas [ 0, 1]  green 14/24
Etc/GMT+12       -12:00  deltas [-1, 0]   green 12/24
```

**No fixed-offset zone yields a constant non-zero delta.** So any test whose
positive control asserts `local_date != utc_date` under one hard-coded `TZ=` is
RED for `24 - |o|` hours a day. `test_phase_86_24_clock_dependence.py:247/:261`
was authored and measured at 07:45-08:20 UTC, inside its own green window, and
was measured RED at 16:51 UTC in the very next step.

Hour-independent shapes: pick the zone from the current UTC hour; run both
directions and require the shift to land in one; or inject a local (non-global)
clock.

**Direction is not decorative.** For an *equality* between a local-domain and a
UTC-domain date, `-1` and `+1` both break it, so either direction detects the
bug. For *ordered* date logic (staleness `anchor < today`, rollover, partitions,
trading-day gating) they hit opposite branches: behind-UTC makes a UTC anchor
look FUTURE, ahead-UTC makes it look STALE. CEST 00:00-02:00 is **ahead** (+1);
`Pacific/Midway` is **behind** (-1).

**Why:** the 86.24 artifacts asserted Midway reproduces the CEST window. It
reproduces the opposite direction, and only for part of the day.
**How to apply:** whenever a fixture names a timezone, ask for BOTH the
direction it produces and the UTC hours over which it holds. Related:
[[project_clock_dependent_tests_86_24]].

## 2. This repo has a SECOND virtualenv that defeats exact-match filters

`.venv.py313.bak` sits next to `.venv`. Every `if ".venv" in path.parts` filter
lets it through, because it is a different string.

```
repo **/*.py kept by EXACT-match  ".venv" filter : 23183
repo **/*.py kept by PREFIX-match ".venv*" filter:  1052
git ls-files '*.py'                             :  1051
```

The exact-match filter admits **22,131 vendored files (95.5%)**. Prefix-match
lands within ONE file of git's own answer (the difference is an untracked
first-party file).

`git check-ignore -v .venv.py313.bak` -> `.gitignore:16:.venv*/` -- **git
already knows**; only the hand-rolled filters do not. pytest's shipped default
`norecursedirs` is `["*.egg", ".*", "_darcs", "build", "CVS", "dist",
"node_modules", "venv", "{arch}"]` -- the **glob `.*`**, not the name `.venv`.
Ruff ships `.venv` in its default exclude AND layers `respect-gitignore` on top,
because a fixed name list is known-incomplete.

Known sites of the bug: `test_phase_86_24_clock_dependence.py:205`,
`scripts/qa/verify_unused_imports_86_26.py:78`,
`backend/tests/test_phase_82_6_bridge_design.py:140`. The repo's own correct
precedent (still a hand-maintained list): `scripts/governance/lint_limits_usage.py:79`.

Also: `REPO.glob("conftest.py") + REPO.glob("**/conftest.py")` double-counts the
root file -- `**` already matches zero directories.

**Why:** a repo-wide guard whose population is 94-95% vendored is green by
accident; its true subject was 2 files.
**How to apply:** before trusting any repo-wide scan, print its denominator and
compare against `git ls-files`. Related:
[[feedback_measure_dont_assert_claims]], [[feedback_guard_from_instance_not_class]].

## 3. A digest recorded inside a "verbatim" block goes stale on a legitimate edit

`live_check_86.24.md:156` records `5c1ce1116769d118` for
`test_phase_86_2_replay_poison_row.py`; current is `fb97b52ecf7fb5be`. The edit
was legitimate and documented (`da9263d6`), and the same artifact records the
edit at `:75`. The header at `:3` (`Code commit: d5180e27`) is likewise two
commits behind.

Practice, from arXiv:2605.12087 (supersession is a semantic relation; a
superseding step creates a NEW artifact and flips the prior one's `status` to
`superseded`) plus SLSA provenance (record the resolved digest *beside* the
mutable ref so drift is detectable). The in-repo template already exists:
`live_check_86.24.md` §D strikes its withdrawn claim through and dates the
correction; §F did not get the same treatment.

**Why:** a reader re-running the checker today sees a different digest and cannot
tell expected drift from "someone wrote to a tracked source".
**How to apply:** regenerate the block and mark the old one superseded; never
silently overwrite, never leave it standing. Related:
[[project_silent_glob_archive_86_29]].
