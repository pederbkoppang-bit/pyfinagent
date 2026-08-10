---
name: clock-dependent-tests-86-24
description: Three tests flipped at midnight = TWO different bugs 2h apart; kill-switch staleness is correct-by-design; no static method has validatable recall
metadata:
  type: project
---

Step 86.24 (tests changing colour with the wall clock). Findings that are not
re-derivable by reading the code cold.

**The three "midnight" tests were TWO distinct bugs whose flip instants were 2h
apart.** `test_phase_82_0_macro_ingestion.py` (2 tests) = a **local-vs-UTC
clock-domain mismatch**: production reads `datetime.now(timezone.utc).date()`
(`data_ingestion.py:344`, `:375`), the tests assert against `date.today()` (LOCAL).
On a CEST box that disagrees for exactly 00:00-02:00 local, every night, and heals --
recurring, self-healing, 2h wide. `test_phase_86_2_replay_poison_row.py:115` = a
classic **time bomb**: fixture pins `sod_date='2026-08-09'`, judged by
`kill_switch.py:986` against UTC now; it went red at 00:00 UTC = 02:00 CEST -- i.e.
the exact instant the other two healed -- and never heals. A single remedy for
"the midnight tests" is wrong for one of them.

**Why:** conflating them produces a fix that either freezes the clock (killing the
staleness guard) or rewrites the wrong side of the comparison.

**How to apply:** when a test flips "at midnight", ask WHICH midnight. Compare the
timezone of the clock read on EACH side before touching anything.

**The kill-switch daily-anchor staleness is correct-by-design -- do not "fix" it.**
Measured: `evaluate_breach` returned `any_breached: True` with `armed: False`. The
rule is per-LEG (`kill_switch.py:865-867`); the trailing high-water leg is not
date-scoped and keeps firing. It was installed against a MEASURED live incident (a
two-day move reported as a same-day loss, `:857-861`), the order-placing gate reads
`baselines_present` not `armed` (`:868-881`), and phase-85.6 gave it an out-of-band
exit (`paper_trader.py:1220-1300`). See [[kill-switch-armed-36-9]] and
[[stale-anchor-disarm-85-5-1]].

**No static derivation of the date-dependent population has validatable recall, and
this is structural.** Measured over 457 test files: own-clock AST scan = 49 files and
**MISSES** the poison-row test (it contains zero clock calls -- the clock read is in
PRODUCTION). Date-literal regex = 129 files. "literal AND imports a clock-reading
production module" = 90 files, catches both, best static option. Union = 123.
Externally: 66 catalogued test smells across 22 tools, **none** time/date-related and
**no Python tool at all** (arXiv 2104.14640). The recall-validatable method is a
**differential run at a shifted clock**; a single `+25h` offset flags all three known
positives. Prior art: Debian reproducible-builds runs a live 398-day + 6h23m + TZ
GMT+12/GMT-14 variation; ChaosAPI (OOPSLA 2026) shows clock-API perturbation beats
rerunning on yield AND efficiency. **Rerunning cannot find this class at all** --
same wall clock, same verdict.

**Measured regex trap:** `\b20\d{2}-\d{2}-\d{2}\b` misses **40 of 169** files here,
because ISO datetimes (`"2026-08-09T00:00:00+00:00"`) fail the trailing `\b` on the
`T`. Drop the trailing boundary.

**Real debate in the literature, don't pick one side blindly:** thoughtbot + GitLab
MR 77474 say replace the hard-coded date with one derived from now; Atomic Spin
argues the opposite (a pinned troublesome date exercises the edge case on EVERY run).
Resolution = both: relative date for the "current anchor" test, explicitly-past
literal for a SEPARATE "stale path" test. Adopting only the relative form silently
deletes the staleness coverage.

Brief: `handoff/current/research_brief_86.24.md`.
