# Experiment Results — phase-36.8

**Step:** `36.8` (P0) — archive merge + DARK `reset_peak` = permanent flatten/pause lockout.
Date 2026-07-26. Contract: `handoff/current/contract_36.8.md`.
Research: `handoff/current/research_brief_36.8.md` — `gate_passed: true`, 8 sources read in full
(floor 5), 30 URLs, recency scan. Spawned BEFORE the contract; contract written BEFORE any code.

---

## The research refuted this step's own suggested fix, with sources

The step text offered "a live-file freshness/authority marker" as one option. Every reference
implementation the brief read solves this with an **in-stream boundary marker**, not file recency —
Fowler's rejected-events, Kafka KIP-101 leader epoch, PostgreSQL timelines, Kleppmann's fencing
tokens — and **PostgreSQL explicitly prefers the ARCHIVE over the live directory**, which directly
refutes live-file-precedence. The design follows the research.

It also **corrected the step's stated failure scenario**, which matters for honest scope: *"a book
legitimately re-anchors lower"* is **not reachable through `update_peak`**, because that method only
ratchets up and cannot write a lower peak while the merged peak is higher. The reachable production
door is the **swallowed archive-scan exception** at `kill_switch.py:104-105` (or an absent-then-present
`handoff/audit/`): boot with the archives unreadable → anchor a fresh low peak from `None` → boot
again with them readable → the old high peak returns and cannot be lowered. Criterion 1 asks for the
row shapes directly, which is what the test builds; this paragraph exists so nobody reads the test as
proof that `update_peak` provides the route.

## Criterion 1 — the defect, recorded BEFORE the fix

Whole file against unfixed code: **10 failed, 12 passed**. The headline failure, verbatim:

```
______ test_phase_36_8_a_fresh_marked_anchor_beats_a_higher_archived_peak ______
        _write(archive / "kill_switch_audit-v3.jsonl",
               _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
        _write(live,
               _row("2026-07-26T10:00:00+00:00", "peak_update", nav=18000.0,
                    anchor=True, prior_peak=None))

>       assert ks.KillSwitchState().snapshot()["peak_nav"] == 18000.0
E       assert 24666.57 == 18000.0

backend/tests/test_phase_36_8_archive_merge_authority.py:87: AssertionError
```

`24666.57` is the stale archived peak winning over a fresh, marked anchor. With `reset_peak` DARK and
`update_peak` ratchet-only, that phantom high-water mark could never be lowered — `flatten_all` +
`pause` on the first cycle after restart, permanently.

## What shipped

| File | Change |
|---|---|
| `backend/services/kill_switch.py` | `update_peak` STAMPS the anchor-from-`None` case (`anchor: true`, `prior_peak: null`); an ordinary ratchet stays unmarked. `_load_from_audit`'s `peak_update` branch ASSIGNS at a row where `anchor is True` and RATCHETS otherwise. New `_apply_authoritative_peak(raw, source)` — the single guarded path for **every** assignment to `_peak_nav` — routes both the new anchor branch and the pre-existing `peak_reset` branch. `reset_peak`'s DARK gate byte-untouched. |
| `scripts/housekeeping/{verify_handoff_layout,backfill_handoff_archive}.py` | new `AUDIT_KEEP_GLOBS = ("kill_switch_audit*.jsonl",)` in BOTH, with the measurement that justifies refusing a cap written into the comment. |
| `backend/tests/test_phase_36_8_archive_merge_authority.py` | new, **26 collected** (`pytest --collect-only`); autouse live-file write-protect fixture ported from the 36.7 module. |

### Why a FIELD on `peak_update` and not a new event name

A new event name would have to be added to `_BASELINE_EVENTS` or `36.12`'s
`baseline_history_exists` probe goes blind to it — a coupling that would have been easy to miss and
would silently weaken 36.12. A field avoids it entirely.

### Why NOT consume 36.12's `baseline_anchor_on_lost_history`

The step text asked whether 36.8 should consume it. **No**, and the reason is semantic, not
technical: that event marks an **accident** — 36.12 exists precisely to flag such an anchor as a
fiction — so granting it authority to lower a real high-water mark is the *under*-conservative
direction. It would also break `test_phase_36_12_the_new_event_is_replay_inert`. The authorized
re-anchor stays `peak_reset`, token-gated, which matches the measured industry practice the brief
found: withdrawal-driven resets are authorized by a discrete payout event, never automatically.

## Criterion 3 — a cap is REFUSED, on measurement, not preference

| measured (read-only, 2026-07-26) | value |
|---|---|
| corpus | **897 rows across 5 files** |
| where today's live baselines come from | **100% from the ARCHIVES** |
| where the true peak `24666.57` lives | the **OLDEST** file |
| `peak_reset` rows ever written | **zero** |
| boot cost | **0.95 ms total, 1.06 µs/row** |

An oldest-first cap would delete the row the kill switch depends on. So growth is **accepted as a
bounded, measured cost** and the archives are declared do-not-prune in both housekeeping scripts,
pinned by an AST-parsing test that fails if the two declarations drift. All five files are
git-tracked — the existing recoverability backstop.

## Criterion 2 — 36.7 is not regressed, asserted two ways

- Synthetic: a higher TRUE historical peak in the archives still beats a later, lower, **unmarked**
  live row (`24666.57` over `23838.19`, where assignment-replay would give `23838.19`).
- **Against the REAL corpus**: the test copies every real audit file into a tmp tree and asserts the
  restored peak equals `max(peak_update)` across them. All 20 real rows are unmarked, so this is the
  behaviour the live book depends on today.
- At suite level: the immutable `-k kill_switch` selector includes 36.7's entire module and it is
  green.

## Verification

```
$ python -m pytest backend/tests/test_phase_36_8_archive_merge_authority.py -q
26 passed

$ python -m pytest backend/tests/ -q -k kill_switch            # IMMUTABLE
94 passed, 1 skipped, 2148 deselected

$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader'   # wider net
116 passed, 1 skipped, 2126 deselected
```

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after every run;
`git status` clean on both live audit files throughout.

## Mutation matrix — 7 mutations, 7 killed, 0 survivors (baseline `26 passed`)

Counts DERIVED from one batch run at the final baseline. Each mutant asserts its pattern matched
exactly once and that the source changed; `kill_switch.py` is mutated **in memory** with
`_AUDIT_PATH` redirected to tmp **before** the module is built (that is how an evaluator wrote 54 rows
into live safety state today); the housekeeping mutant is on disk with sha256 restore-verified.

| # | Mutation | Result |
|---|---|---|
| M1 | the marked anchor loses authority and merely ratchets | KILLED `2 failed, 24 passed` |
| M2 | EVERY `peak_update` assigns — the 36.7 regression | KILLED `7 failed, 19 passed` |
| M3 | `anchor` truthiness instead of `is True` | KILLED `4 failed, 22 passed` |
| M4 | `_apply_authoritative_peak` assigns unguarded | KILLED `13 failed, 13 passed` |
| M5 | the writer stops marking the anchor | KILLED `1 failed, 25 passed` |
| M6 | `peak_reset` bypasses the guard (36.15's defect restored) | KILLED `6 failed, 20 passed` |
| M7 | one housekeeping script drops the archive protection (disk) | KILLED `1 failed, 25 passed`; sha256 restored `516478006960` |

M2 is criterion 2's guard and M1 is criterion 1's — the two directions of the same boundary. M3
exists because `is True` is a deliberate identity check and a truthiness mutant would otherwise
survive: a future schema change or a hand-edited row carrying `anchor: "yes"` must not acquire the
power to lower the high-water mark.

**Ceiling, stated:** "0 survivors" licenses only *"these 7 mutations were killed"* — it is not a
claim that no vacuous guard remains. This step's own history is the argument for saying so: 36.12's
neighbouring call site produced a survivor in five consecutive cycles, each after closure was
declared.

## Scope honesty

- **`36.15` (P1) partly delivered here.** Routing `peak_reset` through the guarded helper closes its
  code defect — M6 is the regression lock, and the malformed-`peak_reset` parametrization asserts all
  six shapes. Its step text is annotated so its executor **re-measures criterion 1 before assuming a
  defect remains**; what is left there is the reproduce-first record and the `_coerce_nav` semantics
  decision. Not silently absorbed, not left to collide.
- **The swallowed archive-scan exception** (`:104-105`) is the real production route to this failure,
  and distinguishing "no archive dir" from "archive dir unreadable" is NOT fixed here — the second is
  not a safe reason to anchor a new peak. Flagged for its own step.
- **The marker is forward-only.** The 20 existing unmarked rows stay ratchet-only by design, so a
  pre-existing stale peak is still correctable only by `peak_reset` (owed token, `79.6`).
- `:8000` never restarted or POSTed to; `:3000` never driven.
