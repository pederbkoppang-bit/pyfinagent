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

backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py:87: AssertionError
```

`24666.57` is the stale archived peak winning over a fresh, marked anchor. With `reset_peak` DARK and
`update_peak` ratchet-only, that phantom high-water mark could never be lowered — `flatten_all` +
`pause` on the first cycle after restart, permanently.

## What shipped

| File | Change |
|---|---|
| `backend/services/kill_switch.py` | `update_peak` STAMPS the anchor-from-`None` case (`anchor: true`, `prior_peak: null`); an ordinary ratchet stays unmarked. `_load_from_audit`'s `peak_update` branch ASSIGNS at a row where `anchor is True` and RATCHETS otherwise. New `_apply_authoritative_peak(raw, source)` — the single guarded path for **every** assignment to `_peak_nav` — routes both the new anchor branch and the pre-existing `peak_reset` branch. `reset_peak`'s DARK gate byte-untouched. |
| `scripts/housekeeping/{verify_handoff_layout,backfill_handoff_archive}.py` | new `AUDIT_KEEP_GLOBS = ("kill_switch_audit*.jsonl",)` in BOTH, with the measurement that justifies refusing a cap written into the comment. |
| `backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py` | new, **35 collected** (`pytest --collect-only`); autouse live-file write-protect fixture ported from the 36.7 module. |

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
$ python -m pytest backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py -q
35 passed

$ python -m pytest backend/tests/ -q -k kill_switch            # IMMUTABLE
129 passed, 1 skipped, 2126 deselected
```

All **35** of this module's tests are now INSIDE that selector (cycle 1 shipped with **zero**
of them selected; measured `pytest -k kill_switch --collect-only | grep -c test_phase_36_8`).

```
```

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after every run;
`git status` clean on both live audit files throughout.

## Mutation matrix — 14 mutations, 14 killed, at ONE baseline (`35 passed`)

**Every row below was re-run in a single batch AFTER the last test landed.** The cycle-3 Q/A caught
the previous version claiming "one batch run at the final baseline" while 9 of 13 rows had actually
been measured at an earlier baseline — a provenance claim that was false even though the kills held.
This table is regenerated from one batch; the counts are derived from its output, not carried
forward.

| # | Mutation | Result @ baseline `35 passed` |
|---|---|---|
| M1 | the marked anchor loses authority and merely ratchets | KILLED `2 failed, 33 passed` |
| M2 | EVERY `peak_update` assigns — the 36.7 regression | KILLED `9 failed, 26 passed` |
| M3 | `anchor` truthiness instead of `is True` | KILLED `4 failed, 31 passed` |
| M4 | `_apply_authoritative_peak` assigns unguarded | KILLED `13 failed, 22 passed` |
| M5 | the writer stops marking the anchor | KILLED `2 failed, 33 passed` |
| M6 | `peak_reset` bypasses the guard | KILLED `6 failed, 29 passed` |
| M8 | stamp authority even when the replay was incomplete *(cycle-1 regression)* | KILLED `2 failed, 33 passed` |
| M9 | the replay claims a complete history unconditionally | KILLED `6 failed, 29 passed` |
| MXL | stat the archive dir instead of listing it *(cycle-2 route)* | KILLED `1 failed, 34 passed` |
| MX4 | per-file read failure not recorded as incomplete | KILLED `1 failed, 34 passed` |
| MX3 | `_history_complete` class default flipped to True | KILLED `1 failed, 34 passed` |
| MX2 | drop `prior_peak=None` from the anchor row | KILLED `2 failed, 33 passed` |
| **MXP** | **a parse failure drops history silently *(cycle-3 route)*** | KILLED `3 failed, 32 passed` |
| M7 | one housekeeping script drops the archive declaration (disk) | KILLED `1 failed, 34 passed` |

Disk mutants restore their file with sha256 re-verified. `M8`, `MXL`, `MX4` and `MXP` are the four routes the completeness gate has had to close (an earlier
revision of this line cited a mutant `M17`, which does not exist in this step's matrix — that label
belongs to phase-36.12's) — see the table below.

**Ceiling, stated plainly:** this licenses *"these 14 mutations were killed at this baseline"*. It is
a lower bound on the guard set, never a claim about the code. Independent passes found survivors in
my set twice; assume a fifth route until someone has looked for it.

### FOUR routes to the same regression, each found by an evaluator after I declared closure

| # | Route | Why the previous gate missed it |
|---|---|---|
| 1 | anchor stamped on the `_peak_nav is None` accident | no gate at all — authority was unconditional |
| 2 | archive dir absent / unreadable | gate existed but keyed on the wrong signal |
| 3 | archive dir **unlistable** (`chmod 000`) | `is_dir()` True and `Path.glob` returns empty **without raising** |
| 4 | archive file **unparseable** | per-*file* failures recorded; per-*line* failures dropped silently |

The invariant was correct from cycle 2 onward — *complete is False whenever this replay may be
missing history it cannot see* — and I under-applied it three times running. Each fix was right and
each left a neighbouring hole, which is the same shape as 36.12's five-position call site.

## Cycle-2 follow-up (post-Q/A-1 FAIL) — I shipped a safety regression and it caught it

Cycle 1 returned **FAIL** on two BLOCKs. Both were mine, and the first was serious.

### BLOCK 1 — the marker granted authority to an ACCIDENT

The Q/A executed this against the REAL corpus:

```
boot A (archives readable)       -> peak 24666.57
boot B (archive scan FAILS)      -> peak None; the cycle calls update_peak(18000.0),
                                    which stamped {"anchor": true, "prior_peak": null}
boot C (archives readable again) -> peak 18000.0     <-- the true mark, destroyed
```

Permanently: `reset_peak` is DARK and `update_peak` only ratchets up. **In the UNSAFE direction** —
a lower peak shrinks `trailing_dd_pct`, i.e. less protection. Pre-36.8 code returns `24666.57` for
the identical sequence.

**The self-contradiction is the part worth recording.** This contract REJECTED consuming 36.12's
`baseline_anchor_on_lost_history` on the grounds that *"that event marks an ACCIDENT ... granting it
authority to lower a real high-water mark is the under-conservative direction"* — and then stamped a
marker on **the same `_peak_nav is None` trigger** and gave it exactly that authority. In production
no path writes an *intentional* lower anchor (the authorized one is `peak_reset`, DARK), so the new
authority could only ever have been produced by the accident.

**Fix: authority is gated on whether the replay actually saw everything.** `_read_audit_rows` now
returns `(rows, complete)`; `complete` is False if any source failed to read **or** the archive
directory is absent (a brand-new install and a lost mount are indistinguishable from there, and only
one is safe). `_load_from_audit` records it, and `update_peak` stamps `anchor: true` **only when the
replay was complete** — otherwise it writes an unmarked row and logs loudly, so a later complete boot
restores the true peak. `_history_complete` defaults to **False** at class level: a state built
without a replay has verified nothing. That default immediately caught one of my own earlier tests
constructing a state implicitly, which is the guard working.

Guarded by `..._a_transient_archive_failure_cannot_destroy_the_true_peak` (the Q/A's exact sequence),
its `COMPLETE` counterpart, and `..._a_real_replay_sets_history_complete_both_ways`. Mutant **M8**
restores the regression and dies.

### BLOCK 2 — the immutable gate selected ZERO of my tests

Measured: `pytest -k kill_switch --collect-only | grep -c test_phase_36_8` → **0**, while 36.7's 33
were selected. Neither the filename nor any test name contained `kill_switch`, breaking the
convention both sibling modules follow — so every guard and the whole mutation kill-signal sat
outside this step's own regression gate. Renamed to
`test_phase_36_8_kill_switch_archive_merge_authority.py`; now **26 of 26 selected**, immutable
command **123 passed, 1 skipped**.

### The WARNs

- **`AUDIT_KEEP_GLOBS` is a declaration, not an enforcement** — the Q/A is right that nothing
  consumes it and neither script has a prune path over `handoff/audit/`. It is a tripwire against a
  future one, and M7 proves only that the two declarations agree. Stated plainly rather than left to
  read as protection.
- **An archive-resident `anchor: true` row can lower a live peak.** True, and now deliberate: with
  the completeness gate, such a row can only have been written by a boot that saw every source, and
  `ts` order decides. Disclosed rather than left as an undocumented capability.

### And a bad mutant of my own, disclosed

My first `M9` inserted `_unused = None` — semantically **inert**, so its "survival" carried zero
information. A mutant that cannot change behaviour is not evidence. Rewritten to make the replay
claim completeness unconditionally; it dies `1 failed, 28 passed`.

## Cycle-3 follow-up (post-Q/A-2 FAIL) — the regression had a THIRD route

Cycle 2 returned **FAIL** and was right again: my completeness gate closed two doors and missed a
third. The Q/A executed it against a tmp copy of the real corpus:

```
chmod 000 handoff/audit/   ->  archive.is_dir() is STILL True
                           ->  Path.glob() returns EMPTY *without raising*
                           ->  complete = True   (a history the replay never read)
                           ->  anchor:true stamped -> boot C = 18000.0
```

`pathlib.glob` swallows the directory `PermissionError`, so an **unlistable** archive is
indistinguishable from an **empty** one — and the `except` I was relying on at `:104-105` never even
fires. My own stated rule ("a brand-new install and a lost mount are indistinguishable, and only one
is safe") applied to this case too, and I had not applied it.

**Root-cause fix: LIST the directory, do not merely stat it.** `os.listdir(archive)` raises exactly
where `glob` stays silent. Guarded by `..._an_UNLISTABLE_archive_dir_is_incomplete_not_empty`
(a real `chmod 000`, skipped under root where chmod is a no-op), and mutant **MXL** — reverting to the
stat-only check — now dies.

### Three of its six mutants survived mine, and all three are now closed

| its mutant | why it survived | now |
|---|---|---|
| **MX4** — delete `complete = False` from the per-file read-failure handler | only the absent-dir branch had a test; the unreadable-FILE half was unguarded although its differential is the same destroyed peak | KILLED `1 failed, 31 passed` — new `..._an_unreadable_archive_FILE_is_incomplete` |
| **MX3** — flip the `_history_complete` class default to `True` | the artifacts credit that default as load-bearing safety, and nothing pinned it | KILLED `1 failed, 31 passed` — new `..._the_class_default_for_history_complete_is_conservative` |
| **MX2** — drop `prior_peak=None` from the anchor write | **tautological assertions**: `dict.get("prior_peak") is None` is satisfied by the key being ABSENT, so both assertions passed with the field deleted | KILLED `2 failed, 30 passed` — assertions changed to `"prior_peak" in row and row["prior_peak"] is None` |

MX2 is the one worth remembering: `.get(k) is None` can never detect a missing `k`. Two of my
assertions were shaped that way and neither could fail.

**Criterion 5 overclaim, acknowledged.** My "9 killed, 0 survivors" met three survivors in one
independent pass. The Goodenough–Gerhart caveat was already in the artifact and it was still an
overclaim in substance — the honest form is *"these mutations were killed"*, and the count is only
ever a lower bound on the guard set, never a statement about the code.

**Also owed and now done:** `live_check_36.8.md` was stale cycle-1 evidence carrying three numbers
that no longer reproduce; it is refreshed from measurements taken this cycle.

## Cycle-4 follow-up (post-Q/A-3 FAIL) — the FOURTH route

Cycle 3 confirmed the chmod-000 route is closed (its own probe: boot C = 24666.57, row unmarked) and
refuted one of my suspicions with measurement (`os.listdir` and `Path.glob` agree on membership
across dir modes `0o444/0o555/0o111/0o644`). Then it found route 4.

**Parse failures were invisible to the completeness gate.** `_read_audit_rows` recorded a per-FILE
read failure but its per-LINE handlers — `except Exception: continue` and
`if not isinstance(row, dict): continue` — dropped history silently with `complete` left True. A file
that opens as UTF-8 but holds unparseable lines therefore looked *empty* rather than *unreadable*,
the anchor claimed authority, and once the file became parseable again the restore returned 18000.0
over the true 24666.57. Executed by the Q/A, not inferred.

Both handlers now record the skip. Guarded by three new tests (unparseable lines, non-dict rows, and
the end-to-end differential), and mutant **MXP** reverts it and dies `3 failed, 32 passed`.

**Its two WARNs are also closed:** the verification block was stale (it printed `29`/`123` against a
measured `32`/`126` — the same class flagged in `live_check` a cycle earlier, in the other file this
time), and the matrix's "one batch at the final baseline" provenance was false for 9 of 13 rows. Both
fixed by regenerating from a single batch at the true final baseline rather than editing numbers.

## Cycle-5 status — FAIL, and the right response is NOT a sixth patch

The cycle-4 Q/A found **route five**, measured end-to-end in four sub-cases, all with the identical
destroyed peak (`bootC = 18000.0` where a pure ratchet over the same end-state rows gives
`24666.57`):

| | source that is **present-but-silent** |
|---|---|
| (a) | an archive file **truncated to 0 bytes**, then restored from git — the recoverability backstop this artifact itself cites |
| (b) | the **LIVE file absent** — `_audit_source_paths` appends it only `if _AUDIT_PATH.exists()`, so its absence never sets `complete=False`, while an absent archive *dir* does. A literal asymmetry against my own rule |
| (c) | an archive file whose name misses the glob (`…jsonl.bak`) |
| (d) | archive rows in a nested `handoff/audit/2026-06/` subdir |

Production relevance is not hypothetical: `handoff/audit/kill_switch_audit.jsonl` is the OLDEST
archive and the **only** holder of the live book's true peak `24666.57` (the live file has **zero**
`peak_update` rows). Truncate or misplace it, boot once, and the anchor is stamped at that day's
lower NAV; restore it and the true mark is outranked forever.

### The architectural conclusion, which is the actual deliverable of this cycle

The Q/A said it better than I would have: *"closing hole #5 by hand is the same move that produced
holes #2–#5."* Five routes, five hand-closed enumerations of "ways a source can fail". The gate is
**negatively derived** — it starts at `True` and drops to `False` on each failure mode I remembered
— so it is only ever as good as my imagination, and five independent passes have each beaten it.

**The next cycle must change the design, not add a sixth `complete = False`.** The shape that kills
all five routes at once: **an anchor may claim authority only if it names what it superseded** —
i.e. `prior_peak` must be a real value the writer actually observed, never `None`. Every one of the
five routes produces an anchor from `None`, so none of them could ever be authoritative. It also
matches the research's own finding that the authorized re-anchor is `peak_reset` (token-gated) and
that **no production path writes an intentional lower anchor at all** — which means 36.8's marker
should probably be production-dead by construction, and criterion 1's test (which builds the row
directly) is what exercises it.

I am **not** implementing that here. It is a redesign of a P0 safety path, it deserves its own
research-informed pass with a fresh Q/A, and shipping it at the end of a long session is exactly how
routes 2–5 got made.

### Two record defects it also found, both fixed

- `live_check_36.8.md:74` contradicted itself inside one sentence (14 killed vs "licenses these 13";
  "four the cycle-2 Q/A found" vs "found three survivors"). Corrected, **with the attribution fixed**:
  cycle 2 found three (MX4/MX3/MX2), MXP came from cycle 3.
- `experiment_results` cited a mutant `M17` that appears nowhere in this step's matrix — that label
  belongs to 36.12's. Removed.

### And it refuted my own bet, which is worth recording

I predicted `baseline_history_exists` (36.12) would misfire on an incomplete replay. It **measured
otherwise**: the tuple is unpacked correctly, nothing raises, and the probe returns `True` against
the real corpus. It did find a genuine latent issue there — the probe discards the completeness
signal while its docstring promises a fail-closed reading — but confirmed it is **pre-existing, not a
36.8 regression**, so it is filed as its own step rather than absorbed.

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
