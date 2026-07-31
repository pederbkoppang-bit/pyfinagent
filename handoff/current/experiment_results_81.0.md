# Experiment Results — phase-81.0

**Status: GENERATE complete. Step NOT closeable. Do not flip to `done`.**

The four code changes are built and independently verified. The step nonetheless fails its own
immutable verification command, for a reason that is **my error in authoring the criteria**, not a
defect in the change. Details in §4.

---

## 1. What was built

| Change | Files |
|---|---|
| A1 verdict-gate audibility | `.claude/hooks/lib/verdict_gate.py`, `.claude/hooks/auto-commit-and-push.sh`, `backend/tests/test_phase_71_3_verdict_gate.py`, `scripts/housekeeping/backfill_handoff_archive.py`, `scripts/housekeeping/verify_handoff_layout.py` |
| A4 harness_log warn mode | `.claude/hooks/lib/harness_log_gate.py`, `.claude/hooks/auto-commit-and-push.sh`, `.claude/hooks/lib/harness_log_gate_test.sh` |
| A2 boot-path honesty | `.claude/skills/masterplan/SKILL.md` |
| A3 dead-context removal | deleted `.claude/context/sessions/` (23 files), deleted `.claude/context/push-credential-diagnosis.md`, edited `.claude/context/known-blockers.md`, edited `.claude/settings.json` |

Out of scope and untouched, as the contract required: `live_check_gate.py` (owned by 75.5.10);
`HARNESS_LOG_GATE_ENABLED` (owner-gated by 38.4) — **confirmed still unset**.

## 2. Verbatim verification output

```
$ python -m pytest backend/tests/test_phase_71_3_verdict_gate.py backend/tests/test_phase_38_4_hook_gate.py -q
..................                                                       [100%]
18 passed in 0.05s

$ bash .claude/hooks/lib/harness_log_gate_test.sh
PASS: case 1 -- gate disabled returns proceed
PASS: case 2 -- gate enabled + token present returns passed
PASS: case 3 -- gate enabled + token missing + MODE=block returns skip
PASS: case 3b -- gate enabled + token missing + default mode returns warn
PASS: case 3c -- warn and skip are distinguishable
PASS: case 4 -- missing log file returns proceed (fail-open)
PASS: case 5 -- prefix-match guard (38.6 does not match phase=38.6.1)
ALL PASS

$ python3 scripts/go_live_drills/incident_log_p0_test.py
DRILL PASS: 6/6 incident-log-P0 scenarios verified

$ python3 scripts/housekeeping/verify_handoff_layout.py
handoff layout FAIL -- 126 invariant violation(s):     <-- SEE §4
```

Case-arm ordering (criterion 2), from the live file:
```
30:        no_input)
41:        proceed|*)
```

Gate still disabled (criterion 5): `grep -c HARNESS_LOG_GATE_ENABLED .claude/settings.json` → `0`.

## 3. Mutation evidence (criterion 9)

**A1 / ROLLING_KEEP** — the guard goes red without the fix:
```
WITH fix   _is_rolling_keep(evaluator_critique_36.7.json) = True
WITHOUT fix (old exact-set membership test)              = False
rolling name in ROLLING_KEEP                             = True
```
Backfill dry-run after the fix proposes **zero** `evaluator_critique*.json` moves (before: the
rolling JSON was swept, which is how the gate went dark).

**A4 / warn token** — case 3c asserts `block` and `warn` modes return *different* tokens; collapsing
them turns the test red. The token is reachable (`warn`) and the 3-arg call still returns the
38.4-pinned `skip`.

**A2 / render** — replayed the skill's own block against the live masterplan:
```
## STATUS DRIFT -- 2 unrecognised status value(s): 'merged', None
## Coverage: 118 of 263 open steps shown above -- 145 NOT shown
## Next actionable: 5.2 -- ... (status=pending)
grep -c '^\[.\] phase-5 ' → 1     # the named phase IS printed
```
Before the change that grep returned 0 — the render recommended a step from a phase it never
printed. Criterion 7 (`phase-77` absent-key, `phase-36` deferred) is covered by the STATUS DRIFT
line: `None` is phase-77's missing status, now counted as open rather than dropped.

## 4. WHY THE STEP CANNOT CLOSE — my authoring error

`verify_handoff_layout.py` reports **126 violations, all pre-existing.** It flags the entire
`contract_<sid>.md` / `research_brief_<sid>.md` / `live_check_<sid>.md` naming convention, because
`STEP_ID_RE = ^(?:phase-)?([0-9.]+)[-.].*\.md$` expects the **inverse** form (`4.5.9-contract.md`).
Flagged files date to 2026-07-26 and earlier; **none were authored by this step**. My own
`contract_81.0.md` and `research_brief_81.0.md` are flagged by the same rule.

I included that checker in 81.0's immutable verification command. It was **already red before this
change**, so the command cannot exit 0 no matter what this step does. Criteria are immutable
(`CLAUDE.md`: "Never edit verification criteria"), so the correct outcome is: leave 81.0 `pending`,
record this, and let Q/A + the operator decide between (a) a corrective step that fixes the
regex/convention mismatch so the checker can go green, or (b) re-scoping 81.0 under a new id with a
verification command bounded to the files this change actually touches.

This is the `feedback_gate_scope_and_disclosure_completeness` failure mode, committed by me while
writing the criteria: **a gate is only green on the scope the change defines**, and I reached for a
repo-wide checker to verify a five-file change.

## 5. Discovered defect — to be queued as its own step

The `STEP_ID_RE` / `archive-handoff.sh:160` glob expect `<sid>-name.md`; every artifact written
since at least phase-75 uses `name_<sid>.md`. Two independent consumers therefore disagree with the
tree: the layout verifier reports 126 false strays, and the archive hook's per-step glob matches
nothing. This is the same root cause the 2026-07-30 audit found stranding 21 `contract_*.md` files.
Per `feedback_queue_discovered_defects_in_masterplan` this gets its **own** research-gated step; it
is deliberately **not** absorbed into 81.0.

Also still owed from the research brief: a periodic **proof test** asserting each gate still
receives real input. Without it this repair has the same half-life as the last one.

## 6. Honest tradeoff

The `/masterplan` render grew from 132,714 to ~148,859 chars — it got **bigger because it stopped
hiding work** (deferred phases are now shown). Per-step name truncation was deliberately not
attempted: it is absent from 81.0's criteria, and adding unrequested scope mid-step is the failure
this contract was written to avoid. It is the obvious follow-up.
