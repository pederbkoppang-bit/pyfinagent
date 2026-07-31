# Experiment Results — phase-81.2 — Gate input resolution hardening

## 1. What changed

| File | Change |
|---|---|
| `.claude/hooks/lib/verdict_gate.py` | +`resolve_verdict_source()`, +`gate_decision_with_source()`, +`--resolve` CLI mode. `gate_decision()` byte-unchanged. |
| `.claude/hooks/auto-commit-and-push.sh` | verdict dispatch uses `--resolve`; logs a NOTE when the verdict came from the archive. |
| `backend/tests/test_phase_81_2_verdict_resolution.py` | NEW — 8 tests. |

**Scope, DERIVED from git — corrected after Q/A cycle 1 flagged the original wording as false.**

The three files above are 81.2's OWN edits, and that claim is verified: filtering the working tree
by mtime >= `contract_81.2.md` returns exactly `verdict_gate.py`, `auto-commit-and-push.sh`,
`test_phase_81_2_verdict_resolution.py` (plus this step's own handoff artifacts and two
hook-appended audit JSONLs).

**But 81.2 does not ship alone.** `git add -An` reports **37 paths**, plus **24 deletions already
staged in the index** via `git rm`. The auto-commit hook runs `git add -A`, so all of it commits
under 81.2's subject. The full ride-along:

- **Sibling step 81.0 (uncommitted, GENERATE-complete, cannot close):**
  `scripts/housekeeping/backfill_handoff_archive.py`, `scripts/housekeeping/verify_handoff_layout.py`,
  `.claude/hooks/lib/harness_log_gate.py`, `harness_log_gate_test.sh`,
  `backend/tests/test_phase_71_3_verdict_gate.py`, `.claude/skills/masterplan/SKILL.md`,
  `.claude/context/known-blockers.md`, `.claude/settings.json`, and the 24 staged deletions
  (`.claude/context/sessions/` ×23 + `push-credential-diagnosis.md`).
- **The Fable-policy correction (this session, unrelated to phase-81):** `CLAUDE.md`,
  `.claude/agents/qa.md`, `.claude/agents/researcher.md`, `session-start-fable-tripwire.sh`.
- `.claude/masterplan.json` (81.0 + 81.1-dropped + 81.2 entries).
- Runtime-generated: `handoff/audit/*.jsonl`, `handoff/away_ops/*`, `handoff/cycle_history.jsonl`,
  `handoff/kill_switch_audit.jsonl`, `handoff/.cycle_heartbeat.json`.

The original wording — *"Nothing else. scripts/housekeeping/** untouched"* — was true of 81.2's own
edits and false of the commit. That is the `feedback_audit_the_commit_not_the_diff` failure class:
`git add -A` ships the whole tree under this step's name, so a scope claim must be derived from
`git add -An`, not from what I intended to touch.

**What 81.2 genuinely did NOT touch** (the claims that matter for the research blockers, re-verified
by `git diff --name-only HEAD`): `.claude/hooks/archive-handoff.sh` — absent from the diff, so
blocker B1's hook race is avoided by construction — and `.claude/hooks/lib/live_check_gate.py` —
absent, so 75.5.10's territory is untouched. `scripts/housekeeping/**` **is** modified, but by 81.0,
not by 81.2.

## 2. Verbatim verification output

```
$ python -m pytest backend/tests/test_phase_81_2_verdict_resolution.py \
      backend/tests/test_phase_71_3_verdict_gate.py \
      backend/tests/test_phase_38_4_hook_gate.py -q
..........................                                               [100%]
26 passed in 0.06s

$ bash .claude/hooks/lib/harness_log_gate_test.sh
ALL PASS

$ python3 scripts/go_live_drills/incident_log_p0_test.py
DRILL PASS: 6/6 incident-log-P0 scenarios verified

FULL COMMAND EXIT: 0
```

End-to-end against the **live** tree:
```
$ python3 .claude/hooks/lib/verdict_gate.py --resolve 36.7 $PWD/handoff
passed
current:per-step

  step 36.7 -> decision=passed    source=current:per-step
  step 81.2 -> decision=no_input  source=none
  step 99.9 -> decision=no_input  source=none

$ python3 .claude/hooks/lib/verdict_gate.py handoff/current/evaluator_critique_36.7.json 36.7
passed          # legacy 2-arg mode unchanged
```

## 3. Mutation matrix (criterion 8) — run on COPIES, live file never mutated

```
BASELINE (unmutated):
  archived-CONDITIONAL -> hold : True
  precedence current-first     : True

MUTATION 1 -- archive branch removed from the candidate chain:
  archived-CONDITIONAL -> hold : False   <-- guard went RED

MUTATION 2 -- resolution order reversed:
  precedence current-first     : False   <-- guard went RED

MUTATION MATRIX: PASS -- both guards can fail
```
Executed against temp copies in the session scratchpad, per
`feedback_executor_sees_mutation_transients`. `git diff --stat` on the helper shows only the
intended change (99 insertions), never a mutated state.

## 4. Criteria walkthrough

| # | Criterion | Evidence |
|---|---|---|
| 1 | archive resolution gates | `test_conditional_found_only_in_archive_still_holds` |
| 2 | resolution order asserted, first hit wins | `test_per_step_current_beats_rolling_current_beats_archive` — three sources with three *different* verdicts; removing each winner in turn walks the chain |
| 3 | source reported | `gate_decision_with_source` returns the label; hook logs a NOTE on `archive:*`; live run shows `current:per-step` vs `none` |
| 4 | archived CONDITIONAL still holds | criterion-1 fixture reproduces the 2026-07-24 sweep shape |
| 5 | no-input still fails open | `test_no_input_anywhere_fails_open` asserts `!= "hold"` |
| 6 | strictly additive | `test_legacy_two_arg_signature_unchanged`. **Baseline stated explicitly (corrected after Q/A cycle 1):** 10/10 assertions in the CURRENT 71.3 suite pass. That suite was modified today at 15:45 by sibling step **81.0**, which went 9→10 tests and deliberately inverted `test_missing_json_fails_open_proceed` (`proceed` → `no_input`). Against the COMMITTED HEAD suite, 8/9 pass and that one is red **by 81.0's design**, not by regression. 81.2's own edit to `verdict_gate.py` is purely additive — two new functions plus a `--resolve` arm in `main()`; `gate_decision()` is unchanged **by 81.2** but is NOT byte-unchanged vs HEAD, because 81.0 changed its file-absent branch first. |
| 7 | no writes/moves | `test_resolution_performs_no_writes_or_moves` snapshots the tree before/after |
| 8 | mutation | §3 |
| 9 | existing suites green | 26 passed; gate test ALL PASS; drill 6/6 |

## 5. Root cause, stated once, in full

`STEP_ID_RE` (`^(?:phase-)?([0-9]+(\.[0-9]+)*)[-.].*\.md$`) matches **0 of 127** `.md` files in
`handoff/current/` — it expects the legacy `<sid>-name.md` form while every artifact written since
~phase-75 uses the inverse `name_<sid>.md`. `backfill_handoff_archive.py:154-158` therefore sets
`sid=None` for every step artifact and calls `_move(p, MISC)`. Commit `fa9aaf8e` (2026-07-24,
"archive 315 misc … out of handoff/current") executed that sweep;
`handoff/current/evaluator_critique.json` went to `handoff/archive/misc/`; the verdict gate's single
literal path stopped resolving; a miss fails open; the gate ran dark for 13 consecutive step closes.

**81.2 does not fix the regex.** It removes the gate's dependence on any one location, because the
sweep is ongoing (`archive/misc/` grew +60 files in the six days to 2026-07-31) and the naming
reconciliation belongs to 75.11.4. A control that only works while nobody tidies up is not a
control.

## 6. Status of the sibling steps

- **81.0** — GENERATE complete, all four changes verified, **cannot close**: I put
  `verify_handoff_layout.py` in its immutable criteria and that script is red for 128 pre-existing
  reasons. Research independently confirmed it stays red (109 violations) even after a regex
  widening. Needs operator arbitration: fix the convention (75.11.4) or re-scope 81.0 under a new id.
- **81.1** — **dropped before any work**, on its own research gate: duplicate of 75.11.4 *and*
  would have raced `live_check_gate.py` into dropping commits. Rationale recorded in the masterplan.

## 7. Defects discovered, to be queued as their own steps

Per `feedback_queue_discovered_defects_in_masterplan` — not absorbed here:

1. `harness_state_reader.py:71` — `_resolve_handoff_file` typed `-> Path | None`, all five callers
   (`:82,:99,:112,:125,:138`) call `.exists()` with no None guard → `AttributeError`.
2. `smoke_test_4_17_2.py:60` — a **third** naming convention (`phase-*-research-brief.md`), matches
   0 files, `assert briefs` raises today.
3. `.claude/skills/masterplan/SKILL.md:222` — a **fourth** drift: instructs writing
   `handoff/contract_{id}.md` at handoff root; nothing reads that path.
4. `backfill_handoff_archive.py:174` vs `:211` — moves ambiguous files to misc while the summary
   prints "left in current/ for manual review".
5. `archive-handoff.sh:160` uses `${sid}` not the `short_sid` computed at `:127` → a `phase-`
   prefixed id produces the dead pattern `phase-phase-6.1-*.md`.
6. **Criteria conflict needing operator resolution:** 75.11.4 pins *"the closing step's OWN
   artifacts (including `live_check_<sid>.md`) land in its archive directory"*, which directly
   contradicts `live_check_gate.py`'s current-dir-only lookup. Shipping 75.11.4 as written would
   reintroduce the disarm. The 81.2 pattern (teach the gate to search) is the resolution.
7. Still owed from the 81.0 brief: a periodic **proof test** asserting each gate still receives real
   input. Without it, silent disarm recurs — it is a Dangerous-Undetected failure class.
