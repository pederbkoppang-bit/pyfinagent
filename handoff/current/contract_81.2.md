# Contract — phase-81.2 — Gate input resolution hardening

**Written 2026-07-31 BEFORE any GENERATE work.** Masterplan `81.2` = `pending` at write time.

## 1. Research gate

`handoff/current/research_brief_81.1.md` — `gate_passed=true`, tier complex, 8 external sources read
in full, 37 URLs, recency scan performed, 24 internal files, audit-class coverage `dry=true` after
2 dry rounds. Commissioned for step 81.1; **its findings retired 81.1 and produced this step.**

The brief's `consumer_inventory` is the load-bearing artifact: it enumerates every code path that
parses a handoff artifact name, ranked by what a miss does. Two entries decide this contract:

- **Destructive on a miss:** `backfill_handoff_archive.py:154` → `_move(p, MISC)`. 125 files move
  out of `handoff/current/` on a dry-run *today*. This is the mechanism that swept
  `evaluator_critique.json` on 2026-07-24.
- **Fail-open on a miss:** `verdict_gate.py` resolves **one literal path**. Anything that moves that
  file disarms the gate silently and permanently.

### Blockers that shaped this scope

- **B1 (hard)** — widening `archive-handoff.sh`'s move glob races `live_check_gate.py`. Both are
  PostToolUse hooks under the same `Write`/`Edit` matchers, and Anthropic's hooks doc says *"All
  matching hooks run in parallel"*, contradicting `auto-commit-and-push.sh:264`'s assumption that
  archive-handoff "runs ahead of us in the chain". Moving `live_check_<sid>.md` mid-flight →
  `"skip"` → `exit 0` **before `git add -A`** → commit, changelog and push all dropped,
  intermittently. **So this step does not touch that glob.**
- **B3** — pending P1 **75.11.4** already owns the housekeeping/naming reconciliation by name.
  **So this step does not touch `scripts/housekeeping/**`.**
- **75.5.10** owns `live_check_gate.py` content hardening. **Not touched.**

### The reframe this step is built on

Every previous framing treated the dead gate as *"the file ended up in the wrong place."* The brief
shows that is the wrong lesson: the file will keep moving (archive/misc grew **+60 files in six
days**, so the sweep is active, not historical). The durable fix is to stop the gate depending on a
single location. Research mitigation (ii), adopted verbatim: *"teach `live_check_gate.py` /
`verdict_gate.py` to fall back to `handoff/archive/phase-<sid>/` so they are order-independent."*

## 2. Hypothesis

A control that resolves exactly one path is disarmed by any file move and reports the disarm as
success. If the verdict gate resolves an ordered chain of locations and names the source that
answered, then no housekeeping run, archive sweep, or hook race can silently disarm it — and a
verdict answered from the archive becomes visible rather than indistinguishable from a fresh pass.

## 3. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json` 81.2

1. `verdict_gate resolves a verdict from handoff/archive/phase-<sid>/ when handoff/current/ holds nothing for that step -- i.e. a step whose artifacts were archived still gates`
2. `resolution order is asserted: per-step current beats rolling current, rolling current beats archive; the first hit wins and later sources are not consulted`
3. `the resolved SOURCE is reported alongside the decision, so a verdict answered from the archive is distinguishable in the log from one answered from handoff/current/`
4. `a CONDITIONAL/FAIL verdict found ONLY in the archive still returns 'hold' -- the disarm-by-move failure mode is closed, demonstrated by a fixture that reproduces the 2026-07-24 sweep`
5. `no-input-anywhere still returns 'no_input' and still fails open (never 'hold')`
6. `gate_decision remains callable with its existing 2-argument signature and every phase-71.3 assertion still passes unchanged -- the new capability is strictly additive`
7. `the helper performs NO writes and NO moves: asserted by a fixture that snapshots the tree before and after a resolution call`
8. `MUTATION: removing the archive branch makes the archived-verdict fixture go red, and removing the ordering makes the precedence fixture go red -- both demonstrated by executed output, not argued`
9. `the two existing gate suites (71.3, 38.4) and the incident-log drill remain green`

**Verification command (immutable):**
```
source .venv/bin/activate && python -m pytest backend/tests/test_phase_81_2_verdict_resolution.py backend/tests/test_phase_71_3_verdict_gate.py backend/tests/test_phase_38_4_hook_gate.py -q && bash .claude/hooks/lib/harness_log_gate_test.sh && python3 scripts/go_live_drills/incident_log_p0_test.py
```

**Scope note, learned from 81.0's failure:** this command contains only a dedicated test file plus
the suites this change can actually affect. It deliberately does **not** include
`verify_handoff_layout.py` — that script is red for 128 pre-existing reasons unrelated to this
change, and putting it in 81.0's criteria is why 81.0 cannot close.

## 4. Plan

1. `verdict_gate.py` — add `resolve_verdict_source(step_id, handoff_root) -> (Path|None, str)` and
   `gate_decision_with_source(...) -> (decision, source)`. `gate_decision()` untouched.
2. `main()` — 2-arg legacy mode byte-identical; new 3-arg mode prints decision on line 1 and source
   on line 2, so the shell's single-token `case` still reads line 1.
3. `auto-commit-and-push.sh` — use the resolution mode and log the source.
4. `backend/tests/test_phase_81_2_verdict_resolution.py` — new.
5. Mutation matrix last, per `feedback_executor_sees_mutation_transients`.

## 5. Discovered defects to queue separately (NOT absorbed here)

- `harness_state_reader.py:71` — `_resolve_handoff_file` is typed `-> Path | None` but all five
  callers (`:82,:99,:112,:125,:138`) call `.exists()` with no None guard → `AttributeError`. Masked
  only because the rolling files still exist. In this fix's blast radius.
- `smoke_test_4_17_2.py:60` — a **third** convention (`phase-*-research-brief.md`), matches 0 files,
  `assert briefs` raises today.
- `.claude/skills/masterplan/SKILL.md:222` — a **fourth** drift: writes `handoff/contract_{id}.md`
  at handoff root; nothing reads it.
- `backfill_handoff_archive.py:174` vs `:211` — moves ambiguous files to misc while printing "left
  in current/ for manual review". The summary contradicts the action.
- `archive-handoff.sh:160` uses `${sid}` not the `short_sid` computed at `:127`, so a `phase-`
  prefixed id yields the dead pattern `phase-phase-6.1-*.md`.
- 75.11.4's criterion *"the closing step's OWN artifacts (including `live_check_<sid>.md`) land in
  its archive directory"* **directly conflicts** with `live_check_gate.py`'s current-dir-only
  lookup (B1). Two immutable criteria sets in conflict — needs written operator resolution before
  either ships.
