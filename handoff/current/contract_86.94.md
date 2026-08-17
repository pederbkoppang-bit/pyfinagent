# Contract — phase-86.94, cycle 4

**Step:** `86.94` — corpora and windows defined relative to `now` are not
reproducible: `git log --since=<bare date>` is applied at the CURRENT time of
day, so at least one measurement frozen into an immutable criterion cannot be
regenerated.

**Cycle:** 4 (prior: FAIL, FAIL, CONDITIONAL — parked overnight at the 3-attempt
rail). **Attempt budget this session: 3.**

---

## 1. Research gate — PASSED (enforced, not self-reported)

`Workflow({scriptPath: '.claude/workflows/research-gate.js'})`, run
`wf_c533d502-21e`. The script RECOMPUTED the verdict and cross-checked the
self-report against the brief on disk; **they agreed**.

```
gate_passed: true      violations: []      self_report_disagreed: false
sources_floor_ok: 7 >= 5        urls_floor_ok: 23 >= 10
recency_scan_ok                 all_7_claimed_sources_present_in_brief
brief_on_disk_ok: handoff/current/research_brief_86.94_cycle4.md
                  (18346 chars, independently read)
brief_status_in_brief: COMPLETE
```

Brief: `handoff/current/research_brief_86.94_cycle4.md`. Sources read in full:
Google SWE-book ch.12 (brittleness taxonomy), arXiv 2306.02319 + 2511.11999
(mutation kill-reason taxonomy), arXiv 2507.15892 (StaAgent — metamorphic
testing *of a static analyser's rules*), in-toto v1 statement spec, SLSA v1.0
verifying-artifacts, cherryleaf docs-as-tests (2026-08).

**The gate changed the design.** It produced F3 (below), which I did not have and
which invalidates the first fix I had drafted.

---

## 2. Starting state — the guard is RED, and that is this cycle's first evidence

The park note records "the shipped guard is ALL GREEN 45/0". **That is stale.**
Measured at preflight this morning and again after the gate:

```
FAILED: 42 passed, 3 failed
```

All three failures are the same assertion — the `mentions_reviewed` equality at
`verify_no_sliding_windows_86_94.py:544-551` disagreeing with the pins at
`:227-231`:

| member | pinned | measured |
|---|---|---|
| `backend/slack_bot/scheduler.py` | 282 | **283** |
| `scripts/qa/verify_decision_log_86_97.py` | 6 | **9** |
| `scripts/harness/frontend_route_inventory.py` | 49 | **50** |

Cause, measured: since the pinning commit `964b0255` (2026-08-17T00:51:13+02:00)
exactly three handoff files were added, and two of them merely **name** the
guarded scripts — `handoff/current/overnight_halt.md` (the park note itself) and
`handoff/current/day_report_2026-08-17.md`, then `handoff/current/day_halt.md`
this morning. **None quotes any figure derived from any window.** The guard went
`45 -> 44` *inside the commit that recorded it green*, and `44 -> 42` when this
morning's deviation record was written. Full account:
`handoff/current/day_halt.md`.

---

## 3. Hypothesis

The three findings the cycle-3 evaluator named are all instances of one defect:
**a claim is bound to the wrong referent, and a kill is credited to the wrong
mechanism.** Fixing the binding (not the numbers) closes all three and also
closes the brittleness that turned the guard red overnight.

- `quoted_as_evidence` is bound to *nothing* — only `isinstance` is checked.
- `mentions_reviewed` is bound to `name in text` over the **working tree**, which
  is both the wrong property (criterion 4 asks about a quoted *figure*, not a
  filename) and an unreproducible corpus.
- The `[4]` cells are bound to `h[3] == "SLIDING"` only, never to `h[2]`, so no
  cell can say *which* mechanism killed it.

---

## 4. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

1. "the drift is REPRODUCED first, by EXECUTION and WITHOUT PINNED FIGURES: run the bare-date command twice at times of day that differ by at least an hour and show the two counts DIFFER, and show that the midnight-pinned form differs from both. Do NOT copy a specific count into this criterion -- by this step's own thesis no such count can be regenerated, and an earlier revision of this criterion pinned 621/592/706, which measured 560/712 the same day. That revision was the identical trap this step exists to close, committed inside the criterion written to prevent it"
2. "the class is enumerated FROM SOURCE, not hand-listed: the enumeration rule is written down, the command is quoted with its output, and each member is classified as REPRODUCIBLE or SLIDING with the reason per member"
3. "the enumeration finds its own known member -- the pre-86.91 form of replay_changelog_rule_86_68.py, recoverable from git -- and a scan that cannot is a FAILED gate"
4. "for each SLIDING member, state whether any figure derived from it has been quoted in a masterplan criterion, an audit_basis, a handoff artifact or CHANGELOG; a member whose numbers were never quoted is lower risk and may be left, but that judgement must be stated rather than silent"
5. "any figure found to be unreproducible is CORRECTED IN EVERY FILE THAT CARRIES IT, not merely annotated in one -- a correction must replace, not accompany"
6. "a regression guard is added that would go RED if a new bare-date or now-relative window is introduced into a measurement script, and it is mutation-tested with the control observed GREEN first"
7. "verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS"

**Immutable command** (unchanged):
`bash -c 'source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'`

**Disclosed, as in cycle 3:** this command runs the *86.91* checker and
**cannot fail on any defect in this step's class**. It was green throughout and
proves only that 86.94's work did not break 86.91. The real evidence is
`live_check_86.94.md`. This is a defect in the step's filing, not a claim of
coverage.

---

## 5. Plan

### P1 — criterion 4 becomes falsifiable AND reproducible (evaluator finding (a); brief F3/F4/F5)

Replace `mentions_reviewed: int` with `figure_probes: [regex]` and assert
**`quoted_as_evidence == bool(probe hits)`**.

- Each probe is derived **from the emitting expression**, not from my phrasing:
  `scheduler.py` → `_git_today()` at `:501-507` → `d["commits_today"]`, rendered
  by `formatters.py:102-109`; `verify_decision_log_86_97.py` → the
  `commits=N decision lines=N gap=N` triple; `frontend_route_inventory.py` →
  `opens_30d` / `"usage_source": "git_activity_30d"`.
- **The corpus becomes git-tracked files only.** Brief F3: `handoff/` holds
  49,094 `.md` of which only 5,167 are tracked — **43,927 (89.5%) gitignored**
  via `.gitignore:80`. A count over "whatever is on this disk" is a number about
  a machine exactly as `--since=<bare date>` is a number about a clock. The
  allowlist's own smoking-gun citation
  (`handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md`)
  is itself gitignored.

**Pre-measured** (probe 3, both corpora): all three probe sets discriminate.
Tracked-only, `frontend_route_inventory` drops 71 hits → **5**, still non-empty,
in tracked `handoff/archive/phase-4.7.0/` and `phase-4.7.1/`. So the True claim
survives the corpus repair rather than depending on ignored files.

### P2 — a mutation cell for the fail-closed `<unparsed>` branch (finding (b); brief F6)

Measured 4/4 shapes that reach `:374-379`: argv-list with a variable value, the
f-string-element form, `--since=` with an empty value, `--after` + variable. Add
cells asserting `h[2] == "<unparsed>"`.

### P3 — every cell carries its mechanism (finding (c); brief F7)

Value-classification cells assert `h[2] != "<unparsed>"`; fail-closed cells
assert `h[2] == "<unparsed>"`.

### P4 — corrections that REPLACE (criterion 5)

- `:372-379`'s motivating comment is **stale**: it says the space form
  `--since 2026-08-11` reaches the fail-closed path. It no longer does —
  `window_value()` returns `('2026-08-11', True)` once `PLAUSIBLE_VALUE` matches.
- The allowlist entry presents **paraphrases inside quote marks**
  (`usage_source: git_activity_30d`, `/portfolio 2 /login 1`); the file actually
  reads `"usage_source": "git_activity_30d"` and the second is line-wrapped.
- Every carrier of a `mentions_reviewed` figure must be swept by the **claim
  class with a known-member recall test**, not by my own wordings.

---

## 6. Mutations — named BEFORE the work, each to be RUN with the control observed green

| id | mutation | required outcome |
|---|---|---|
| M-A | restore the fail-OPEN `continue` in place of the `<unparsed>` append | RED (today: **SURVIVED**) |
| M-B | neutralise `VALUE_ARGV_RE` (argv value-parse leg) | RED (today: **SURVIVED**) |
| M-C | neutralise `WINDOW_RE`'s argv alternative (visibility leg) | RED (today: kills exactly the 2 argv cells) |
| M-D | `frontend_route_inventory` `quoted_as_evidence` True→**False** | RED (today: **SURVIVED**) |
| M-E | `scheduler` `quoted_as_evidence` False→**True** | RED (today: **SURVIVED**) |
| M-F | widen the corpus back to untracked files | RED (reproducibility regression) |
| M-G | delete a figure probe from a True member | RED (drift on the *relevant* corpus still re-opens) |

M-A/B/D/E surviving today is the measured basis for the change. **Scoring is by
FAIL-SET DELTA against BASE**, because BASE is currently dirty; a plain
green/red comparison would be meaningless.

---

## 7. Explicit non-goals / rails

- **R8:** `mentions_reviewed` is REPLACED by a strictly stronger predicate, not
  deleted to get green. Proof obligation: M-D and M-E must go from SURVIVED to
  KILLED. If they do not, the change is a loosening and must be reverted.
- **R5:** no edits to `.claude/agents/qa.md`, `qa-verdict.js`,
  `research-gate.js`.
- Criterion 7: no verdict semantics touched; no masterplan step flipped by this
  work other than 86.94 itself, and only on a Q/A PASS.
- The window rule itself (`WINDOW_RE`, `classify`) is **not** relaxed. No
  allowlist member is added or removed.

## 8. References

- `handoff/current/research_brief_86.94_cycle4.md` (gate `wf_c533d502-21e`)
- `handoff/current/day_halt.md` — preflight deviation, the red's provenance
- `.claude/masterplan.json` 86.94 notes — park diagnosis + the three
  post-verdict corrections (W1/W2/W3) that were never re-graded
- Probes: `probe_86_94_attribution.py`, `probe2_86_94.py`, `probe3_witness.py`,
  `probe4_wrongbool.py` (scratchpad; outputs transcribed into
  `live_check_86.94.md`)
