# Contract — phase-36.8

**Step id:** `36.8` (phase-36, **P0**, `harness_required: true`)
**Title:** *Archive merge + DARK `reset_peak` = permanent flatten/pause lockout on a legitimately
re-anchored book.*
**Written:** 2026-07-26, BEFORE any code was changed for this step. Research → contract → generate.

## TIER

| field | value |
|---|---|
| Tier | **T3** |
| Model | Opus 5, effort `max` |
| Rationale | P0 on kill-switch replay semantics; the failure mode is a permanent engine stop. |

## Research gate — PASSED

`handoff/current/research_brief_36.8.md`, tier `moderate`: `external_sources_read_in_full: 8`
(floor 5), `snippet_only_sources: 11`, `urls_collected: 30`, `recency_scan_performed: true`,
`internal_files_inspected: 9`, **`gate_passed: true`**. Spawned before this contract.

**It refuted the fix shape this step's own text suggested, with sources.** The step proposed "a
live-file freshness/authority marker" as one option. Every authority the brief read solves this with
an **in-stream boundary marker**, not file recency — Fowler's rejected-events, Kafka KIP-101 leader
epoch, PostgreSQL timelines, Kleppmann's fencing tokens — and **PostgreSQL explicitly prefers the
ARCHIVE over the live directory**, which directly refutes live-file-precedence. Design follows the
research, not the step text.

**It also corrected the step's stated failure scenario — this matters for honesty about scope.**
"A book legitimately re-anchors lower" is **not reachable through `update_peak`**, because
`update_peak` cannot write a lower peak while the merged peak is higher (it only ratchets up). The
reachable production door is the **swallowed archive-scan exception** at `kill_switch.py:104-105`
(or an absent-then-present `handoff/audit/`): boot with the archives unreadable, anchor a fresh low
peak from `None`, then boot again with the archives readable and the old high peak returns. The
criterion-1 test constructs the row shapes directly, which is legitimate and is what the criterion
asks for — but the artifacts must state the real production route rather than implying `update_peak`
provides it.

**Measured facts that decide criterion 3** (read-only, by the researcher):
- **897 rows across 5 files.** `handoff/kill_switch_audit.jsonl` (live) + 4 under `handoff/audit/`.
- **100% of today's live baselines come from the ARCHIVES**, and the true peak `24666.57` comes from
  the **OLDEST** file.
- **Zero `peak_reset` rows have ever existed.**
- **Boot cost 0.95 ms total, 1.06 µs/row.**

## Hypothesis (falsifiable)

`_audit_source_paths` merges the archives unconditionally and `peak_update` replay `max()`es across
the merged stream, so a stale archived peak outranks a fresh, intentional, lower one — and with
`reset_peak` DARK and `update_peak` ratchet-only, nothing can lower it again. If the *writer* stamps
the anchor-from-`None` case on its row and the *replay* treats a marked anchor as an authority
boundary that ASSIGNS (unmarked rows continuing to ratchet), then a fresh anchor wins over stale
history while 36.7's true-peak restore is byte-preserved.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `A test reproduces the exact failure: fresh peak_update in the live file at a LOWER value than an archived peak, then a boot restore, and asserts the RESTORED peak reflects the fresh live value, not the stale archived one -- this test must FAIL against the current unconditional max()-merge (record the failing output verbatim)`
2. `36.7's original defect stays fixed: a genuinely OLDER live peak plus a genuinely HIGHER true historical peak in the archives still restores the higher true peak (do not regress the fix this step depends on)`
3. `No cap-less unbounded archive growth is left undocumented -- either a pruning/cap policy is added, or the boot-cost-scales-with-archive-count risk is explicitly accepted and measured`
4. `reset_peak's DARK-by-default behavior is untouched (still gated on kill_switch_peak_reset_enabled)`
5. `MUTATION-TEST: reverting the freshness/precedence fix must fail the new test`

**Verification command (immutable):**
```
source .venv/bin/activate && python -m pytest backend/tests/ -q -k kill_switch
```

**live_check (immutable):** *A test log showing: (a) the original 36.7 restore-true-peak behavior
still works, (b) the new re-anchor-respects-fresh-live-data behavior now works, both against real
archived file shapes.*

## Design — decided, with the rejected options named

**CHOSEN: an in-stream authority boundary. A `peak_update` row that represents an anchor-from-`None`
is MARKED; the replay ASSIGNS at a marked row and ratchets everywhere else. Unmarked ⇒ ratchet.**

1. **Mark at the writer.** `update_peak` stamps the distinction when `self._peak_nav is None`. That
   is the only place the two cases are still distinguishable — the forensic gap `kill_switch.py`
   already names in prose and `36.12` filed against.
2. **Boundary resets the fold.** In `_load_from_audit`, a marked anchor ASSIGNS; later rows ratchet
   up from it; earlier rows are superseded because `_read_audit_rows` sorts by `(ts, src, line)`.
   That sort is the hinge — Kafka leader-epoch / PostgreSQL timeline semantics in one branch.
3. **Unmarked ⇒ ratchet**, so all **20** existing `peak_update` rows keep 36.7's behaviour exactly.
4. **One guarded helper for BOTH assignment branches.** 36.8 must not add a second
   assignment-semantics branch while the first is unguarded: a value that does not coerce to a
   positive finite float is ignored and logged loudly.
5. **A FIELD on `peak_update`, not a new event name.** A new name would have to be added to
   `_BASELINE_EVENTS` or `36.12`'s `baseline_history_exists` probe goes blind to it. A field avoids
   the coupling entirely.

**REJECTED — live-file precedence.** Refuted by the research: PostgreSQL prefers the archive.
**REJECTED — boot-time-only scoping / read-once-then-discard.** Destroys 36.7's whole purpose.
**REJECTED — consuming `36.12`'s `baseline_anchor_on_lost_history` as authoritative.** It would break
`test_phase_36_12_the_new_event_is_replay_inert` and is semantically backwards: that event marks an
**accident** (36.12 exists to flag the anchor as a fiction), so letting it lower a real high-water
mark is the under-conservative direction. The *authorized* re-anchor stays `peak_reset`, token-gated
— which matches measured industry practice (withdrawal resets are authorized by a discrete payout
event, never automatic).
**REJECTED — a pruning cap (criterion 3).** Actively dangerous, and the measurement proves it: the
true peak lives in the **oldest** file, so a cap would delete the row the switch depends on. Instead:
record the measured boot cost, and extend the existing housekeeping allowlist idiom to name
`handoff/audit/kill_switch_audit*.jsonl` as safety-relevant-do-not-prune, pinned by a test in the
style of the existing two-script allowlist-agreement test. All five files are git-tracked, which is
the existing recoverability backstop.

## Files to change

| Path | Change |
|---|---|
| `backend/services/kill_switch.py` | `update_peak` marks the anchor-from-`None` row; `_load_from_audit`'s `peak_update` branch assigns at a marked row and ratchets otherwise; new `_apply_authoritative_peak` helper guarding BOTH assignment branches; `reset_peak`'s DARK gate untouched. |
| `scripts/housekeeping/backfill_handoff_archive.py`, `scripts/housekeeping/verify_handoff_layout.py` | extend the allowlist idiom to protect `handoff/audit/kill_switch_audit*.jsonl`. |
| `backend/tests/test_phase_36_8_*.py` | new. **Must** use the 36.7 module's `ks_tmp_audit` / `isolated_state` fixtures and port the autouse live-file write-protect fixture. |

## Anti-patterns guarded

1. **Guard-that-cannot-fail** — every new guard gets a named mutation, both directions.
2. **A second unguarded assignment branch** — the helper exists precisely to prevent that.
3. **Regressing the step this one depends on** — criterion 2 is asserted against the REAL corpus, not
   only synthetic fixtures.
4. **Silently making an accident authoritative** — the 36.12 event stays replay-inert.
5. **Writing live safety state from a test** — the write-protect fixture is ported, not re-invented.

## Out of scope

- **`36.15`** (the `peak_reset` replay branch's missing None-check) stays its own step, but this
  step's guarded helper **may deliver part of its code fix**. 36.15's text will be annotated so its
  executor re-measures criterion 1 before assuming there is still a defect.
- **The swallowed archive-scan exception** at `:104-105` — distinguishing "no archive dir" from
  "archive dir unreadable" is arguably its own step, and the second is not a safe reason to anchor a
  new peak. Flagged, not fixed.
- **`36.9`** and **`36.16`** do not collide (checked).

## Risk after this step passes

- The marker only helps rows written **after** it ships; the existing 20 unmarked rows stay ratchet-only
  by design, so a pre-existing stale peak is still only correctable by `peak_reset` (owed token, 79.6).
- Archive growth remains uncapped by deliberate choice; the accepted risk is boot cost, now measured.
