---
name: replay-abort-total-disarm-86-2
description: phase-86.2 -- the verification command EXITS 0 with the defect present; the OverflowError window is BOUNDED 310..4300 digits because json.loads already guards above it; the per-row skip idiom already exists one layer up
metadata:
  type: project
---

Step 86.2 (`kill_switch._load_from_audit` aborts on `OverflowError`, stranding
BOTH protective legs). Five things that are non-obvious and would mislead an
executor who reasoned instead of measured.

**1. The immutable verification command is a MEASUREMENT HARNESS, not a gate.**
`scripts/diagnostics/measure_sod_date_reachability.py` exits **0 today with the
defect fully present**. `main()` returns 1 only when the HEALTHY control fails to
breach; case E's `any_breached=False` on a 20% drawdown is *printed and narrated*
but never asserted. So "the verification command is green" proves nothing about
this defect. Class lesson: **before treating a step's `verification.command` as
the gate, check what can actually make it non-zero** -- see
[[feedback_immutable_criteria_must_be_green_able]].

**2. The exposure is BOUNDED and measurable, not open-ended.**
`OverflowError` escapes `_coerce_nav`'s `(TypeError, ValueError)` only for a JSON
**integer literal of 310 to 4300 decimal digits**. Above 4300, `json.loads`
itself raises `ValueError` (`sys.int_max_str_digits`), which the per-line handler
already catches. `1e400` / `Infinity` / `NaN` / huge *strings* all become `inf` or
`nan` and are killed by the existing `math.isfinite` check. Enumerate the escape
set by execution; do not reason about it.

**3. The correct idiom already exists ONE LAYER UP in the same file.**
`_read_audit_rows` wraps each line in its own `try` and, on failure, does
`complete = False; continue` -- skip loudly AND record that history was unseen.
`_load_from_audit` has one `try` around the entire `for row in rows:`. Any fix
that skips a row without also setting `complete = False` reopens the
anchor-authority hole that five prior Q/A passes closed.

**4. `isolated_state` (36.7) deliberately does NOT set the class-level attrs.**
It builds a detached state via `object.__new__` and hand-sets only 8 fields.
`_baseline_provenance` / `_sod_provisional` / `_history_complete` survive as
CLASS-level defaults -- that is *why* they are class-level. New instance state
added to the replay must be class-level too, or ~34 fixtures AttributeError
inside `_snapshot_locked`.

**5. PostgreSQL is the adversarial source against "just skip it".**
For *authoritative-state* recovery (not a rebuildable projection) the default is
to REFUSE: `ignore_invalid_pages=off` PANICs and aborts recovery; skipping is
superuser-gated, restart-only, and labelled "may cause crashes, data loss,
propagate or hide corruption". Kafka Connect's default is likewise
`errors.tolerance=none`. Neither system permits the third state this defect
produces -- **abort halfway and then return normally, serving a silently
truncated state**.

**Why:** the 85.5.1 author got the ordering wrong (malformed row LAST -> nothing
stranded -> opposite conclusion) and shipped a claim that had to be retracted; the
same file now carries an in-code comment about it.

**How to apply:** on any replay/recovery-loop step, (a) run the verification
command first and find out what makes it fail, (b) enumerate the reachable bad-input
set by execution, (c) look one layer up for the guard that already exists, and
(d) place the poison record FIRST in sort order or the reproduction is vacuous.

Related: [[project_stale_anchor_disarm_85_5_1]],
[[project_flag_accident_landmine_86_1]],
[[reference_vacuous_type_guards_on_bq_string_columns]],
[[feedback_measure_dont_assert_claims]].
