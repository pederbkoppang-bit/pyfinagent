# Contract — phase-86.2

**Step:** `86.2` — *P1 TOTAL DISARM: one oversized JSON int aborts the entire
kill-switch audit replay and strands BOTH protective legs.* **Cycle 189**,
2026-08-09.

**STATUS: PLAN COMPLETE.** Enforced gate returned `gate_passed: true` from
`wf_ce7c54a7-367` **before** any GENERATE work — recomputed, not self-reported:

```
sources_floor_ok: 9 >= 5 | urls_floor_ok: 39 >= 10 | recency_scan_ok
listed_sources_consistent: 9 >= 9
brief_on_disk_ok: research_brief_86.2.md (42230 chars, independently read)
all_9_claimed_sources_present_in_brief   |  self_report_disagreed: false
```

### THREE CORRECTIONS the gate made to this contract — one is load-bearing

**C1 — the immutable verification command EXITS 0 TODAY, with the defect fully
present.** Case E is *printed and narrated* (`measure_sod_date_reachability.py:167-183`)
but **never asserted**; only the HEALTHY control can fail the script
(`:184-186`). **My §3 RED baseline is a witness, not a test.** So criterion 1's
"reproduced FIRST as a red test" **must be a NEW pytest** — the diagnostic
cannot satisfy it, and treating a passing script as the reproduction would have
been exactly the kind of green-that-proves-nothing this project keeps finding.
§4 is amended accordingly.

**C2 — the correct idiom ALREADY EXISTS one layer up.** `_read_audit_rows:247-258`
already wraps each line in its own `try` **and sets `complete = False`**. So the
fix is to make the *apply* loop match a pattern the *parse* loop already uses —
and, critically, **a skip that omits the `complete = False` bookkeeping reopens
the anchor-authority hole five prior Q/A passes closed.** That bookkeeping is
not optional politeness; it is the thing that keeps `armed` honest.

**C3 — the external record is adversarial to a naive skip, and that sharpens
criterion 3.** PostgreSQL redo **PANICs** by default; Kafka Connect defaults to
`errors.tolerance=none`. **No source permits this defect's third state — abort
halfway and then return normally as though the replay succeeded.** The choice is
between *refuse* and *skip-with-loud-bookkeeping*; what exists today is neither.

---

## 1. Research-gate summary

**Brief:** `handoff/current/research_brief_86.2.md`. Self-reported envelope:
9 read in full, 39 URLs, recency scan performed, 6 internal files,
`coverage.rounds: 3`, `dry_rounds: 1`.

**Main re-verified the load-bearing mechanics independently before writing this:**

```
_coerce_nav  except (TypeError, ValueError)   kill_switch.py:127
broad swallow  except Exception -> logger.warning("audit load failed")   :394-395
float(10**400) -> OverflowError: int too large to convert to float   (measured)
```

### The findings that shape the fix

- **F1 — `OverflowError` is unreachable from `(TypeError, ValueError)` by
  construction.** It is `ArithmeticError`'s child, a *sibling* branch. Measured
  in this repo's own 3.14.4 interpreter:
  `isinstance(OverflowError(), (TypeError, ValueError)) == False`.
- **F2 — the exposure is a PRECISELY BOUNDED window**, enumerated over every
  JSON-decodable value: **a JSON integer literal of 310–4300 decimal digits.**
  Below that, `float()` succeeds. `1e400` / `Infinity` / `NaN` are already safe —
  rejected by the `math.isfinite` guard at `:139`. **Above 4300 digits
  `json.loads` itself raises `ValueError`**, caught by the per-line handler at
  `:249`. That bound is a fact the tests can assert rather than hand-wave.
- **F3 — `(TypeError, ValueError)` is a folk tuple.** `decimal.InvalidOperation`
  is also an `ArithmeticError`. ⇒ widen to `ArithmeticError`, which covers
  `OverflowError` *and* the decimal family, rather than bolting on one class.
- **F5 — "aborted halfway" is the real hazard**, and it is structural: the abort
  spans **rows**, not one value. Widening `_coerce_nav` alone still leaves the
  `:394` handler able to eat a whole replay on the next unanticipated fault.
  ⇒ **a per-row `try` inside the loop** is the actual fix; `:394` stays as a
  last-resort net but must stop being the thing that eats a per-row fault.
- **F4 — the fail-safe channel already exists as a flag, not a refusal.**
  `self._history_complete` (`:279`). A skipped row must set it **False**, so the
  existing consumers (`update_peak:625-630`, the anchor gate at `:362-363`) keep
  doing their job. This is criterion 3's subject.
- **F6 — severity is inverted.** The replay-failure log is `WARNING` (`:395`)
  while `_apply_authoritative_peak` logs a *single ignored value* at `ERROR`
  (`:422`). A stranded replay is strictly worse. Raise it, and name the row
  index and source path.
- **Criterion 2 needs a row on BOTH sides.** Case E today has well-formed rows
  only *after* the malformed one. The test must add one *before* it too.

---

## 2. Immutable success criteria — verbatim from `.claude/masterplan.json`

1. the failure mode is reproduced FIRST as a red test (malformed row first in ts order -> both legs stranded -> any_breached False on a real breach), so the fix has something to turn green
2. _coerce_nav no longer lets a malformed value abort the replay: a bad row is skipped and logged, and every WELL-FORMED row before AND after it still applies -- proven by a test with rows on both sides
3. the replay's fail-safe direction is stated and tested: whatever a malformed row does, it must never leave a leg reporting armed=True on a baseline it did not actually load
4. a mutation reverting the widened except makes the reproduction test fail again
5. no threshold changed, no leg re-armed on an unloadable baseline; fresh Q/A PASS

**Verification command (immutable):**
```
bash -c 'source .venv/bin/activate && python scripts/diagnostics/measure_sod_date_reachability.py'
```

**live_check:** `live_check_86.2.md` with the verbatim before/after of case E,
and the verbatim mutation transcript.

---

## 3. Measured RED baseline — captured BEFORE any change

The immutable command, verbatim, on the live tree:

```
CASE E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)
  replayed snapshot : sod_nav=None sod_date=None peak_nav=None
  current_nav       : 80.0   (drop vs sod: None%)
  armed             : False
  daily_baseline_missing=True daily_baseline_stale=False
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : False  (0.0%)
  >>> any_breached      : False
```

**A 20% drawdown and nothing fires.** Every other measured degraded state keeps
`any_breached: True` via the date-independent trailing leg; **case E alone
reaches `False`.**

The diagnostic isolates itself (`tempfile.TemporaryDirectory` at `:48`, asserts
`str(ks._AUDIT_PATH).startswith(td)` at `:54`, restores at `:72`). Live journal
`90e0303130fc…` before and after — verified.

---

## 4. Plan

1. **Widen the coercion** — `_coerce_nav`'s except tuple to include
   `ArithmeticError` (covers `OverflowError` + the decimal family). Behaviour
   unchanged for every value that already worked: a malformed value still
   returns `None`.
2. **Per-row isolation** — wrap each row's application in the replay loop in its
   own `try`, so a fault can never span rows. `:394` remains a last-resort net.
3. **Fail-safe direction (criterion 3)** — a skipped row sets
   `_history_complete = False`, so no leg can report `armed: True` on a baseline
   it did not actually load. **Assert this, do not assume it.** Per C2 this
   mirrors what `_read_audit_rows:247-258` already does for the PARSE layer, and
   omitting it would reopen the anchor-authority hole five prior Q/A passes
   closed — so it is tested directly, not inferred from the skip working.
4. **Loudness (F6)** — the skip logs at `ERROR` naming the row index and source
   path; the replay-failure log is raised from `WARNING`.
5. **Tests — a NEW pytest file**, because per C1 the immutable command cannot
   host the reproduction (it exits 0 with the defect present). Following the
   phase-86.1 idioms that landed today (autouse live-journal byte-identity
   fixture, detached `KillSwitchState`, redirect-then-assert-derivation,
   pinned flags):
   - the RED reproduction (malformed row **first** in ts order) — this is what
     criterion 1 actually requires;
   - well-formed rows **before AND after** the bad one, both applying (criterion 2);
   - the 310/4300-digit **boundary** asserted from F2;
   - criterion 3's fail-safe assertion;
   - criterion 4's mutation.

## 5. Scope decision the brief asked for explicitly

`update_peak` (`:611`, `:633`) and `reset_peak` (`:697`) coerce with a **bare
`float(nav)`**, so the *same* `OverflowError` is reachable from the **in-memory**
path. **That is OUT OF SCOPE here and it is not ambiguous:** `kill_switch.py:406-408`
already flags it, and it is masterplan step **36.19**, which exists and is
pending. 86.2 is the **replay** path. Recorded rather than left implicit, because
the brief specifically warned that leaving it ambiguous is how it gets lost.

## 6. Non-goals

- No threshold, limit or gate changed. No leg re-armed on an unloadable baseline.
- **Not** turning `peak_reset` into a ratchet, and not touching
  `_apply_authoritative_peak`'s existing guard.
- No production behaviour change for any value that already coerced successfully.
- No `backend/.env`, no flag promotion, `historical_macro` untouched.
- If 86.2 adds instance state to the replay it **must be class-level** — the
  36.7 fixtures build detached states via `object.__new__`, and an instance-only
  attribute breaks them (the phase-36.12 trap, already recorded at `:160/169/174`).

## 7. References

`handoff/current/research_brief_86.2.md` · `scripts/diagnostics/measure_sod_date_reachability.py`
(case E, `:130-146`) · `backend/tests/test_book_safety_69.py` (86.1 idioms, landed today) ·
`backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` (`ks_tmp_audit`, `isolated_state`) ·
masterplan `86.2`, `36.19`
