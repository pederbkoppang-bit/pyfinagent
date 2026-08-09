# Contract — phase-86.1

**Step:** `86.1` — *P1 LIVE-STATE LANDMINE: `test_peak_reset_dark_by_default`
calls `reset_peak` on the REAL kill-switch singleton, and is safe only because
an operator flag is currently OFF.* **Cycle 188**, 2026-08-09.

---

## 1. Research-gate summary

**Brief:** `handoff/current/research_brief_86.1.md` (36,998 bytes).
**Gate: PASSED** — and it is the **first live run of the new Workflow rail**
(`wf_9880694c-d30`, shipped by 36.27 earlier today):

```
gate_passed: true   (ENFORCED, not self-reported)
sources_floor_ok: 8 >= 5 | urls_floor_ok: 44 >= 10 | recency_scan_ok
listed_sources_consistent: 8 >= 8
brief_on_disk_ok: 36790 chars, independently read
all_8_claimed_sources_present_in_brief
self_report_disagreed: false
```

**Four findings that change this step — every one RE-VERIFIED by Main at source
before being acted on:**

1. **The isolation asymmetry is INVERTED.** The **flag-ON** arm
   (`test_peak_reset_active_when_token_enabled`, `:195-207`) **IS** isolated —
   it redirects `_AUDIT_PATH` at `:196` and builds a fresh
   `ks.KillSwitchState()`. The **flag-OFF** arm (`:186-192`) is the unguarded
   one. The step text implies the opposite. *Confirmed by reading both.*
2. **A SECOND landmine, not in the step text.** With the flag ON,
   `assert out is None` at `:191` goes **RED** — so this file's greenness is
   coupled to operator configuration in *both* directions, not just its safety.
3. **The `get_state` patch at `:188` is vacuous BY IDENTITY.**
   `st = ks.get_state()` binds the real singleton at `:187`, *then*
   `monkeypatch.setattr(ks, "get_state", lambda: st)` re-points the accessor at
   the object it already holds. It changes nothing. And module functions read
   `_state` **directly** (`:793/:995/:1033/:1047/:1053`), so patching the
   accessor could not have isolated them anyway.
4. **A redirect alone is a HALF fix.** `reset_peak` assigns
   `self._peak_nav = float(new_peak)` **before** auditing, so with the flag ON
   the **in-memory singleton** is corrupted even if the audit row goes to tmp.
   ⇒ the fix must use a **DETACHED state**, not merely a redirected path.

**Re-derived line numbers** (the step text's `~694` is stale): `reset_peak` at
`kill_switch.py:670`; the DARK `return None` at `:694`, **before** the lock;
assign at `:697`; `_append_audit` at `:698-700`; `_AUDIT_PATH :48`;
`_audit_archive_dir() :89`; `_audit_source_paths() :94`; `_BASELINE_EVENTS :709`.

**Severity, measured (Main re-verified):** the live journal holds **ZERO peak
rows** — 62 rows = 44 `pause` + 10 `resume` + 8 `sod_snapshot`. All 20
`peak_update` rows and the 24666.57 max live in `handoff/audit/` archives;
`peak_reset` has **never** fired. So a `peak_reset` row written today **wins the
`ts` merge-sort outright** and destroys 24666.57 permanently — **trailing trip
point 22199.9 → 11110.5.**

---

## 2. Immutable success criteria — verbatim from `.claude/masterplan.json`

1. the test no longer touches the real singleton: it redirects _AUDIT_PATH (and _audit_archive_dir) to tmp BEFORE constructing state, and a precondition asserts _audit_source_paths() contains no live file
2. a MUTATION proves it: force kill_switch_peak_reset_enabled True in the test's own settings and show that the pre-fix form WOULD have written a peak_reset row to the live journal while the fixed form writes only to tmp -- demonstrated, not argued
3. the live handoff/kill_switch_audit.jsonl is byte-identical before and after the whole file runs, asserted by the test run itself rather than checked by hand
4. the stale 'omitted three' docstring figure is corrected to seven
5. no assertion is weakened and no other test changes status; fresh Q/A PASS

**Verification command (immutable):**
```
bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'
```

**live_check:** `live_check_86.1.md` with before/after line count and sha256 of
the live journal across a full run of the file, plus the verbatim mutation
transcript showing the pre-fix form writing a `peak_reset` row under a forced-ON
flag.

---

## 3. Measured baseline

```
handoff/kill_switch_audit.jsonl   62 lines
sha256 90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653
event census: {'pause': 44, 'resume': 10, 'sod_snapshot': 8}   peak rows: 0
```

`settings.py:39` — `kill_switch_peak_reset_enabled: bool = Field(False, ...)`.
**Measured False**, which is the only reason the landmine is inert.
**79.6 (KS-PEAK-RESET) is APPROVED but NOT APPLIED.** This step must land first.

`_snapshot_locked` returns **9** keys — `paused, pause_reason, sod_nav,
sod_date, peak_nav, paused_at, auto_resume_alerted_at, baseline_provenance,
sod_provisional` — so a mock carrying 2 of them omitted **seven**, not three
(criterion 4).

---

## 4. Plan

1. **Fix the OFF arm** (`test_peak_reset_dark_by_default`):
   - redirect `_AUDIT_PATH` to `tmp_path` **BEFORE** constructing any state
     (`__init__` replays, so ordering is load-bearing);
   - build a **DETACHED** `KillSwitchState`, never `ks.get_state()` — required by
     finding 4, since a redirect alone leaves `:697` corrupting the singleton;
   - **drop the vacuous `get_state` patch** (finding 3) rather than leave a line
     that reads like isolation and provides none;
   - add a **precondition** asserting `_audit_source_paths()` contains no live
     file (criterion 1).
   - `_audit_archive_dir()` is **derived from `_AUDIT_PATH`** (`:89-91`), so one
     redirect covers both — assert that rather than assume it.
2. **Close the second landmine** (finding 2): the OFF arm must assert the DARK
   behaviour against a **pinned** flag value, so the test states the
   configuration it depends on instead of inheriting the operator's.
3. **Criterion 3** — an autouse byte-compare fixture over the live journal, so
   the *test run itself* asserts it, mirroring
   `test_phase_36_12_...::_live_audit_file_is_write_protected`.
4. **Criterion 4** — "omitted three" → "omitted seven".
5. **Criterion 2** — the mutation, see §5.

## 5. Criterion 2 — how the mutation is run WITHOUT writing to the live journal

Criterion 2 asks to *"show that the pre-fix form WOULD have written a
`peak_reset` row to the live journal"*. Read literally against the operator's
file, satisfying it means **destroying the 24666.57 high-water mark** — the
exact harm this step exists to prevent, and it would break criterion 3 in the
act.

**How it will be demonstrated instead — with the REAL code path, not an
argument:**

- Copy the live journal **byte-for-byte** into `tmp_path`.
- Point `_AUDIT_PATH` at that copy and run the **pre-fix form verbatim**
  (`st = ks.get_state()`; `st.reset_peak(12345.0, trigger="flatten")`) with
  `kill_switch_peak_reset_enabled` forced **True**.
- Show a real `peak_reset` row appended to the copy, and show the replayed peak
  collapsing 24666.57 → 12345.0 on that copy.
- Assert the **live** file is byte-identical throughout.
- Separately assert the module default `_AUDIT_PATH` **is** the live file, so
  the copy stands in for it by construction rather than by assertion.

That is strictly stronger than 86.3's (a)∧(b) split: the destructive write is
**actually performed**, by the real production code, on a byte-identical
stand-in. **This is a deliberate deviation from the literal wording, disclosed
here rather than taken silently.** The criterion text is not edited. If Q/A
judges the stand-in insufficient, that is a legitimate CONDITIONAL and I will
not argue it away.

## 6. Non-goals

- **Not** applying 79.6 / flipping `kill_switch_peak_reset_enabled`. Operator-gated.
- **Not** changing `reset_peak` or any production code. This is a test-isolation
  defect; the production dark-by-default behaviour is correct.
- **Not** closing the filesystem channel generally — that is `86.6`, which now
  also carries the live cycle-lock instance found today.
- No assertion weakened, no threshold touched, no `backend/.env` write.

## 7. References

`handoff/current/research_brief_86.1.md` ·
`handoff/current/killswitch_cluster_reconciliation_2026-08-09.md` ·
`backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` (`isolated_state`,
`ks_tmp_audit`) · `test_phase_36_12_...` (autouse byte-compare) ·
masterplan `86.1`, `86.6`, `79.6`
