# Experiment results — phase-86.1

**Step:** `86.1` — P1 live-state landmine: `test_peak_reset_dark_by_default`
calls `reset_peak` on the REAL kill-switch singleton, safe only because an
operator flag is OFF. **Cycle 188**, 2026-08-09.

## 1. What changed

`backend/tests/test_book_safety_69.py` only. **No production code.** No
threshold, no flag, no `backend/.env`.

| Change | Criterion |
|---|---|
| `test_peak_reset_dark_by_default` rewritten: `_AUDIT_PATH` → `tmp_path` **before** any state is constructed; a **DETACHED** `KillSwitchState`, never `ks.get_state()`; the vacuous `get_state` patch **deleted**; the flag **pinned** to the value the test asserts about; preconditions asserting `_audit_archive_dir()` is derived from the redirect and `_audit_source_paths()` holds no live file | 1 |
| NEW `test_phase_86_1_the_pre_fix_form_really_would_have_destroyed_the_live_peak` | 2 |
| NEW function-scoped autouse `_live_kill_switch_journal_is_byte_identical` | 3 |
| docstring "omitted three" → "omitted **seven**" | 4 |

## 2. The four research findings, all re-verified at source

1. **The isolation asymmetry is INVERTED.** The flag-**ON** arm (`:195-207`) IS
   isolated; the **OFF** arm was not. The step text implies the opposite.
2. **A second landmine:** with the flag ON, `assert out is None` goes **RED** —
   greenness was coupled to operator config in both directions. Now pinned.
3. **The `get_state` patch was vacuous BY IDENTITY** — `st` was bound to the
   real singleton on the line above, so re-pointing the accessor at the same
   object changed nothing; module functions read `_state` directly anyway.
4. **A redirect alone is a HALF fix** — `:697` assigns `self._peak_nav` before
   auditing, so the in-memory singleton is corrupted even when the row goes to
   tmp. **Demonstrated, not argued** (§3).

Re-derived: `reset_peak` at `kill_switch.py:670`, DARK `return None` at `:694`
**before** the lock, assign `:697`, `_append_audit` `:698-700`. The step text's
`~694` referred to the whole function and is stale.

`_snapshot_locked` returns **9** keys (measured), so a 2-key mock omitted
**seven** — criterion 4.

## 3. Verification — verbatim

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'
16 passed in 1.48s
```
Live journal `90e0303130fc…` → `90e0303130fc…`, **62 lines**, unchanged.

Adjacent kill-switch files: **87 passed**, journal unchanged.

`ruff`: `I001` autofixed; **`SIM117` is PRE-EXISTING** — proven by running ruff
on a HEAD copy inside `backend/tests/`, and `git diff | grep -c
"cycle_lock_failed_acquire"` = **0**. Left unfixed as out-of-scope.

**Criterion 2 — the destructive write actually performed**, by production
`reset_peak`, on a byte-identical copy (full transcript in `live_check_86.1.md`):

```
row appended             : {"event": "peak_reset", "old_peak": 24666.57, "new_peak": 12345.0, "trigger": "flatten"}
in-memory singleton peak : 12345.0  (was 24666.57)
peak after REPLAY        : 12345.0
trailing trip point      : 22199.9 -> 11110.5
LIVE journal byte-identical throughout : True
```

`peak rows in the journal today: 0` — so the row wins the `ts` merge-sort
outright; nothing later overrides it.

## 4. Honest limits

- Criterion 2 was run against a **copy**, not the operator's journal. Doing it
  literally destroys 24666.57 and breaks criterion 3 in the act. The copy is
  proven identical, and `_AUDIT_PATH`'s module default is proven to be the live
  file, so it stands in by construction. **Disclosed in `contract_86.1.md` §5.**
  A CONDITIONAL here would be legitimate.
- The autouse guard is a **detector**, not a preventer — bytes are on disk when
  it fires. The general preventer is **86.6**.
- **Not claimed:** that no other file has this flag-inert-today shape. My census
  methods produced a false negative on this very file earlier today.
- The mutation test drives the real singleton and restores `_peak_nav` in a
  `finally`. It deliberately does **not** rebuild the singleton — at that point
  `_AUDIT_PATH` still points at the decoy, so rebuilding would replay it and
  install 12345.0 permanently, turning the cleanup into the leak.
