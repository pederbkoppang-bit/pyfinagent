---
name: flag-accident-landmine-86-1
description: Step 86.1 — the flag-ON arm was isolated and the flag-OFF arm was not; patching get_state() was vacuous BY IDENTITY; the 24666 peak lives only in the ARCHIVES, not the live journal
metadata:
  type: project
---

Step 86.1 (`test_book_safety_69.py::test_peak_reset_dark_by_default`). Four
findings that are not derivable by re-reading the code casually:

1. **The isolation asymmetry is INVERTED.** The flag-**ON** arm
   (`test_peak_reset_active_when_token_enabled`, :195-207) IS fully isolated;
   the flag-**OFF** arm (:186-192) is NOT. The author isolated the arm they knew
   would write and left unisolated the arm they *reasoned* could not — and that
   reasoning reads a value the operator owns. Generalise: **isolation must be
   established by the fixture, never derived from the value under test.** A test
   asserting "this is a no-op" needs MORE isolation than one asserting "this
   writes", because the no-op assertion is exactly the claim that can go false.

2. **Patching the accessor was vacuous TWICE.** `st = ks.get_state()` at :187
   binds the real singleton BEFORE `monkeypatch.setattr(ks, "get_state", lambda:
   st)` at :188 — so the lambda returns the same object; a no-op by identity.
   And module-level fns read the global `_state` DIRECTLY
   (`kill_switch.py:793/:995/:1033/:1047/:1053`), never via `get_state()`.
   Check for identity-vacuity before believing any accessor patch.

3. **A redirect-only fix is a HALF fix.** `_append_audit` is a `@staticmethod`
   reading the module global `_AUDIT_PATH` at call time (:440) — so the FILE
   needs the redirect; but `reset_peak` also assigns `self._peak_nav` at :697,
   so the in-memory singleton needs `monkeypatch.setattr(ks, "_state", fresh)`
   too. `_audit_archive_dir` is DERIVED (:89-91) and needs nothing.
   `KillSwitchState.__init__` ends with `_load_from_audit()`, so the redirect
   must precede construction.

4. **MEASURED 2026-08-09: the live `handoff/kill_switch_audit.jsonl` holds ZERO
   peak rows** (62 rows: 44 pause / 10 resume / 8 sod_snapshot). All 20
   `peak_update` rows and the 24666.57 max live in `handoff/audit/*.jsonl`
   archives. `peak_reset` has NEVER fired. So a `peak_reset` written today wins
   the `ts` merge-sort outright and destroys 24666.57 permanently (the ratchet
   cannot heal downward) — trailing trip point 22199.9 → 11110.5 at a 10% limit.

**Why:** filed during the 86.1 research gate; the phase-85.5.1 gate found the
landmine and the 86.3 conftest explicitly declined to close the filesystem
channel. **How to apply:** when auditing any test that touches a
safety-critical singleton, ask "is this safe, or merely currently-configured
safe?" and check the OTHER arm's isolation as the baseline. See
[[kill-switch-36-9-armed-semantics]], [[kill-switch-36-12-traps]].
