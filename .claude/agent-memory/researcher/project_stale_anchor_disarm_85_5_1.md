---
name: stale-anchor-disarm-85-5-1
description: phase-85.5.1 — the kill-switch guard was already CORRECT (per-leg selective degradation); the defect was a mock omitting 3 snapshot keys. Plus the OverflowError replay-abort path, the worktree baseline technique, and a latent live-peak mutation in a "dark by default" test.
metadata:
  type: project
---

Step 85.5.1 (`test_book_safety_69.py::test_valid_nav_still_breaches` RED). Measured
2026-08-08 with an in-process script; live journal md5 unchanged.

**Why:** the step framed it as "the kill switch DISARMS instead of firing on a real 20%
breach", implying a broken guard. Three premises were wrong, and finding that out took a
measurement, not reading.

**How to apply:** the reusable classes below outlive this step.

1. **"The switch disarms" almost always means ONE LEG disarms.** `evaluate_breach`
   (`kill_switch.py:749-882`) evaluates the daily and trailing legs independently and ORs
   them at `:876`. On a stale/None `sod_date` the trailing leg still fires:
   `any_breached=True`, only `daily_loss_breached=False`. Before accepting a "the guard
   didn't fire" premise, print EVERY leg flag, not `any_breached`. The RED test failed on
   its first conjunct alone.

2. **The guard was already right — the fix was the test.** `kill_switch.py:774-785`
   explicitly rejects a wholesale `if not sod or not peak: return disarmed`. That is
   textbook IEC 61511 selective bypass ("bypass a single channel rather than the entire
   trip logic … preserves available redundancy"). When a safety module's comments state a
   design choice that matches the standards literature, suspect the TEST before the guard.
   The defect was `test_book_safety_69.py:80` mocking `snapshot` to a dict missing
   `sod_date`, `baseline_provenance` AND `sod_provisional` — a shape `_snapshot_locked`
   (`:450-473`) can never produce. Repo-wide there is exactly ONE snapshot mock.

3. **`_sod_date_is_stale` refuses to double-name an absence** (`:922-923`): when
   `sod_nav` is None it returns **False**. So "startup before the first anchor" is
   `daily_baseline_MISSING`, NOT stale — a different flag, a different UI branch. Do not
   bucket them together.

4. **NEW reachability path — `OverflowError` aborts the whole audit replay.**
   `_coerce_nav` (`:114-141`) catches only `(TypeError, ValueError)`; a JSON integer too
   large for a float (`float(10**400)`) raises `OverflowError`, which the outer handler at
   `:394-395` swallows — so every row AFTER the bad one is never applied and both anchors
   freeze at the last good row. Log signature: `kill_switch: audit load failed: int too
   large to convert to float`.
   A TORN pair is NOT reachable: `_sod_nav` (:299) and `_sod_date` (:313) are in the same
   branch with no raising call between them.

5. **phase-85.6 NARROWED the stale window, it did not widen it.** The Step-0 provisional
   roll (`paper_trader.py:1276-1301`) re-arms the daily leg the same cycle; the upgrade at
   `:1415-1449` (reading the DURABLE `sod_provisional`, replayed from the audit row) closes
   the multi-session-stale-value hazard. Pre-85.6 the daily leg could sit disarmed all day
   waiting for Step 5.5. Any later fix must not fight that path.

6. **Baseline a suite without corrupting live journals: use a `git worktree`.** Every
   polluting path derives from the module's own location — `kill_switch._AUDIT_PATH`
   (`:48`, `Path(__file__).resolve().parents[2]`), `cycle_health._HISTORY_PATH`/
   `_HEARTBEAT_PATH` (`:36-37`), `cycle_lock._LOCK_PATH` (`:53`) — so one worktree
   relocates them ALL, no per-test monkeypatching. `handoff/kill_switch_audit.jsonl` and
   `handoff/.cycle_heartbeat.json` are git-TRACKED. **Assert the precondition** before
   trusting the run: `python -c "import backend.services.kill_switch as k; print(k._AUDIT_PATH)"`
   must print a path under the worktree. Copy-then-`git checkout --` is REPAIR not
   prevention and races a live backend that may be writing concurrently.
   Diff failure SETS (`grep -E "^(FAILED|ERROR)" | sort`), never counts — a bare "26" hides
   a 1-for-1 swap.

7. **A "dark by default" test can become a live mutation the day the flag flips.**
   `test_book_safety_69.py:86-92` calls `st.reset_peak(12345.0)` on the REAL singleton with
   no `_AUDIT_PATH` redirect. Safe only because `kill_switch_peak_reset_enabled` is
   measured False (`reset_peak` returns None at `:694` before locking or appending). If the
   owed KS-PEAK-RESET token is approved, it writes a `peak_reset` row and drops the live
   peak from ~24666 to 12345, replayed on every future boot. Queued as its own P1.
   Class: grep for tests that touch a production singleton guarded only by a DARK flag.

Brief: `handoff/current/research_brief_85.5.1.md`. Related: [[kill-switch-deadlock-85-6]],
[[kill-switch-36-9-armed-semantics]], [[kill-switch-36-13-alternate-path]].
