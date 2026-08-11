---
name: failing-test-triage-86-5
description: phase-86.5 triage of the standing backend/tests failures — 17 not 26, NONE flaky, and three failures where the test is wrong and the code is right
metadata:
  type: project
---

Measured 2026-08-11, HEAD `5759914c`, live tree:
`python -m pytest backend/tests/ -q -p no:randomly` → **`17 failed, 3417 passed`** in 401s.

**Why the title says 26 and I measure 17.** The 26 baseline (2026-08-08) was
`26 failed, 3017 passed`; I measure 17 failed / **3417** passed — **+400 tests**. The
populations are not comparable. 26 is not wrong, it is **stale**. Never reconcile two
counts without checking the denominator moved.

**NONE of the 17 is a flaky test.** 100% deterministic across four runs (full suite,
isolated, `TZ=Pacific/Kiritimati`, `TZ=Etc/GMT+12`): **0 order-dependent, 0
clock-dependent**. So the flaky-test literature's *detection* half does not apply here —
only its *taxonomy* and *disposition* halves do. Do not import iDFlakies' 50.5%
order-dependent prior; the OD share is ecosystem-specific (JS studies find "very few").

**Why: the three failures where acting on the test would do harm.**
1. **4 tests assert a code default but measure an env override.** `settings.py:46,342`
   default `False`; `backend/.env:83,84` set both `=true`. `Settings()` loads the .env.
   "Fixing" by flipping either side silently **disarms two armed money-path features**.
   The fix is `Settings(_env_file=None)`.
2. **`test_c6_no_launchctl_bootstrap_executed_in_ops_scripts` is a broken PROBE, not a
   caught defect.** Its scanner (`test_phase_75_sre_ops.py:360-368`) skips `#` lines but
   is **heredoc-blind**; the hit at `reissue_cc_oauth_token.sh:117` sits inside a
   `cat <<EOF` opened at `:110` and is printed, never executed — the script deliberately
   does not automate bootout/bootstrap (away-ops rail 9). Editing the script to satisfy
   the test would **delete the operator's restart instructions**.
3. **A quarantine that exists only in a docstring is not a quarantine.**
   `test_phase_23_2_6...backend_log_has_skipping_buy_evidence` documents itself as
   "quarantined per the requires_live convention" — but carries **no
   `@pytest.mark.requires_live`**, and its only escape is a `size < 100` skip that never
   fires because `backend.log` is 14MB.

**The 86.3/36.28 live-kill-switch class is 0 of 17.** All six files 86.5's audit_basis
named as live-pause-coupled are green now, and a full run left
`handoff/kill_switch_audit.jsonl` untouched (66 lines, mtime 16h before the run). That
is most of the 26→17 drop — record it as already-resolved, never re-file it.

**Why:** the goal asked for grouping by ROOT CAUSE; module is not root cause, and three of
the eight groups invert the obvious remedy.
**How to apply:** before "fixing" any red test, ask which of usable / broken-but-repairable /
obsolete it is, and whether the probe or the subject is wrong. `xfail(strict=True,
raises=...)` is the quarantine primitive — it still RUNS and turns an unexpected pass into
a suite failure; `skip` does not execute at all. See
[[a-red-check-may-indict-the-probe]], [[measure-dont-assert-claims]],
[[zsh-no-word-splitting]] (the `$IDS` splat produced "no tests ran" mid-measurement).

Brief: `handoff/current/research_brief_86.5.md` (37 sources read in full, 18 rounds, dry).
