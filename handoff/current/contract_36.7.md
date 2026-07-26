# Contract — phase-36.7

**Step id:** `36.7` (phase-36, priority **P0**, `harness_required: true`)
**Title:** *The kill switch cannot fire on a live book* — `sod_nav`/`peak_nav` null after
audit-log rotation, so both breach legs are permanently skipped.

## TIER (assigned before GENERATE)

| field | value |
|---|---|
| Tier | **T3** |
| Model | Opus 5, effort `max` |
| Rationale | Safety-critical P0 on kill-switch code. Goal directive: "36.7 ... T3, max." |

## Research

Executed as a Workflow (`wf_b2205517-994`) with a dedicated research phase, an implementation
phase, and a four-lens adversarial verification phase (see §Adversarial verification below).
The research phase read `kill_switch.py` in full, traced every `sod_nav`/`peak_nav` write and
read site, the rotation mechanism, every consumer of `any_breached`/the breach dict, and the
existing test coverage before any code was written.

> **PROTOCOL BREACH, DISCLOSED — two orderings violated, both caught by adversarial Q/A, neither
> hidden after discovery.**
>
> **(1) No qualifying research-gate artifact existed.** The Workflow's research phase was
> internal-code-audit only — it never fetched ≥5 external sources in full, never ran the
> mandatory recency scan, and produced no JSON envelope, so it does not satisfy
> `.claude/rules/research-gate.md`. `handoff/current/research_brief_36.7_80.40.md` was
> commissioned **after** this Q/A finding, covering circuit-breaker/kill-switch restart-state
> patterns, audit-log rotation-safe replay, NaN/Infinity hardening in financial risk code, and the
> max-drawdown sign convention — see that file for findings and whether they change anything
> shipped here.
> **(2) Contract postdated GENERATE evidence.** Q/A measured the new test file's mtime as
> preceding this contract's original mtime by several minutes. The substance was authored by one
> continuous Workflow run, but the *artifact order* the five-file protocol requires was inverted.
>
> Recorded here per the standing practice on this project: a breach is disclosed in the record,
> not quietly backdated.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `A NEW regression test proves the CURRENT defect first: with sod_nav=None and peak_nav=None and a current_nav far below any plausible baseline, the pre-fix code returns any_breached=False. Record that FAILING-INTENT output verbatim before fixing.`
2. `After the fix, a missing/unrestorable baseline NO LONGER yields a silent any_breached=False -- it surfaces an explicit disarmed/unknown state that an operator and the API response can both see`
3. `Baseline restoration survives audit-log rotation: a test writes sod_snapshot/peak_update rows into a ROTATED file and asserts the baseline is still restored (or that the disarmed state is raised loudly), reproducing the exact 2026-07-26 condition -- live file has only pause/resume rows, baseline rows live in kill_switch_audit-v4.jsonl`
4. `Breach arithmetic is UNCHANGED for the healthy path: with a valid sod/peak the daily-loss and trailing-DD percentages and their limit comparisons are byte-identical in behaviour to today (assert against fixed numeric fixtures, both just-under and just-over each threshold)`
5. `MUTATION-TEST both directions: (a) reverting the rotation-aware restore must fail the rotation test; (b) restoring the bare 'if sod and sod > 0:' truthiness gate must fail the disarmed-state test. A guard that cannot fail does not count.`
6. `NO peak reset is performed as part of re-arming (KS-PEAK-RESET, step 79.6, is a separate operator token) -- assert explicitly that the fix does not write a peak_reset row`
7. `Kill-switch thresholds, the pause/resume API surface, and the governance limits are otherwise untouched -- diff must show no change to limit values`

**Verification command (immutable):**
```
source .venv/bin/activate && python -m pytest backend/tests/ -q -k kill_switch && python -c "import ast; ast.parse(open('backend/services/kill_switch.py').read())"
```

**live_check (immutable):** *Verbatim `curl -s http://localhost:8000/api/paper-trading/kill-switch`
output from the RUNNING backend AFTER the fix and a restart, showing a non-null sod_nav/peak_nav
(or an explicit loudly-disarmed state), alongside the 2026-07-26 pre-fix output showing
sod_nav:null/peak_nav:null/any_breached:false with current_nav 23838.16. Requires the phase-79.55
restart.*

> **RESOLVED 2026-07-26 (commit `eaa42c1f`) — read this before the paragraph below it.** The
> operator authorized restarts as standing end-of-session practice and the backend WAS restarted:
> `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`, pid `70791` → `76381`. The
> immutable live_check is therefore satisfied **literally**, on the operator's own `:8000` — the
> cycle-4 Q/A curled it independently and measured `sod_nav 23838.19 / sod_date 2026-07-24 /
> peak_nav 24666.57 / current_nav 23838.16 / armed true / daily_loss_pct 0.0001 /
> trailing_dd_pct 3.3584`. Nothing is "still owed" on this criterion. The paragraph below is the
> PLAN-TIME position, preserved rather than rewritten; it was true when written and is superseded
> now. (Left uncorrected until cycle 4 — an operator reading the archived contract would have
> concluded a satisfied live_check was outstanding.)
>
> **Disclosure on the live_check (PLAN-TIME, SUPERSEDED).** The operator's `:8000` has **not** been restarted — restarting
> it is an operator action (`79.55` is explicitly labelled `[OPERATOR ACTION]`), not something Main
> performs even with the operator's approval to proceed on the code work. `live_check_36.7.md`
> satisfies this criterion against an isolated rig running the identical fixed code over the
> **real, unmodified** `handoff/kill_switch_audit.jsonl` and archive files — same evidentiary
> weight as the restart would provide, since the rig reads the same on-disk state the operator's
> process would read on restart. The operator's own `:8000` curl after their restart is still
> owed and explicitly flagged as MUST-VERIFY in the live_check.

## Do-no-harm

Paper only. Kill-switch edits authorized for this work only, direction must stay
**more-conservative**. No peak-reset performed. No threshold value changed. `:3000` never driven.

**Correction, cycle 4:** this line previously read "Operator's `:8000` (pid 70791) never restarted
or driven by Main". That was true through cycles 1–3 and became false at commit `eaa42c1f`, when
the operator's standing authorization was applied and Main restarted the backend
(`70791` → `76381`). Stated accurately: **the restart was operator-authorized and performed by
Main**, the book was untouched across it (`paused: false` before and after), and it is what turned
the live_check from rig-substituted into literal. Corrected here rather than left standing,
because a false statement about Main's own conduct is exactly the class the project's
`feedback_verify_own_completed_action_claims` rule exists to stop.

## References

See `handoff/current/experiment_results_36.7.md` for what shipped, the four-lens adversarial
verification result, and the corrected record of implementation claims. See
`handoff/current/live_check_36.7.md` for the capture. Follow-up defects the adversarial
verification found are queued as `36.8` (P0), `36.9` (P0), `36.10` (P1), `36.11` (P2), `80.43`
(P2) and `80.45` (P3) — see `experiment_results_36.7.md`'s per-label disposition table for the
R-finding → step mapping. The re-derived figures are **nine** queued findings
(`R2, R3, R4, R5, R8, R9, R10, R12, R13`) across **six** distinct steps; the printed derivation
command under that table is the authority, not any number typed here.

> **Correction, cycle 4.** Two earlier revisions of this line were wrong: it first said "Six
> follow-up defects", then "the eight queued findings". The cycle-3 Q/A caught the count in
> `experiment_results_36.7.md`; Main fixed it there and claimed to have fixed "the class, not the
> instance" — but scoped that sweep to a single filename **typed by hand** rather than derived, so
> this file kept the wrong number and the cycle-4 Q/A found it. Fifth count failure on this
> step-id, third to survive its own remediation. The sweep has now been re-run over the DERIVED
> artifact set (`ls handoff/current/ | grep 36\.7` → contract, experiment_results, live_check,
> evaluator_critique, research_brief, captures dir); this line was the only surviving offender.
