# Contract — phase-85.5.1 (P1 BOOK SAFETY)

**Step id:** 85.5.1 · **Cycle:** 185 — 2026-08-09
**Order:** RESEARCH done → **this contract** → GENERATE (not started) → Q/A → log → flip.
Closed earlier this session: **85.4** (`8aa3f52e`), **85.6** (`cb34a7c0`).

---

## 1. Research gate

`handoff/current/research_brief_85.5.1.md` — `gate_passed: true`, **7** sources
read in full, ~40 URLs, recency scan performed.

### The gate corrected THREE of the step's/my premises. Build against the gate.

| Premise | Measured reality |
|---|---|
| "the kill switch DISARMS instead of firing on a real 20% breach" | **Overstated.** On a stale/None anchor the switch **does** fire — `any_breached=True` via the trailing leg. Only the **daily leg** disarms. The RED test fails on its first conjunct alone. |
| "the GUARD needs fixing" | **No.** The minimal change to `kill_switch.py` is **NONE**. `evaluate_breach` already computes `daily_leg_unevaluable` (`:810`) and `trailing_baseline_missing` (`:798`) independently, gates each leg separately (`:859`, `:865`) and ORs at `:876`; `:774-785` states this as deliberate. It is textbook **IEC 61511 selective bypass** — bypass a single channel, not the whole trip logic, with the bypass visible and logged. Anything that re-arms the daily leg on a stale anchor re-introduces phase-36.9 F1 verbatim. |
| "85.6 may have widened reachability" | **It NARROWED it.** With the Step-0 provisional roll, `armed=True` and a 20% drop fires the daily leg. Pre-85.6 the leg was disarmed from 00:00 UTC until Step 5.5 of a *surviving* cycle; now only until Step 0. |

**Therefore `85.5.1` MUST NOT touch `paper_trader.py:1276-1301` or `:1413-1449`** —
that is 85.6's machinery and it is working.

## 2. Criterion 1, answered — the reachability table

Six paths, measured (script + verbatim output in the brief §5):

| | Path | Reachable? |
|---|---|---|
| A/B | legacy `sod_snapshot` with no `date` and an absent/unparseable `ts` (`kill_switch.py:298-316` — **not** `:285-295` as I had written) | mechanism live; **no such row exists today** |
| **C** | **UTC rollover — yesterday's anchor, before the day's first roll** | **YES. Every UTC day. No fault required.** This is the severity driver, not A/B. |
| **E** | **NEW, not on anyone's list.** `_coerce_nav` (`:131`) catches only `(TypeError, ValueError)`; a 401-digit JSON int raises `OverflowError`, swallowed at `:394` → **the entire replay aborts** and today's rows never apply. Strands the **peak** too. Observed: `kill_switch: audit load failed: int too large to convert to float` | mechanism live |
| F | startup before the first anchor | **Mis-bucketed by me.** That is `daily_baseline_**missing**`, not stale: `_sod_date_is_stale` returns False when `sod_nav` is None (`:922-923`), deliberately refusing to double-name an absence |

A **torn pair is unreachable**: `_sod_nav` (`:299`) and `_sod_date` (`:313`) are set
in the same branch with no raising call between them.

**Severity, bounded:** exposure is drawdowns in `[daily_limit, trailing_limit)` =
**[4%, 10%)** — the trailing leg covers everything above. **Exception:** in case E
the same fault also strands the peak, so BOTH legs die and `any_breached=False`
for any drawdown by design (`:770-773`), leaving only the BUY gate at `:1372`.

## 3. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. the question 'can sod_date be None or stale in PRODUCTION, not merely in a test mock' is answered with a measurement (e.g. the observed value of the daily anchor at backend startup before the first anchor write, and across a date rollover), and that measurement is recorded verbatim in the step evidence rather than reasoned about
>
> 2. the fix follows the answer: if production cannot reach a None/stale anchor, the mock is corrected to the real _snapshot_locked() contract (kill_switch.py:444 includes sod_date); if production CAN reach it, the GUARD is fixed so an evaluable breach still fires, and a test drives the production path that reaches the disarm rather than a mock of it
>
> 3. no book-safety assertion is weakened to obtain green: test_valid_nav_still_breaches must still assert that a 20% drawdown against both sod and peak yields daily_loss_breached True, trailing_dd_breached True, any_breached True and a falsy nav_invalid
>
> 4. a mutation proves the guard bites -- reverting the fix makes the test fail again, demonstrated in the evidence
>
> 5. backend/tests/test_book_safety_69.py is fully green (0 failed) and no other test changes status vs the pre-change baseline, which is measured and recorded; fresh Q/A PASS

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'`

This one **can** fail. It is a real gate.

## 4. Criterion-by-criterion plan

**C1 — record the measurement, verbatim.** Ship the gate's reachability script as
`scripts/diagnostics/measure_sod_date_reachability.py` so the answer is
re-runnable rather than a paragraph, and paste its output into the evidence.

**C2 — the answer is "production CAN reach it", and the criterion's second branch
is satisfied WITHOUT touching the guard.** Read literally, it requires two things:
(a) *"an evaluable breach still fires"* — already true and now measured: the
trailing leg fires and `any_breached` is True; and (b) *"a test drives the
production path that reaches the disarm rather than a mock of it"* — this is the
actual work. Replace the `st.snapshot` monkeypatch with a **real
`KillSwitchState`** on a redirected `_AUDIT_PATH`, using the idiom already at
`test_phase_36_9_kill_switch_armed_liveness.py:63-88`.

**I will not "fix" a guard the gate measured as correct.** Re-arming a leg that
cannot be evaluated is precisely phase-36.9 F1 and would be a safety regression
dressed as a green test. If Q/A reads criterion 2 as compelling a guard change,
that disagreement belongs in the critique — I will not pre-emptively weaken the
switch to avoid it.

**C3 — nothing weakens.** Measured case F: a same-day anchor yields daily 20%
True, trailing 20% True, `any_breached` True, no `nav_invalid`. All four
assertions pass **at full strength**; the fix is entirely input-side.
`TODAY_UTC` must be **computed, never hardcoded**, or the test rots at midnight.

**C4 — mutation.** Revert the fixture to the 2-key mock → the test must go RED
again. Plus a mutation on the guard itself to prove it still bites.

**C5 — baseline in a `git worktree`.** All four polluting constants derive from
`Path(__file__).parents[N]` (`kill_switch.py:48`, `cycle_health.py:36-37`,
`cycle_lock.py:53`), so ONE worktree relocates them all — no per-test
monkeypatching. **Assert the precondition first** (`k._AUDIT_PATH` must be under
the worktree) or the isolation is assumed rather than proven. Diff failure
**SETS, not counts**. Do NOT copy-then-`git checkout --`; that races the live
backend and the PreToolUse hook is right to block it.

## 5. Hazards and hard prohibitions

- **Do not weaken any assertion or threshold to obtain green.** Criterion 3 is
  explicit and the whole point of the step.
- **Do not re-arm the daily leg on a stale anchor.**
- **Do not touch 85.6's machinery** (`paper_trader.py:1276-1301`, `:1413-1449`).
- **The book is currently RESUMED and armed.** Do not disturb it, do not trigger
  a cycle, do not restart a service.
- Running `test_book_safety_69.py` **alone** is safe — the gate confirmed every
  test in it is read-only or redirects to `tmp_path` first.

## 6. Defects found by the gate that are OUT of scope — queue, do not absorb

Per the standing rule each gets its own research-gated step, not a prose mention:

1. **P1 — `test_peak_reset_dark_by_default` (`test_book_safety_69.py:86-92`) calls
   `st.reset_peak(12345.0)` on the REAL singleton with no redirect.** Safe today
   only because `kill_switch_peak_reset_enabled` is measured `False`, so
   `reset_peak` returns at `:694` before locking. **The day the KS-PEAK-RESET
   token is approved, that test writes a `peak_reset` row and drops the live peak
   from ~24666 to 12345, replayed on every boot.** A latent live-state landmine
   armed by an unrelated operator decision.
2. **P2 — `_coerce_nav` (`:131`) should catch `OverflowError`**; today one
   oversized int aborts the whole audit replay (case E) and strands both legs.
3. **P3 — no time limit on how long the daily leg may sit disarmed.** IEC 61511
   Cl. 16.2.3 requires associated operation limits including duration for a
   bypass.

## 7. References

- `handoff/current/research_brief_85.5.1.md` (gate output, incl. the §5 script)
- `backend/services/kill_switch.py:131, :298-316, :394, :444, :694, :770-785, :798, :810, :859, :865, :876, :922-923`
- `backend/tests/test_book_safety_69.py:79-80, :86-92`
- `backend/tests/test_phase_36_9_kill_switch_armed_liveness.py:63-88` (the isolation idiom to reuse)
- `handoff/current/experiment_results_85.6.md` §12-§13 (why 85.6 narrowed reachability)
