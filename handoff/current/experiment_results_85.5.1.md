# Experiment results — phase-85.5.1 (P1 BOOK SAFETY)

**Step id:** 85.5.1 · **Cycle:** 185 — 2026-08-09
**Contract:** `handoff/current/contract_85.5.1.md` · **Gate:** `research_brief_85.5.1.md` (`gate_passed: true`, 7 sources read in full, ~40 URLs)

## Headline: the guard was never broken. The FIXTURE was.

The RED test monkeypatched `st.snapshot` to return only
`{"sod_nav", "peak_nav"}`. The real `_snapshot_locked()` emits **three** more
contract keys — `sod_date` (`:465`), `baseline_provenance` (`:471`) and, since
phase-85.6, `sod_provisional` (`:472`). So the test handed `evaluate_breach` a
state with `sod_date=None`, which the phase-36.9 liveness guard **correctly**
reads as an unevaluable daily leg and disarms — and then asserted that leg fires.

`backend/tests/test_book_safety_69.py`: **1 failed / 13 passed → 15 passed, 0 failed.**
Not one assertion weakened.

---

## 1. What was built

| File | Change |
|---|---|
| `scripts/diagnostics/measure_sod_date_reachability.py` | **NEW.** Walks every production path to a None/stale anchor through the real replay and evaluates a real breach at each. Asserts its own isolation first. This is criterion 1's evidence and it is re-runnable. |
| `backend/tests/test_book_safety_69.py` | `test_valid_nav_still_breaches` now drives a **real `KillSwitchState`** on a redirected audit path, anchored to a **computed** today. **NEW** `test_stale_anchor_disarms_the_daily_leg_but_the_trailing_leg_still_fires` — the missing half of the contract. |
| `scripts/qa/mutation_matrix_85_5_1.py` | **NEW.** 5 mutations, all killed; asserts the live journal is byte-identical afterwards. |

**No production file was changed.** `git diff` over `backend/services/` for this
step is empty. That is the finding, not an omission.

## 2. Criterion 1 — answered with a measurement, not reasoning

Full verbatim output in `live_check_85.5.1.md` §1. Summary of the six paths:

| | Path | Reachable? | What the switch does |
|---|---|---|---|
| **C** | **UTC rollover** — yesterday's anchor before today's first roll | **YES, every day, no fault required** | daily leg disarmed, **trailing leg fires**, `any_breached=True` |
| A/B | legacy row: no `date` key, unparseable `ts` (`:298-316`) | mechanism live; no such row exists today | same as C |
| F | startup before any anchor | yes | named `missing`, **not** `stale` (`:922-923` deliberately refuses to double-name an absence); trailing fires |
| **E** | **oversized JSON int aborts the entire replay** | mechanism live | **BOTH legs stranded — `any_breached=False` on a 20% drawdown** |
| — | torn pair (`sod_nav` set, `sod_date` not) | **unreachable** — same branch, no raising call between `:299` and `:313` | — |
| HEALTHY | same-day anchor, 20% drawdown | control | daily True, trailing True, `any_breached` True |

**Severity is bounded to drawdowns in `[daily_limit, trailing_limit)` = [4%, 10%)**,
because the trailing leg is date-independent. Above 10% the switch fires
regardless. **Case E is the exception and is a separate, worse defect** (§6).

### I got case E wrong first, and my own output caught me

My first construction put the malformed row **last**, so the good rows had
already applied: the run printed `armed=True` with both legs firing, directly
contradicting the summary line I had written above it. Ordered **first**, it
reproduces: `sod_nav=None sod_date=None peak_nav=None`, `any_breached=False`.
The verdict block now derives its wording from the measured result instead of
restating the expectation.

## 3. Criterion 2 — the fix follows the answer, and the answer says do not touch the guard

Production **can** reach it, so criterion 2's second branch applies. Read
literally it asks two things:

1. **"the GUARD is fixed so an evaluable breach still fires"** — already true, and
   now measured. `evaluate_breach` computes `daily_leg_unevaluable` (`:810`) and
   `trailing_baseline_missing` (`:798`) independently, gates each leg separately
   (`:859`, `:865`) and ORs at `:876`. `:774-785` states this as deliberate. The
   research gate identified it as textbook **IEC 61511 selective bypass** — bypass
   the single unevaluable channel, never the whole trip logic, and log the bypass.
2. **"a test drives the production path that reaches the disarm rather than a mock
   of it"** — this was the actual work, and it is done.

**I did not change the guard, and that is a deliberate decision, not an
oversight.** Re-arming a leg that cannot be evaluated is precisely the phase-36.9
F1 defect (`armed: true` must mean the leg can fire *now*); it would be a safety
regression dressed as a green test. The contract stated this in advance so the
decision could not be reverse-engineered after the fact.

## 4. Criterion 3 — nothing weakened

All four assertions stand at full strength and still demand a 20% drawdown
against **both** baselines produce `daily_loss_breached True`,
`trailing_dd_breached True`, `any_breached True`, falsy `nav_invalid`:

```python
r = ks.evaluate_breach(80.0, 4.0, 10.0)  # 20% down vs sod AND peak
assert r["daily_loss_breached"] is True and r["trailing_dd_breached"] is True
assert r["any_breached"] is True and not r.get("nav_invalid")
```

The change is entirely **input-side**. Two details that matter:

- **`today` is computed, never hardcoded.** A literal date would rot at midnight
  and start disarming the very leg the test exists to prove fires. Mutation
  **M1b** pins this.
- A **real state cannot omit a contract key** the way a hand-written dict can —
  and this one already omitted three. The fixture is now drift-proof by
  construction rather than by vigilance.

## 5. Criterion 4 — the mutation transcript, 5/5

Full transcript in `live_check_85.5.1.md` §3.

**M4 was LIVE on its first run, and it is the most useful result of this step.**
Hardcoding `_sod_date_is_stale` to `return False` — the phase-36.9 F1 regression,
i.e. a liveness guard that never disarms — **survived the entire file**. A
book-safety suite that proves the switch FIRES but never that it correctly
DISARMS cannot tell a working kill switch from a disabled one.

Closed by adding
`test_stale_anchor_disarms_the_daily_leg_but_the_trailing_leg_still_fires`, which
pins all four halves of the bypass contract: daily suppressed, `armed` False,
**trailing still fires**, `any_breached` still True.

**M1's target became ambiguous** once that test existed, and the harness
**refused to mutate** rather than silently picking a site — the uniqueness guard
doing its job. Re-anchored on a line unique to the RED test.

## 6. Criterion 5 — no other test changed status, measured as a SET

Measured in a detached **git worktree** with the isolation **proven before
measuring** (all four polluting constants relocated), both arms in the same
environment with one variable:

```
BEFORE (no fix): 20 FAILED, 4 ERROR
AFTER  (fix)   : 19 FAILED, 4 ERROR
FIXED   : backend/tests/test_book_safety_69.py::test_valid_nav_still_breaches
NEWLY FAILING: (none)
ERROR set identical: True
CRITERION 5 SET DIFF: PASS
```

The worktree counts (19/20) differ from the live tree's 26 because the worktree
lacks gitignored files, most importantly the 32.5MB `backend.log`. That is why
the comparison is worktree-vs-worktree. The phase-85.5 cycle already lost time to
reading that environment difference as a regression; this run does not repeat it.

**The live kill-switch journal was 54 lines before and after every run in this
step**, and the mutation matrix now asserts that as a post-condition.

## 7. Defects found, queued rather than absorbed

Per the standing rule each gets its own research-gated masterplan step:

1. **P1 — a latent live-state landmine armed by an unrelated operator decision.**
   `test_peak_reset_dark_by_default` (`test_book_safety_69.py:86-92`) calls
   `st.reset_peak(12345.0)` on the **real singleton** with no redirect. It is safe
   **only** because `kill_switch_peak_reset_enabled` is measured `False`, so
   `reset_peak` returns at `:694` before locking. **The day the KS-PEAK-RESET
   token is approved, that test writes a `peak_reset` row and drops the live peak
   from ~24666 to 12345, replayed on every boot.** I did not fix it here: doing so
   inside a step whose criterion 5 demands "no other test changes status" would be
   exactly the scope creep that criterion exists to prevent.
2. **P1 — `_coerce_nav` (`:131`) catches only `(TypeError, ValueError)`.** One
   oversized JSON int raises `OverflowError`, is swallowed at `:394`, aborts the
   whole audit replay, and strands **both** legs — measured: `any_breached=False`
   on a 20% drawdown. This is the only measured path to a *total* disarm.
3. **P3 — no time limit on how long the daily leg may sit bypassed.** IEC 61511
   Cl. 16.2.3 requires associated operation limits including duration.

## 8. Lint gate (qa.md §1a)

```
$ uvx ruff check --select F821,F401,F811 "${FILES[@]}"   # git-derived scope, non-empty guard
All checks passed!
ruff_exit=0
```
