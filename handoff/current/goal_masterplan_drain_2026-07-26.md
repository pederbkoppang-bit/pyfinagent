# Goal — Masterplan drain, cycle 2 (set 2026-07-26)

**Supersedes** `goal_masterplan_drain_2026-07-25_DRAFT.md`. That goal ran one long session; this
one carries its measured state forward so nothing is re-derived from a stale snapshot.

**Operator authorization, verbatim (2026-07-26):** *"you have my approval to do it also write the
new goal prompt which will drain the masterplan in a new session"* — granted in reply to a
four-item ask list. **The four items are therefore AUTHORIZED and are no longer blockers:**
`36.7`+`80.40` (kill switch, both ends), `79.55`+backend restart,
`tools_nonfinite_fail_safe_enabled`, and the `80.5` disposition. See "Authorized this cycle".

---

## FIRST: read these, in this order

1. `handoff/current/goal_masterplan_drain_2026-07-26.md` (this file — authoritative)
2. `CLAUDE.md` — harness protocol, MAS layers, Fable policy, effort policy
3. `.claude/rules/` — at minimum `research-gate.md`, plus `frontend.md` +
   `frontend-layout.md` for UI work, `backend-api.md` + `backend-services.md` for backend
4. `handoff/current/done_definition_evidence_2026-07-26.md` — per-item evidence ledger
5. `handoff/current/p0_triage_2026-07-26.md` — 16-step P0 re-audit (zero drops survived refutation)
6. `handoff/current/tier_ledger_2026-07-26.md` — what actually ran at which tier last cycle

## Measured scope (re-derived 2026-07-26 — RE-DERIVE AGAIN, another session writes this file)

Definition: `status == "pending"`. **224 open / 21 phases** —
`{P0: 23, P1: 46, P2: 96, P3: 40, P4: 8, unset: 11}`. **53 are `harness_required: false`**
(operator actions, almost all phase-79) → **171 executor steps**.

Open by phase: 5(11) 27(2) 35(1) 36(1) 43(1) 44(6) 58(1) 61(4) 62(4) 63(3) 65(2) 68(7) 72(9)
73(13) 74(4) 75(33) 76(14) 77(3) 78(17) 79(54) 80(34).

> A naive "not done" count gives **~244 / 32 phases** because it also counts `deferred`(15),
> `blocked`(1), `merged`(2), `superseded`(4). Both are correct; **state which you mean.**
> Reconciliation method: `handoff/current/count_reconciliation_2026-07-26.md`.

## THE HEADLINE: the kill switch is broken at BOTH ends

Do this first. It is the only finding where a safety mechanism the whole go-live path assumes is
provably non-functional on a live book.

- **`36.7` (P0) — the mechanism cannot fire.** Measured read-only:
  `/api/paper-trading/kill-switch` returns `sod_nav: null, peak_nav: null` against
  `current_nav: 23838.16`. `kill_switch.py:311/:317` gate both breach computations behind
  `if sod and sod > 0:` — `None` is falsy, so `any_breached` is `False` **for any NAV**. Cause:
  log rotation left the live audit file with 8 pause/resume lines and zero `sod_snapshot`/
  `peak_update` rows; the baselines sit in `handoff/audit/kill_switch_audit-v4.jsonl` where
  `_load_from_audit` never looks. **It will recur on every rotation.**
- **`80.40` (P0) — the indicator has never had data.** `/api/paper-trading/performance` does not
  return `max_drawdown_pct` at all (`sharpe_ratio` is present, so this is not a null-perf case).
  `max_drawdown` appears nowhere in `paper_trading.py` or `perf_metrics.py`; only `backtest.py`
  exposes it. Pre-`80.36` the absence rendered a green **`SAFE`** via `?? 0` because `0 > -10` —
  **on a healthy backend, every day.**

Fix both, then reconcile the thresholds: the cockpit row says **-15%** while
`kill_switch.py` uses a **10%** trailing-DD limit. Decide which is authoritative; do not
silently pick.

## Authorized this cycle (was blocked, now granted)

| # | Action | Notes |
|---|---|---|
| A1 | **Edit `backend/services/kill_switch.py`** for `36.7` | Direction must stay *more conservative*. A missing baseline must fail LOUD, never silently disable the breach test. **Do not** peak-reset while re-arming — that is the separate owed `KS-PEAK-RESET` token (`79.6`). |
| A2 | **Answer `79.55` and restart the backend** | pid `70791` started 2026-07-25 11:39:05 and predates 78.2, 80.1, 80.2, 80.3, 80.31, 80.4. **No paper cycle has run under the current binary.** Restarting makes 11 P0s measurable. |
| A3 | **Flip `tools_nonfinite_fail_safe_enabled`** | Proven working live (classifier forced to `NEUTRAL` was overridden → `ERROR`). Currently dark. Prefer the Settings API over a raw `.env` edit if the field is exposed. |
| A4 | **`80.5` disposition** | Recommended: close the four *verified* donut fixes under a fresh step id; leave `80.5` carrying the failed WCAG attempt, already re-queued as `80.35` with all three dead ends recorded. |

**Everything else in phase-79 (53 steps) remains operator-only.** Batch as an ask list; a recorded
DECLINE closes one as validly as doing it.

## State carried forward

**Closed and pushed last cycle (8):** `80.2`, `80.1`, `80.27`, `80.31`, `80.3`, `80.4` (P0s) plus
`80.31`(P2). Repo clean, 0 ahead of origin at handoff.

**Committed WIP — finish these, do not restart them:**

| step | state |
|---|---|
| `80.5` | 4 donut defects fixed + verified live (card height 215→215, one tooltip, rounding agrees, hole no longer hover-active). **Three consecutive CONDITIONALs** → next Q/A pass is in auto-FAIL territory. WCAG machinery removed and re-queued as `80.35`. See A4. |
| `80.11` | Single-flight + 401 epoch guard + probe timeout + cookie-first short-circuit. Session probes **11 → 3** measured on the rig; criterion 1 wants ≤2. **The rig cannot demonstrate the cookie path** (`LIGHTHOUSE_SKIP_AUTH` = no NextAuth cookie) — inject a cookie or capture an authenticated session. |
| `80.36` | Risk Monitor no longer fabricates. 6 tests, 5 mutations killed, both live captures taken. Needs a fresh Q/A → log → flip. |

**Queued from measurement last cycle (9 new steps):** `36.7`(P0), `80.40`(P0), `80.39`(P1),
`80.36`(P1), `80.33`(P2), `80.34`(P2), `80.35`(P2), `80.37`(P2), `80.38`(P2).

## Wave order (each step = FULL harness cycle)

1. **`36.7` + `80.40`** — the kill switch, both ends. T3, `max`.
2. **`79.55` + restart** (A2), then immediately re-measure done-definition items 1 and 3 — both
   are already *proven on a rig* and only need the live binary.
3. **Finish the WIP:** `80.36` (Q/A → flip), `80.5` (A4), `80.11` (authenticated capture).
4. **Remaining P0s, oldest first:** 27(2), 61(2), 62(3), 63(2), 65(2), 68(3), 72(2). **Re-read
   each** — `p0_triage_2026-07-26.md` found zero obsolete but five needing *rescope*.
5. **phase-80 P1 money surfaces:** `80.7` → `80.6` in that order, then `80.8`–`80.10`,
   `80.28`(the 12.4s event-loop freeze), `80.29`, `80.30`, `80.39`. T3 for 80.7/80.8/80.9.
6. **P1 tail** across 43, 44, 61, 62, 63, 68, 72, 73, 75, 76, 78. T2.
7. **P2/P3/P4 tail.** Triage hard: **an obsolete P3 is closed as `dropped` with a reason, not
   executed.** phase-5 (11 open, `deferred`) and phase-13 are oldest — decide drop-vs-do first.

## Tiering (unchanged; the ledger says it was NOT applied last cycle)

On the Max rail model choice costs **$0 metered** — tier for quota/latency, never trade
correctness for a zero saving. **Prefer an effort step-down to a model step-down.**

- **T4 Fable 5** — hardest correctness-critical only. Per-invocation
  (`agent({model:'fable', effort:'xhigh'})`), **never** a roster repin (exhaustion is a HARD FAIL,
  `harness_log.md:27346`). Every T4 call needs an Opus fallback; an errored/empty return is
  **NO VERDICT, not PASS**. Pins silently fall back to Opus in headless/cron — never *require* it.
- **T3 Opus 5 `xhigh`/`max`** — other P0s, audit research, adversarial verify.
- **T2 Opus 5 `high`** — default P1/P2.
- **T1 Haiku 4.5** — mechanical single-file edits only. Rejects `effort`; 200K ctx so it
  **cannot** read `masterplan.json` — slice it with a script. In doubt → **T2**.

**WRITE THE TIER INTO `contract.md` AS A NAMED FIELD BEFORE GENERATE.** Last cycle's ledger
found Fable was authorized and **never used** — zero T4 invocations — because the tier was never
recorded and silently defaulted. A tier that isn't written down is a tier that didn't happen.

## Non-negotiable

- **Full harness per step:** researcher (≥5 sources read in full) → `contract.md` (criteria
  **byte-for-byte**, verified programmatically) → GENERATE → **ONE fresh Q/A** → `harness_log.md`
  append → masterplan flip. No self-eval. No verdict-shopping on unchanged evidence.
- **DO-NO-HARM:** paper only; `historical_macro` FROZEN; no optimizer runs; stops, sector caps,
  DSR≥0.95 and PBO≤0.5 byte-untouched. Kill-switch edits are authorized **only** for `36.7`/`80.40`
  and only in the more-conservative direction.
- **UI claims need a Playwright capture** on the isolated skip-auth `:3100` rig with its own
  `PLAYWRIGHT_DIST_DIR` — never the operator's `:3000`. Restore `tsconfig.json` +
  `next-env.d.ts` after; both get rewritten by `next dev`.
- **`git add -An` before every flip.** The hook stages the whole tree under your step's name.
- **Mutation-test every guard. Measure, don't assert.**

## Lessons this cycle paid for — read these, they each cost real time

1. **Verify the experiment applied before interpreting its result.** A StrictMode experiment
   returned a byte-identical capture that looked like a clean refutation. The compiled bundle still
   read `StrictModeIfEnabled = true ? …` — a stale `PLAYWRIGHT_DIST_DIR` chunk meant the config
   change never reached the client. An identical outcome is equally consistent with "hypothesis
   wrong" and "change never took effect". `rm -rf` the audit dist dir, then grep the bundle, *then*
   trust the capture.
2. **Fix the CLASS, not the instance.** `80.5` cycle 1 fixed the exact line named; the same diff
   shipped a second instance of the same defect, and cycle 3 found a third introduced by the fix
   for the second. Root-cause it (there it was one state conflating hover and focus) or you will
   pay per-line, forever.
3. **A guard that resets what it guards cannot fail.** Two `80.11` tests called a reset helper that
   nulled the very ref the mutation removed — both passed against a build with the fix deleted.
   Mutation-test found both.
4. **Mutate the intended line.** A `replace(old, new, 1)` matching the first of three identical
   strings produced two **false kills** in `80.5`'s matrix. Assert the anchor count, or index by
   line number.
5. **Discriminate on PRESENCE, not VALUE**, and resolve **per row, not per card** (`80.36`). `0` is
   a legitimate healthy reading; a card-level bail hides a live breach in a sibling row.
6. **Re-run the WHOLE matrix when tests are added**, or its rows silently become claims.
7. **Verify your own past-tense claims.** Four false "I did it" statements last cycle
   (`"verbatim"`, `"queued"` ×2, a mutation total), two of them *introduced by the fix for the
   previous one*. See `feedback_verify_own_completed_action_claims`.
8. **Editing `next.config.js` — even restoring it — briefly takes the operator's `:3000` to `000`**
   because both dev servers watch it. It self-recovers; re-poll, don't panic-restart.

## Done-definition (HARD STOP)

Not "all 224 closed" — that is a multi-week program. This goal closes when **every open P0 is PASS
or explicitly deferred with a recorded reason**, the phase-80 P1 money surfaces are closed, and the
phase-79 ask list is batched. Closing evidence — carried forward with last cycle's measured status:

| # | Evidence | Status at handoff |
|---|---|---|
| 1 | `/api/signals/AAPL` → 200, 12 keys, **+ loop heartbeat** | payload **PASS**; heartbeat **FAILS** — 12.4s event-loop block, owned by `80.28` |
| 2 | NaN payload → NOT-SUFFICIENT; classifiers `ERROR`/`NO_DATA` | **PASS** live (flag on); dark in prod → A3 |
| 3 | raising route → 500 + CORS + `nosniff` + PerfTracker row | **PASS** on rig; needs the restart to be live |
| 4 | `/agent-map` edges, **0** React Flow warnings @1440×900 | **PASS** |
| 5 | donut hover → **0** layout shift | **PASS** measured; `80.5` Q/A outstanding |
| 6 | **≤2** `/api/auth/session` per 20s view | 11 → **3**; needs an authenticated capture |
| 7 | backend stopped → no page fabricates a fact | Risk Monitor **fixed**; `80.36` Q/A outstanding |
| 8 | per-step tier ledger exists | **PASS** — and it must record a *deliberate* tier this cycle |

## Stop conditions

**SOFT STOP:** 12 cycles, or a genuinely new operator-blocking gate → write the summary + a crisp
ask and stop. Note that last cycle mis-reported three items as operator-blocked when they were
executor work: **before declaring a block, check whether you can measure or fix it on a rig you own.**

**HARD STOP:** any change that moves the live book; any `80.27`-class change that is not strictly
fail-safe; a Fable *roster* repin; or a kill-switch change that makes the system *less*
conservative. Check `git log` after every background-agent notification and re-verify the working
tree before any flip.
