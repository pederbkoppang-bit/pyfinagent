# Operator ask list — 2026-07-26

Produced at the end of the `/goal` masterplan drain. Everything below is **blocked on you**
— no executor can clear these. Ordered by what unblocks the most.

---

## P0-URGENT — the kill switch is currently inert on a live book

**This was found outside any step's scope. Nothing in the backlog would have caught it.**

Your DO-NO-HARM rule says the kill switch stays byte-untouched, so **I did not fix it.**
It is queued as a masterplan step and reported here.

Measured live, read-only, 2026-07-26:

```
$ curl -s http://localhost:8000/api/paper-trading/kill-switch
{
    "paused": false,
    "sod_nav": null,          <-- baseline missing
    "peak_nav": null,         <-- baseline missing
    "current_nav": 23838.16,  <-- real money-equivalent book
    "breach": { "daily_loss_breached": false,
                "trailing_dd_breached": false,
                "any_breached": false }
}
```

**Mechanism** — `backend/services/kill_switch.py:311,317`:

```python
if sod and sod > 0:      # None is falsy -> whole branch skipped
if peak and peak > 0:    # None is falsy -> whole branch skipped
```

With both baselines `None`, neither breach is ever computed, so `any_breached` is `False`
**for any NAV**. A 50% drawdown would not pause the book.

**Cause** — `_load_from_audit` (`kill_switch.py:61-109`) restores the baseline by replaying
`sod_snapshot` / `peak_update` rows from `handoff/kill_switch_audit.jsonl`. That file was
rotated and now contains **8 lines, all `pause`/`resume`, zero baseline rows**:

| file | contents |
|---|---|
| `handoff/kill_switch_audit.jsonl` (**live, the one it reads**) | `{'pause': 4, 'resume': 4}` — **no baseline** |
| `handoff/audit/kill_switch_audit-v4.jsonl` | `{'sod_snapshot': 1}` — 2026-07-24, nav 23838.19 |
| `handoff/audit/kill_switch_audit-v3.jsonl` | 26 `sod_snapshot`, 5 `peak_update` |

The baseline rows still exist — they are just in archived files the loader never reads.
**This will recur on every rotation.**

**What I need from you:** authorization to fix it, since it touches kill-switch code.
The fix direction is *more conservative* (re-arming a breaker that currently cannot fire).
Two parts: (a) make `_load_from_audit` tolerate rotation, (b) make a missing baseline
**fail closed / loudly** instead of silently disabling the breach test.

Related and still owed since phase-69: **`KS-PEAK-RESET`**. Directly relevant now — a
re-armed switch must not be silently peak-reset in a way that suppresses a real drawdown.

---

## 1. Answer phase-79.55 and restart the backend — unblocks 11 P0s

`79.55` is an open **RESTART BLOCKER** (rail-model tier confirmation). Until it is answered
and `:8000` restarted, live evidence is impossible for: **27.6, 27.6.3, 61.2, 61.3, 63.4,
65.2, 65.4, 68.1, 68.3, 72.0.1, 72.0.2**.

PID `70791` started **2026-07-25 11:39:05** and predates phase-78.2, 80.3, 80.31 and 80.4.
**No paper cycle has run under the current binary.** This is the single highest-leverage
unblock in the backlog.

## 2. Refresh the Claude Code credential

70 `AUTH-DEAD` entries; both 2026-07-25 away sessions skipped. The `KILL SWITCH: RESUME`
path terminates in a dead session today — i.e. your remote stop/resume is currently broken.

## 3. Tokens and decisions

| # | Ask | Notes |
|---|---|---|
| 3.1 | `TEST TOKEN: PING` **sent from Slack** (user `U0A078KP4FQ`, channel `C0ANTGNNK8D`) | Closes 62.2 criterion 3. Authorizes zero env changes. Do the 62.1 slack-bot kickstart first. |
| 3.2 | `PAPER_SYNTHESIS_INTEGRITY_ENABLED` | 61.2. Settings-UI flippable, **no restart**. Promote this one first. |
| 3.3 | `PAPER_POSITION_RECOMMENDATION_FIX_ENABLED` | 61.2. **Only after 3.2** — it revives a SELL path that can sell healthy holdings on a synthetic HOLD. |
| 3.4 | `PAPER_AVG_ENTRY_FX_FIX_ENABLED` | 61.3. Not in `_FIELD_TO_ENV` → manual `.env` + restart. |
| 3.5 | One-cycle `paper_analyze_top_n` bump to ≥15 | 27.6 criterion 5 has no denominator at the current value of 5. Money-path config change. |
| 3.6 | `paper_learn_loop_enabled` decision | DEF-001's real remedy. Ship dark + token per 63.4 criterion 2. |
| 3.7 | `tools_nonfinite_fail_safe_enabled` | Owed from 80.27. Currently DARK. |
| 3.8 | Cost decision for 72.0.2 | Gemini fail-forward moves live cycle cost off $0. |
| 3.9 | Cloud Function redeploy approval (Quant CF) | 27.6.3 — or approve the cheaper orchestrator-tolerance alternative. |
| 3.10 | macOS hardening: `AutomaticallyInstallMacOSUpdates` → 0, `sudo pmset -a sleep 0` | Away launchd jobs fire daily; these are live requirements. |
| 3.11 | `rm -rf frontend/.next-audit-3100` | Leftover Playwright rig build dir. Trivial. |

**Do NOT ask yet** (prerequisites unlanded): `EXEC-BACKEND: ALPACA_PAPER`,
`ALPACA-RESET: APPROVED` (68.3 — triple-blocked), `65.2 EU SCREENER: ON` (current
mechanism would be a measured no-op).

Plus **~50 phase-79 operator actions** (`harness_required: false`) — those are yours by
design, not executor work.

---

## Money-path warning you should see before 27.6 is worked

93 of 97 July full-path rows persist `final_score = 0.0`, and
`portfolio_manager.py:216` reads `analysis.get('final_score', 0)` inside `decide_trades`.
**A 0.0 score is part of what is keeping the book at HOLD.**

So: any work that makes the full path produce real scores will make the system
**measurably less conservative — it will start trading again.** 27.6's criteria measure
*persistence*, not *content*, so passing 27.6 must **not** be read as "the full pipeline
works". The synthesis parse-failure root cause (`orchestrator.py:1683-1685`) should land
**before** 27.6/27.6.3, or fixing the orchestrator failures just swaps real lite scores
(9/9 positive) for empty full scores — a regression dressed as a fix.

---

## Triage headline: nothing in the P0 backlog is obsolete

A 13-agent read-only re-audit of all 16 open executor P0s, with **adversarial refutation of
every drop-class verdict**, produced **zero drops**. All five candidate drops were rejected
on refutation. The only legitimate reductions are **rescopes**, not drops (62.7, 63.4,
65.2, 65.4, 72.0.1).

Full evidence: `handoff/current/p0_triage_2026-07-26.md`.
