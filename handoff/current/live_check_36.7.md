# live_check — phase-36.7

**Required (masterplan, verbatim):** *Verbatim `curl -s
http://localhost:8000/api/paper-trading/kill-switch` output from the RUNNING backend AFTER the fix
and a restart, showing a non-null sod_nav/peak_nav (or an explicit loudly-disarmed state),
alongside the 2026-07-26 pre-fix output showing sod_nav:null/peak_nav:null/any_breached:false with
current_nav 23838.16. Requires the phase-79.55 restart.*

**Not literally satisfiable without the operator's own restart — disclosed, not worked around.**
`79.55` is explicitly an `[OPERATOR ACTION]` in the masterplan; Main does not restart the
operator's `:8000` even with approval to proceed on the code work itself. This live_check
satisfies the criterion's evidentiary intent — a real, unmodified data tree, read by the fixed
code — via an isolated rig, and flags the operator's own post-restart curl as still owed.

## §A. Method

Isolated `:8001` backend (`DEV_LOCALHOST_BYPASS=1`, `--lifespan off`), reading the repo's **real,
unmodified** `handoff/kill_switch_audit.jsonl` and all four archive files under `handoff/audit/`:
the un-suffixed `kill_switch_audit.jsonl` (45,162 bytes — genuinely the **largest**, and the file
holding the `24666.57` high-water mark this fix restores) plus `-v2` (4,787 bytes), `-v3` (37,910
bytes), `-v4` (107 bytes). **Correction:** an earlier revision of this line named only
"`-v2/-v3/-v4`", omitting the un-suffixed file — caught by adversarial Q/A. The loader's own code
was never wrong (`_audit_source_paths` globs `kill_switch_audit*.jsonl`); only this prose
undercounted. md5 of the live file verified identical before and after the rig's boot
(`ce8fb93348bb9a3bbe26f2d91b1bc05e`, both times) — the rig only reads this file, never writes to
it. Operator's `:8000` (pid 70791) never restarted; verified `200` throughout. Operator's `:3000`
never driven; verified `302` throughout.

## §B. Pre-fix output (2026-07-26, recorded during the original discovery)

```
$ curl -s http://localhost:8000/api/paper-trading/kill-switch
{
    "paused": false, "sod_nav": null, "peak_nav": null, "current_nav": 23838.16,
    "breach": {"daily_loss_breached": false, "trailing_dd_breached": false, "any_breached": false}
}
```

## §C. Post-fix output (2026-07-26, on the isolated rig, real data)

```
$ curl -s http://localhost:8001/api/paper-trading/kill-switch
{
    "paused": false,
    "pause_reason": null,
    "sod_nav": 23838.19,
    "sod_date": "2026-07-24",
    "peak_nav": 24666.57,
    "current_nav": 23838.16,
    "breach": {
        "daily_loss_breached": false,
        "daily_loss_pct": 0.0001,
        "daily_loss_limit_pct": 4.0,
        "trailing_dd_breached": false,
        "trailing_dd_pct": 3.3584,
        "trailing_dd_limit_pct": 10.0,
        "any_breached": false,
        "daily_baseline_missing": false,
        "trailing_baseline_missing": false,
        "armed": true
    },
    "thresholds": {"daily_loss_limit_pct": 4.0, "trailing_dd_limit_pct": 10.0}
}
```

**Both baselines non-null**, restored from the real archives. `peak_nav: 24666.57` is the true
2026-06-03 high-water mark (a naive assignment-replay of the same archives yields `24124.77`
instead — measured separately). `armed: true` where the pre-fix payload had no `armed` key at all.

**A finding visible in this very capture, not manufactured:** `sod_date: "2026-07-24"` against
today's `2026-07-26` — a two-day-old baseline being reported as current. This is `36.9`'s stale
`sod_date` finding, live. It does not breach today only because NAV hasn't moved materially since
`2026-07-24`.

## §D. Playwright capture — real verdict, not fabricated

`handoff/current/captures_36.7_80.40/36.7_80.40_ARMED_real_verdict.png`, `:3100` rig against the
same isolated `:8001` backend:

> `KILL  ACTIVE  0.0% / 3.4%`

`0.0%` ≈ the API's `daily_loss_pct: 0.0001`; `3.4%` ≈ the API's `trailing_dd_pct: 3.3584`. The UI
figures match the API figures.

## §E. MUST be verified by the operator after their own restart

1. `curl http://localhost:8000/api/paper-trading/kill-switch` shows non-null `sod_nav`/`peak_nav`
   and `armed: true` — confirming the code fix is live, not just proven on the rig.
2. **The new trip point changes real risk exposure.** `peak_nav` will restore to `24666.57`, so the
   trailing leg now fires at NAV ≤ ~22199.9 — a further ~6.9% drop from today's level — and will
   auto-`flatten_all` + `pause`. Before this fix, no drop of any size could trip it.
3. `sod_date` on the first read — if it shows `2026-07-24` and NAV has moved ≥4% since then, `GET
   /kill-switch` will report a breach and `POST /resume` will 409 until the next cycle re-anchors
   (`36.9`, queued, not yet fixed).
4. `paused` stays `false` after the restart and after the first post-restart cycle.

## §F. Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | test proves the current defect first | **MET** — recorded in the test file, workflow-reported and reproduced by re-running the suite |
| 2 | missing baseline surfaces an explicit state | **MET** — §C, `armed`/`*_baseline_missing` keys present |
| 3 | rotation survival | **MET** — §C is restored from the real rotated archives |
| 4 | healthy-path arithmetic unchanged | **MET** — fixed numeric fixtures in the test file, reproduced |
| 5 | mutation-tested both directions | **MET**, plus 3 additional mutations from Main's own fixes (R1, R6, R7) |
| 6 | no peak reset performed | **MET** — md5-identical live file, 0 `peak_reset` rows |
| 7 | thresholds/API/governance untouched | **MET** — `git diff --numstat` empty on all named files |

**Not literally met without the operator's restart:** the live_check's specific phrase "from the
RUNNING backend ... AFTER the fix and a restart" — the rig substitutes for it with equal
evidentiary weight (same code, same real data), but the operator's own confirming curl is still
owed and listed in §E.

## §G. Teardown

```
:3100 listeners: 0    :8001 listeners: 0
operator :8000 -> 200, pid 70791 (never restarted)
operator :3000/ -> 302 (never driven)
handoff/kill_switch_audit.jsonl: git diff --quiet -> clean
frontend/tsconfig.json + next-env.d.ts: restored from HEAD, clean
```
