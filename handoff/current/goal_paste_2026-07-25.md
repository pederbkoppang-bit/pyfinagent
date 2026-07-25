# GOAL — masterplan drain + tiering (2026-07-25)

FIRST read `handoff/current/goal_masterplan_drain_2026-07-25_DRAFT.md` (authoritative) + CLAUDE.md + `.claude/rules/`.

## Scope
222 open steps / 21 phases — P0 26, P1 45, P2 91, P3 40, P4 8. 53 are `harness_required:false` operator actions (phase-79), NOT executor work. **Re-derive these counts** — another session also writes the file.

## Tiering (operator directive)
On the Max rail model choice costs $0 metered — tier for quota/latency, never trade correctness for a zero saving. **Prefer an effort step-down over a model step-down** (Opus 5 low/medium are strong).
- **T4 Fable 5** — hardest correctness-critical only: 80.27 (`max`); 80.1/80.2 + money-path P0s (`xhigh`).
- **T3 Opus 5 `xhigh`/`max`** — other P0s, audit research, adversarial verify.
- **T2 Opus 5 `high`** — default P1/P2.
- **T1 Haiku 4.5** — mechanical single-file edits only. Rejects `effort`; 200K ctx so it CANNOT read `.claude/masterplan.json` (1.84MB) — slice it with a script.
In doubt → T2.

### Fable: authorized, weekly-capped
Operator 2026-07-25: *"you can use fable 5 if neccacry as it is free on the max subscrition based on weekly limits"* (logged as FABLE 5 SCOPED-AUTHORIZE).
- **Per-invocation only**: `agent({model:'fable', effort:'xhigh'})`. **Do NOT repin `.claude/agents/*.md`** — exhaustion is a HARD FAIL (`harness_log.md:27346`: the 69.4 Q/A died on "reached your Fable 5 limit" via an inherited pin).
- Every T4 call needs an Opus fallback; an errored/empty return is NO VERDICT, not PASS.
- FABLE-HEADLESS: pins silently fall back to Opus in headless/cron — never require Fable.
- CLAUDE.md's "Fable must use max, never xhigh" is STALE (67.6 whitelisted it, `llm_client.py:1634`); Fable supports `xhigh`. Queued as 80.32.

## Order (each step = FULL harness cycle)
1. **80.2 FIRST.** Until 500s carry CORS/OWASP headers and reach PerfTracker, every failure reads as "backend down" and is invisible in the latency panel — you cannot measure your own fixes.
2. **NaN family together: 80.1 + 80.27 + 80.31.** 80.27 is NOT closed by 80.1: sanitising NaN at the response boundary turns the 500 green while leaving the trading path poisoned AND hiding it. TWO leak sites — `sector_analysis.py:34` and `quant_model.py:63`.
3. 80.3 (agent-map: zero edges), 80.4 (false "Disconnected").
4. Remaining P0s oldest-first: phases 27, 61, 62, 63, 65, 68, 72 — re-read each; several are stale.
5. phase-80 P1s: 80.5, then **80.7 → 80.6 in that order**, then 80.8–80.11, 80.28–80.30.
6. P1 tail (43,44,61,62,63,68,72,73,75,76,78) → P2/P3/P4 tail. An obsolete P3 gets `dropped` with a reason, not executed.
Phase-79's 54 operator actions are not executor work — batch into an ask list.

## Non-negotiable
- Full harness/step: researcher (≥5 sources) → contract.md → GENERATE → ONE fresh Q/A → log → flip. No self-eval or verdict-shopping.
- DO-NO-HARM: paper only; no `.env` edits, no flag flips, `historical_macro` FROZEN; kill-switch/stops/sector-caps/DSR/PBO byte-untouched.
- **80.27 is the only step changing live decisions — FAIL-SAFE ONLY** (non-finite → ERROR/NO_DATA, never a new trade). Less conservative = HARD STOP.
- UI claims need a Playwright capture on the skip-auth `:3100` rig — never the operator's `:3000`; restore `tsconfig.json` + `next-env.d.ts` after.
- Mutation-test every guard. Measure, don't assert. `git add -An` before each flip.
- Do NOT re-chase the 10 dead ends recorded in the phase-80 steps.

## Preconditions
Operator reviews phase-80 (32 steps, uncommitted). `rm -rf frontend/.next-audit-3100`. Confirm `model: opus` → **Opus 5** is intended.

## Done
Every open P0 PASS or deferred-with-reason; phase-80 P1 money surfaces closed; batched ask list. Evidence: signals 200 with 12 keys + loop-heartbeat; NaN payload → NOT-SUFFICIENT; a raising route 500s with CORS + nosniff + a PerfTracker row; agent-map 0 React Flow warnings; donut hover 0 layout shift; ≤2 session probes/20s; backend stopped and no page fabricates a fact.
