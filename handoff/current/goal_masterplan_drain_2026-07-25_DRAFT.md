# Goal draft — Masterplan drain with model/effort tiering (set 2026-07-25)

**Operator prompt (verbatim, 2026-07-25):**

> now write the masterplan goal prompt i asked about earlier
> i meant for the whole masterplan work including your new findings. you should
> change to cheaper models for easy task and complex task you should use the best
> model available (fable 5 or opus 5)

Supersedes `goal_phase80_ui_defect_burndown_DRAFT.md` (phase-80 only), which is folded
in below as one wave. Nothing here is committed or pushed.

## North star

(masterplan.json::goal) *Ship an Intelligence Engine trading system that maximizes Net
System Alpha = Profit − (Risk Exposure + Compute Burn) by dynamically shifting capital to
the highest-earning strategy, recursively self-improving under hard risk caps, within a
15-slot daily Claude-routine budget.* **This goal's lens:** the backlog has grown to 222
open steps across 21 phases while the operator-facing instruments have been quietly
lying (phase-80). Drain the backlog, and spend model capability where correctness is
load-bearing rather than uniformly.

## Measured scope (2026-07-25, re-measure before acting)

**222 open steps across 21 phases** — `{P0: 26, P1: 45, P2: 91, P3: 40, P4: 8, unset: 12}`.
**53 carry `harness_required: false`** (operator actions, almost all of phase-79) — those
are *not* executor work. So roughly **168 executor steps + 54 operator decisions**.

Open by phase: 5(11) 13(1) 27(2) 35(1) 43(1) 44(6) 58(1) 61(4) 62(4) 63(3) 65(2) 68(7)
72(9) 73(13) 74(4) 75(33) 76(14) 77(3) 78(18) 79(54) 80(31).

These counts are a point-in-time snapshot from a script over `.claude/masterplan.json`.
**Re-derive them; do not inherit them** — a concurrent writer added two steps mid-session
while this was being written (`feedback_audit_the_commit_not_the_diff`).

## Model + effort tiering (the operator's directive, made mechanical)

**Two facts change how to apply "cheaper for easy, best for complex":**

1. **On the Claude Code Max rail, model choice costs $0 metered.** Layer-3 harness work
   (Main + Researcher + Q/A, and every Workflow fan-out agent) runs flat-fee. Downgrading
   a model there saves **quota and latency, not dollars**. The real-dollar lever is
   Layer-2 in-app roles (per-ticker, metered) — which this drain mostly does not touch.
   So tier for *speed and quota*, and never trade correctness for a saving that is zero.
2. **On Opus 5, `effort` is the better cheap lever than model.** Opus 5's `low`/`medium`
   are unusually strong — often beating prior models at `xhigh`. Dropping effort keeps
   the best model and still cuts tokens and latency. **Prefer an effort step-down to a
   model step-down.**

### Current pins (measured)

`.claude/settings.json` → `effortLevel: xhigh`, `fallbackModel: [claude-sonnet-5,
claude-haiku-4-5]`. Layer-3 `researcher.md` / `qa.md` → `model: opus`.
**Note the alias drifted:** `opus` resolves to the *latest* Opus, which is now
**Claude Opus 5**, not the 4.8 those pins were written against. Confirm that is intended.

### The tiers

| Tier | Use for | Model | Effort |
|---|---|---|---|
| **T4 — scarce** | The few hardest correctness-critical steps: `80.27` (NaN → live trading verdict), `80.1`, `80.2`, and any P0 on the money path where a wrong answer moves the book | `fable` (Fable 5) | `xhigh` (`max` for 80.27) |
| **T3 — max** | The rest of the P0s; audit-class research; adversarial verify | `opus` (Opus 5) | `xhigh`, `max` where correctness ≫ cost |
| **T2 — default** | Normal P1/P2 build steps with a clear spec | `opus` (Opus 5) | `high` |
| **T1 — cheap** | Mechanical, deterministic, single-file edits — palette swaps, `scrollbar-thin`, favicon, gitignore, label/timezone text, doc-only steps, most P3/P4 | `haiku` | *(omit — Haiku 4.5 rejects `effort`)* |

### Fable 5 — AUTHORIZED, but it is the scarce resource

**Operator, verbatim (2026-07-25):** *"you can use fable 5 if neccacry as it is free on
the max subscrition based on weekly limits"* — recorded in `handoff/harness_log.md` as
`FABLE 5 SCOPED-AUTHORIZE`. This is the fresh free-window confirmation CLAUDE.md's Fable
policy requires. It is **scoped use, not a permanent repin** — no `FABLE PERMANENT:
AUTHORIZE` token was given, so the SessionStart tripwire and revert doctrine stay intact.

Because it is free-but-weekly-capped, Fable is the one tier bounded by **quota, not
dollars** — so spend it deliberately on the hardest steps rather than uniformly. Three
constraints from this project's own log, all load-bearing:

- **Exhaustion is a HARD FAIL, not a downgrade.** `harness_log.md:27346` records the 69.4
  Q/A failing outright with *"reached your Fable 5 limit"* because the workflow `qa`
  agentType inherited the `qa.md` Fable pin.
- **Therefore: spend Fable PER INVOCATION, never by repinning the roster.** Use
  `Workflow agent({model:'fable', effort:'xhigh'})` or Agent-tool `model: fable` on the
  specific T4 step. Do **not** set `model: fable` in `.claude/agents/researcher.md` /
  `qa.md` — that puts every future Q/A on the capped budget and breaks the evaluator when
  it runs out. Per-invocation also needs no scheduled revert and no session restart
  (roster pins snapshot at session start; per-invocation does not).
- **FABLE-HEADLESS:** Fable pins are unavailable in headless/cron runs and silently fall
  back to Opus (`harness_log.md:27085`, `:27096`). Away-ops and scheduled runs cannot rely
  on Fable — never design a step that requires it.

**Every T4 invocation needs an Opus fallback path.** If a Fable agent returns empty or
errors on the limit, re-run at T3 (`model: 'opus'`) rather than treating the failure as a
verdict — an errored return is NO VERDICT, never PASS.

**Correction to CLAUDE.md (queue the doc fix; do not repeat the stale rule).** The Fable
bullet says `xhigh` silently downgrades to `high` for non-Opus-4.8/4.7 models (citing
`llm_client.py:1507-1512`) and that "Fable roles must use `max`, never `xhigh`". Both the
behavior and the line numbers are stale — phase-67.6 whitelisted Fable and Sonnet 5, and
the live guard at `backend/agents/llm_client.py:1634` is
`not model_id.startswith(("claude-opus-4-8","claude-opus-4-7","claude-fable-5","claude-sonnet-5"))`,
with `MODEL_EFFORT_FALLBACK` carrying `("claude-fable-5","xhigh")`
(`backend/config/model_tiers.py:335`). **Fable supports `xhigh`.**

**Fable prompting caveat that matters for this harness specifically:** Anthropic's
guidance is that prompts written for prior models are *often too prescriptive for Fable
and reduce output quality*. Our harness prompts are highly prescriptive by design. On the
first T4 step, A/B the prompt with the step-by-step scaffolding stripped down to goal +
constraints before concluding Fable underperforms. Also: Fable turns can run many minutes
— the 2026-07-09 stall watch (`project_phase67_fable_window`) applies; watch transcript
mtime and fall back to Opus rather than waiting indefinitely.

### Two hard constraints on the cheap tier

- **Haiku 4.5 is 200K context. `.claude/masterplan.json` is ~1.7MB (~420K tokens).**
  A cheap-tier agent physically cannot read it. **Never hand the masterplan to T1** —
  slice it with a Python script (as the phase-80 install did) and pass only the step.
- **Haiku 4.5 does not accept the `effort` parameter** — passing it errors. T1 has no
  effort knob; if a task needs one, it is not a T1 task.

### Assigning a tier (do this per step, not per phase)

T1 only if **all** hold: single file, no money path, no live decision path, no research
gate, and the success criteria are mechanically checkable. **If in doubt, use T2** — on a
flat-fee rail a wrong cheap call costs a rework cycle and buys nothing.

## Scope — wave order (each step = FULL harness cycle)

1. **Wave 0 — unblock the instruments.** `80.2` (500s carry no CORS/OWASP headers and
   never reach PerfTracker) **first**: until it lands, every other failure in this drain
   is misreported as "backend is down" and is invisible in `/api/observability/latency`.
   T3.
2. **Wave 1 — the NaN family, together.** `80.1` + `80.27` + `80.31`. **`80.27` is not
   closed by `80.1`** — sanitising NaN at the response boundary turns the 500 green while
   leaving the pipeline path poisoned *and hiding it*. Fix at source (two leak sites:
   `sector_analysis.py:34` **and** `quant_model.py:63`). T3, `max`.
3. **Wave 2 — remaining phase-80 P0s.** `80.3` (agent-map renders zero edges), `80.4`
   (false "Disconnected"). T3.
4. **Wave 3 — the other open P0s, oldest first.** phase-27(2), 61(2), 62(3), 63(2),
   65(2), 68(3), 72(2). Re-read each before starting: several are months old and the
   world moved (phase-72's premise was credit exhaustion, since diagnosed). T3.
5. **Wave 4 — phase-80 P1 money surfaces.** `80.5` (the operator-reported donut reflow),
   then `80.7` → `80.6` **in that order** (adjacent code; the NAV-basis fix changes what
   the formatter renders), then `80.8`, `80.9`, `80.10`, `80.11`, `80.28`, `80.29`,
   `80.30`. T3 for 80.7/80.8/80.9 (money surfaces), T2 for the rest.
6. **Wave 5 — the P1 tail across phases** 43, 44, 61, 62, 63, 68, 72, 73, 75, 76, 78. T2.
7. **Wave 6 — the P2/P3/P4 tail.** Triage hard here: **a P3 that no longer matters should
   be closed as `dropped` with a reason, not executed.** phase-5 (11 open, status
   `deferred`) and phase-13 (1) are the oldest — decide *drop vs do* before spending a
   cycle. T1 where the criteria above allow, else T2.
8. **Operator actions (phase-79, 54 steps) are not in this drain.** They need the human.
   Surface them as a batched ask list; a recorded DECLINE closes one as validly as doing
   it. Do not let them block executor waves.

## Founding principles (non-negotiable)

- **Full harness per step:** researcher FIRST (≥5 sources in full + recency scan) →
  `contract.md` (criteria verbatim) → GENERATE → ONE fresh Q/A → `harness_log.md` append
  → masterplan flip. No self-evaluation; no verdict-shopping.
- **DO-NO-HARM — the live book does not move.** Paper-only; no `backend/.env` edits, no
  flag flips, no optimizer runs (`historical_macro` stays FROZEN); kill-switch limits,
  stops, sector caps, DSR≥0.95 and PBO≤0.5 stay byte-untouched.
- **`80.27` is the one step that changes live decision behaviour — FAIL-SAFE ONLY.** A
  non-finite input must yield `ERROR`/`NO_DATA` (→ fewer, more-gated trades), never a new
  trade. Anything that could make the system *less* conservative is out of scope and a
  HARD STOP.
- **Tier down on effort before tiering down on model**, and never tier down a step whose
  failure mode is a wrong number on a money surface.
- **Every UI claim needs a Playwright capture** against the running app. Use the isolated
  skip-auth `:3100` + `PLAYWRIGHT_DIST_DIR` rig — never the operator's `:3000` — and
  **restore `tsconfig.json` + `next-env.d.ts` afterwards** (`next dev` rewrites both).
- **Mutation-test every guard.** phase-80 exists partly because the one test touching
  currency strips separators before asserting and cannot fail. A guard that cannot fail
  does not count.
- **Measure, don't assert.** Re-derive every count in this document — including the
  222/168/54 split — and reconcile by symmetric diff against what you actually changed.
- **`git add -An` before every flip.** The auto-commit hook stages the whole tree under
  your step's name; a foreign session nearly shipped phase-80's 31 un-gated steps under
  phase-78.2.
- **Do not re-chase the 10 recorded dead ends** written into the phase-80 step text.

## Operator-gated (ask, never assume)

Any LLM spend beyond the $0 Max rail; **repinning the roster to Fable** in
`.claude/agents/*.md` (per-invocation Fable is authorized above; a *pin* still needs its
own token + a scheduled revert, and risks the hard-fail documented at
`harness_log.md:27346`); pip installs; BQ `DROP` /
unqualified `DELETE` / backfills; `launchctl` changes; the 54 phase-79 operator actions;
`80.25` (the `browser_evaluate` capability decision); and any change to live trading
behaviour that is not strictly fail-safe.

## Done-definition (HARD STOP)

Not "all 222 closed" — that is a multi-week program. This goal closes when **every open
P0 across every phase is PASS or explicitly deferred with a recorded reason**, the
phase-80 P1 money surfaces (80.5–80.11) are closed, and `cycle_block_summary.md` carries
a batched operator ask list. Closing evidence, because it is what the audit measured:

1. `GET /api/signals/AAPL` → **200** with all 12 signal keys, and an event-loop heartbeat
   test proving the loop stayed responsive.
2. A NaN-poisoned payload is classified **NOT-SUFFICIENT** by `info_gap`, and both
   classifiers return `ERROR`/`NO_DATA` rather than `NEUTRAL`.
3. A deliberately-raising route returns a 500 carrying the CORS header, `nosniff`, and a
   `PerfTracker` record visible in `/api/observability/latency`.
4. `/agent-map` draws its edges with **zero** React Flow console warnings at 1440×900.
5. Hovering any donut slice produces **zero** layout shift (identical bounding boxes).
6. One cockpit page view issues **≤2** `/api/auth/session` requests over 20s.
7. With the backend **stopped**, no page fabricates a fact.
8. A per-step tier ledger exists (step-id → tier → model/effort) so the tiering policy can
   be audited rather than assumed.

## Stop conditions

**SOFT STOP:** 12 cycles OR any operator-blocking gate → write the summary + a crisp ask
and stop. **HARD STOP:** any change that would move the live book, any `80.27` change
that is not strictly fail-safe, or a Fable *roster repin* (per-invocation use is fine). Check
`git log` after every background-agent notification, and re-verify the working tree
before any flip.

## Preconditions

- Operator reviews **phase-80** (31 steps, uncommitted) before an executor picks it up.
- `rm -rf frontend/.next-audit-3100` (Main's `rm` was policy-denied; now gitignored).
- Confirm the `model: opus` → **Opus 5** alias drift on `researcher.md` / `qa.md` is
  intended, and decide whether `effortLevel: xhigh` stays the session default under the
  new tiering.
- Evidence for phase-80 is in `handoff/current/captures_ui_audit_2026-07-25/`.
