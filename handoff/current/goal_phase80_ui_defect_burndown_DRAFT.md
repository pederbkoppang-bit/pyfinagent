# Goal draft — Phase-80: UI/API defect burn-down (operator-requested Playwright audit)

**Operator prompt (verbatim, 2026-07-25):**

> i would like you to audit with playwright mcp these design bugs and other bugs. when done
> add them to our masterplan for later fix. [3 screenshots] picture one and 2 on the home
> page picture one makes the rest of the page drop down dou to a tooltips when hoovering the
> ring. picture 3 shows that it is not possible to fetch signals. there are problably more
> bugs which i havnt told you therefor audit all functions possible with playwright whcih we
> can later fix in our masterplan step

Follow-ups, verbatim: *"add this to playwright mcp later No browser_evaluate in this MCP
build."* (→ step 80.25) and *"you have to include all tabs on each page as well"* (→ every
tab on every page was visited; the audit was re-run to cover them).

**Status:** phase-80 is INSTALLED in `.claude/masterplan.json` (31 steps, all `pending`,
5×P0 / 10×P1 / 16×P2) but **UNCOMMITTED and UNREVIEWED**. This goal doc is the DRAFT that
frames its execution. Nothing in phase-80 has been executed.

## North star

(masterplan.json::goal) *Maximize Net System Alpha = Profit − (Risk Exposure + Compute
Burn).* **This goal's lens:** the cockpit is the instrument the operator reads before
touching money, and the audit proved the instrument lies in three directions at once —
a dead endpoint reports as "backend is down", a data outage renders as a confident NEUTRAL
trading verdict, and a failed cost query renders as a verified $0.00 spend. Fix what the
operator sees before trusting what it says. Compute-burn is in scope too: one page view
issues 11 redundant session probes and one request freezes the whole event loop for ~17s.

## Scope — in order (each step = FULL harness cycle)

Ordering is load-bearing, not cosmetic. Waves 0–1 must not be reordered.

0. **80.2 FIRST — it is the instrument, not a fix.** Until 500s carry CORS + OWASP headers
   and reach PerfTracker, every other failure in this phase is misreported to the operator
   as "backend is down" and is invisible in `/api/observability/latency`. You cannot measure
   your own fixes before this lands.
1. **The NaN family — 80.1 + 80.27 + 80.31 land TOGETHER.** One root cause, three surfaces.
   **The trap:** 80.1 can be "fixed" by coercing non-finite floats to null at the response
   boundary, which turns the 500 green while leaving 80.27's pipeline path fully poisoned —
   and hiding it. **80.27 is NOT closed by 80.1.** Fix at the source (`sector_analysis.py:34`
   AND `quant_model.py:63` — there are TWO leak sites), then make the classifiers and the
   `info_gap` gate reject non-finite input.
2. **P0 dead surfaces:** 80.3 (`/agent-map` renders zero edges — `AgentNode` never renders a
   `<Handle>`), 80.4 (MAS Dashboard reads "Disconnected" on a healthy SSE stream).
3. **Operator-reported + money surfaces:** 80.5 (the donut reflow from picture 1 — includes
   the `fill="transparent"` hit-test bug that fires it from dead space), then 80.7 → 80.6
   **in that order** (they touch adjacent code; the NAV-basis fix changes what the formatter
   renders), then 80.8, 80.9, 80.10.
4. **Load + latency:** 80.28 (event-loop blocking — this IS the 17s), then 80.11 (session
   stampede + three uncapped poll loops + duplicate fetch owners).
5. **Stop lying to the operator:** 80.30 (failure paths that fabricate specific facts),
   80.14 (no-data painted as success), 80.13 (blank tabs with no empty state).
6. **Accessibility:** 80.29 (the shared `DataTable` makes the Glass Box rationale drawer
   mouse-only; six money-sliders have no accessible name).
7. **P2 tail, any order:** 80.12, 80.15, 80.16, 80.17, 80.18, 80.19, 80.20, 80.21, 80.22,
   80.23, 80.24, 80.26.
8. **80.25 is an operator decision, not an executor task** — enabling `browser_evaluate`
   contradicts the standing `docs/runbooks/browser-mcp.md` no-code-exec guardrail (issue
   #1495 RCE). A recorded REJECT closes it just as validly as enabling it.

## Founding principles (non-negotiable)

- **Full harness per step:** researcher FIRST (≥5 sources read in full + recency scan) →
  `contract.md` (criteria copied verbatim) → GENERATE → ONE fresh Q/A → `harness_log.md`
  append → masterplan flip. No self-evaluation; no verdict-shopping.
- **DO-NO-HARM — the live book does not move.** Paper-only; no `backend/.env` edits, no flag
  flips, no optimizer runs (`historical_macro` stays FROZEN); kill-switch limits, stops,
  sector caps, DSR≥0.95 and PBO≤0.5 stay byte-untouched.
- **80.27 is the one step that changes live decision behaviour — it must be FAIL-SAFE ONLY.**
  A non-finite input must produce `ERROR`/`NO_DATA` (→ fewer, more-gated trades), never a new
  trade. If a change could make the system *less* conservative, it is out of scope.
- **Every UI claim needs a Playwright capture** against the running app (CLAUDE.md binding
  rule + `qa.md` §1c). Code reading is not UI evidence. Use the isolated skip-auth `:3100` +
  `PLAYWRIGHT_DIST_DIR` rig — **never** touch the operator's `:3000`, and **restore
  `tsconfig.json` + `next-env.d.ts` afterwards** (`next dev` rewrites both to point at the
  audit distDir).
- **Mutation-test every guard.** phase-80 exists partly because the ONE test touching
  currency (`PortfolioAllocationDonut.test.tsx:177`) strips all separators before asserting
  and is structurally incapable of failing. A guard that cannot fail does not count —
  see `feedback_mutation_test_guards_and_fixtures`.
- **Measure, don't assert.** Several steps quote counts (call sites, endpoints, sliders,
  edges). **Re-derive every census yourself** and reconcile it by symmetric diff against what
  you actually changed. Do not inherit a number from the step text —
  see `feedback_measure_dont_assert_claims`.
- **Do NOT re-chase the 10 recorded dead ends.** Eight workflow candidates were refuted by
  adversarial verifiers and two of Main's own were self-refuted; all are written into the
  phase-80 text (notably: the `TRANSACTION COST "0,1"` comma is Chromium's locale rendering
  of a valid `0.1`; there are ZERO emoji violations; `formatCurrency` is NOT unreachable for
  USD; `short_interest.py` is dark behind a disabled flag).
- **`git add -An` before every flip.** The auto-commit hook stages the whole tree under your
  step's name — a foreign session nearly shipped these 31 un-gated steps under phase-78.2.
  See `feedback_audit_the_commit_not_the_diff`.

## Operator-gated (ask, never assume)

Any LLM spend beyond the $0 Claude Code Max rail; pip installs; BQ `DROP` / unqualified
`DELETE` / backfills; `launchctl` changes; **80.25** (the `browser_evaluate` capability
decision); and any change whose effect on live trading behaviour is not strictly fail-safe.

## Done-definition (HARD STOP)

All 31 steps PASS **or** are explicitly deferred with a recorded reason + phase-80 flipped
`done` + `cycle_block_summary.md` refreshed with a crisp operator ask list. Non-negotiable
closing evidence, because it is what the audit measured:

1. `GET /api/signals/AAPL` returns **200** with all 12 signal keys — and an event-loop
   heartbeat test proves the loop stayed responsive during it.
2. A NaN-poisoned enrichment payload is classified **NOT-SUFFICIENT** by `info_gap`, and
   both classifiers return `ERROR`/`NO_DATA` rather than `NEUTRAL`.
3. A deliberately-raising route returns a 500 that carries the CORS header, `nosniff`, and a
   `PerfTracker` record visible in `/api/observability/latency`.
4. `/agent-map` draws its edges with **zero** React Flow console warnings at 1440×900.
5. Hovering any donut slice produces **zero** layout shift (identical bounding boxes).
6. One cockpit page view issues **≤2** `/api/auth/session` requests over 20s.
7. With the backend **stopped**, no page fabricates a fact — no `$0.00 / zero billed jobs`,
   no `No harness cycles yet`, no `needs at least 2 daily snapshots`, no permanent `Loading…`.

## Stop conditions

**SOFT STOP:** 12 cycles OR any operator-blocking gate → write the summary + a crisp ask and
stop. **HARD STOP:** any change that would move the live book, or any 80.27 change that is
not strictly fail-safe → stop and ask. Check `git log` after every background-agent
notification (the `ad349f57` lesson), and re-verify the working tree before any flip.

## Preconditions before Wave 0

- Operator reviews phase-80 (it is uncommitted; another session nearly swept it up).
- `rm -rf frontend/.next-audit-3100` (the audit build dir; Main's `rm` was policy-denied).
  It is now gitignored via `.next-*/`, so it cannot pollute `git status` meanwhile.
- Evidence for every finding is in `handoff/current/captures_ui_audit_2026-07-25/`
  (30 screenshots + the raw 20s network log behind the session-stampede measurement).
