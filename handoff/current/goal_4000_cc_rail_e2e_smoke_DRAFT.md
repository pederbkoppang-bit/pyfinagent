# GOAL 4000 (DRAFT) -- E2E smoke: run the app's Claude LLM traffic on the Claude Code Max plan, with the metered API key dark

Status: DRAFT prompt for operator review. When approved, this text becomes the
seed prompt for the phase-4000 masterplan goal, verbatim. Written 2026-08-06
from a live code audit; every file:line pointer below was verified this session.
Executor assumption: you have NO memory of the conversation that produced this
file. Everything you need is either in this file or reachable from its pointers.

---

## 1. Operator intent (the "why" -- keep in the contract verbatim)

- **The binding problem is TRADE THROUGHPUT, not just cost.** The app is
  currently doing ~1-2 trades per week. The operator needs ~100 round-trip
  trades to consider the app tested and ready for live production, and there
  is plenty of available cash -- so at the current cadence the validation
  sample takes a year to accumulate. Phase-72's money recon already found the
  root cause of the app sitting in ~97% cash: Anthropic API credits dead since
  2026-05-17 killed the Claude-role signal paths (plus the meta-scorer
  rail-bypass). Reviving the Claude rail is the highest-leverage throughput
  fix on the table.
- Anthropic API credits have been dead since 2026-05-17 (phase-72 money recon).
  We are on the Claude Max plan (weekly limits, flat-fee) and no credit top-up
  is planned, so the `anthropic_api_key` path in the app is a dead rail.
- Gemini (Vertex) is metered per-token.
- The Claude Code Max plan is already paid for. Goal: serve the app's
  Claude-role LLM traffic on the Max-plan rail (`claude` CLI subprocess), prove
  it end to end against the RUNNING app, and -- if the smoke passes -- make the
  rail the operator-confirmed steady state so the metered key path stays dark
  and the signal paths come back to life.
- This goal PROVES and PROMOTES an existing route. It does not build a new one.
- **North-star metric for the goal:** round-trip trade cadence. Baseline
  (measure, don't assert, from `financial_reports.paper_trades`): trades/week
  over the last 4 weeks. Target direction: a measurable cadence increase in
  the post-flip observation window, on the path toward the 100-round-trip
  validation sample. HONESTY CLAUSE: the rail being dead is a NECESSARY-fix
  suspect, not proven sufficient. Known co-suspects that also throttle trade
  count: `paper_analyze_top_n=5` binding (phase-82 finding), the phase-70 S3
  silent BUY-gates + monosector funnel, and label-quality issues. If cadence
  does not recover after rail-ON, the readout must say so and queue the
  co-suspects as their own research-gated steps -- not stretch this goal.

## 2. Ground truth -- what already exists (verify, never rebuild)

| Thing | Where | State |
|---|---|---|
| CC-rail client | `backend/agents/claude_code_client.py` (821 lines): `claude_code_invoke()` at :297, `ClaudeCodeClient(LLMClient)` at :571. Shells `claude --print --output-format json`; uses Max-subscription auth from `~/.claude/`; no `ANTHROPIC_API_KEY` needed. Binary resolution handles launchd PATH (`_resolve_claude_binary`). | Shipped 2026-05-26 |
| Routing seam | `backend/agents/llm_client.py:2108-2150` -- flag ON + Claude-class model -> `ClaudeCodeClient`; fallthrough raises a routing-breach RuntimeError. Second breach guard on `advisor_call` at :2253-2270. Flat-fee rows are excluded from the metered-spend metric per the note at :432. | Shipped |
| The gate flag | `backend/config/settings.py:176` `paper_use_claude_code_route: bool = False`. Flippable at runtime via `/api/settings/` PUT. Its description says "Testing-phase only; flip to False before flipping real_capital_enabled to True" -- see Preconditions. | Default OFF; MEASURE the live value via GET /api/settings, do not assume |
| Rail guard | phase-66.1: per-cycle probe gate + circuit breaker (`claude_rail_breaker_threshold`, default 20) + exactly-one P1 page. Born from the 2026-06-15..07-06 outage (162 doomed calls/cycle, three weeks, zero pages). | Shipped |
| Timeout/retry | `claude_code_timeout_s` default 150s (phase-61.2); `claude_code_empty_retry_max` default 2 (effective only when `paper_synthesis_integrity_enabled=True`). | Shipped |
| Explicit --model | Since phase-78.2 every rail call passes `--model` for the CONFIGURED tier. Before that the rail silently ran whatever `/model` was last set to (measured: opus-5[1m] doing the work while `llm_call_log` recorded `claude-haiku-4-5`). | Shipped, but see 79.55 blocker |
| Covered call sites | Six signal overlays (meta_scorer, news_screen, macro_regime, pead_signal, analyst_narrative_scorer, call_transcript_gpr), lite trader (`autonomous_loop.py:2453`), lite risk judge (`:2534`), ticket queue via `agent_model_map`. | Per 79.55 text; re-derive from code, the lines move |
| SDK-caller bridge | `scripts/ops/anthropic_max_bridge.py` (phase-76.9.2) + `run_nightly.sh` flag wiring, tested E2E in `backend/tests/test_phase_76_9_2_max_bridge.py`. This is the OTHER Max-rail mechanism (for anthropic-SDK callers, e.g. nightly autoresearch). | Shipped + tested; OUT of this goal's scope |
| Existing unit tests | `backend/tests/test_claude_code_client.py`, `test_phase_66_1_rail_guard.py`, `test_phase_61_2_decision_integrity.py`, `test_phase_75_llm_rail.py`, `test_phase_78_1_c_block_rail.py`. All mock the subprocess. NONE of them exercise the real CLI against the real running app -- that gap is exactly this goal. | Green but blind to the live path |

## 3. Hard preconditions (gates BEFORE any GENERATE)

1. **Masterplan 79.55 (P0, pending, RESTART BLOCKER) must be answered first.**
   The operator must confirm per-role rail model tiers ("RAIL TIERS: AS
   CONFIGURED" in `handoff/harness_log.md`, or name re-pins). The smoke runs
   whatever tiers are configured; running it before 79.55 is answered would
   ship the silent re-tiering that 79.55 exists to prevent.
2. **Operator token to flip `paper_use_claude_code_route` ON for the smoke
   window**, plus an explicit decision on the flag's standing doctrine: its
   description currently says "Testing-phase only; flip to False before
   real_capital_enabled". Making the rail the steady state means AMENDING that
   description with operator sign-off -- that is a deliberate doc+settings
   change inside this goal, not a drive-by.
3. **Budget rail (non-negotiable):** the Max weekly pool is SHARED with the dev
   harness, this session, and every Workflow fan-out. The smoke is bounded:
   max 2 tickers, max ~30 rail calls, one cycle. No loops-until-green against
   the live rail. Past the 50% Fable/weekly ceiling usage silently goes
   metered, which breaks the standing $0-metered constraint.
4. **Do not break the working engine** (live money state: +20% NAV; biggest
   risk is regression INTO a working engine). Record the pre-smoke flag state
   via GET /api/settings; on any FAIL, restore it in the same session and say
   so in the readout.

## 4. The deliverable -- one repeatable E2E smoke, run against the REAL app

Build `scripts/qa/smoke_cc_rail_e2e.py` (or extend an existing scripts/qa
harness if one fits -- audit first). It must, in order:

1. **Preflight:** backend :8000 healthy; `claude` binary resolves via
   `_resolve_claude_binary`; `claude --print` probe succeeds on Max auth
   (assert the envelope's auth/costUSD shape, not just exit 0); record
   pre-smoke `paper_use_claude_code_route` value.
2. **Flip the flag ON via the real API** (`PUT /api/settings/`), never by
   editing .env -- the E2E claim is "the operator's runtime path works".
3. **Drive a bounded real analysis** through the running backend (1-2 tickers
   through the path that exercises the rail call sites -- derive the cheapest
   entrypoint from code: lite path or a single-ticker analysis endpoint).
   `cache.preload_macro()` rule applies if a backtest-adjacent path is used.
4. **Prove, with evidence, all of Section 5.**
5. **Restore or keep the flag per the operator's Section-3.2 decision**, and
   print a machine-readable PASS/FAIL summary with per-check verdicts.

## 5. Evidence the smoke MUST produce (goes in `live_check_<step>.md`, verbatim outputs)

- **E1 -- rail actually used:** for the smoke window, every Claude-role row in
  `pyfinagent_data.llm_call_log` carries the CC-rail marker. Derive the marker
  from code (measure, don't assert) and state the row-selection rule with the
  count ("N of N rows in window W match rule R" -- both halves under ONE rule).
- **E2 -- metered path dark:** zero rows/requests on the Anthropic-direct
  client in the window, and the metered-spend metric does not increment
  (flat-fee rows are excluded by design at `llm_client.py:432` -- cite the
  actual behavior, and remember `session_cost_usd` is a gauge: never SUM it).
- **E3 -- model truth:** for at least one rail call, capture the CLI envelope's
  `modelUsage` (it is a MAP -- iterate/sum across ALL keys, never take the
  first) and show the model that did the work equals the configured tier AND
  equals what `llm_call_log` recorded. This is the phase-78 defect class;
  the smoke must show it stays fixed.
- **E4 -- output integrity:** the analysis result is schema-compliant and
  non-degraded (no synthetic 0.0/HOLD, no `final_synthesis.error`), and the
  decision path consumed it (a signals/strategy_decisions row or equivalent).
- **E5 -- guard health:** rail-guard probe passed, breaker not tripped,
  `consecutive_failures == 0` at end of window.
- **E6 -- quota math for the operator:** per-call tokens + would-be costUSD
  from the envelopes, extrapolated to a full 13-ticker cycle x current cycle
  cadence, expressed as %-of-weekly-Max-pool per day. This number is the
  actual answer to "can the app live on the plan without starving the dev
  harness" -- the readout is incomplete without it.
- **E7 -- UI (only if any claim touches the UI):** Playwright capture behind
  the NextAuth wall per the standing rule; otherwise state "no UI claims".
- **E8 -- trade-cadence baseline (pre-registered BEFORE the flip):** trades
  and round-trips per week over the trailing 4 weeks from
  `financial_reports.paper_trades` (bounded, date-filtered query), written
  into the contract as the baseline the observation window is judged against.
  Pre-registering it before the flip is what makes the post-flip comparison
  evidence instead of narrative.

## 6. Candidate immutable criteria -- freeze ONLY after a measured dry pass

Per `feedback_immutable_criteria_must_be_green_able`: run the verification
command and see it green BEFORE writing it into masterplan.json. Candidates:

1. `scripts/qa/smoke_cc_rail_e2e.py` exits 0 against the running backend with
   all E1-E6 checks individually PASS (machine-readable output).
2. E1 count rule: 100% of Claude-role rows in the smoke window match the rail
   marker rule (state rule + both counts).
3. E3: envelope-observed model == configured tier == llm_call_log model for
   >=1 sampled call.
4. Metered-spend delta over the window == 0.
5. Flag doctrine text updated in `settings.py` + operator token recorded in
   `handoff/harness_log.md` (only if the operator chose keep-ON).

Express repo-wide facts as measured DELTAs over the smoke window, never as
absolute repo states that 128 unrelated things can redden.

## 7. Kill / rollback criteria (pre-registered)

- Probe fails or breaker trips during the smoke -> FAIL, restore flag, queue
  defect with the envelope stderr; do NOT retry-loop against the live rail.
- E3 mismatch (envelope model != logged model) -> FAIL; this is a truth
  defect, not a smoke flake. Queue its own research-gated step.
- E6 shows a full cadence would exceed ~30% of the weekly pool -> smoke can
  still PASS technically, but keep-ON becomes an explicit operator decision
  with the number in front of them -- never default to ON.
- Any metered increment observed -> FAIL regardless of other checks.

## 8. Explicitly OUT of scope for goal 4000

- **Migrating the 28-agent Gemini pipeline onto the rail.** Google Search
  Grounding is Gemini-only and degrades on Claude; the Gemini 2.5 family
  retirement (2026-10-16) forces a decision soon anyway. E6's quota math is
  the INPUT to that decision. If the operator wants it, it is a separate
  research-gated goal ("4001") with a decision brief before any code.
- Re-testing the 76.9.2 SDK bridge (already E2E-tested; different mechanism).
- `real_capital_enabled` interplay beyond amending the flag description.
- Any model-tier re-pinning (that is 79.55, an operator judgment).

## 9. Suggested step split (harness protocol applies in full to each)

- **4000.1** Research gate + baseline measurement: live flag state, current
  llm_call_log rail-marker shape, cheapest bounded entrypoint, pre-registered
  E1-E6 check definitions with their selection rules written down. No code.
- **4000.2** Build the smoke script + dry-run it with the flag in its CURRENT
  state (expected: it correctly reports "rail OFF" -- proves the checks can
  fail). Mutation-test at least E1 and E2 (a guard that cannot fail does not
  count; mutate the production seam, not a helper).
- **4000.3** Operator gate: 79.55 answer + flag-flip token. Then the live
  smoke window, evidence capture, restore-or-keep, readout with E6 math.
- **4000.4** If keep-ON: amend the flag description + CLAUDE.md note, record
  the operator token, and add the steady-state monitoring hook (the 66.1
  guard already pages; verify it is wired on the now-permanent path).
- **4000.5** Observation window (7-14 days, rail ON): trade-cadence readout
  vs the E8 baseline. Report round-trips/week, projected weeks-to-100
  round-trips at the observed cadence, and Max-pool burn vs the E6 estimate.
  If cadence has NOT moved, do not iterate here -- queue the co-suspects
  from Section 1 (paper_analyze_top_n, silent BUY-gates, monosector funnel)
  as separate research-gated steps with the measured evidence attached.

## 10. Standing rules that bite on exactly this goal (do not rediscover them)

- Research gate before every step, even "we've been here before" (it is true
  that phases 61/66/76/78 covered this ground -- tier may be `simple`, the
  phase may not be skipped).
- Q/A via the Workflow rail (`.claude/workflows/qa-verdict.js`); persist the
  verdict the turn it returns; lean prompts (the rail drops ~180K-token ones).
- Write-first on all subagent briefs.
- `git add -An` before every status flip (the hook ships the whole tree).
- Masterplan edits via Write/Edit only, never Bash (the auto-commit hook
  matches Write/Edit).
- No emojis anywhere.
