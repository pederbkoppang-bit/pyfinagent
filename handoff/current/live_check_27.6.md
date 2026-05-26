# Live Check -- Step 27.6 -- BLOCKED on cycle-timeout-vs-rail mismatch (2026-05-27 01:20)

## STATUS: BLOCKED (not PASS, not FAIL — structural blocker discovered)

The Claude Code rail (operator-approved cycle-5 ship) is operationally
functional: 225 successful `claude_code_invoke` calls / 0 errors / 0
"Full orchestrator failed" lines / `model=claude-sonnet-4-6` confirmed.
BUT the autonomous-loop's hard wall-clock budget `paper_cycle_max_seconds`
(currently 3600s = 1 hour) is too tight for the Claude Code rail's
per-call latency (~30s × ~7 calls/ticker × 13 tickers >= 45 min, plus
serial dependencies inside the orchestrator). The cycle TIMED OUT at
exactly 3600s with 7 of 13 tickers analyzed.

Quoting the autonomous_loop terminal log line verbatim:
```
00:50:08 E [autonomous_loop] Paper trading cycle TIMED OUT after 3600s
```

Step 27.6 cannot close PASS until either (a) `paper_cycle_max_seconds`
is increased to ~7200s or 10800s (settings change; no code), OR (b) the
Anthropic API direct rail is restored (operator credit top-up), OR
(c) the universe is reduced.

## Step under verification

`27.6 — End-to-end smoke verify: full path on Claude` (P0)

## Per-criterion evidence (final state, cycle 6)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | model = claude-sonnet-4-6 via settings API | **PASS** | `curl /api/settings/` returns `gemini_model: claude-sonnet-4-6` |
| 2 | full cycle completed status=completed | **FAIL** | cycle TIMED OUT after 3600s; status=timeout |
| 3 | lite_mode=False in Step 3 log | **PASS** | `23:50:32 Step 3 -- Analyzing 4 new + 9 re-evals (lite_mode=False)` |
| 4 | zero "Full orchestrator failed" lines | **PASS** | `grep -c "Full orchestrator failed"` returns 0 in the fresh-cycle window |
| 5 | min 14/15 analyses persisted to analysis_results | **FAIL** | 7 of 15 universe tickers persisted (AMD, CIEN, GEV, KEYS, MU, QCOM, STX); cycle timed out before completing the remaining 6-8 |
| 6 | OutcomeTracker step 9 attempted | **UNKNOWN** | cycle timed out before reaching step 9 (per the 15-step pipeline order) |

Three criteria PASS, two FAIL, one UNKNOWN. 27.6 cannot close PASS.

## Cycle 6 fresh-cycle counters (verbatim)

```
cycle_id:         (timed-out cycle, started 2026-05-26T23:50:07+02:00)
claude_code calls started:    225
claude_code calls succeeded:  225 (100% success rate; no rail-side errors)
claude_code errors:           0
--max-tokens rejections:      0 (cycle-5 fix held)
Full orchestrator failed:     0 (rail bypassed the credit-exhausted direct API)
BQ rows persisted:            7 tickers in analysis_results (AMD, CIEN, GEV, KEYS, MU, QCOM, STX)
Final outcome:                cycle TIMED OUT after 3600s during DELL Alt Data Agent
```

## Root cause: rail latency vs cycle wall-clock budget

The Claude Code rail's CLI-subprocess overhead (~30s/call) is ~5×
slower than the previous `api.anthropic.com` direct rail. The
orchestrator's per-ticker pipeline ALSO has serial dependencies
(enrichment → debate → risk → synthesis) that can't trivially
parallelize across tickers. With 13 tickers and ~7 LLM calls each
running serially, the math is:
  13 tickers × ~7 calls × ~30s/call ≈ 2700-3500s in the best case
  +
  per-ticker serial dependencies (enrichment must finish before debate,
  debate before risk, etc.) effectively serializes the pipeline within
  each ticker, blocking parallel ticker fan-out.
  
Net result: a full-orchestrator pass with 13 tickers DOES NOT fit
inside the 3600s wall-clock. The cycle timed out completing 7 tickers.

## Operator path forward (one of)

1. **Bump `paper_cycle_max_seconds` to 7200s (2h) or 10800s (3h).** Lowest-friction; no code change beyond exposing the field in `backend/api/settings_api.py` allow-list. Adds masterplan step `38.12` (queued below).
2. **Restore Anthropic-direct rail.** Requires operator to top up Anthropic credits + flip `paper_use_claude_code_route=false` via settings UI. Returns the loop to the credit-exhaustion failure mode of cycles 1-4 unless credits are kept fresh.
3. **Lite-mode for the testing phase.** Flip `lite_mode=true` via settings UI; reduces per-ticker calls from ~7 to ~2. Trade-off: violates 27.6 criterion #3.
4. **Reduce universe.** Drop `paper_screen_top_n` + `paper_analyze_top_n` so fewer tickers per cycle. Trade-off: violates 27.6 criterion #5 (must hit 14 of 15 SP500-eligible tickers).

Recommended path: **option 1 (timeout bump)**. Operator-low-risk; preserves the testing-phase Max-subscription cost model.

## Why this artifact is BLOCKED (not PASS, not FAIL)

This artifact is HONEST evidence of the cycle's state. Two criteria
fail; that would normally be a FAIL verdict. But the failure is
structural (timeout-vs-rail mismatch), not a code defect. Path forward
is a single settings flip (option 1) which Main can implement in
cycle 7. Until that ships, 27.6 stays pending.

Masterplan step `27.6.status` STAYS `pending`. Main is HOLDING the flip.

## Cycle 6 commit + Q/A trail

- Researcher (borrowed): cycles 3, 4, 5 (no new external research; cycle 6 was operational closure, not a code-change cycle).
- Contract: `handoff/current/contract.md` (cycle 5 + 6 consolidated).
- Generate: this artifact + cycle-5 code now live in the backend.
- Q/A: pending (next agent spawn).
- Log: pending (appends to `handoff/harness_log.md` after Q/A).
- Commit: pending (after harness_log).

## Cross-references

- Cycle 5 rail-verification artifact: `handoff/current/live_check_cycle_5_rail_verification.md`.
- Researcher (cycle 4): `handoff/current/research_brief_phase_claude_code_stdin_fix.md`.
- Researcher (cycle 3): `handoff/current/research_brief_phase_claude_code_routing.md`.
- The `paper_cycle_max_seconds` field declaration: `backend/config/settings.py:31`.
- The timeout enforcement: `backend/services/autonomous_loop.py:219, :226, :1128`.
