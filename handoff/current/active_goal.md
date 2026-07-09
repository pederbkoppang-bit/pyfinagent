# Active Goal -- goal-phase67-fable-window (PRIMARY through Sunday 2026-07-12) + goal-phase68-real-fill-runway (queued)

Refreshed 2026-07-10 (operator /goal installed phase-68). Attended rules apply:
CLAUDE.md harness protocol per step, metered LLM costs need Peder's approval, push to
main, no feature branches.

## Primary: goal-phase67-fable-window (masterplan phase-67; spec: handoff/current/goal_phase67_fable_window.md)

Why: FREE Fable 5 on the Max plan until ~Sun 2026-07-12. Fable is the UPGRADE ENGINE,
not the runtime model: it audits + rewrites the harness/MAS artifacts during the window;
the improvements persist on Opus 4.8 after the pins revert. Deliverable = durably better
researching, bug-catching, and does-it-actually-run verification.

Strict step order:
- 67.1 Q/A verification depth (P0): retire dead 55s cap; backend lint + runtime-smoke
  gates; remove stop_hook_active auto-PASS; reconcile CONDITIONAL-recovery contradiction
- 67.2 bug-catching (P0, after 67.1): consumer-contract-break heuristic + fix the
  verified NameError at agent_definitions.py:396 (+ behavioral test)
- 67.3 researcher WRITE-FIRST discipline (P1): codify incremental-brief writing; prune
  stale scaffolding; NO floor weakened
- 67.4 revert (P0, SCHEDULED 2026-07-12): pins -> opus, KEEP all artifacts. Any session
  on/after 07-12 treats this as its top P0.

Boundaries (binding): $0 metered (Fable via free Max rail only; NO in-app Fable pins);
harness stays exactly 3 agents; full five-file protocol per step; research-gate floors
immutable; trailing stops / sector caps / kill-switch / gate thresholds untouched;
progress claims cite tool results.

## Rider in flight: 66.2 close (from Cycle-76 addendum)

Criterion-1(a) MET 2026-07-09 (scheduled cycle 603e287c: AMD+MU BUYs,
APPROVE_REDUCED recorded; evidence live_check_66.2.md section 9). Closing fresh Opus
Q/A spawned this session; on PASS: log-last Cycle 77 -> flip 66.2 -> phase-66 DONE ->
operator summary.

## Prior goals -- state

- goal-phase66-reactivation: 66.0/66.1/66.3/66.4/66.5 done; 66.2 closing (above).
  Engine IS trading again (2 positions ~$1,560). Optional operator convenience: add the
  synthesis-integrity + rj-shape flag lines to backend/.env (authorized, no longer
  blocking).
- goal-phase61-churn-integrity: 61.1 done; 61.2-61.5 pending -- resume AFTER the
  phase-67 window work (the Fable-tuned harness makes them cheaper/safer).
- goal-away-ops (62-65): 62.2/62.6/62.7 pending operator tokens; 63/64/65 dispositioned
  by 66.5.

## Open operator asks

TEST-TOKEN-62.2 (`TEST TOKEN: PING` in C0ANTGNNK8D), WEBHOOK, AUTORESEARCH-SPEND, FRED
key rotation (due), SDK-CREDIT (before any next away window). NEW: `FABLE PERMANENT:
AUTHORIZE` only if Fable should outlive 2026-07-12 on personal credits (default: revert).

## Cycle ledger

- 2026-07-09: /goal fable-window set; phase-67 installed; Fable pins applied (effect
  next session); audit findings verified (NameError repro, no-linter, dead 55s cap).
- 2026-07-09 evening: **66.2 CLOSED (Cycle 77) -> phase-66 COMPLETE, engine trading**;
  67.1 PASS (Cycle 78: lint+runtime-smoke gates, tiered budget, auto-PASS backdoor
  removed, recovery reconciled); 67.2 PASS (Cycle 79: consumer-contract-break
  heuristic, live NameError fixed + 5 tests); 67.3 PASS (Cycle 80: write-first
  codified, wrong DSR pruned).
- 2026-07-10: latest-features audit -> 67.6 PASS (Cycle 81: Fable/Sonnet-5 API traps
  de-mined incl. orchestrator twin; SDK pin 0.87->0.96 downgrade bomb defused; 14
  request-shape tests) + 67.5 PASS (Cycle 82: fallbackModel chain; SessionStart
  Sunday tripwire = 67.4 now SELF-ENFORCING; MCP approvals restored 7x
  Pending->Connected -- headless fleet had been MCP-blind since v2.1.196; doc truth).
  **Phase-67 remaining: 67.4 Sunday 2026-07-12 revert ONLY** (pins -> opus, KEEP all
  artifacts; tripwire-enforced) -- top P0 for any session on/after that date.
- 2026-07-10: **phase-68 Real-Fill Runway INSTALLED** (operator /goal; 8 steps
  pending; spec handoff/current/goal_phase68_real_fill_runway.md). QUEUED BEHIND
  phase-67 P0s; 67.4 untouchable. 68.0 research spawned same session (calendar-gated
  before Sunday). Tokens filed: ALPACA-RESET-68.2, EXEC-BACKEND-68.3 (decide after
  shadow drift), PAPER-LEARN-LOOP-68.4 (awaiting measured cost), STALE 35-44 amended
  onto the existing triage ask (68.7). Window plan: 68.5/68.4 briefs Sat; Sun commit
  briefs then 67.4; 68.1+ continues on Opus.
