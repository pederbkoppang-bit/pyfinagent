# Triage: phases 63/64/65 (away backlog) -- 66.5 deliverable, Cycle 70, 2026-07-07

Status: AWAITING OPERATOR SIGN-OFF. No masterplan edit takes effect until the
operator approves (reply in-session or `TRIAGE 63-65: APPROVED` /
`TRIAGE 63-65: AMEND <notes>` via the bot channel). Basis:
research_brief_66.5.md (ground-truth audit 2026-07-07 + SRE/testing canon).

## Summary: 12 KEEP (5 re-anchored), 2 MERGE, 0 DROP

The away plan's substance survives; what died was its cadence (PM-slot/digest
wiring) and one diagnosis superseded by 66.2's broader version. Everything is
sequenced BEHIND the 66.1->66.2 P0 chain (stop starting, start finishing).

## Disposition table

| Step | Disposition | One-line rationale |
|---|---|---|
| 63.1 Playwright walk of 22 routes | KEEP | Operator's screenshot bugs remain unaddressed; 22 routes verified still accurate; attended cadence replaces AM-slot wiring. |
| 63.2 BQ cross-check of displayed numbers | KEEP (re-anchored) | Valid as designed, but run AFTER 66.2 resolves data staleness (historical_macro ~103d, paper_trades freeze) or every number-check false-positives on stale inputs. |
| 63.3 Verified defect register | KEEP (seeded) | File absent; seed with the 8 defects already found: _resolve_claude_binary docstring mismatch (claude_code_client.py:56), .env-bleed test isolation (61.1), auth-latch paged:false no-retry (66.4 note), auto-commit hook silent stalls (12 INVOKED / 0 pushes on 07-06 alone), changelog trailing-commit race, historical_macro ~103d stale, alpaca short_market_value -13,842.89 (66.2 carries), paper_portfolio single-US-row (66.2 carries). |
| 63.4 Fix queue execution | KEEP (re-anchored) | "One fix per AM slot" becomes "one fix per harness cycle, register-priority order" under attended operation. |
| 63.5 Regression re-walk | KEEP | Unchanged; end-of-queue gate. |
| 64.1 Functional-E2E Playwright project | KEEP (resequenced) | tests/e2e-functional absent, config has ONE project -- step is still greenfield; stabilize-before-you-pave: build AFTER the engine trades again (post-66.2), weekends as designed. |
| 64.2 Functional specs for 22 routes | KEEP (absorbs 64.5 nightly-runner leg) | Suite + its nightly execution belong together; <15-min budget unchanged. |
| 64.3 Backend gap tests | KEEP | Kill-switch state machine + currency paths remain untested; unaffected by away-mode assumptions. |
| 64.4 Multi-market fixture e2e | KEEP (repoint dependency) | depends_on 65.1 must repoint to 66.2 (funnel counters now land there). |
| 64.5 CI wiring + nightly runner | MERGE -> 64.2 (CI leg) | Nightly-runner leg was PM-session-cadence-specific; CI leg (credential-free subset in e2e-smoke.yml) folds into 64.2's definition of done. |
| 65.1 EU zero-trades funnel diagnosis | MERGE -> 66.2 | Subsumed: 66.2's immutable criterion already demands per-gate candidate counts for ALL markets (the freeze is no longer EU-specific); 65.1's per-ticker counter design carries over as 66.2 input. |
| 65.2 Per-market screener thresholds (dark) | KEEP | Still the right dark fix IF the 66.2 funnel shows per-market screener rejection; gated on that diagnosis. |
| 65.3 US+KR health baseline | KEEP (re-anchored window) | Since-06-01 window is ~70% trade-freeze; re-anchor to a clean window (e.g. 05-01..06-10 pre-freeze + post-reactivation) or the baseline mostly measures the outage. |
| 65.4 Three-market proof | KEEP (wall-clock re-gated) | Still the goal's capstone proof; week-3 wall-clock gate re-anchors to "3rd week after 66.2 closes". |

## Proposed masterplan edits (EXACT; applied only after sign-off)

1. 65.1 -> status "merged" + note "subsumed by 66.2 funnel criterion (triage 66.5)".
2. 64.5 -> status "merged" + note "nightly-runner leg -> 64.2; CI leg folded into 64.2 name/criteria".
3. 64.4 depends_on_step: "65.1" -> "66.2".
4. 64.2 name += "; absorbs 64.5 (CI wiring + nightly execution)".
5. 63.2/63.4/65.3/65.4 names prefixed "(post-66.2)" re-anchor notes; no criteria changes (immutable).
6. No step deleted; no status flipped to done.

## Operator questions (answer with sign-off)

- Q1: Approve the table + edits above? (`TRIAGE 63-65: APPROVED` or in-session.)
- Q2: Away plists (away-session-am/pm, away-watchdog) are STILL ARMED and firing
  daily. Keep them running as the autonomous execution engine while you're home
  (they now do useful masterplan work at $0 marginal, and the PM slot can close
  66.1 tonight), or disarm to operator-driven-only? Recommendation: KEEP ARMED --
  they are the mechanism that closes wall-clock-gated evidence without you.
- Q3: 63.3's register seeds include the auto-commit hook stalls (12 INVOKED / 0
  pushes tonight) -- promote to an early fix-queue item? Recommendation: YES, it
  costs manual pushes every cycle.
