# Active Goal -- goal-phase66-reactivation (primary)

Refreshed 2026-07-06, return-day session. Operator BACK (in-session approval "do it");
the away window (2026-06-15..07-06) and its rails are OVER. Normal attended rules apply:
CLAUDE.md harness protocol per step, LLM API (metered) costs still require Peder's
approval, no feature branches, push to main.

## Primary: goal-phase66-reactivation (masterplan phase-66; spec: handoff/current/goal_phase66_reactivation.md)

Why: portfolio is 100% cash at NAV $23,997.71 -- zero expected alpha until reactivated.
The Claude decision rail (cc_rail) was dead 06-15..07-06 (ECONNRESET, then 401 expired
OAuth) with no circuit breaker and no page; the same credential killed 34 consecutive
away sessions. Two outcomes: (1) the engine analyzes and trades again through its normal
gates; (2) any decision-path or credential death pages within one cycle, forever.

Strict step order (do not start N+1 while N has an unmet P0 criterion):
- 66.0 recovery re-baseline (backlog pushed, pending_tokens dispositioned, recovery loop exited)
- 66.1 rail restore: probe-gate + circuit breaker + single P1 page; fallback dark
- 66.2 redeploy capital via the NORMAL path only (first honest BUY or verified funnel diagnosis)
- 66.3 cost-truth (phantom $0.50 failure-cost fix; sentinel = dollars actually billed)
- 66.4 credential-expiry paging within 24h (drill-proven)
- 66.5 phase-63/64/65 triage (planning-only, operator sign-off)

Boundaries (binding): trailing-stop engine untouchable; hysteresis family banned absent
`HYSTERESIS: AUTHORIZE`; trading behavior changes config-gated default OFF; never
manufacture trades or evidence; scheduled-run evidence for scheduled-job claims (39.1
lesson); progress claims cite tool results.

## Prior goals -- state

- goal-away-ops (phases 62-65): 62.1 done; 62.2/62.6/62.7 pending (62.2 needs the
  operator `TEST TOKEN: PING`); phases 63/64/65 executed 0% (credential death) -- their
  disposition is 66.5's job. Away rails (away-ops-rules.md) NO LONGER BINDING except
  where restated above; away plists remain loaded and harmless (healthy sessions on a
  clean tree do one masterplan step per the standing prompts).
- goal-phase61-churn-integrity: 61.1 done (Cycle 66); 61.2-61.5 pending -- resume AFTER
  phase-66 P0s (a churn fix is worthless while the book is 100% cash).
- phase-58.1 $25 window: expired during the away window; do not disturb its artifacts.

## Open operator asks (handoff/away_ops/pending_tokens.json)

MAS-PLIST resolved (mv done 2026-07-06). Still open: TEST-TOKEN-62.2 (`TEST TOKEN: PING`
in C0ANTGNNK8D), WEBHOOK, AUTORESEARCH-SPEND, FRED key rotation (now due), SDK-CREDIT
(re-decide before any next away window). METERED-BREACH closes via 66.0 note + 66.3 fix.

## Cycle ledger

- 2026-07-06: return-day analysis (100%-cash + credential-death findings); phase-66
  installed; backlog sweep commit; 66.0 cycle begins.
