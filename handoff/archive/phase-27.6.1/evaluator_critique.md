# Evaluator Critique — phase-27.6

Q/A subagent: `qa` (aa86f7cd0a7722772), 2026-05-17, single pass on cycle-#9 evidence.

## Verdict: CONDITIONAL

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "violated_criteria": [
    "zero_Full_orchestrator_failed_lines_for_the_cycle",
    "contract_pre_generate_present_and_27.6_specific",
    "experiment_results_27.6_authored"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "state": "14 of 14 tickers hit 'Full orchestrator failed', lite Claude rescued all 14",
      "constraint": "zero_Full_orchestrator_failed_lines (masterplan 27.6 success_criteria)",
      "severity": "BLOCK"
    }
  ]
}
```

## What's GOOD

- Immutable verification command rc=0 (cycle_id `d73f5129`, lite_mode False, analyses_persisted 14).
- **phase-27.1 schema fix INDEPENDENTLY CONFIRMED CORRECT on Claude**: zero `additionalProperties` errors in 14+ Anthropic API calls. This is the most expensive prior work and it's banked.
- BQ persistence: 14 unique tickers in cycle window (audit-trail clean).
- Status: `completed` (no timeout, no cost-budget trip). Lite Claude fallback (per 27.3) caught every full-path failure.
- Anti-self-match line is honest one-line addition, not coincidence-grep gaming. Anti-rubber-stamp confirmed.

## What BLOCKS the flip

1. **`zero_Full_orchestrator_failed_lines` strictly violated** — 14/14 tickers failed full path. Two NEW orthogonal upstream bugs surfaced (not regressions of 27.1-27.5):
   - **SEC EDGAR 429** on `https://www.sec.gov/files/company_tickers.json` — 8 tickers. Concurrency=8 firing simultaneous bulk-file downloads without a courtesy User-Agent or per-host throttle. Real code defect under SEC fair-access policy.
   - **QuantAgent `'NoneType' object has no attribute 'get'`** — 6 tickers (STX, COHR, INTC, DELL, SNDK, WDC). Claude-pathway-specific data flow returns None where a dict was expected. Cycle #8 Gemini had 0 of these.
2. **contract.md is stale** (dry-run optimizer Cycle 1 shell from 02:25, not 27.6-specific) — protocol breach per `feedback_contract_before_generate.md`.
3. **experiment_results.md is stale** (27.5 trilogy content) — same protocol breach.

## Required follow-up steps (proposed)

- **phase-27.6.1** — SEC EDGAR rate-limit + courtesy guard. Cache `company_tickers.json` for 24h (changes daily) + per-host throttle (≤2 simultaneous to `sec.gov`) + User-Agent header per SEC policy.
- **phase-27.6.2** — QuantAgent NoneType safety. Defensive `.get()` / `or {}` on upstream dep that returns None on Claude path. Regression test for STX/COHR/INTC/DELL/SNDK/WDC.
- **phase-27.6.3** — Re-run Claude full-path smoke. Same 27.6 gate but with both fixes; success_criterion `zero_Full_orchestrator_failed_lines` must hold cleanly. Authors proper 27.6.3-specific contract + experiment_results + live_check.

**Recommendation:** do NOT flip 27.6 done. Inject sub-steps, fix, re-run, fresh Q/A.
