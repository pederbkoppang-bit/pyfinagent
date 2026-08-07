# Operator ask list — 2026-08-07 (full-day autonomous drain)

One row per decision. A recorded DECLINE closes the linked step as validly as
doing it. Items accumulate during the day; recommendations are the executor's.

| # | Decision | Linked steps | Recommendation |
|---|---|---|---|
| 1 | PROMOTE-66.2 flags (approved 2026-07-09, never applied) | 79.1 | Apply — approval already recorded, only the mechanical write is owed |
| 2 | Anthropic direct-API credit decision (metered rail dead since 05-17) | 79.3, 27.6, 85.1 | Decide either way; the Max-rail migration (85.1) reduces the stake |
| 3 | AUTORESEARCH_USE_MAX_RAIL=1 | 79.4 | Apply — routines are $0 on credits-off basis (phase-85 measurement) |
| 4 | OPS-BRIDGE-BOOTSTRAP + claude-code-proxy plist rebind | 79.5 | Operator-only, batch with next at-machine session |
| 5 | Phase-69 activation tokens still owed: KS-PEAK-RESET, sign_safe_overlays, regime_net_liquidity, historical_macro un-freeze | 79.6 + phase-69 record | No change urged today; re-listing so they are not lost |
| 6 | Alpha Vantage archive: point-in-time INVALID for backtests (19x density discontinuity) + non-commercial ToS. Use anyway for any purpose? | 83.0 scope (c) | Recommend NO for backtest corpus; GDELT/EDGAR are measured-clean. Corpus writer ships source-agnostic so this decision stays open without rework |
| 7 | Remaining phase-79 operator-only steps (53 pending, harness_required:false) | phase-79 | Reviewed in batch at next at-machine session; none executor-actionable |

| 8 | Qualified DELETE of test pollution in `pyfinagent_data.news_articles`: 9 `source='stub'` rows (phase6_e2e runs) + 2 `source='fixture'` rows (83.0.1 live_check: the 2022 backfill row + the quarantined NULL row), plus 9 matching `news_sentiment` rows ($0 spent, 6 are scorer error records). Purge queued as step 83.0.7 with WHERE-qualified predicates (`source IN ('stub','fixture')`) after the streaming buffer drains. | 83.0.7 | Approve the qualified DELETE (never unqualified); executor runs it under 83.0.7's live_check |

## Recorded during the day

- 82.23 closed as `superseded` by 82.27 (no decision needed — operator already
  chose the re-spec 2026-08-04; listed for visibility). Commit cb4b3c52.
