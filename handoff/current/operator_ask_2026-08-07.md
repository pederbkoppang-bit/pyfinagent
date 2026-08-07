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

| 9 | 27.6 + 27.6.3 (P0 Claude full-path smoke) are executor-BLOCKED: they need a full-orchestrator cycle on Claude, which requires either the direct-API credit decision (#2 / 79.3) or the CC-rail promotion (4000.4, whose 79.55 flip was held for a quiet tree). Not skipped — sequenced behind that decision. | 27.6, 27.6.3, 79.3, 4000.4 | Decide #2, or authorize the 4000.4 rail-promotion branch; either unblocks both smokes |

| 10 | **Promote `paper_synthesis_integrity_enabled` (61.2)** — the fabricated-neutral defect is FIRING: 142 of 170 analysis rows in 40 days (83.5%) are `0.00/HOLD` masking `Failed to parse final report.`, last 2026-08-06. The fix has been built+dark since 07-08; C (company_name) is already live-proven. RISK TO WEIGH (AWS REL05-BP01): promotion routes synthesis failures to the lite fallback, which today carries a measured 8.2% of traffic (14/170 via $._path; post-promotion up to 91.8%) — the untested-fallback-becomes-primary hazard; the live_check will measure the post-promotion share. Companion flags `paper_position_recommendation_fix_enabled` (E — only WITH A, never alone) and the RiskJudge-ctx flag ride the same decision. Requires backend restart + 1 scheduled cycle for the post-fix live_check. | 61.2 | Promote A (+F); hold E until A's first clean cycle; decline = 61.2 stays CONDITIONAL (2 of 3 before auto-FAIL) |

| 11 | **85.2 criterion 7 is unreachable as written** — the research gate measured a full secretless run at 46 failed + 4 errors from causes OUTSIDE the step's scope (8 kill-switch-state failures via the TRACKED audit file, DefaultCredentialsError reaching unmarked live-BQ tests), so even with both in-scope blockers fixed (4 settings defaults + the slack-bolt 1.30 aiohttp extra removal) the workflow stays red, and C7 forbids closing on a red workflow. Both out-of-scope causes are queued (85.2.1, 85.2.2). | 85.2, 85.2.1, 85.2.2 | Either sequence 85.2 AFTER 85.2.1+85.2.2 (then C7 is reachable — recommended, no re-spec needed), or re-spec C7 the 82.27 way |

| 12 | **SECURITY (85.3.3 / DD-4)**: the away-watchdog launchd plist embeds a literal `CLAUDE_CODE_OAUTH_TOKEN` on disk outside the secrets store (found by the 85.3 research gate). The queued step moves it out and checks git history for exposure — if the plist was ever committed with the token, rotation is needed. No action needed until that check runs; listed now so the finding is not buried. | 85.3.3 | Expect a rotation decision if the history check comes back positive |

## Recorded during the day

- 82.23 closed as `superseded` by 82.27 (no decision needed — operator already
  chose the re-spec 2026-08-04; listed for visibility). Commit cb4b3c52.
