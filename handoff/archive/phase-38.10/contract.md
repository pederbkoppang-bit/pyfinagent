# Cycle 10 Contract -- Step 38.10 (Slack digest regression -- evidence-only closure)

**Generated:** 2026-05-27T19:32+02:00.

**Step id:** `38.10` -- Slack digest regression: Portfolio +$0.00 (+0.0%) + Recent Analyses 0.0/10.

**Cycle class:** Evidence-only verification. NOT a trading-policy change. NOT a code change.

## Research gate
- Researcher: `a2c9fc63d5bb07df7`, tier=simple-to-moderate, gate_passed=true.
- Output: `handoff/current/research_brief_phase_38_10_slack_digest.md`.
- Sources read in full: 5 (Pythonworld None-handling, Sambath sentinel objects, APScheduler official docs, Promethium data observability 2026, Slack Block Kit designing).
- URLs collected: 15.
- Recency scan: performed (three-variant query discipline).
- Internal files inspected: 9 with file:line anchors.

## Findings from researcher
**The operator-flagged bugs are STALE DATA, NOT a serialization defect.**

The phase-71 commit `b9a1b772` (2026-05-22) shipped:
1. `formatters.py:342-344` nested-envelope unwrap (`portfolio_data.get("portfolio") if isinstance(...) else portfolio_data`).
2. `autonomous_loop.py:1309-1310` final_weighted_score -> final_score fallback.
3. `bigquery_client.py:264-268` + `api/models.py:96` column alignment.

The phase-72 commit added the `(as of close YYYY-MM-DD)` label at `formatters.py:359-360,414`.

The operator's 2026-05-26 23:47 screenshot captures a digest sent BEFORE these fixes propagated to the live slack-bot process. The live system, post-fix, renders correctly.

## Hypothesis
Running `format_morning_digest()` + `format_evening_digest()` against the CURRENT live API responses will produce Block Kit output with non-zero Portfolio $ + non-zero scores. If this is true, step 38.10 closes with evidence, no code change.

## Plan steps
1. Probe live `/api/paper-trading/portfolio` and `/api/reports/?limit=5` for current data shape.
2. Import formatters and run `format_morning_digest(portfolio, reports)` + `format_evening_digest(portfolio, [])` against live data.
3. Capture verbatim Block Kit output to `live_check_38.10.md`.
4. If Portfolio + Recent Analyses both show non-zero values -> closure satisfied per success criteria.

## Success criteria (verbatim from masterplan 38.10)
- morning_digest_portfolio_dollars_nonzero_when_NAV_is_nonzero
- evening_digest_portfolio_dollars_nonzero_when_NAV_is_nonzero
- recent_analyses_scores_reflect_actual_final_score_field_not_0.0
- live_check_38_10_quotes_a_post_fix_slack_message

## Out-of-scope (researcher-flagged)
- Optional follow-up: `scheduler.py:199-220` lacks `misfire_grace_time` / `coalesce=True` (cosmetic robustness improvement; does NOT cause the regression).
- Stale-bytecode hypothesis: if regression reproduces on a POST-fix digest, `launchctl kickstart` the slack-bot daemon (matches `feedback_npm_install_requires_launchctl_kickstart.md` pattern).

## Memory-rule compliance
- ZERO code changes (evidence-only cycle).
- ZERO new deps.
- ZERO emojis introduced (Slack emoji codes like `:sunrise:` are intentional Block Kit syntax, not Unicode emoji).

## References
- `handoff/current/research_brief_phase_38_10_slack_digest.md`
- `backend/slack_bot/formatters.py:323-388` (format_morning_digest)
- `backend/slack_bot/formatters.py:391-430` (format_evening_digest)
- `backend/tests/test_phase_slack_digest_71.py` (316 lines, all three regressions covered by automated tests)
