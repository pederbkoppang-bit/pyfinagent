# phase-4.9.9 red-team audit

Run at: 2026-04-19T06:12:14.445825+00:00

| Test | Pass | Evidence |
|------|------|----------|
| unsigned_mutation_blocked | True | verify_limits_tag step present in .github/workflows/limits-tag-enforcement.yml -- CI blocks limits.yaml diffs lacking a GPG-signed annotated tag (phase-4.9.1). |
| bad_strategy_blocked | True | promote_strategy raised=PromotionBlocked | msg=no gauntlet report for strategy 'intentionally_bad_strategy' at /Users/ford/.openclaw/workspace/pyfinagent/handoff/gauntlet/intentionally_bad_strategy/report.json; added to 30-day  | blocklist_rows=1 at /var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/phase4_9_redteam_blocklist.jsonl |
| evidence_logged | True | (see bad_strategy_blocked row above) |

Overall: REDTEAM_PASS
