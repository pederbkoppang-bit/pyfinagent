# Sprint Contract -- Cycle 38: Batch fix 6 follow-ups

Research gate: done (3 agents in parallel, summaries in harness_log.md).

Fixes:
#1 Patent -> BQ `patents-public-data.patents.publications` drop-in.
#2 nlp_sentiment ERROR payload + rules fallback.
#3 Auth root cause: backend/api/auth.py:108-118 silent-None; require DEV_DISABLE_AUTH=1 to opt-in.
#4 Test-env auth: Pattern 3 cookie-signed JWE via Auth.js encode() + Playwright storageState.
#5 Slack digest_test.py module + env-aware exit.
#6 spacy/unstructured deferred to phase-4.8.8.

Success criteria: each fix independently verifiable; re-run 4.6.6 verbatim PASS; 4.6.3 still PASS; 4.6.4 still PASS; new auth path returns 401 when DEV_DISABLE_AUTH unset.

Plan: execute in order 3,1,2,5,4,6; then EVALUATE each; re-run 4.6.6; LOG; commit.
