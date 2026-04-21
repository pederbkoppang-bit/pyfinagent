# Experiment Results — phase-7 / 7.7 (Revelio license doc)

**Step:** 7.7 **Date:** 2026-04-20 **Cycle:** 1.

One new doc: `docs/compliance/revelio-license.md` (~140 lines, 8 sections matching reddit-license.md template with Revelio-specific content). Second per-vendor doc under the framework introduced in phase-7.5.

```
$ test -f docs/compliance/revelio-license.md && echo "DOC OK"
DOC OK

$ python3 -c "open('docs/compliance/revelio-license.md','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Immutable criterion `test -f docs/compliance/revelio-license.md` → PASS.

Key deltas vs reddit-license:
- MSA-based contract (not OAuth click-through).
- GDPR Art. 28 DPA explicit.
- Batch delivery (S3/Snowflake/GCS), no per-request rate limit.
- Workforce Dynamics is company-aggregate; Sentiment has quasi-identifiers.
- In-scope datasets: Sentiment Analysis + Workforce Dynamics only (not all 6 Revelio products).
