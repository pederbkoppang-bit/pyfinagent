# Contract — phase-8.5.6 Promotion path to paper-live (5-day shadow)

**Immutable:** `python scripts/harness/autoresearch_promotion_test.py` exit 0.

Plan:
1. `backend/autoresearch/promoter.py` — `Promoter` with `promote`, `position_size`, `on_dd_breach`, `SHADOW_MIN_DAYS=5`, `DD_TRIGGER=0.10`.
2. `scripts/harness/autoresearch_promotion_test.py` — 3 cases matching criteria + aggregate PASS.
