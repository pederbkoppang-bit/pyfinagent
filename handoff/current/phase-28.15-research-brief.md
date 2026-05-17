# phase-28.15 Research Brief — Social media velocity in screener
**Date:** 2026-05-17
**Tier:** simple
**Step:** phase-28.15 (Candidate Picker Expansion — lift existing social_sentiment.py velocity into screener pre-filter)
**Audit basis:** supplement Gap 2; existing backend/tools/social_sentiment.py at line 95 already computes velocity = recent_avg - older_avg but is wired to Layer-1 enrichment only. 2025 DNUT case: 500% StockTwits spike preceded 90% pre-market move.
