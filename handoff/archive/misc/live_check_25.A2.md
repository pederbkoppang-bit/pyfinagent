# Live-check placeholder — phase-25.A2

**Step:** 25.A2 — Wire bq.save_report into full pipeline
**Date:** 2026-05-12

## Live-check field
> "Frontend /reports page shows non-zero recent rows after next full-pipeline cycle"

## Pre-deployment evidence
- 8/8 verifier PASS
- Full path now returns `_path: "full"`; persist guard accepts both
- Function renamed; all callsites updated; zero legacy refs
- Same `_persist_analysis` helper persists to BQ `analysis_results` via existing `bq.save_report`

## Post-deployment verification
1. Set `lite_mode=False` in `.env` (or settings)
2. Trigger autonomous cycle
3. Visit frontend `/reports` page → expect non-empty list
4. Or query directly: `SELECT * FROM financial_reports.analysis_results WHERE analysis_date > '<deploy>' ORDER BY analysis_date DESC LIMIT 5`

**Audit anchor for next bucket:** 25.A (decouple RiskJudge with independent LLM call).
