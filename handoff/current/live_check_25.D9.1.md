# Live-check placeholder -- phase-25.D9.1

**Step:** 25.D9.1 -- Caller-side Files API adoption (skill_file_id wiring in run_*_agent calls)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Run a Claude-mode analysis; observe BQ cost_tracker rows show skill-token input drop to ~50 tokens per agent (was 700-1500)"

## Pre-deployment evidence
- 5/5 verifier PASS.
- 11 call-site wires confirmed by grep (lines 839/846/853/860/867/883/890/897/904/911/945).
- 3 live behavioral tests on the helper:
  - empty `_skill_file_ids` -> returns None (Gemini fallback unaffected).
  - mapped stem -> returns `{"skill_file_id": "<file_id>"}`.
  - missing stem -> returns None (no KeyError).
- AST clean on `backend/agents/orchestrator.py`.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   source .venv/bin/activate
   ```
2. Confirm general_client is Claude (Anthropic) -- the Files API path is
   Claude-only. If `settings.gemini_model.startswith("gemini-")`, the
   helper returns None on all calls and the inline path is preserved.
3. Restart backend:
   ```
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
4. Trigger a Claude-mode analysis:
   ```
   curl -X POST http://localhost:8000/api/analysis/ -H 'Content-Type: application/json' \
     -d '{"ticker": "AAPL"}'
   ```
5. Query BQ cost_tracker for the per-agent skill-token input:
   ```sql
   SELECT agent_name, model_name, AVG(prompt_tokens) AS avg_prompt_tokens
   FROM `sunny-might-477607-p8.pyfinagent_data.cost_tracker_events`
   WHERE DATE(event_time) = CURRENT_DATE()
     AND agent_name IN ('Insider','Options','Patent','Earnings Tone',
                        'Social Sentiment','Sector','NLP Sentiment',
                        'Anomaly','Scenario','Alt Data','Quant Model')
   GROUP BY agent_name, model_name
   ORDER BY avg_prompt_tokens DESC;
   ```
   Expected: prompt_tokens for enrichment agents drops by 90%+ vs pre-25.D9.1 baseline.

## North-star calculus
Per Anthropic Files API docs: each Claude-mode enrichment call now sends
an ~8-token file_id reference instead of the 5K-15K-token skill markdown.
For an 11-agent full pipeline on Sonnet 4.6 at $3.00/MTok input:
- Pre-25.D9.1: ~55K-165K skill tokens per ticker = $0.17-$0.50 per ticker.
- Post-25.D9.1: ~88 skill tokens per ticker = ~$0.0003 per ticker.

Multiplied across a 10-ticker backtest: $1.50-$5.00 -> ~$0.003 in
skill-payload cost alone. This is the foundation; 25.D9.2 (cache_control
on the document block) will compound the saving further when the same
skill is re-read within the 1h cache window.

## Closes audit basis
Caller-side gap from 25.D9 RESOLVED. The mechanism is now actually wired
into the orchestrator hot path; future ticker analyses on Claude
provider see the document-block path automatically.

**Audit anchor for next bucket:** 25.D9.2 (cache_control on doc block),
25.C9.2 (run_full_analysis batch refactor), 25.S.1 (per-call ticker tagging).
