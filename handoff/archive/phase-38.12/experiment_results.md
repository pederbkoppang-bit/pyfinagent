# Experiment Results -- Cycle 7 (closure correction)

**Date:** 2026-05-27
**Cycle 7 step shipped:** masterplan `38.12` "Bump paper_cycle_max_seconds for Claude Code rail" -- PASS, flipped done.
**Cycle 7 closure attempt for 27.6:** FAIL per Q/A `abbcca28fb3536a63`. 27.6 stays pending. Step `38.13` added as the load-bearing follow-up.

## What changed (2 modified backend files + masterplan changes)

### MODIFIED

1. `backend/config/settings.py` -- raised `paper_cycle_max_seconds` default 1800.0 -> 7200.0 (cycle-6 BLOCKED context + cycle-7 rationale in field-doc).
2. `backend/api/settings_api.py` -- exposed `paper_cycle_max_seconds` to the HTTP layer in 4 places (FullSettings, SettingsUpdate w/ `ge=300, le=21600`, `_FIELD_TO_ENV`, `_settings_to_full`).

### MASTERPLAN

3. `.claude/masterplan.json`:
   - `38.12.status` flipped `pending` -> `done` (with `closure_note` documenting that the timeout bump landed but didn't unblock 27.6).
   - `38.13` ADDED under phase-38 (P0, harness_required=True). Title: "Wire Claude Code rail into AnalysisOrchestrator's full pipeline (orchestrator + downstream make_client consumers)". Full audit_basis + verification.command + success_criteria + live_check fields.
   - `27.6.status` UNCHANGED at `pending` (Q/A returned RECOMMEND-KEEP-PENDING).

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read())"
exit=0

$ python -c "import ast; ast.parse(open('backend/api/settings_api.py').read())"
exit=0

$ curl -s http://localhost:8000/api/settings/ | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('paper_cycle_max_seconds'))"
7200.0

$ grep -c "paper_cycle_max_seconds" backend/api/settings_api.py
4

$ python3 -c "from google.cloud import bigquery; c=bigquery.Client(project='sunny-might-477607-p8'); [print(f'rows={r.n} tickers={r.t}') for r in c.query('SELECT COUNT(*) n, COUNT(DISTINCT ticker) t FROM \`sunny-might-477607-p8.financial_reports.analysis_results\` WHERE DATE(analysis_date)=CURRENT_DATE()').result()]"
rows=13 tickers=13

$ grep -E "08:31:33.*cycle complete" backend.log
08:31:33 I [autonomous_loop] Paper trading cycle complete: NAV=$23767.00, P&L=18.83%, trades=0, cost=$1.3000
```

## Why 27.6 FAILed (Q/A finding)

All 13 BQ rows persisted today are LITE-FALLBACK signatures (`standard_model=NULL`, `deep_think_model=NULL`, `debate_rounds_count=NULL`, flat $0.10 cost). The cycle-7 cycle ran, fell back to lite-mode after 11 of 13 tickers hit `Full orchestrator failed: credit balance is too low` (Anthropic-direct rail, NOT claude_code CLI), then the lite-fallback `_run_claude_analysis` path -- which DOES honor `paper_use_claude_code_route=True` (cycle 5's dispatch) -- wrote the 13 rows via the Claude Code rail.

So the rail IS operational. But ONLY in the lite-mode fallback. The full orchestrator pipeline (`AnalysisOrchestrator.run_full_analysis`) does NOT route through `make_client`'s claude_code branch correctly. That's 38.13's scope.

## Q/A trail

- Q/A `abbcca28fb3536a63` returned FAIL on cycle 7's 27.6 closure attempt + RECOMMEND-KEEP-PENDING.
- Critique written to `handoff/current/evaluator_critique.md`.
- Cycle 7 itself (the 38.12 ship) is PASS in isolation -- but does not advance the goal's stop condition because 27.6 stays pending.

## Cycle 8 scope (the path forward)

1. Full-codebase audit pass per operator memory rule: trace every LLM call site in `backend/agents/orchestrator.py`, `backend/agents/debate.py`, `backend/agents/risk_debate.py`, downstream consumers.
2. Identify paths that bypass `make_client` OR pass model_name values not matching the `claude-` prefix guard (the gemini-2.5-pro deep_think_model path is suspicious -- it routes to Gemini Vertex AI, not Anthropic OR claude_code).
3. Identify where the orchestrator falls into Anthropic-direct (the credit-exhausted rail) despite the flag being set.
4. Ship the routing fix.
5. Re-trigger cycle.
6. Verify BQ rows have `standard_model LIKE 'claude%'` (per 38.13's verification.command).
7. Re-attempt 27.6 closure.

## Memory-rule compliance

- ZERO frontend changes.
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.

## Honesty + accountability

Operator memory entry `feedback_full_codebase_audit_before_changes.md` flagged exactly this failure mode 18 hours ago: "Q/A LLM-judgment cannot grep what isn't pointed at -- it audits what's in scope, not what's downstream. When a cycle touches LLM-call shape (model, prompt, args), also audit: every formatter that consumes the response, every orchestrator path that reads or transforms the shape." Cycle 7's earlier optimistic PASS claim violated this rule. Cycle 8 will satisfy it by including the explicit full-pipeline trace BEFORE claiming PASS.

## Open follow-up backlog (still / now)

- `38.10` Slack digest envelope regression (operator screenshot)
- `38.11` Recent Reports table bugs (operator screenshot)
- `38.13` (P0, new) Wire Claude Code rail into AnalysisOrchestrator's full pipeline
- `46.0-46.8` Market Insight page (operator-approved scoping, awaits implementation green-light)
- contract.md collision deconfliction (9th overwrite today by autonomous-loop sprint -- elevate to P0?)
- Concurrent autonomous-loop cycles (cron fired overlapping cycles during cycle 7)
