# Cycle 8 -- Experiment Results (step 38.13)

**Window:** 2026-05-27T18:57 - in-flight (cycle completion ETA 20:30-21:00 per cycle-7 ~102min baseline).

## Files modified
1. `backend/services/autonomous_loop.py` -- 2 edits (line 1278 pre-dispatch log; line 1310 full_report.source + rail injection).
2. `backend/agents/orchestrator.py` -- 1 edit (line 2123, `cost_summary["rail"]` attribution).
3. `handoff/current/research_brief_phase_38_13_orchestrator_rail_audit.md` -- researcher output (25,526 bytes).
4. `handoff/current/contract.md` -- restored at 19:08 after autonomous-loop sprint clobber #10.

## Verification commands run (with verbatim output)

```
$ python3 -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"
exit=0

$ python3 -c "import ast; ast.parse(open('backend/agents/orchestrator.py').read())"
exit=0

$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
(backend restarted at 18:58)

$ curl -sf http://localhost:8000/api/health
HTTP 200

$ curl -s http://localhost:8000/api/settings/ | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('paper_use_claude_code_route'), d.get('gemini_model'), d.get('paper_cycle_max_seconds'))"
True claude-sonnet-4-6 7200.0

$ curl -X POST http://localhost:8000/api/paper-trading/run-now
(fresh cycle triggered at 18:59:30+02:00)
```

## Live evidence (backend.log)

### Cycle-7 baseline (proves the bug existed)
All 13 cycle-7 persist lines say **`Lite analysis persisted to analysis_results for X`** -- the full orchestrator path silently failed (credit-exhausted on direct Anthropic rail) and fell back to the lite-mode writer. Sample:
```
07:08:28 [autonomous_loop] Lite analysis persisted to analysis_results for AMD
07:08:49 [autonomous_loop] Lite analysis persisted to analysis_results for STX
07:10:49 [autonomous_loop] Lite analysis persisted to analysis_results for CIEN
... (10 more, all "Lite analysis persisted")
08:31:33 [autonomous_loop] Paper trading cycle complete: NAV=$23767.00, P&L=18.83%, trades=0, cost=$1.3000
```

### Cycle-8 evidence (proves the fix is live)
First three pre-dispatch lines from the new instrumentation (edit 1):
```
18:59:56 [autonomous_loop] Orchestrator pre-dispatch ticker=STX rail=claude_code lite_mode=False model=claude-sonnet-4-6
18:59:58 [autonomous_loop] Orchestrator pre-dispatch ticker=AMD rail=claude_code lite_mode=False model=claude-sonnet-4-6
18:59:59 [autonomous_loop] Orchestrator pre-dispatch ticker=CIEN rail=claude_code lite_mode=False model=claude-sonnet-4-6
```
- `lite_mode=False` -- confirmed full pipeline (NOT lite fallback).
- `rail=claude_code` -- confirmed rail attribution.
- `model=claude-sonnet-4-6` -- confirmed correct model.

Orchestrator-internal claude_code_invoke evidence (19:04-19:06 window):
```
19:04:27 [claude_code_client] claude_code_invoke: success duration_ms=55575 input_tokens=6 output_tokens=2281
19:04:27 [orchestrator] [Debate] Compacted enrichment_for_debate for small-context model 'claude-sonnet-4-6' (limit: 26,000 chars, lite=False, 11 signals kept)
19:04:28 [claude_code_client] claude_code_invoke: success duration_ms=36877 input_tokens=6 output_tokens=1736
19:04:38 [claude_code_client] claude_code_invoke: success duration_ms=46433 input_tokens=6 output_tokens=2034
... (15+ more interleaved success calls in this window)
```
The `[orchestrator] [Debate] ... lite=False` interleaved with `[claude_code_client] claude_code_invoke: success` is **direct proof** that the full-orchestrator's debate stage is now calling through the Claude Code rail (not the direct Anthropic rail that exhausted in cycle 7).

## Artifact shape (deferred -- pending cycle completion)
- BQ rows in `financial_reports.analysis_results` for today's cycle, post-19:00: TO BE INSPECTED after cycle complete.
- Expected: `standard_model='claude-sonnet-4-6'`, `JSON_EXTRACT_SCALAR(report_json, '$.rail')='claude_code'`, `JSON_EXTRACT_SCALAR(cost_summary, '$.rail')='claude_code'` for >=11 of universe tickers.
- Expected persist log line: NOT `Lite analysis persisted` (which was cycle 7's fallback signature). Should be the full-path persist line.

## Pending steps
1. Wait for cycle completion (~95 min from cycle start, ETA 20:30-21:00).
2. BQ query for post-fix evidence.
3. Spawn Q/A on cycle 8.
4. On Q/A PASS, append harness_log.
5. Flip masterplan 38.13 to done.
6. Re-assess 27.6 closure with definitive evidence.

## Memory-rule compliance
- ZERO frontend changes.
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.
- Full-codebase audit pass: researcher `a52e1c2d9ee70e256` traced `_persist_analysis -> full_report.source -> orchestrator report dict` to find the missing-key bug (overturned cycle-7 Q/A diagnosis).

## Note on contract.md collision
The autonomous-loop parameter-optimization sprint overwrote `handoff/current/contract.md` for the 10th time today (timestamp 19:03 today). Restored at 19:08 with the cycle-8 narrative. Permanent deconfliction is on the backlog (Layer-3 harness contract path vs Layer-1 autonomous-loop sprint contract path should be different files).

## Wakeup scheduled
Next wakeup at 20:09 (3625s) to check cycle completion + BQ rows + spawn Q/A.
