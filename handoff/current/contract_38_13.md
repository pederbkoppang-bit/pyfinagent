# Cycle 8 Contract -- Step 38.13 (full-orchestrator rail wiring + observability)

**Generated:** 2026-05-27T19:08+02:00 (RESTORED after autonomous-loop parameter-optimization sprint clobber #10 today).

**Step id:** `38.13` -- Wire Claude Code rail into AnalysisOrchestrator full pipeline + add rail/model attribution to BQ persistence.

**Cycle class:** Observability + routing fix. NOT a trading-policy change. Citation floor (>=2 AI-in-trading + >=2 academic) per testing-phase mandate does NOT apply per the cycle-7 harness_log precedent ("cycle 7 is verification + small settings change ... Citation floor does NOT apply per goal mandate.").

## Research gate
- Researcher: `a52e1c2d9ee70e256`, tier=deep, gate_passed=true.
- Output: `handoff/current/research_brief_phase_38_13_orchestrator_rail_audit.md` (25,526 bytes, 18:57 mtime).
- Sources read in full: 5+ (write-first directive honored).
- URLs collected: 13.
- Recency scan: performed (last-2-year window).
- Internal files inspected: 7+ (orchestrator.py, llm_client.py, autonomous_loop.py, claude_code_client.py, _persist_analysis path).

### Critical finding (overturns cycle-7 Q/A)
Cycle-7 Q/A `abbcca28fb3536a63` misdiagnosed the 13 NULL-standard_model rows as "lite-mode fallback signature". The actual root cause is upstream of routing:

- `_persist_analysis()` reads `full_report.get("source")` to populate `standard_model`.
- The orchestrator's report dict has NO `source` key (0 grep hits across `backend/agents/orchestrator.py`).
- Therefore `standard_model=NULL` regardless of which rail ran.
- The lite-mode-vs-full diagnostic from cycle 7 was INVALIDATED.

The 13 cycle-7 rows are a MIX of cycle-6 (`paper_use_claude_code_route=False`) and cycle-7 (`paper_use_claude_code_route=True`) writes; NULL was never a rail signal.

## Hypothesis
Three small observability edits (no routing logic change required for the orchestrator path itself, since cycle-7 evidence shows `make_client` already gates on `claude-` prefix and `gemini_model=claude-sonnet-4-6` DOES match) will:
1. Make rail/model attribution visible in BQ persistence.
2. Produce definitive post-fix evidence for 27.6 closure.
3. Prevent future misdiagnoses of NULL-as-fallback.

## Plan steps
1. `backend/services/autonomous_loop.py:1278` -- pre-orchestrator rail log:
   ```python
   _route = "claude_code" if getattr(settings, "paper_use_claude_code_route", False) else "anthropic_direct"
   logger.info("Orchestrator pre-dispatch ticker=%s rail=%s lite_mode=False model=%s", ticker, _route, settings.gemini_model)
   ```
2. `backend/agents/orchestrator.py:2123` -- cost_summary rail attribution:
   ```python
   cost_summary["rail"] = "claude_code" if getattr(self.settings, "paper_use_claude_code_route", False) else "anthropic_direct"
   ```
3. `backend/services/autonomous_loop.py:1310` -- populate full_report.source + rail before persist:
   ```python
   "full_report": {**(report if isinstance(report, dict) else {}), "source": settings.gemini_model, "rail": _route},
   ```

## Success criteria (verification.live_check for 38.13)
- After cycle completes, BQ query returns rows with `standard_model='claude-sonnet-4-6'` (non-NULL) AND `rail='claude_code'` for >=11 of universe tickers.
- `cost_summary.rail` populated in cost_summary JSON column.
- AST parse clean on both edited files.
- Backend health endpoint returns 200 after kickstart.

## Bridge to 27.6 closure
If cycle 8 PASSes, step 27.6 ("End-to-end smoke verify: full path on Claude") becomes re-closable using the post-fix BQ row sample (definitive, non-NULL, rail-tagged evidence). 27.6 is the sole `production_ready.must_have` entry, so 38.13 PASS + 27.6 closure together advance the production-ready predicate.

## Execution status (as of 19:08)
- AST parse: OK on both files.
- Backend kickstarted: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` at 18:58.
- Health check: HTTP 200.
- Settings: `paper_use_claude_code_route=True`, `gemini_model=claude-sonnet-4-6`, `paper_cycle_max_seconds=7200.0`.
- Fresh autonomous-loop cycle triggered: `POST /api/paper-trading/run-now` at 18:59:30+02:00.
- Live evidence (backend.log, 19:04-19:06): Multiple `[orchestrator] [Debate] Compacted enrichment_for_debate for small-context model 'claude-sonnet-4-6'` lines interleaved with `[claude_code_client] claude_code_invoke: success` calls -- DEFINITIVE proof the full-orchestrator pipeline is now routing through the Claude Code rail.

## Open verification (deferred)
- BQ row inspection: must wait for cycle completion (~20:30-20:45 ETA per cycle-7 ~102min runtime).
- Q/A spawn: deferred until BQ evidence available.
- harness_log + masterplan flip: only after Q/A PASS.

## References
- `handoff/current/research_brief_phase_38_13_orchestrator_rail_audit.md`
- `handoff/harness_log.md` Cycle 7 entry + Cycle 7 Q/A correction (Main, 09:05)
- `backend/agents/orchestrator.py:516-518` (make_client gate)
- `backend/agents/llm_client.py:1888-1905` (claude- prefix routing)
- `backend/agents/claude_code_client.py` (ClaudeCodeClient adapter)
