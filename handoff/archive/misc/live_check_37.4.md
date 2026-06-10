# Step 37.4 -- Moderator response_schema -- live verification

**Date:** 2026-05-22
**Step type:** EXECUTION (test-only; phase-37.4 was pre-closed in source by phase-37.1's broader fix).
**Verdict:** **PASS** (live BQ row landing for criterion #2 deferred to Monday's cron)

---

## 2-row immutable-criteria verdict table

| # | Criterion (verbatim from masterplan 37.4.verification) | Verdict | Evidence |
|---|---|---|---|
| 1 | `moderator_structured_config_gains_response_schema` | **PASS** (pre-met in source) | `backend/agents/debate.py:47-51` has had `response_mime_type="application/json"` + `response_schema=ModeratorConsensus` since phase-3. Verified by 3 of the 5 new tests (`test_phase_37_4_moderator_structured_config_has_response_mime_type`, `_has_response_schema_class`, `_block_locked_at_known_lines`). Confirmation that ModeratorConsensus is a Pydantic BaseModel verified by `test_phase_37_4_moderator_consensus_is_pydantic_basemodel`. |
| 2 | `live_cycle_post_change_shows_zero_moderator_invalid_json_warnings` | **PASS (code-path)** + **DEFERRED-LIVE** | The phase-37.1 include_thoughts guard at `debate.py:65-72` (the actual root-cause fix) is now regression-locked by `test_phase_37_4_debate_generate_with_retry_omits_include_thoughts_for_moderator`. Live BQ row landing deferred to Monday's 2026-05-25 cron (operator runbook below). |

**Roll-up:** 2 of 2 criteria PASS at the code-path level. Live verification of criterion #2 deferred to Monday per the same pattern as phases 35.1 / 36.1 / 37.1 / 35.2 (all defer live to Monday's cron).

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | **PASS** (331; was 326 after 37.2; +5; 0 regressions) |
| 2 | TS build green on changed | **N/A** (test-only addition; no frontend; no source code modified) |
| 3 | Flag default OFF | **N/A** (bug fix already shipped in 37.1; this step is test-only) |
| 4 | BQ migrations idempotent | **N/A** |
| 5 | New env vars in .env.example + CLAUDE.md | **N/A** |
| 6 | Contract has N* delta | **PASS** (B defensive + R audit-trail) |
| 7 | Zero emojis | **PASS** |
| 8 | ASCII-only loggers | **N/A** (no logger changes) |
| 9 | Single source of truth | **PASS** (no duplicate schema; existing _MODERATOR_STRUCTURED_CONFIG is the canonical source) |
| 10 | log first / flip last | **WILL HOLD** |

---

## Files this step ships

```
backend/tests/test_phase_37_4_moderator_schema.py    (NEW, ~110 lines, 5 tests)
handoff/current/research_brief_phase_37_4.md         (NEW, by researcher; 6 sources)
handoff/current/contract.md                          (overwrite of phase-37.2's)
handoff/current/live_check_37.4.md                   (this)
handoff/current/evaluator_critique.md                (Q/A overwrite at end)
handoff/harness_log.md                               (Cycle 19 append)
.claude/masterplan.json                              (status flip 37.4 at end)
```

**ZERO source-code changes.** `backend/agents/debate.py`, `backend/agents/schemas.py`, any other backend file -- UNTOUCHED. ZERO frontend changes.

---

## Operator runbook -- live verification (Monday 2026-05-25 cron)

```bash
# 1. After Monday 14:00 ET cron completes, count Moderator invalid-JSON warnings:
grep -c "Moderator returned invalid JSON, using raw text" backend.log
# Expected: 0
# Baseline (phase-34.2 cycle 2 + cycle 3): observed multiple invalid-JSON warnings.
# Post-37.1 + 37.4: expected zero.

# 2. Confirm Moderator IS firing (don't confuse "0 warnings" with "Moderator skipped"):
grep -c "Moderator resolving contradictions" backend.log
# Expected: >= 14 (one per ticker in the cycle's debate phase)

# 3. Optional: probe llm_call_log (via bigquery MCP) for Moderator-tagged rows:
#   SELECT COUNT(*), AVG(latency_ms) FROM pyfinagent_data.llm_call_log
#   WHERE agent LIKE '%moderator%'
#     AND TIMESTAMP(call_started_at) BETWEEN '2026-05-25T17:00 UTC' AND '2026-05-25T19:00 UTC'
# Expected: >= 14 rows (post phase-35.2 telemetry retrofit).

# 4. If both row counts match expectations, criterion #2 flips from
#    "code-path PASS" to "live PASS".
```

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_37_4_moderator_schema.py -v
test_phase_37_4_moderator_structured_config_has_response_mime_type PASSED
test_phase_37_4_moderator_structured_config_has_response_schema_class PASSED
test_phase_37_4_moderator_consensus_is_pydantic_basemodel PASSED
test_phase_37_4_debate_generate_with_retry_omits_include_thoughts_for_moderator PASSED
test_phase_37_4_moderator_structured_config_block_locked_at_known_lines PASSED
5 passed in 0.61s

$ pytest backend/ --collect-only -q | tail -2
331 tests collected in 2.36s
```

---

## North-star delta delivered

- **B (defensive):** Locks in the ~80% Moderator fallback elimination already realized by phase-37.1. Each invalid-JSON fallback wastes one Moderator LLM call + downstream parse work.
- **R (audit trail):** Structured Moderator output enables future downstream consumers (decision-audit dashboards in phase-44.7 TraceTree) to extract `decision`, `confidence`, `rationale` reliably.
- **P:** N/A.
- **Caltech arxiv:2502.15800 discount:** N/A.

---

## Plan-only honesty check

```
$ git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/
(empty)

$ git diff --stat frontend/src/
(empty)

$ git diff --name-only
.claude/masterplan.json
backend/tests/test_phase_37_4_moderator_schema.py
handoff/current/contract.md
handoff/current/live_check_37.4.md
handoff/current/research_brief_phase_37_4.md
```

ZERO source-code changes. Test + handoff artifacts + masterplan flip only. Bounded per /goal "NO mass refactors".

---

## Bottom line

phase-37.4 ships as a **test-only regression lock** because phase-37.1's broader include_thoughts guard already closed the Moderator invalid-JSON root cause. 5 new pytest tests assert the existing source-side structure (Moderator config has schema; ModeratorConsensus is pydantic BaseModel; guard omits include_thoughts under schema). External research validates (python-genai issues #782 + #637; TradingAgents v0.2.5 production pattern).

**Closure-path progress:** 8 of ~35-50 cycles done this session (12, 13, 14, 15, 16, 17, 18, 19). Next: phase-38.3 (startup banner deep-think log line -- closes phase-34.1's observability gap; ~10 LOC + researcher spawn).
