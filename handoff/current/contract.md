# Contract — phase-43.0 Production-Ready DoD Audit (14 criteria)

**Cycle:** 12 | **Date:** 2026-05-28 | **Step:** 43.0 (P1, H) | **Author:** Main

---

## Research-Gate Summary

- Researcher subagent: `a9547514da955b875`
- Brief: `handoff/current/research_brief_phase_43_0_dod_audit.md`
- `gate_passed: true` — 6 external sources read in full, 13 snippet-only URLs, recency scan present, 3-variant queries documented, 16 internal files mapped.
- Headline finding: expected NOT_PRODUCTION_READY; 8 confirmed PASS + 3 drift candidates (DoD-6/7/9) + 1 contested (DoD-14) + 1 structurally unverifiable (DoD-2) + 1 confirmed FAIL (DoD-1).

## Hypothesis

The 14-criterion audit will return **NOT_PRODUCTION_READY** today, with concrete PASS counts as follows after live verification:

| Bucket | Expected count | DoDs |
|--------|----------------|------|
| Confirmed PASS | 8 | DoD-3, DoD-4 (tiered), DoD-8, DoD-10, DoD-11, DoD-12, DoD-13, + DoD-9 (drift confirmable via cycle_history.jsonl) |
| Drift PASS (live-check pending) | 2 | DoD-6 (phase-35.1 shipped writer), DoD-7 (phase-37.1 shipped schema) |
| Contested | 1 | DoD-14 (3 LLM categories not explicitly tagged) |
| Structurally unverifiable today | 1 | DoD-2 (no walk-forward result vs paper-trading comparison artifact) |
| Confirmed FAIL | 1 | DoD-1 (autoresearch cron exit 1 — 9 consecutive ERROR days; phase-39.1 pending) |
| Live-probe required | 1 | DoD-5 (`/api/paper-trading/freshness` Unknown-band probe) |

**Best-case PASS: 10 of 14. Worst-case PASS: 8 of 14. Either way: NOT_PRODUCTION_READY.**

## Immutable Success Criteria (verbatim from `.claude/masterplan.json` 43.0)

1. `all_14_DoD_criteria_PASS`
2. `audit_file_carries_verbatim_evidence_per_criterion`
3. `qa_confirms_no_silent_drops`
4. `operator_approval_recorded_for_PRODUCTION_READY_declaration`

**Verification command (immutable):**
```bash
test -f "handoff/current/production_ready_audit_$(date +%Y-%m-%d).md" && grep -qE 'PRODUCTION_READY|NOT_PRODUCTION_READY' "handoff/current/production_ready_audit_$(date +%Y-%m-%d).md"
```

**Live-check:** `production_ready_audit_<date>.md` IS the deliverable; live verification = read it.

## Plan Steps

1. **Inspect environment state** — `launchctl list | grep pyfinagent`, `cat handoff/cycle_history.jsonl | tail`, `ls handoff/autoresearch/ | tail -10`, confirm backend alive (`curl -sf http://localhost:8000/health || curl -sf http://localhost:8000/api/health`).
2. **Per-DoD live verification** — execute the evidence command from the brief's Section 6 table for each of DoD-1..DoD-14. For each: record the verbatim command + verbatim output (truncated to relevant lines) + PASS/FAIL/UNKNOWN classification + cite file:line or commit hash where applicable.
3. **Handle DoD-2 honestly** — no walk-forward results JSON carrying paper-trading Sharpe comparison exists; mark UNKNOWN, document the structural gap, point to `backend/services/perf_metrics.py:186 compute_sharpe_gap()` as the function that COULD evaluate this if walk-forward results carried the paper-trading comparison column. Do NOT mark PASS without evidence.
4. **Handle DoD-14 honestly** — re-grep SKILL.md for explicit `LLM0[1-9]:2025|LLM10:2025` tags; if missing for LLM04/05/09, mark CONTESTED (= literal-FAIL per the verbatim criterion text) and flag SKILL.md tagging gap as a documentation follow-up (not in this step's scope to fix mid-audit).
5. **Render `production_ready_audit_2026-05-28.md`** — must contain (per immutable criterion #2) verbatim evidence per DoD; must contain (per the verification grep) the literal token `PRODUCTION_READY` or `NOT_PRODUCTION_READY` on a line.
6. **Render `experiment_results.md`** — summarize what was probed, file list touched, verbatim verification command output, artifact shape.
7. **Verify against immutable criteria** — re-read criterion #1 honestly: if any DoD ≠ PASS, the step cannot satisfy `all_14_DoD_criteria_PASS`, so step 43.0 itself stays `pending` (not `done`). The deliverable can land + Q/A can verify the deliverable is accurate — but the step closes ONLY when all 14 PASS. This is the right behavior per immutable-criteria discipline.
8. **Spawn Q/A** — single subagent, 5-item harness audit + LLM judgment + deterministic verification grep.
9. **Append `handoff/harness_log.md`** Cycle 12 block AFTER Q/A PASS, BEFORE any masterplan status edit.
10. **Masterplan status policy** — 43.0 stays `pending` with `audit_completed: true` annotation in cycle 12 log; DoD-1 / DoD-2 / DoD-5 / DoD-6 / DoD-7 / DoD-14 each get a follow-up cycle pointer (existing pending steps in phase-35/36/37/39).

## What this cycle will NOT do

- Fix DoD-1 (`phase-39.1` is owner-gated; widening to cover `langchain_huggingface` requires its own cycle).
- Fix DoD-14 (SKILL.md tag additions are a doc edit out of scope for the audit cycle).
- Re-run a 30-day walk-forward backtest to populate DoD-2 evidence (compute budget; separate cycle).
- Add the LLM04/05/09 tags to SKILL.md (separate doc-edit cycle).
- Probe BQ via destructive queries — only `COUNT(*)` and `SELECT ... LIMIT 5` style reads, all gated by per-call user approval.

## Stop-Condition Contribution

43.0 IS the production_ready stop-condition gate. The audit DELIVERABLE landing (production_ready_audit_2026-05-28.md) is the cycle output; the GATE PASS (all 14 criteria PASS) is what flips 43.0 to `done`. Cycle 12 produces the deliverable; the gate PASS is contingent on remaining phase-35/36/37/39 work.

## Anti-Patterns to Avoid (citing project auto-memories)

- `feedback_no_emojis` — no emojis in audit file or evidence sections.
- `feedback_contract_before_generate` — this contract is being written BEFORE GENERATE.
- `feedback_log_last` — harness_log.md append comes AFTER Q/A PASS, BEFORE status flip.
- `feedback_qa_harness_compliance_first` — Q/A prompt will begin with 5-item harness audit.
- `feedback_harness_rigor` — do not rig PASS by ignoring contested/unverifiable criteria.
- `feedback_full_codebase_audit_before_changes` — this IS the full audit; surface bugs not hide them.
- `feedback_never_skip_researcher` — researcher spawned + gate passed BEFORE contract.

## References

- `.claude/masterplan.json:phase-43.0` (immutable spec)
- `handoff/current/master_roadmap_to_production.md` §6 (14 DoD criteria source)
- `handoff/current/research_brief_phase_43_0_dod_audit.md` (this cycle's research gate)
- Anthropic harness-design canon: https://www.anthropic.com/engineering/harness-design-long-running-apps
- OWASP LLM Top 10 (2025): https://genai.owasp.org/llm-top-10/
- `backend/services/perf_metrics.py:186` `compute_sharpe_gap()` (DoD-2)
- `backend/services/kill_switch.py:275` `check_auto_resume()` (DoD-3)
- `backend/services/paper_trader.py:530` `check_scale_out_fires()` (DoD-8)
- `backend/services/autonomous_loop.py:1975` phase-35.1 outcome_tracking writer (DoD-6)
- `backend/agents/orchestrator.py:115-116` Risk Judge response_schema (DoD-7)
- `backend/services/cycle_lock.py` + `autonomous_loop.py:150,167` (DoD-13)
- `backend/config/model_tiers.py:66` + `settings.py:30` (DoD-10)
- `scripts/qa/ascii_logger_check.py` (DoD-12)
- `.claude/skills/code-review-trading-domain/SKILL.md:56-100,213,230` (DoD-14)
- `handoff/cycle_history.jsonl` (DoD-9)
- `handoff/autoresearch/2026-05-28-ERROR-topic08.md` (DoD-1)
