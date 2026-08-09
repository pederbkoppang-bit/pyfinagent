---
name: decision-input-integrity-61-2
description: Step 61.2 research (2026-08-09) — the step's stated trigger (the critic) is REFUTED by BQ; the real emitter is the synthesis-draft parse failure; 61.2 is already BUILT and DARK behind two flags
metadata:
  type: project
---

Step 61.2 ("never persist synthetic 0.00/HOLD"). Research gate run 2026-08-09.
Three findings that change how the step must be planned.

**1. The step mis-attributes its own trigger — twice.** It first blamed a
claude_code timeout, then (2026-08-09 update) blamed the critic:
`"Critic returned unparseable JSON after retry -- proceeding with the UNREVIEWED
draft"`. BigQuery refutes both. Over 40 days: 185 rows, **153 are
`final_score=0.0` + `recommendation='HOLD'`, and all 153 carry
`$.final_synthesis.error = "Failed to parse final report."`** — a 153/153 exact
match. `critic_degraded=true` appears on 45 rows, of which **3 carry a real
score > 0**. So critic degradation is ORTHOGONAL: the CRWD row the step cites as
"real" (5.75) is itself `critic_degraded=true`, and the PANW 0.0 row it cites as
"caused by the critic" has `critic_degraded=false`. The real emitter is
`backend/agents/orchestrator.py:1681-1688` — the SYNTHESIS draft failing
`_parse_json_with_fallback`, not the critic. A critic-scoped fix would break the
working path and miss all 153 rows.

**Why:** the step's evidence update reasoned from co-occurring log lines
(3 critic warnings bracketing a save) instead of from the row's own JSON.
The discriminating column was one `JSON_VALUE(..., '$.final_synthesis.error')`
away.

**How to apply:** for any "which failure path fired" question on
`financial_reports.analysis_results`, query
`JSON_VALUE(full_report_json,'$.final_synthesis.error')` and
`'$.final_synthesis.critic_degraded'` together and CROSS-TABULATE them against
`final_score`. Log-line adjacency is not attribution. Note the column is
`final_score` (not `overall_score`/`score`) and the dataset is
`financial_reports` in **us-central1**.

**2. 61.2 is already built and DARK.** Criteria 1/4/6 sit behind
`paper_synthesis_integrity_enabled=False`, criterion 5 behind
`paper_position_recommendation_fix_enabled=False`; criteria 2 and 3 already
shipped ungated (`claude_code_timeout_s=150`, company_name non-NULL since
07-09). 495-line test file + `live_check_61.2.md` already exist; the
verification command collects 72 tests and its `test -f` leg passes today.
`harness_log.md` Cycle 173 records **CONDITIONAL #2** — a third Q/A on unchanged
blocker evidence MUST auto-FAIL per CLAUDE.md. The step is an operator
promotion decision plus a small residue, not a build.

**3. The residue that is genuinely NOT built:** `backend/tasks/analysis.py:210-214`
and `backend/api/analysis.py:210-214` fabricate unconditionally
(`final_score=synthesis.get("final_weighted_score", 0)`,
`recommendation=rec_obj.get("action","N/A")`) — outside every flag. Plus SIX
downstream `or 0` / `or "HOLD"` coercions re-create the fabrication even if the
persist path is fixed (`conflict_detector.py:87,115`, `formatters.py:180`,
`scheduler.py:1069`, `portfolio_manager.py:140,182`).

**The casing tell is a formatting accident.** `'HOLD'` uppercase comes from the
literal default in `rec.get("action", "HOLD")` at `autonomous_loop.py:2065`;
genuine rows write `'Hold'`. It is 153/153 accurate today and 0 false positives
over 40 days, but any prompt/model change breaks it silently — never build a
consumer on it.

**Free columns for an explicit degraded marker** (measured NULL/empty on 179/179
rows since 2026-07-01): `data_quality_score` (FLOAT), `critic_review` (STRING),
`bias_flags`, `synthesis_iterations`, `groupthink_flag` (the table's ONLY
BOOLEAN), `recommendation_justification`. `overall_reliability` and `risk_level`
are occupied. There is no dedicated status column.

Related: [[project_decision_input_integrity_61_2]] supersedes the stale
2026-06-11 anchors in the masterplan node — `_BUY_RECS` is at
`portfolio_manager.py:63` (not :50), the signal_downgrade rule at `:154-157`
(not :127), `paper_trader` rec-selection at `:447-457` (not :305).
See also [[project_learnings_61_4]], [[feedback_measure_dont_assert_claims]].
