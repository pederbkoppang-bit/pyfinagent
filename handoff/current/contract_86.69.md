# Contract -- step 86.69

**Step:** 86.69 (P0) -- 81% of analyses are persisted as an EMPTY row scored
0.0 and labelled HOLD; a dated regression that began 2026-06-11 (corrected
from the audit-basis window 06-12/06-15 by the research gate), and it is why
the book stopped trading.

## Research-gate summary (what changes the plan)

Gate PASSED (`wf_8515024d-2fe`, recomputed; 9 sources read in full, 31 URLs,
recency scan; brief `research_brief_86.69.md`, 40,219 chars):

1. **Cause ESTABLISHED -- hypothesis (i) confirmed, (ii)/(iii) refuted.**
   phase-60.1 (`fa62b5fe`, 2026-06-11) RESTORED the full pipeline after
   gemini-2.0-flash's 2026-06-01 server-side retirement had silently forced
   every analysis onto the healthy LITE path. The `_path` provenance field
   60.1 added is a free deploy marker: last unstamped row 2026-06-10 18:38Z,
   first stamped 2026-06-11 10:17Z, zero-scores start the same day. All 211
   empties are `_path=full`; lite produced 0/19.
2. **The row is FABRICATED at `autonomous_loop.py:2179` (uppercase `HOLD`)
   and `:2190-2192`** -- not the lite writer (title-case `Hold`). Every empty
   carries `final_synthesis.error='Failed to parse final report.'` and no
   `scoring_matrix` (211/211 sensitive, 38/38 specific on live data).
3. **The second half is the SAME cause**: BUY conversion 2.6% on the full
   path vs 36.8% on lite -- a mixture effect; no threshold change exists.
4. **The fix already exists and is DARK**: phase-61.2's fail-closed guard at
   `:2163-2171` behind `paper_synthesis_integrity_enabled=False`
   (`settings.py:206-208`, "DARK until operator promotion"); its own
   docstring names this exact defect. Every other guard in the file detects
   AFTER the write (audit-after-publish ordering, detection `:1301` vs write
   `:1249`).
5. **Transport**: the full path runs on the CC rail (`$.rail=claude_code`,
   257 rows, 84.4% zero-score) whose `--json-schema` is post-hoc validated,
   not constrained -- the Gemini-shaped `_SYNTHESIS_STRUCTURED_CONFIG` is
   honoured by nothing. 2,859 parse failures across ALL agents (synthesis
   only 9.2%) -- a larger surface, QUEUED as its own step, not absorbed.
6. **Hazards**: `_DOWNGRADE_RECS` contains `HOLD` (a downgrade path must not
   resurrect the fabrication); NULL recommendation is already plumbed to the
   frontend.

## Hypothesis

Recording a failed synthesis as an ABSENCE (NULL recommendation/score + the
error preserved) at the fabrication site itself -- unconditionally, because
the criteria mandate exactly this write-boundary change -- stops the book
reading fabricated HOLDs, makes the failure visible to the operator, and
recovers nothing by itself (the parse failure still loses the analysis);
armed 61.2 semantics and the CC-rail schema fix are operator-gated /
queued follow-ups. Post-fix measurement requires the restarted process and
accrues from the next scheduled cycles.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

1. the CAUSE of the 2026-06-12/15 break is identified and demonstrated, not hypothesised: name the specific change or failure and show the mechanism in source or in a reproduction, and state explicitly which of the three listed hypotheses it confirms or refutes
2. the persist call site that writes final_score=0.0 with an empty summary and recommendation='HOLD' is READ and quoted with file:line -- the current inference from log co-occurrence is explicitly replaced by source evidence or corrected
3. a failed or degraded analysis is no longer recorded as a valid verdict: show that the absence is represented as an absence at the persistence boundary, and prove no consumer reads it as a HOLD, deriving the consumer set rather than asserting it
4. the fix is measured against the SAME populations used to find the defect -- analysis_results by analysis_date, and the JSON-format backend log lines -- and the post-fix zero-score share is reported next to the 81.2% POST and 37.8% PRE baselines with the query that produced each
5. the recovery is decomposed the same way the defect was: report the post-fix share of analyses producing a real score AND the BUY conversion among them, since both halves broke and fixing only the emptiness recovers about 3.3x of a 13.2x collapse
6. NO gate is loosened and NO risk threshold is changed to produce the improvement -- the risk judge's 61.9% unparseable-response fallback rate is a separate defect and is NOT in scope for this step
7. NO flag is promoted and NO .env is written by this step; operator-gated changes are recorded as numbered asks
8. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

## Plan

1. **Criteria 1+2 (discharge from the gate, re-verified by Main):** re-verify
   the deploy-marker dating and quote the fabrication sites from source at
   HEAD; state hypothesis (i) confirmed / (ii),(iii) refuted with the
   refuting measurements.
2. **Criterion 3 (the write-boundary change):** at the fabrication site,
   replace the fabricated `0.0 / '' / 'HOLD'` with an ABSENCE row --
   `recommendation=None`, `final_score=None`, `summary=''` retained,
   `analysis_failed=True` + the preserved `final_synthesis.error` -- so the
   row says "failed", never "the most common valid verdict". Sweep the
   consumer set DERIVED from source (`decide_trades`, `_fold_degraded_for_
   trading`, `_degraded_scoring_check`, meta-scorer, signal attribution,
   frontend types, BQ readers of `recommendation`/`final_score`): each must
   treat NULL as absence (skip), never as HOLD; guard the `_DOWNGRADE_RECS`
   path against resurrecting HOLD from NULL. Fault-injected boundary test:
   inject `Failed to parse final report.` into the REAL code path and assert
   the persisted row shape.
3. **Criteria 4+5 (measurement, honestly bounded):** the fix takes effect at
   the next backend restart (batched to session end per standing operator
   instruction) and the shares accrue from the next scheduled cycles. This
   step reports the fault-injected boundary proof now, publishes the exact
   queries beside the 81.2%/37.8% baselines, and the live shares are
   reported as they accrue -- the same starved-measurement shape 86.74's C6
   carries, stated rather than hidden.
4. **Criterion 6:** no gate touched; the risk-judge fallback (61.9%) stays
   out of scope.
5. **Criterion 7 (numbered asks):**
   - ASK-1: promote `paper_synthesis_integrity_enabled` (arms the 61.2
     fail-closed raise; its predicate measured 211/211 + 38/38). Operator
     decision + restart timing.
   - ASK-2: queue the CC-rail structured-output mismatch (2,859 parse
     failures across agents; synthesis 9.2%) as its own step.
   - ASK-3: restart timing -- session end (default) vs immediate (first
     post-fix cycle tonight).
6. **Criterion 8:** mutation cells -- restore the fabricated HOLD at the
   site (test must go red); drop the `analysis_failed` marker (red); make a
   consumer read NULL as HOLD (red); control green first, byte-identical
   restore.

## Numbered operator asks

ASK-1 / ASK-2 / ASK-3 as above (plan step 5).

## References

`research_brief_86.69.md` (sections A1-A5, B1-B3, D1-D4; 9 sources incl.
Google SRE data-integrity + cascading-failures, AWS Write-Audit-Publish,
arXiv 2606.08162 / 2606.05806 / 2408.02442v3 on LLM structured-output
reliability); `q1_binding_constraint_86.59.md` (populations A/B, the row
signature); phase-61.2 (`settings.py:206-208`); phase-60.1 (`fa62b5fe`);
phase-78.1 CC-rail measurement (auto-memory).
