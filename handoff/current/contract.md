# Contract — step 71.3 (harden Q/A judgment + machine-readable verdicts, WITHIN the single Q/A role)

**Phase:** phase-71 | **Step:** 71.3 | **Priority:** P2 | harness_required: true | depends_on: 71.0 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** harness-protocol / qa.md rubric edit + evaluator_critique.json emission.
$0, local-only, NO production/live-loop change. Edits `.claude/agents/qa.md` → separation-of-duties + roster note.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0), run wf_f8bce7de-c96. Envelope: **gate_passed=true**,
tier=moderate, **6 external sources read in full**, 10 snippet-only, 24 URLs, recency scan, 12 internal files.
Brief: `research_brief_71.3.md`. Grounding HOLDS. **Reconciliation resolved** (the crux): phase-71.0 DROPPED #8a
(worst-of-N **self-consistency** = N IDENTICAL samples, "no independent signal / correlated bias") but KEPT #8b
(the adversarial red-team leg). 71.3 implements ONLY the **"FROM N LENSES"** reading — the single Q/A judges the
claimed PASS from N DISTINCT adversarial lenses (correctness / does-it-reproduce / scope-honesty) and takes the
WORST. This (i) satisfies the criterion + the `adversarial`/`worst-of-N` grep, and (ii) is CONSISTENT with 71.0
(which objected to N-identical resampling, NOT perspective diversity). Grounded: arXiv:2505.19477 (perspective-
diverse meta-judge resists bias), arXiv:2508.06709 (self-bias correlated across a judge's own samples), Anthropic
judge-rubric "completeness" + doer/judge separation. GENERATE MUST explicitly NEGATE the N-identical #8a.

## Plan (line-anchored)

### A. `.claude/agents/qa.md` (edits → separation-of-duties + roster note in harness_log)
1. **Contract-completeness dimension** — add a bullet to the §4 LLM-judgment list (after "Research-gate
   compliance", ~L247) requiring the Q/A to map EVERY contract/immutable criterion → covering evidence in
   `experiment_results.md`; an uncovered criterion = `Missing_Assumption` that caps the verdict. Add a
   "Contract completeness" row to the Quality-criteria table (~L304-311) to unambiguously land the `completeness`
   grep token.
2. **Adversarial worst-of-N-LENSES leg (P0/P1 money-path only)** — new subsection "### 4a. Adversarial
   worst-of-N-LENSES verdict (P0/P1 money-path only)" after §4: the SAME single Q/A evaluates the claimed PASS from
   N DISTINCT lenses (correctness / does-it-reproduce / scope-honesty) and takes the WORST verdict — WITHIN the
   single Q/A role (no fourth agent, no re-split). MUST state: "worst-of-N over N distinct LENSES, NOT the
   N-identical self-consistency resampling (#8a, dropped in phase-71.0 as correlated / no independent signal)."
   Lands the `adversarial` + `worst-of-N` grep tokens.
3. **evaluator_critique.json emission** — note (after the Output-format schema, ~L267, + a Constraints clause
   ~L315-318) that MAIN persists the returned verdict VERBATIM to `handoff/current/evaluator_critique.json`
   (injecting `step_id` + `cycle_num`; transforming `checks_run` to an object map) alongside the .md. **Q/A stays
   read-only** — Main is the scribe (mirrors the verbatim .md transcription → no-self-eval holds). Do NOT edit the
   71.1-owned qa-verdict.js VERDICT_SCHEMA.

### B. `docs/runbooks/per-step-protocol.md` §4 EVALUATE
Add a paragraph (after the "Returns the JSON schema" block, ~L148) documenting the completeness dimension + the
P0/P1 N-lens leg (distinct lenses, NOT N-identical) + the `evaluator_critique.json` persistence; add the JSON
filename to the LOG/flip step notes.

### C. Status-flip gate reads JSON (criterion 2)
- **Persist evaluator_critique.json for 71.3 itself** (dogfood) from this step's Q/A return value.
- **Fail-open `verdict_gate.py`** under `.claude/hooks/lib/` mirroring `live_check_gate.py` (proceed on
  missing/unreadable JSON = fail-open; passed iff `verdict=="PASS"` and `ok==true`; else WARN + hold-push), wired
  into `auto-commit-and-push.sh` after the existing gate block — IF reading the existing gate confirms a clean
  fail-open mirror (else fall back to documented Main-discipline read, which still satisfies "the gate CAN read it
  deterministically"). The gate must NEVER break the masterplan Write (fail-open discipline).

## Immutable success criteria (verbatim from masterplan.json 71.3)

1. qa.md rubric includes a contract-completeness dimension and (for P0/P1 money-path steps) an adversarial
   worst-of-N / self-consistency verdict, both explicitly WITHIN the single Q/A role (no fourth agent, no re-split)
2. Q/A emits a machine-readable evaluator_critique.json alongside the markdown with the verdict schema; the
   status-flip / live_check gate can read it deterministically
3. The single-Q/A-per-step rule and file-based handoffs are preserved

Verification command (immutable):
`bash -c 'grep -Eqi "contract completeness|completeness" .claude/agents/qa.md && grep -Eqi "worst-of-n|self-consistency|adversarial" .claude/agents/qa.md docs/runbooks/per-step-protocol.md'`

## Boundaries (binding)
$0; local-only; NO production/live-loop change (harness-protocol + docs + a fail-open hook only). WITHIN the single
Q/A role — exactly-3-agents, NO re-split, NO fourth agent. The dropped N-identical #8a is NOT re-introduced (the
N-lens leg explicitly negates it). Do NOT edit the 71.1-owned qa-verdict.js VERDICT_SCHEMA. Any hook is FAIL-OPEN
(never breaks the masterplan Write / push). Q/A stays read-only (Main persists the JSON). Separation of duties +
roster snapshot: edits qa.md → harness_log requests Peder review + verify_qa_roster_live.sh next session (the
Workflow path reads qa.md from disk live; only the Agent-tool type snapshots). historical_macro FROZEN.

## References
research_brief_71.3.md; design_harness_mas_71.md §71.3 (kept #6+#8b, DROP #8a); harness_proposals.json;
arXiv:2505.19477 + arXiv:2508.06709; qa-verdict.js VERDICT_SCHEMA; live_check_gate.py (mirror pattern).
