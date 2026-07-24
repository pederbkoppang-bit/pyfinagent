# Contract — Step 78.0 (Anthropic call-site CENSUS, audit-class)

Date: 2026-07-25 | Cycle: 158 | Executor: Main (Opus 5) | Research gate: **PASSED (audit-class)**

## Research-gate summary

`handoff/current/research_brief_78.0.md` — **audit-class** sweep, 9 rounds,
`coverage.dry = true` (loop-until-dry: rounds continued until two consecutive rounds
surfaced zero new call sites beyond de-dup), envelope `gate_passed: true`. The sweep
found 28 call sites across A–K, including the 4 previously-missed raw-SDK sites named
in the step scope.

Gate outputs consumed here:
- The A–K census with file:line anchors.
- `handoff/current/census_78_decision_draft.md` — Main's staged decision draft.
- The confirmed exclusion of MCP servers (verified zero outbound LLM calls) and the
  standing rule that `_inventory.json` is NEVER ground truth.

## Hypothesis

A per-role routing decision table, grounded in re-derived anchors and measured
volumes, converts "route everything to the Max rail" from a slogan into an
executable, prioritized queue — and makes the *justified* stay-metered rows explicit
so nobody re-litigates them later.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. Census covers EVERY area in the step scope incl. the 4 previously-missed raw-SDK
   sites, with file:line anchors re-derived (not copied from the goal text)
2. Audit-class research gate ran with coverage.dry=true (loop-until-dry; a fixed-list
   census under-covers the tail)
3. Every role row carries a decision from {max_rail_cli, max_rail_proxy, stay_metered}
   with a stated reason; volume/frequency measured from llm_call_log or honestly
   marked unmeasurable
4. advisor_call and BatchClient rows are stay_metered with the no-CC-equivalent
   evidence cited
5. The census names which follow-up step (78.x, authored in this step's close) owns
   each max_rail decision

Immutable verification command:
```
python3 -c "import json; c=json.load(open('handoff/current/census_78.json')); assert len(c['roles'])>=12, ...; assert all(r.get('decision') in ('max_rail_cli','max_rail_proxy','stay_metered') for r in c['roles']), 'undecided rows'"
```

## Plan

1. **Re-derive every anchor MECHANICALLY** (criterion 1). Not an LLM re-read: a script
   that opens each claimed `file:line` and asserts the expected symbol is present
   there, reporting the ACTUAL line and any drift. A grep is stronger evidence than a
   second opinion.
2. **Measure volumes** (criterion 3) with a verbatim 30d `GROUP BY provider, model,
   agent` over `pyfinagent_data.llm_call_log`.
3. **Audit instrumentation per site** — for each call site, does it write an
   `llm_call_log` row at all? Required so that "0 rows" is not silently reported as
   "0 calls" for uninstrumented paths (the honest-unmeasurable half of criterion 3).
4. **Generate both artifacts from ONE source of truth** (`census_78.json` →
   `census_78.md`) so the human-readable and machine-readable copies cannot drift.
5. Assign every row a decision + reason + owning follow-up step (criteria 3, 5), with
   advisor_call and BatchClient carrying explicit no-CC-equivalent evidence
   (criterion 4).
6. Author the 78.1+ follow-up steps, each executor-tagged and research-gated.
7. Q/A → `harness_log.md` append → masterplan flip (log LAST).

## Boundaries

- **READ-ONLY deliverable.** This step changes NO production code and flips NO flags.
  Everything it finds becomes a follow-up step, per
  `feedback_queue_discovered_defects_in_masterplan`.
- Decisions are recommendations; the operator owns every actual routing flip.
- MCP servers remain out of scope (verified zero outbound LLM calls).

## References

- `handoff/current/research_brief_78.0.md` (audit-class gate, coverage.dry=true)
- `handoff/current/census_78_decision_draft.md` (staged draft)
- Anchor re-derivation + volume + instrumentation evidence: `handoff/current/live_check_78.0.md`
- phase-72 rail-bypass root-cause (the C-block's money relevance)
