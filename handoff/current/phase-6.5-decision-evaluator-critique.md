# Q/A Evaluator Critique — phase-6.5-decision (Path D)

**Verdict token:** `qa_phase65_decision_v1`
**Date:** 2026-04-19
**Cycle:** 1 (first Q/A on this meta-decision — no verdict-shopping)
**Reviewer:** qa (merged qa-evaluator + harness-verifier)

---

## 1. Five-item protocol audit

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawn proof | PASS | `handoff/current/phase-6.5-decision-research-brief.md` present; 7 sources read in full via WebFetch (floor is 5); source table has Tier column (Peer-reviewed / Industry practitioner); three-variant query table explicit (2026 / 2025 / year-less canonical); "Recency scan (2024-2026)" section present with 6 enumerated findings; JSON envelope `gate_passed: true`, `external_sources_read_in_full: 7`, `recency_scan_performed: true`. |
| 2 | Contract PRE-commit | PASS | mtimes: brief 20:46:50 < contract 20:47:50 < masterplan 20:48:06 < experiment-results 20:48:32. Contract authored 16 sec before masterplan edit. |
| 3 | Experiment results present | PASS | Per-step diff table (9 rows, before/after/notes), verbatim JSON-validity check output (`valid`), contract-criterion check table (4 criteria with PASS/PENDING and evidence). |
| 4 | Log-last discipline | PASS | Last block of `handoff/harness_log.md` is "Meta — 2026-04-19 20:30 UTC — phase=6.5 action=RATIFIED" (the phase-level ratification from earlier), NOT a phase-6.5-decision Path-D block. Path-D block is correctly deferred until after this Q/A PASS. |
| 5 | No verdict-shopping | PASS | No prior `phase-6.5-decision-evaluator-critique*.md` exists in `handoff/current/` or `handoff/archive/`. First Q/A on this decision. |

---

## 2. Deterministic checks (A–H)

### A. JSON validity — PASS

```
$ python3 -c "import json; json.load(open('.claude/masterplan.json')); print('ok')"
ok
```

### B. Step status matrix — PASS

```
path_decision: {'selected': 'D', 'decided_at': '2026-04-19T18:48:06.575444+00:00', ...}
  phase-6.5.1     status=pending    sup_by=-               drop_reason=N
  phase-6.5.2     status=pending    sup_by=-               drop_reason=N
  phase-6.5.3     status=dropped    sup_by=phase-7.2       drop_reason=Y
  phase-6.5.4     status=dropped    sup_by=-               drop_reason=Y
  phase-6.5.5     status=dropped    sup_by=-               drop_reason=Y
  phase-6.5.6     status=dropped    sup_by=phase-7.5       drop_reason=Y
  phase-6.5.7     status=pending    sup_by=-               drop_reason=N
  phase-6.5.8     status=dropped    sup_by=phase-6.5.9     drop_reason=Y
  phase-6.5.9     status=pending    sup_by=-               drop_reason=N
```

Matches contract exactly: 4 kept {6.5.1, 6.5.2, 6.5.7, 6.5.9} as pending; 5 dropped with `dropped_reason`; 3 have `superseded_by` pointers (6.5.3→phase-7.2, 6.5.6→phase-7.5, 6.5.8→phase-6.5.9).

### C. Immutable verification preserved — PASS

For each kept step, `verification.command` + `verification.success_criteria` inspected:

- **6.5.1**: `python scripts/migrations/phase_6_5_intel_schema.py` + `['migration_dry_run_exit_0', 'all_intel_tables_defined_in_script', 'schema_test_green']`.
- **6.5.2**: `pytest backend/tests/test_intel_source_registry.py` + `['registry_loads_all_configured_sources', 'scanner_dry_run_returns_candidates', 'tests_green']`.
- **6.5.7**: `pytest backend/tests/test_intel_novelty_client.py` + `['voyage_primary_gemini_fallback_smoke_ok', 'novelty_score_distinguishes_duplicate_vs_novel', 'prompt...']`.
- **6.5.9**: `python scripts/smoketest/intel_e2e.py --fixtures` + `['overall_ok_true', 'at_least_one_record_per_extractor_family', 'novelty_and_digest_stages_pass', ...]`.

These match the 20:30 UTC authoring block in `handoff/harness_log.md`. No mutation.

### D. Path decision metadata — PASS

`phase.path_decision.selected == "D"`. `path_decision.contract = "handoff/current/phase-6.5-decision-contract.md"` and `path_decision.research_brief = "handoff/current/phase-6.5-decision-research-brief.md"` — both files exist on disk (verified via `ls -la handoff/current/`).

### E. Scope check — PASS (with observation)

```
$ git status --short
M .claude/masterplan.json
M handoff/current/evaluator_critique.md
M handoff/current/experiment_results.md
... (+ unrelated ambient changes to backend/* from prior uncommitted work not part of this cycle)
```

**Observation:** backend/* files appear in `git status` but `find backend/ -newer phase-6.5-decision-research-brief.md` returns EMPTY. No backend source file was modified during this phase-6.5-decision cycle (20:46:50 UTC onward). The backend changes are pre-existing ambient state, not this cycle's work. Scope discipline upheld for THIS cycle.

### F. Evidence sanity — PASS

The "WSB high-attention returns -8.5% holding period" claim is attributed in the brief to:
- `sciencedirect.com/science/article/pii/S1057521924006537` — Elsevier *International Review of Financial Analysis* 2024 (peer-reviewed, snippet-only in brief but publisher-tier correct).
- Corroborated by `pmc.ncbi.nlm.nih.gov/articles/PMC10111308/` — Springer *Digital Finance* 2023, read-in-full, giving -7.8% 1-year alpha and Sharpe ~50% of market.

Both URLs resolve to peer-reviewed journal hosts (Elsevier + NIH/PMC Springer mirror), not to blogs or preprint aggregators. The two sources agree directionally. Citation shape is defensible.

### G. Overlap claim sanity — PASS (with ID-format note)

Phase-7 exists; step IDs `7.2` ("13F institutional holdings ingestion") and `7.5` ("Reddit WSB sentiment ingestion") exist and are `status: pending`. The `superseded_by` values in phase-6.5 are written as `phase-7.2` / `phase-7.5`, while the literal step IDs inside phase-7 are `7.2` / `7.5`. Semantically unambiguous (phase-7 prefix is conventional) but the exact string doesn't match. Non-blocking pointer-format inconsistency — flagged for future resolver code.

### H. Anti-rubber-stamp — PASS

The research brief's "Recommendation" section (§ lines 154-172) autonomously proposes Path D with four distinct arguments BEFORE contract authoring. The contract's "Selected path" block (§ lines 16-22) copies Path D faithfully and names it "researcher-recommended." No smuggled-in alternative. Main did not hand-pick evidence; the brief itself explicitly argues against Paths A/B/C with evidence citations (§ lines 166-172). Mutation-resistant.

---

## 3. LLM judgment

**System-goal alignment:** Path D is directly aligned with "maximize Net System Alpha = Profit − (Risk Exposure + Compute Burn)." Dropping 5 source-specific extractors (6.5.3/.4/.5/.6/.8) reduces compute burn by ~55% of phase-6.5 scope with zero alpha cost (peer-reviewed evidence: WSB has negative long-term alpha; academic papers decay 50-60% post-publication). Keeping 6.5.7 (prompt-patch queue → phase-8.5 soft-seed) preserves the one load-bearing path from intel to alpha. This matches the "dynamically shift strategy to whichever is making the most money" directive by redirecting effort from speculative ingestion to the Karpathy-style autoresearch loop (phase-8.5) with ATLAS-style precedent.

**Evidence-based drops:** 6.5.6's drop cites Springer 2023 + ScienceDirect 2024 — both peer-reviewed. 6.5.4's drop cites McLean-Pontiff 2016 (JoF, canonical). 6.5.3's drop cites Meta v. Bright Data (2024) ToS precedent + absence of alpha evidence. All drop rationales rest on published or legal-precedent evidence, not intuition.

**Weakest link in Path D (non-blocking observation):** 6.5.7 (novelty client + prompt-patch queue) currently has **no source-specific extractor feeding it**, because 6.5.3/.4/.5/.6 are all dropped. The queue is supposed to receive ingested reports and score novelty — without extractors, only phase-7 alt-data streams feed it, and phase-7 is oriented toward features-for-backtest, not toward prompt-patches-for-proposer. The risk is that 6.5.7 ships with an empty input pipe. Mitigation already anticipated in the research brief's risk register ("Prompt-patch queue produces low-novelty patches in practice") but worth an explicit follow-up: before 6.5.7 ships, the contract for 6.5.7 must specify which phase-7 step(s) write to `intel_prompt_patches`, or accept that the table will be seeded manually / by an out-of-band extractor deferred to a later phase.

**Missed path (Path E) check:** I considered whether a "Path E" — defer ALL of 6.5 including the scaffolding, and seed phase-8.5 directly from phase-7 features — would be superior. Rejected: 6.5.1 (BigQuery intel schema) + 6.5.7 (novelty client) deliver reusable infrastructure (novelty scoring generalizes to any text input, including phase-7 Reddit/Congress feeds) that phase-7 would otherwise duplicate. Path D is not beaten by Path E. No Path F/G identified.

**Soft-seed wiring:** The contract's §3 correctly specifies that prompt-patches feed 8.5.3 as a *soft-seed* (read-only proposer input), NOT auto-ratification. This preserves the CLAUDE.md/phase-6.5 hard boundary ("no automatic capital allocation changes"). Boundary is intact.

---

## 4. Violations / observations

**violated_criteria:** []

**violation_details (non-blocking observations, per VeriPlan triplets):**

```json
[
  {
    "violation_type": "Overgeneralization",
    "action": "superseded_by: 'phase-7.2' / 'phase-7.5'",
    "state": "phase-7 step IDs are literal '7.2' / '7.5', not 'phase-7.2' / 'phase-7.5'",
    "constraint": "superseded_by should match the exact step-id string to be resolvable by automated tooling"
  },
  {
    "violation_type": "Missing_Assumption",
    "action": "keep 6.5.7 prompt-patch queue",
    "state": "no source-specific extractors remain to populate it; phase-7 writes features, not prompt-patches",
    "constraint": "6.5.7 contract must specify input source(s) or accept empty-pipe at ship time"
  }
]
```

Neither observation rises to FAIL or CONDITIONAL. Both are forward-pointers for future contracts.

**checks_run:** `["audit_5_item", "json_validity", "step_status_matrix", "immutable_verification_preserved", "path_decision_metadata", "scope_check", "evidence_sanity_WSB", "overlap_phase7", "anti_rubber_stamp", "llm_goal_alignment", "weakest_link", "missed_path_E_consideration"]`

---

## 5. Final decision

**PASS** — `qa_phase65_decision_v1`

Path D is evidence-based, matches system goal, preserves immutable criteria on kept steps, and does not smuggle in auto-ratification. Drop rationales cite peer-reviewed sources. Main does not redirect 5 steps on vibes; the research brief carries the argument. Two non-blocking observations recorded for future action (pointer-format, 6.5.7 input-source spec) but neither blocks ratification of the meta-decision.

**JSON envelope:**

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Path D is researcher-recommended with peer-reviewed evidence backing each drop; contract pre-commits immutable criteria; masterplan edit matches contract exactly; no verdict-shopping; no scope creep; weakest-link and missed-path checks clear.",
  "violated_criteria": [],
  "violation_details": [
    {"violation_type": "Overgeneralization", "action": "superseded_by string format", "state": "'phase-7.2' vs literal '7.2'", "constraint": "resolver-safe id match"},
    {"violation_type": "Missing_Assumption", "action": "keep 6.5.7 without source-extractors", "state": "no populator remains after drops", "constraint": "6.5.7 contract must specify input sources"}
  ],
  "certified_fallback": false,
  "checks_run": ["audit_5_item", "json_validity", "step_status_matrix", "immutable_verification_preserved", "path_decision_metadata", "scope_check", "evidence_sanity_WSB", "overlap_phase7", "anti_rubber_stamp", "llm_goal_alignment", "weakest_link", "missed_path_E_consideration"],
  "token": "qa_phase65_decision_v1"
}
```
