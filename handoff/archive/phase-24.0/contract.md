# Sprint Contract — phase-24.0 — Audit Charter + Red-Line Invariants

**Cycle:** phase-24 cycle 1
**Date:** 2026-05-12
**Step ID:** 24.0
**Step name:** Phase-24 audit charter + red-line invariants (gates 24.1-24.13)
**Priority:** P2 (charter — gates all P0/P1/P2 buckets below)
**Depends on:** none
**Harness required:** true
**Audit basis:** `project_system_goal.md` (memory) + `/Users/ford/.claude/plans/sunny-jingling-deer.md` (approved plan 2026-05-12)

---

## Research-gate summary

Researcher subagent ran 2026-05-12. **`gate_passed: true`** with envelope:

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```

Six sources fetched in full via WebFetch (3 canonical Anthropic engineering posts + Claude Code hooks reference + 2 arXiv preprints on financial-AI orchestration). Three-variant search-query discipline applied (current-year frontier / last-2-year window / year-less canonical). Recency scan surfaced 4 new findings in the 2024-2026 window; none supersede the canonical Anthropic patterns. Full brief at `handoff/current/research_brief.md`.

---

## Hypothesis

The 15-bucket structure (24.0 charter + 24.1-24.14 = 14 audit buckets) is sufficient and non-overlapping to cover the entire pyfinagent codebase against the profit-maximization red-line goal: "maximize profit at lowest cost live; dynamically shift strategy to whichever is making the most money" (`project_system_goal.md`).

**Researcher verdict: CONFIRMED** (one documentation gap noted — masterplan.json `depends_on_step: "24.9"` on bucket 24.13 vs master prompt's "24.1-24.9 complete"; non-structural).

---

## Success criteria (immutable — copied verbatim from `.claude/masterplan.json` step 24.0 `verification.success_criteria`)

1. `findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_0_charter_findings_md`
2. `charter_cites_project_system_goal_memory`
3. `charter_embeds_coverage_matrix_for_backend_frontend_infrastructure`
4. `charter_lists_15_buckets_24_0_through_24_14`
5. `charter_documents_research_gate_canonical_url_whitelist_per_bucket`
6. `charter_documents_findings_md_path_convention`
7. `research_gate_envelope_present_with_gate_passed_true`
8. `external_sources_count_at_least_5`
9. `anthropic_harness_design_url_cited_verbatim`
10. `harness_log_has_phase_24_0_cycle_entry`

**Verifier command:** `source .venv/bin/activate && python3 tests/verify_phase_24_0.py`
**Live check:** `ls docs/audits/phase-24-2026-05-12/24.0-charter-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.0-charter-findings.md`

---

## Plan steps

1. **Write findings doc** at `docs/audits/phase-24-2026-05-12/24.0-charter-findings.md` with:
   - YAML frontmatter (bucket, slug, cycle, cycle_date, researcher_gate)
   - Executive summary (1 paragraph TL;DR)
   - Code-grounded findings — confirm 15-bucket structure with file:line anchors to `.claude/masterplan.json:7895-8399`, `scripts/audit/phase_24_audit_prompt.md`, `project_system_goal.md`
   - External-research summary — verbatim URLs for all 6 read-in-full sources
   - Recency scan (2024-2026) section — 4 findings reported
   - Coverage matrix (codebase path → bucket) — full verbatim from research brief
   - Canonical-URL whitelist per bucket (so subsequent buckets' researchers have a single source of truth)
   - Findings-doc path convention (every bucket's expected filename)
   - At least 3 phase-25 candidate steps (charter-level: e.g., automate verifier wiring, schedule a stress-test loop, document the red-line metric)
   - Open questions (e.g., the 24.13 depends_on documentation gap)
   - References (all 16 URLs collected)
2. **Write `experiment_results.md`** at `handoff/current/experiment_results.md` with verbatim verifier output from `python3 tests/verify_phase_24_0.py`.
3. **Spawn Q/A subagent** with the 5-item harness-compliance audit first; let it run the verifier independently and return a verdict envelope.
4. **On Q/A PASS**, append harness_log.md with `## Cycle N -- 2026-05-12 -- phase=24.0 result=PASS` block.
5. **Flip masterplan status to done** — auto-commit-and-push hook fires.

**Phase-24 is READ-ONLY**: no code changes in this cycle. All proposed fixes are phase-25.x candidates emitted in the findings doc.

---

## Live-check gate evidence

After GENERATE, the live_check field requires:
- `ls docs/audits/phase-24-2026-05-12/24.0-charter-findings.md` returns the file
- `head -30 docs/audits/phase-24-2026-05-12/24.0-charter-findings.md` shows the frontmatter + executive summary

Operator records this evidence in `handoff/current/live_check_24.0.md` before auto-push fires.

---

## References

From research brief (read in full via WebFetch):
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://www.anthropic.com/engineering/building-effective-agents
- https://www.anthropic.com/engineering/built-multi-agent-research-system
- https://code.claude.com/docs/en/hooks
- https://arxiv.org/html/2603.13942v1
- https://arxiv.org/html/2512.02227v1

Internal:
- `.claude/masterplan.json` lines 7895-8399 (phase-24 entry)
- `scripts/audit/phase_24_audit_prompt.md` (943 lines — master prompt)
- `project_system_goal.md` (memory — red-line spec)
- `docs/audits/dev-mas-2026-05-11/` (prior-art shape to mirror)
- `CLAUDE.md` (harness MAS layer = Main + Researcher + Q/A)
- `ARCHITECTURE.md` (4-layer architecture)
- `.claude/rules/research-gate.md`
- `.claude/rules/backend-agents.md`
- `.claude/rules/frontend.md`
