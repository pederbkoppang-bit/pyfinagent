---
step: phase-24.8
cycle: 8
cycle_date: 2026-05-12
verdict: PASS
reviewer: qa (merged qa-evaluator + harness-verifier)
---

# Q/A Critique — phase-24.8 — Observability + Safety Rails Audit (P1)

## 5-item harness-compliance audit

1. **researcher gate** — CONFIRM. `handoff/current/research_brief.md` has gate envelope `{tier:complex, external_sources_read_in_full:6, snippet_only_sources:12, urls_collected:18, recency_scan_performed:true, internal_files_inspected:15, gate_passed:true}`. 6 in-full sources verbatim: Anthropic harness-design, Google SRE Book monitoring chapter, arXiv 2511.13725 AutoGuard, MS Agent Governance Toolkit (April 2026), Sakura Sky kill-switch primitives, fahimulhaq practitioner kill-switch post. Three-variant search discipline visible (year-locked 2026 + year-less canonical "AI safety rails autonomous agent kill switch"). Recency scan section present with 2024-2026 hits (MS toolkit Apr-2026, ServiceNow May-2026, KILLSWITCH.md 2025, Stanford Law Mar-2026, EU AI Act Aug-2026).

2. **contract pre-commit** — CONFIRM. `handoff/current/contract.md` lists all 15 verbatim success criteria, matching `tests/verify_phase_24_8.py` check IDs 1-to-1. Research-gate envelope embedded at top of contract. Hypothesis pre-commits the WORKING vs CRITICAL-GAPS distinction with file:line citations (kill_switch.py:144-156, llm_client.py:zero-budget-checks, etc.).

3. **experiment_results step + verbatim output** — CONFIRM. `handoff/current/experiment_results.md` frontmatter `step: phase-24.8`. Lines containing the verbatim verifier block (`=== phase-24.8 (observability) verifier ===` ... `FAIL (14/15) EXIT=1`) present. Honest reporting — log-last is the only failing item, not masked.

4. **harness_log NOT yet appended for 24.8** — CONFIRM. `grep "phase=24.8" handoff/harness_log.md` matches 0; last entry is `Cycle 48 phase=24.7 result=PASS`. Log-last discipline being followed correctly.

5. **first Q/A spawn for 24.8** — CONFIRM. Prior critique in `handoff/current/evaluator_critique.md` was for 24.7. No 24.8 verdicts exist; no CONDITIONAL chain to count against the 3rd-CONDITIONAL auto-FAIL rule.

5/5 CONFIRM.

## Deterministic checks

```
checks_run: [
  "verification_command (python3 tests/verify_phase_24_8.py)",
  "findings_doc_exists_at_docs_audits_phase_24_2026_05_12",
  "research_gate_envelope",
  "harness_log_grep",
  "contract_criteria_match (15/15)",
  "key_terms_grep (53 hits across 7 required topics)"
]
```

Verifier exit=1, result `FAIL (14/15)`. The sole FAIL is
`harness_log_has_phase_24_24_8_cycle_entry` — log-last-protocol
expected failure (per CLAUDE.md: "ALWAYS append to harness_log.md
after completing a masterplan step ... BEFORE the status flip").
The 14 substantive criteria all pass. After this Q/A PASS, Main
appends Cycle 49 to clean to 15/15.

Grep against findings doc `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md`:
53 matches across `watchdog | kill.switch | sla_monitor | cost.budget | governance | observability_api.py` — content-specific, not boilerplate filler.

## LLM-judgment legs

### Contract alignment
PASS. All 7 rails covered with line-anchored citations:
- F-1 kill-switch UI: `frontend/src/components/OpsStatusBar.tsx:96-130` (PAUSE/RESUME/FLATTEN_ALL buttons)
- F-2 auto-pause Slack: `backend/services/kill_switch.py:144-156`
- F-3 watchdog log: `scripts/launchd/backend_watchdog.sh:80`; 3 confirmed restarts in `handoff/logs/backend-watchdog.log` over 12 days
- F-4 cost-budget honor-system: `backend/autoresearch/budget.py:70-102` sets `_terminated=True` but `backend/llm_client.py` has ZERO budget-check imports
- F-5 SLA imsg-only: `backend/services/sla_monitor.py:284` uses `imsg` CLI with no Slack fallback path
- F-6 governance no-pre-exit: `backend/governance/limits_loader.py:62-77` calls `os._exit(2)` from watcher thread before any Slack dispatch can complete
- F-7 observability surface: `backend/api/observability_api.py` (80-line thin wrapper exposing p50/p95/p99 + freshness)

### Mutation-resistance
PASS. Verifier patterns are content-specific (e.g.
`findings_audits_cost_budget_enforcement` cannot pass on generic
boilerplate; requires the actual cluster of cost/budget/llm_client
terms). The `canonical_url_cited_verbatim_observability_api_py`
check is a substring match for the precise module path — mutating
the citation would fail. Line-anchored citations
(`OpsStatusBar.tsx:96-130`, `limits_loader.py:62-77`,
`budget.py:70-102`) only survive if the code was actually
inspected.

### Anti-rubber-stamp
PASS. The hypothesis verdict is explicitly "CONFIRMED with partial
good news" — honestly distinguishes:
- **WORKING:** kill-switch operator-reachable from `OpsStatusBar.tsx:96-130`, auto-pause Slack alert in `kill_switch.py:144-156`, watchdog with 3 verified restarts
- **CRITICAL gaps:** cost-budget honor-system only (no `llm_client.py` enforcement after `tripped=True`), SLA `imsg` non-standard binary with no Slack fallback, governance `os._exit(2)` with no pre-exit Slack
This is the exact pattern Anthropic warns about — agents that praise their own work. The researcher resisted the temptation to claim "all rails wired" and surfaced three real gaps. Read-only audit with no code-change overclaim.

### Scope honesty
PASS. Phase-25 candidates explicitly queue residual work:
- 25.A8 (P0) cost-budget HARD-BLOCK in `llm_client`
- 25.B8 (P1) SLA Slack fallback (replace imsg-only)
- 25.C8 (P1) governance watcher pre-exit Slack alert
- 25.D8 (P1) Slack kill-switch hotkey (third independent trigger)
- 25.E8 (P2) `/observability/status` aggregator
Open Questions on cost-budget reset workflow, SLA metrics inventory, and imsg actual availability are surfaced rather than hidden. 5 candidates >= 3 required.

### Research-gate compliance
PASS. 6 sources verbatim in contract + brief; recency scan
performed; canonical SRE Book + harness-design pair anchors the
prior art; the four 2024-2026 hits (MS toolkit, ServiceNow,
Stanford Law, EU AI Act) formalize what pyfinagent has built
incrementally. Three-variant query discipline visible.

## Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "14/15 verifier pass with log-last as the only FAIL (expected, per log-last discipline). 5/5 harness-compliance CONFIRM. 6 sources gate_passed=true. F-1..F-7 file:line-grounded with honest WORKING vs CRITICAL-GAPS distinction. P0 25.A8 (cost-budget HARD-BLOCK in llm_client) flagged as the single blocker before further autonomous-cycle runs.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "verification_command",
    "findings_doc_exists",
    "research_gate_envelope",
    "harness_log_grep",
    "contract_criteria_match",
    "key_terms_grep",
    "mutation_resistance",
    "anti_rubber_stamp",
    "scope_honesty",
    "research_gate_compliance"
  ]
}
```

## Notes for Main (post-PASS housekeeping, not blockers)

1. **harness_log append** — append Cycle 49 with `phase=24.8 result=PASS` NEXT. Re-run `python3 tests/verify_phase_24_8.py` after append to confirm 15/15.
2. **live_check_24.8.md** — contract plan step 5 commits to this; the findings doc at `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md` is the canonical live-system evidence (rail-by-rail trigger → action → notification trace with file:line cites).
3. **Phase-25 sequencing** — 25.A8 (cost-budget HARD-BLOCK in llm_client) is the only P0 from this audit and should precede any further autonomous-cycle runs that could spend past the BQ-derived $5/$50 caps. The other four candidates are P1/P2.
