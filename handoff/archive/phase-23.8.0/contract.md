---
step: phase-23.8.0
title: Dev-MAS audit remediation Bundle-1 (R-3 / R-4 / R-7; R-6 deferred after research gate)
cycle_date: 2026-05-11
harness_required: true
verification: 'python3 tests/verify_phase_23_8_0.py'
research_brief: (researcher subagent, 2026-05-11; gate_passed=true; 6 sources read in full, 16 URLs, recency scan performed)
audit_basis: docs/audits/dev-mas-2026-05-11/04-remediation.md
---

# Contract — phase-23.8.0

**Step**: phase-23.8.0 — Dev-MAS audit remediation Bundle-1
(R-3 / R-4 / R-7; R-6 deferred after research gate).

**Date**: 2026-05-11.

**Status target**: pending → done.

**Hypothesis**:
The dev-MAS cleanups in R-3 (rename in-app Layer-2 agent labels to
break Layer-3 namespace collisions), R-4 (move `META_PLAN` thresholds
from hardcoded planner-prompt constants to a runtime config file),
and R-7 (document the 28-skill-to-5-drawer-layer-to-3-lite-mode-rows
mapping in `ARCHITECTURE.md`) can be landed in **one cycle** without
regressing any active code path. R-6 (delete two `DEPRECATED` stub
files) is **not landable in this cycle** because both files have
live importers; it is documented as deferred.

## Research-gate summary

Researcher subagent ran 2026-05-11 (tier: moderate). JSON envelope:
`{"external_sources_read_in_full": 6, "snippet_only_sources": 10,
"urls_collected": 16, "recency_scan_performed": true,
"internal_files_inspected": 9, "gate_passed": true}`.

Key external citations (≥5 sources read in full via WebFetch):

1. HARNESS-DOC
   (`https://www.anthropic.com/engineering/harness-design-long-running-apps`)
   — stress-test doctrine: "every component in a harness encodes
   an assumption about what the model can't do on its own, and
   those assumptions are worth stress testing." Grounds the entire
   audit's framing.
2. MULTI-DOC
   (`https://www.anthropic.com/engineering/multi-agent-research-system`)
   — four-component delegation: "Each subagent needs an objective,
   an output format, guidance on the tools and sources to use, and
   clear task boundaries." Grounds R-3 (clear task boundaries =
   non-colliding names).
3. EFFECTIVE-DOC
   (`https://www.anthropic.com/engineering/building-effective-agents`)
   — "you should consider adding complexity _only_ when it
   demonstrably improves outcomes." Grounds R-7 (smaller doc fix
   over bigger code fix when possible).
4. SUBAGENT-DOC (`https://code.claude.com/docs/en/sub-agents`) —
   "Each subagent runs in its own context window with a custom
   system prompt, **specific tool access**, and independent
   permissions." Grounds R-3 (specialize behavior with focused
   prompts).
5. blakecrosley.com (2025 agent-architecture practitioner guide) —
   "externalize all deterministic requirements (thresholds,
   commands, validation rules) to configuration files and hooks,
   never embed them solely in prompts." Grounds R-4.
6. foojay.io (2025/2026 agent best-practices guide) — "all config
   via environment variables, never hardcoded values." Corroborates
   R-4.

Recency scan: no 2024-2026 source argues FOR hardcoded thresholds
in agent system prompts.

### Red-flag findings from research gate (action taken)

The researcher's internal-code audit surfaced two critical
dependencies that the audit document understated:

1. **`backend/agents/meta_coordinator.py` is NOT dead code.** It is
   imported at module level by `backend/services/autonomous_loop.py:19`
   (`from backend.agents.meta_coordinator import MetaCoordinator`)
   and used at lines 50, 462-488, 896-897 (active instantiation +
   health-check + `get_coordinator()` public accessor). A second,
   lazy importer at `backend/agents/skill_optimizer.py:825`.
   Deleting this file breaks paper trading at startup.
2. **`backend/autonomous_harness.py` is NOT dead code.** It is
   imported by `scripts/risk/phase4_9_redteam.py:58` which uses
   `_BLOCKLIST_PATH`, `promote_strategy`, and `PromotionBlocked` for
   active phase-4.9 negative-test enforcement (FINRA Notice 15-09).
   Deleting this file breaks the red-team test script.

**Action: R-6 is descoped from this cycle entirely.** A future
step will refactor `autonomous_loop.py` and `phase4_9_redteam.py`
to remove the imports before either file can be deleted.

A third research-gate finding: the audit's quoted "**stale
scaffolding is dead weight**" is a **paraphrase**, not a verbatim
HARNESS-DOC quote. The actual verbatim quote is "every component
in a harness encodes an assumption about what the model can't do
on its own, and those assumptions are worth stress testing." This
contract cites the verbatim version.

## Plan steps

### R-3 — Rename in-app Layer-2 labels (4 files)

- **R-3a** (`backend/agents/agent_definitions.py:179`): rename
  `name="Ford (Main Agent)"` → `name="Ford (Slack Orchestrator)"`.
- **R-3b** (`backend/agents/agent_definitions.py:271`): rename
  `name="Researcher"` → `name="Slack Researcher"`.
- **R-3c** (`backend/agents/agent_definitions.py:165-167`): update
  Communication agent's inline prose listing of downstream agents
  to use the new display names.
- **R-3d** (`ARCHITECTURE.md` Layer-2 flow diagram around lines
  169-175): update the same labels.
- **R-3e** (`CLAUDE.md` the "🔴 MAS HARNESS LOOP" critical-rule
  bullet): scope "**The MAS is exactly 3 agents**" → "**The
  Harness MAS layer (Layer 3) is exactly 3 agents**" and add a
  sentence pointing at `_inventory.json` as the canonical roster
  for the broader dev MAS.

No `AgentType` enum keys change (`main`, `qa`, `research` remain).
Routing parsers depend only on enum keys, not display names —
verified by the researcher's grep at `agent_definitions.py:372-373`.

### R-4 — Extract `META_PLAN` to runtime config (3 files)

- **R-4a** (NEW `backend/backtest/experiments/meta_plan.json`):
  create JSON with explicit numeric keys
  (`sharpe_target`, `annual_return_min_pct`, `annual_return_max_pct`,
  `max_drawdown_pct`, `max_trades_per_month`,
  `sector_concentration_max_pct`, `cost_stress_multiple`). Values
  initially copied verbatim from the existing hardcoded
  `META_PLAN` text (`planner_agent.py:23-31`) to preserve current
  behavior. Future updates land in the JSON, not the code.
- **R-4b** (`backend/agents/planner_agent.py:23-31`): replace the
  hardcoded `META_PLAN` triple-quoted string with a function that
  reads `meta_plan.json` and renders the same prompt text on each
  `PlannerAgent` instantiation. Cache on the instance.
- **R-4c** (test): add a unit test
  `tests/agents/test_planner_meta_plan_config.py` that asserts
  `PlannerAgent()` reads the JSON and that the rendered
  system prompt contains the same thresholds as the JSON file.

### R-7 — ARCHITECTURE.md 28 → 5 → 3 mapping paragraph (1 file)

- **R-7** (`ARCHITECTURE.md` after the "Total Layer 1 agents: 28"
  line around line 120): insert a new paragraph that maps:
  - 28 Layer-1 skill agents (full pipeline) →
  - 5 progressive-disclosure layers in the rationale drawer
    (Analyst / Bull-Bear debate / Quant / Trader / Risk) →
  - 3 rows in lite-mode (Quant + Trader + RiskJudge) per
    `backend/services/signal_attribution.py:57-157`.
  Cite `handoff/current/phase-23.2.A-agent-rationale-audit.md` as
  the authoritative source. This closes the documentation gap
  driving Symptom 2 in `docs/audits/dev-mas-2026-05-11/03-symptoms.md`.

### R-6 — Deferral note (no code changes; 1 file)

- **R-6 deferred**: append a section to `handoff/harness_log.md`
  noting that R-6 (delete `backend/autonomous_harness.py` +
  `backend/agents/meta_coordinator.py`) cannot proceed without
  first refactoring `backend/services/autonomous_loop.py` and
  `scripts/risk/phase4_9_redteam.py`. This is documented for a
  future step.

### R-5 — Deferred (already user-decided)

- **R-5 deferred**: append a separate section to
  `handoff/harness_log.md` noting that R-5
  (`.claude/agents/qa.md` fail-mode change) is deferred per user
  decision (separation-of-duties forbids the current session from
  both editing qa.md and self-evaluating). Requires a future
  session + Peder review.

## Immutable success criteria (verification)

These criteria are immutable once the masterplan step is written.
A script can check each one.

1. `agent_definitions.py:179` contains the string `"Ford (Slack Orchestrator)"`.
2. `agent_definitions.py:271` contains the string `"Slack Researcher"`.
3. `agent_definitions.py:165-180` (Communication agent prompt block)
   contains the updated display names (no more `"Ford (Main Agent)"`
   or bare `"Researcher,"` references in the inline prose).
4. `ARCHITECTURE.md` no longer contains the literal string
   `"Ford (Main Agent)"` in the Layer-2 diagram area; the diagram
   uses the new labels.
5. `CLAUDE.md` contains the new scoped phrase **"The Harness MAS
   layer (Layer 3) is exactly 3 agents"** and a reference to
   `_inventory.json` as the broader dev-MAS roster.
6. `backend/backtest/experiments/meta_plan.json` exists, is valid
   JSON, and contains all 7 numeric keys listed in R-4a.
7. `backend/agents/planner_agent.py` no longer contains the
   hardcoded `META_PLAN = """STRATEGIC GOAL...` block; instead
   reads from `meta_plan.json` at `PlannerAgent()` instantiation.
8. `tests/agents/test_planner_meta_plan_config.py` exists and
   passes (`python -m pytest tests/agents/test_planner_meta_plan_config.py -v`).
9. `ARCHITECTURE.md` contains the 28→5→3 mapping paragraph after
   the "Total Layer 1 agents: 28" line and references
   `phase-23.2.A-agent-rationale-audit.md`.
10. `handoff/harness_log.md` contains a new "Cycle N — phase=23.8.0"
    section with both deferral notes (R-5 and R-6).
11. **No regressions**: `python -c "import
    backend.agents.planner_agent; import
    backend.agents.agent_definitions; import
    backend.services.autonomous_loop; print('OK')"` exits 0.
12. **No regressions**: `python -c "import
    backend.autonomous_harness; import
    backend.agents.meta_coordinator; print('OK')"` exits 0 (the
    stubs are NOT deleted; they must remain importable).

A verifier script at `tests/verify_phase_23_8_0.py` (NEW) will
assert all 12 claims and exit 0/1.

## Files expected to change

| File | Type | Change |
|---|---|---|
| `backend/agents/agent_definitions.py` | edit | 3 spots: lines 179, 271, 165-167 (R-3a/b/c) |
| `ARCHITECTURE.md` | edit | 2 spots: ~lines 120 (new paragraph, R-7), ~lines 169-175 (R-3d) |
| `CLAUDE.md` | edit | 1 spot: the "🔴 MAS HARNESS LOOP" rule paragraph (R-3e) |
| `backend/backtest/experiments/meta_plan.json` | NEW | R-4a |
| `backend/agents/planner_agent.py` | edit | lines 23-31 hardcoded constant → file read (R-4b) |
| `tests/agents/test_planner_meta_plan_config.py` | NEW | R-4c |
| `tests/verify_phase_23_8_0.py` | NEW | 12-claim source-level assertion script |
| `handoff/current/contract.md` | NEW (this file) | the contract itself |
| `handoff/current/experiment_results.md` | NEW (later) | by GENERATE |
| `handoff/current/evaluator_critique.md` | NEW (later) | by Q/A |
| `handoff/harness_log.md` | append | cycle entry + R-5/R-6 deferral notes (LAST before flip) |
| `.claude/masterplan.json` | edit | add step 23.8.0; flip status to done at end |

## Rollback note

Single-commit revert. The 3 edits + 3 new files form one logical
unit. No infrastructure / hooks / settings.json changed. R-6 not
landed means no `import` failures possible. The verifier script
(claim 11) explicitly checks `autonomous_loop` imports — early
trip-wire if anything regresses.

## Out of scope (explicit)

- **R-1** (live_check field + hook enforcement) — separate cycle.
- **R-2** (TaskCompleted hook delete or promote) — separate cycle.
- **R-5** (qa.md fail-mode change) — separate session + Peder review.
- **R-6** (delete deprecated stubs) — separate cycle (must refactor
  `autonomous_loop.py` + `phase4_9_redteam.py` first).

## References

- `docs/audits/dev-mas-2026-05-11/02-per-agent.md` — Phase 2 audit
  findings (R-3 ↔ F-1/C-2; R-4 ↔ P-1; R-6 ↔ C-A7; R-7 ↔ Symptom 2
  systemic pattern).
- `docs/audits/dev-mas-2026-05-11/03-symptoms.md` — Phase 3 evidence
  trace (Symptom 2 SCOPING_GAP grounded by R-7).
- `docs/audits/dev-mas-2026-05-11/04-remediation.md` — Phase 4
  recommendations + masterplan-step proposal.
- `handoff/current/phase-23.2.A-agent-rationale-audit.md` — prior
  audit cited in R-7 paragraph.
- `backend/agents/_inventory.json` — canonical roster cited in
  R-3e.
- Researcher subagent JSON envelope (above): `gate_passed: true`.
