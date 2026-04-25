---
step: phase-16.20
cycle_date: 2026-04-25
verdict: CONDITIONAL
reviewer: qa
---

# Q/A Critique -- phase-16.20

## Harness-compliance audit (5 items)

1. **Research gate** -- PASS. `handoff/current/phase-16.20-research-brief.md`
   exists. Envelope reports tier=simple, 5 in-full sources read via WebFetch,
   15 URLs collected, dedicated "Recency Scan (2024-2026)" section present
   with 3 directly relevant 2026 hits (anthropics/claude-code #28091,
   openclaw #9938, Anthropic 2026 docs), `gate_passed: true`. Hierarchy mix
   acceptable: 1 official-doc / 2 official-engineering-blog / 1 vendor-doc /
   1 practitioner-blog -- all in-full sources are tier 1-4, no
   community-only padding. Internal code inventory cites exact lines
   (`multi_agent_orchestrator.py` 155-168, 201-218; `llm_client.py`
   1090-1154; `agent_definitions.py` 129/179/228/272). 3-variant query
   discipline visible in the Recency Scan. Spot-check: cited finding #5
   ("`run_orchestrated_round` does not exist") independently confirmed
   below by my own grep + import test. Gate cleared.

2. **Contract-before-GENERATE** -- PASS. `contract.md` mtime
   `2026-04-25 07:00:36`; `experiment_results.md` mtime
   `2026-04-25 07:01:04`. 28-second gap, correct order. The existing
   `evaluator_critique.md` on disk before this overwrite (mtime 06:53:53)
   was the prior 16.19 PASS critique still awaiting archive rotation
   (the `archive-handoff` hook fires on masterplan status flip). Its
   contents were unambiguously labelled `step: phase-16.19`; not a
   16.20 critique pre-empting this Q/A pass.

3. **Experiment results** -- PASS. `step: phase-16.20`. Includes the
   verbatim ImportError stdout, exit code 1, and explicit "Why this is
   a CONDITIONAL, not a forced PASS" section. Honest disclosures section
   (lines 65-73) explicitly invites FAIL if I judge it appropriate
   ("Q/A is being asked to make a judgment call. Do not rubber-stamp.").
   The 0/3 mechanical-criterion table is presented unflinchingly.

4. **Log-last** -- PASS. `grep -c "phase-16.20" handoff/harness_log.md`
   returns `0`. Main has not yet appended the cycle entry, which is
   correct -- log-last fires AFTER my verdict, not before.

5. **No verdict-shopping** -- PASS. Prior on-disk `evaluator_critique.md`
   was the 16.19 PASS critique, not a previous-Q/A-pass on 16.20. This
   is my first and only Q/A spawn for phase-16.20. No fresh-respawn
   pattern on unchanged evidence.

All five compliance items clear.

## Deterministic checks

- **import_error_reproduced**: YES. Verbatim:
  ```
  ImportError: cannot import name 'run_orchestrated_round' from
  'backend.agents.multi_agent_orchestrator'
  (/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py)
  ```
  Exit code 1. Reproduces independently of Main's report.

- **orchestrator_module_top_level_symbols**: only two top-level defs at
  `^(def|async def|class) `:
  - `class MultiAgentOrchestrator:` (line 124)
  - `def get_orchestrator() -> MultiAgentOrchestrator:` (line 1310)
  Module exposes orchestrator-related code (the class and a singleton
  factory) but no `run_orchestrated_round` symbol -- aligns with
  researcher finding #5 and Main's experiment_results.

- **daily_cycle_imports_mas**: NO. `grep -n "multi_agent_orchestrator"
  backend/services/autonomous_loop.py backend/api/paper_trading.py`
  returned zero matches in either file. Main's claim that the daily
  paper-trading cycle does not depend on `multi_agent_orchestrator` is
  verified. MAS layer-2 is genuinely off the Monday paper-trading
  critical path -- the analysis pipeline (Layer 1, 28-agent Gemini,
  `backend/agents/orchestrator.py`) is what feeds signals.

- **anthropic_key_starts_with_oat_or_api**: starts with `sk-ant-oat`
  (len=108, ends `QAAA`). Confirms Main's claim that the current
  `backend/.env` value is the OAuth token, not the API key. Per
  research findings #1-2, this token will produce 401 against the
  Messages API even if `run_orchestrated_round` existed -- a second,
  independent blocker beyond the missing function. The key swap is a
  genuine precondition, not a fabricated obstacle.

## Judgment

- **conditional_or_fail**: **CONDITIONAL** (not FAIL). Defended below.

- **rationale**: Three facts converge on CONDITIONAL being the
  discipline-correct call here, not a soft option:

  (1) The verification command was authored for an earlier,
  in-progress step (16.3) that assumed a function which was never
  written. 16.20's contract was constructed THIS CYCLE around that
  pre-existing aspirational verification command -- meaning 16.20 was
  always-going-to-fail-mechanically as designed. Forcing FAIL on a
  step whose immutable criteria reference a non-existent symbol
  punishes Main for honestly running the broken command instead of
  silently rewriting the criteria (which is forbidden). The
  underlying defect is in 16.3, not in 16.20's execution.

  (2) The "Monday critical path" claim is independently verified --
  zero MAS imports in the daily cycle, signal generation runs through
  Layer 1 Gemini. CONDITIONAL with concrete follow-up beats FAIL that
  blocks UAT 16.21-16.23 from progressing on a step that does not
  gate go-live.

  (3) Main is not asking me to rubber-stamp a passing claim; the
  experiment_results explicitly says "0 of 3 met" and offers FAIL as
  acceptable. CONDITIONAL with mandatory follow-up tickets is the
  documented harness path for "blocker exists, but it is a real
  feature gap not addressable in scope" -- exactly the pattern used
  in cycle 17.4. The discipline that "Q/A holds the line" means
  refusing PASS, not refusing CONDITIONAL when CONDITIONAL is what
  the evidence supports.

  However: CONDITIONAL is conditional. See follow-up-tickets section
  and the closing constraint.

- **not_a_monday_blocker_verified**: YES. `backend/services/autonomous_loop.py`
  and `backend/api/paper_trading.py` contain zero references to
  `multi_agent_orchestrator` (grep verified). The daily cycle's
  signal generation does not pass through Layer 2 MAS. Layer 2 is the
  in-app domain-orchestration tier (Slack queries, ad-hoc operator
  commands), not the trading-decision tier. Main's claim stands.

- **follow_up_tickets_sufficient**: YES, with one addition. The 3
  tickets in experiment_results.md (implement `run_orchestrated_round`,
  swap Anthropic key, add Gemini fallback in `_get_client`) cover the
  structural gap. **Required addition**: ticket #4 below
  ("16.3-revisit"), to ensure 16.3 itself gets a proper close path
  rather than being left as orphaned in-progress.

  Mandatory follow-up tickets to close the CONDITIONAL:
  1. (Main #1) Implement `run_orchestrated_round(ticker, max_iterations)`
     as a public synchronous entry point. Must return dict with
     `iterations` key. Belongs to 16.3-revisit phase, not a sneaked
     16.21 patch.
  2. (Main #2) Anthropic API key swap: user replaces `sk-ant-oat-*`
     with `sk-ant-api03-*` in `backend/.env`; Main bounces backend.
     Pre-condition for 16.3 verification re-run.
  3. (Main #3) Gemini fallback in orchestrator's `_get_client()` so a
     future 401 from Anthropic doesn't kill MAS entirely. Pattern
     exists in `llm_client.py::make_client`.
  4. (Q/A addition) **16.3 status reconciliation.** 16.3 is the parent
     step whose verification command 16.20 inherited. 16.3 must NOT
     be silently closed by 16.20's CONDITIONAL. Either (a) 16.3
     becomes the implementation phase that ticket #1 lands in, or (b)
     16.3 is explicitly cancelled in masterplan.json with a pointer
     to a successor step. No third option. Currently 16.3 is
     `in-progress` (verified) -- this is the correct state and must
     stay that way until tickets #1-2 land.

- **related_step_16_3_status**: **still in-progress** (verified by
  reading `.claude/masterplan.json`: 16.3=`in-progress`,
  16.20=`pending`, 16.21=`pending`, 16.19=`done`). NOT silently
  flipped to done by this cycle. This is the correct state. Main
  MUST NOT include any edit to 16.3's status when flipping 16.20 to
  done. The CONDITIONAL for 16.20 closes 16.20 only; 16.3 remains
  open pending tickets #1 and #2.

## Verdict

```json
{
  "ok": true,
  "verdict": "CONDITIONAL",
  "violated_criteria": [
    "orchestrator_completes_at_least_1_iteration",
    "anthropic_provider_in_log_OR_gemini_fallback_documented"
  ],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "from backend.agents.multi_agent_orchestrator import run_orchestrated_round",
      "state": "module exposes only MultiAgentOrchestrator class + get_orchestrator() factory; symbol absent",
      "constraint": "verification command (immutable) requires the symbol to exist"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "ANTHROPIC_API_KEY=sk-ant-oat-... in backend/.env",
      "state": "OAuth token (sk-ant-oat prefix), 108 chars, rejected by Anthropic Messages API per 2026 docs + GitHub #28091",
      "constraint": "MAS orchestrator requires sk-ant-api03-* console key for non-401 calls"
    }
  ],
  "follow_up_tickets": [
    "Implement run_orchestrated_round(ticker, max_iterations) -> dict with 'iterations' key in backend/agents/multi_agent_orchestrator.py (estimate 2-4h, lands in 16.3-revisit)",
    "User swaps backend/.env ANTHROPIC_API_KEY from sk-ant-oat-* to sk-ant-api03-* console key; Main bounces backend",
    "Add Gemini fallback to MultiAgentOrchestrator._get_client() per llm_client.py::make_client pattern",
    "Reconcile phase-16.3 status: either land tickets 1-2 under 16.3 (preferred) or cancel 16.3 explicitly with successor pointer; do NOT silently flip 16.3 in this cycle"
  ],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "import_reproduction",
    "verification_command",
    "module_symbol_inventory",
    "daily_cycle_dependency_grep",
    "anthropic_key_prefix_inspection",
    "masterplan_status_check",
    "evaluator_critique_history",
    "harness_compliance_5_items"
  ]
}
```

## Closing constraint

Main may flip 16.20 to `done` with this CONDITIONAL **only if** all
four follow-up tickets land in `.claude/masterplan.json` (or an
equivalent task tracker visible to the next session) AND the
`harness_log.md` cycle entry explicitly names the four tickets and
states "16.3 remains in-progress; not closed by this cycle." If
either condition is not met, this CONDITIONAL becomes FAIL.
