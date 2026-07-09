# Contract -- 67.1 Q/A verification-depth upgrade

Step: masterplan phase-67 / 67.1 (P0). Research gate: PASSED (moderate tier;
research_brief_67_1.md -- 7 sources read in full, 17 URLs, recency scan done).

## Research-gate summary

- No hook spawns or times the Q/A subagent today: the only agent-type hook is the Stop
  hook (settings.json:113, its own inline prompt); TaskCompleted retired phase-23.8.2
  (per-step-protocol.md:227-229; audit R-2). Retiring the 55s cap is SAFE; the real
  work bound is Agent-tool maxTurns:30.
- CRITICAL SCOPING (researcher): settings.json:113-134 contains its own
  `{"ok":true,"reason":"loop prevention"}` for Stop/SubagentStop hooks -- there ok:true
  legitimately means "allow the stop". Only qa.md's copy is the auto-PASS bug. DO NOT
  TOUCH settings.json.
- Lint gate proven live pre-change: `uvx ruff check --select F821,F401,F811
  backend/agents/agent_definitions.py` -> `F821 Undefined name json` at :396, exit 1;
  clean file exits 0. uvx = ephemeral, zero venv mutation, stays out of
  backend/requirements.txt (dev-dep precedent).
- qa.md's CURRENT deterministic pytest command is broken twice: `pytest tests/ -v
  --timeout=30` targets the root tests/ tree (7 collection errors; backend/tests is the
  clean 991) AND passes --timeout with pytest-timeout NOT installed (exit 4).
- Backend live on :8000 (/api/health -> 200, auth-exempt) -- valid runtime-smoke target.
- Watermelon warning (researcher R6): two references escape the immutable greps and
  must be fixed for honest compliance: qa.md "55s Q/A budget" phrasing (1b) and
  per-step-protocol.md:255-256 "SendMessage-to-same-agent" drift-mode line.
- Literature: deterministic graders first (Anthropic demystifying-evals); tiered
  cascade evaluation (2026) -- cheap deterministic tiers resolve most checks, LLM judge
  last. Full citations in the brief.

## Hypothesis (falsifiable)

Replacing the flat 55s cap with a tiered work-budget and adding deterministic backend
lint + runtime-smoke gates lets a fresh Q/A catch the undefined-name bug class and
does-not-run failures that current checks (ast.parse only) provably miss, without
breaking any existing gate -- testable by the step's verification command plus a live
ruff run over a known-buggy file (F821 fires) and a clean file (exit 0).

## Success criteria (verbatim from .claude/masterplan.json 67.1 -- IMMUTABLE)

1. ".claude/agents/qa.md contains no global 55-second runtime cap; the verification
   budget is tiered and explicitly permits full pytest + runtime smoke for
   backend-touching diffs"
2. "qa.md defines a deterministic backend gate REQUIRED for diffs touching backend/**:
   (a) an undefined-name-class Python lint (ruff or pyflakes) over changed .py files
   with verbatim exit code, and (b) a runtime smoke that imports each changed module
   inside .venv and, when the diff touches a live API/service path, exercises it
   (endpoint or command) with output captured"
3. "The stop_hook_active escape hatch no longer returns ok:true; loop-prevention exits
   are verdict-neutral (ok:false with an explicit no-evaluation-performed reason) so no
   auto-PASS path remains in the evaluator"
4. "CONDITIONAL/FAIL recovery guidance is consistent across .claude/agents/qa.md,
   docs/runbooks/per-step-protocol.md, and CLAUDE.md: fix blockers -> update handoff
   evidence -> spawn a FRESH Q/A; respawn on UNCHANGED evidence stays forbidden; no
   artifact still mandates SendMessage-to-the-SAME-agent as the recovery path"
5. "A fresh Q/A (running the pre-change roster snapshot) returns PASS on this step's
   diff, and the new lint gate is proven live with verbatim ruff/pyflakes output over
   this step's own changed files"

Criterion-5 note (honest interpretation, decided at PLAN time): this step's own diff is
markdown-only, so the gate's semantics for it are "no changed .py -> lint N/A, exit 0";
to prove the gate LIVE with teeth, the live_check additionally runs the same command
over backend/agents/agent_definitions.py (known-buggy until 67.2) showing F821 fires.
Both outputs verbatim.

## Design (files to modify)

1. `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa.md`
   - §1: fix the broken suite command -> scoped backend/tests invocation (+ timeout once
     pytest-timeout present).
   - NEW §1b-backend "Python lint gate (REQUIRED if diff touches *.py)": uvx ruff check
     --select F821,F401,F811 over changed .py files; verbatim exit code; never pipe to
     tail (PIPESTATUS gotcha).
   - NEW §1d "Backend runtime smoke (REQUIRED if diff touches backend/**)": venv import
     of each changed module; exercise touched live paths (endpoint on :8000 / command);
     output captured.
   - Replace "Maximum runtime: 55 seconds" constraint with the tiered budget
     (deterministic+lint <60s; scoped tests <=5min; runtime smoke <=2min; LLM judgment
     last; work bounded by maxTurns, not wall-clock panic). Also fix the 1b "55s Q/A
     budget" phrase (watermelon).
   - stop_hook_active clause -> `{"ok": false, "verdict": null, "reason":
     "loop-prevention exit; no evaluation performed"}`.
   - Recovery constraint -> canonical: fix blockers -> update evidence -> FRESH Q/A;
     respawn on unchanged evidence forbidden.
2. `/Users/ford/.openclaw/workspace/pyfinagent/docs/runbooks/per-step-protocol.md`
   - Anti-pattern #5 -> canonical fresh-spawn text (mirrors its own §4).
   - Drift-modes line 255-256 -> same canonical text (watermelon).
   - NOT touching the "(sonnet)" diagram label (owned by 67.3).
3. Environment prep (logged, not a criterion): `pip install pytest-timeout` into .venv
   -- required for qa.md's fixed suite command AND for 67.2's immutable verification
   command (`--timeout=60`), which cannot pass without it (researcher-67-2 prereq #1).
4. NOT touched: .claude/settings.json (legit ok:true semantics), CLAUDE.md (already
   canonical on recovery), any trading logic.

## Anti-patterns guarded (from research)

- Watermelon compliance: grep-visible fix while intent-violating phrasings survive --
  guarded by fixing the two grep-escaping references identified in R6.
- Semantic collision: "fixing" settings.json's legitimate ok:true loop-prevention
  because it looks like qa.md's bug -- guarded by explicit DO-NOT-TOUCH scoping.
- Masked exit codes: piping lint output through tail/head loses $? -- gate text
  mandates direct invocation or ${PIPESTATUS[0]}.
- Over-prescription (Fable de-prescription guidance): new gate text states the check
  and the invariant, not a 20-step procedure.

## Out of scope

settings.json; CLAUDE.md recovery text (already canonical); the (sonnet) diagram label
(67.3); the NameError fix itself + skill heuristic (67.2); any backend/** code change;
any trading logic.

## Risk (what can still go wrong after PASS)

- The NEXT session's Q/A runs the modified definition on the Fable pin -- if Fable
  stalls mid-evaluation, the qa.md STALL WATCH clause mandates immediate revert of the
  model pin (the gate changes are model-agnostic and stay).
- uvx requires network on first ruff resolve; mitigated by uv's local cache after the
  first run (proven in this session).
- Separation of duties: this session authors qa.md AND orchestrates its evaluation;
  mitigated by the evaluating Q/A running the pre-change snapshot + Peder-review note
  already filed in the phase-67 setup addendum.
