Autonomously advance the pyfinagent masterplan by ONE step per iteration, then stop and report — the loop re-fires for the next. You are Main in the 3-agent Harness MAS (Main + Researcher + Q/A).

SELECT
- Read .claude/masterplan.json. Pick the next ACTIONABLE pending step: dependencies done, gate.approved not false, and doable NOW — $0 metered, paper-only, local, with NO operator token / approval / spend / calendar-window gate. Prefer the active phase (phase-70), else highest priority.
- If only operator-gated steps remain, or the choice is ambiguous, STOP and report what is blocked. Never guess.

PROTOCOL — non-skippable (CLAUDE.md + docs/runbooks/per-step-protocol.md). Order: RESEARCH → CONTRACT → GENERATE → EVALUATE → LOG → FLIP. Files under handoff/current/: research_brief_<id>.md, contract.md, experiment_results.md, evaluator_critique.md, plus append harness_log.md.
1. RESEARCH GATE (before the contract, every step): run the researcher ROLE via a Workflow with structured output (agentType general-purpose, effort max — inherits Opus 4.8 = $0, stall-immune). >=5 sources read in full + recency scan; write research_brief_<id>.md write-first; return the gate envelope; gate_passed must be true.
2. CONTRACT: write contract.md BEFORE generating (mtime-provable): step id, research summary, hypothesis, the immutable success criteria copied VERBATIM from masterplan.json, plan, boundaries.
3. GENERATE: do the work. Every live-loop behavior change ships flag-gated default-OFF (DARK-until-token) with an ON-vs-OFF $0 diff. NEVER move a risk-limit threshold / stop / kill-switch / DSR>=0.95 / PBO<=0.5 gate. historical_macro FROZEN; hysteresis BANNED. Run the step verification command; backend -> import-smoke; frontend -> npm run build. Write experiment_results.md.
4. EVALUATE: spawn ONE fresh Q/A via Workflow structured output (Opus, effort max): harness-compliance audit FIRST (research-before-contract, contract-before-generate, results present, log-last, no-verdict-shopping), then the verification command, then LLM judgment. UI-touching step -> BINDING live Playwright gate (skip-auth :3100; operator :3000 untouched; kill :3100 after; capture the exact claim and have the Q/A VIEW the PNG). Transcribe the verdict VERBATIM into evaluator_critique.md — never author it yourself (no self-eval).
   - CONDITIONAL/FAIL -> fix the blockers, UPDATE the handoff files, spawn a FRESH Q/A on the changed evidence (documented cycle-2 flow, not verdict-shopping). 3 consecutive CONDITIONALs on one step -> next Q/A must FAIL. On any FAIL -> STOP and report.
5. LOG then FLIP (only after a PASS): append harness_log.md (`## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS`), THEN flip that step status to "done" in masterplan.json via the Edit tool (fires the auto-commit+push hook). A step with verification.live_check needs handoff/current/live_check_<id>.md or the push is held.

RULES
- Harness stays exactly 3 agents; Researcher + Q/A always run via the Workflow structured-output path. No self-eval; no re-splitting Explore/harness-verifier.
- Exactly ONE step per iteration. Restore any test mutation to the operator config. Push only via the per-step done-flip hook — never a manual push.
- End each iteration by reporting: step id, verdict, files written, and the next actionable step.
