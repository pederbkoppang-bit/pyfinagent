# Evaluator Critique — phase-47.8: Opus-4.8 stale-pin sweep

## Q/A verdict (cycle-1, first Q/A on 47.8) — **PASS**

`ok: true`, verdict: **PASS**. First Q/A on step-id 47.8 (brand new step; no prior
CONDITIONAL/FAIL in `harness_log.md` — `grep -c "phase=47.8"` = 0; not subject to
3rd-CONDITIONAL auto-FAIL). Evidence is fresh, not a re-spawn → no verdict-shopping.

### STEP 1 — Harness-compliance audit (5/5 PASS)

1. **Researcher / research gate — PASS.** `research_brief_phase_47_8_opus48_sweep.md`
   present (26,336 bytes). JSON envelope at EOF: `gate_passed: true`,
   `external_sources_read_in_full: 6` (>= 5 floor), `recency_scan_performed: true`,
   `urls_collected: 16`, `internal_files_inspected: 12`. Contract cites it in both the
   research-gate summary (`contract.md:6`) and References (`contract.md:36`).
2. **Contract-before-generate — PASS.** `contract.md` is the 47.8 contract (header
   line 1 "phase-47.8: Opus-4.8 stale-pin sweep"), has step id + research-gate summary
   + verbatim immutable success_criteria (4, matching masterplan + test) + hypothesis +
   8 plan steps + references. mtime ordering: brief 03:55 < contract 04:02 <
   experiment_results 04:09 (research → contract → generate, correct).
3. **experiment_results.md — PASS.** Covers root cause, 11 edits / 9 files, verbatim
   verification output (`ast OK 9 files` + `11 passed`), behavioral-confirmation list,
   success-criteria mapping, and a "Scope honesty / deferred" section.
4. **Log-last discipline — PASS.** No `phase=47.8` block in `harness_log.md` yet
   (`grep -c` = 0). Correct ordering — orchestrator appends AFTER this PASS, before the
   status flip. Absence is correct, NOT a defect.
5. **No verdict-shopping — PASS.** First Q/A on 47.8; fresh evidence.

### STEP 2 — Deterministic checks (reproduced, not trusted)

- **Immutable command — EXIT_CODE=0.** Ran the EXACT
  `verification.command` from masterplan phase-47.8 verbatim:
  `ast OK 9 files` + `11 passed in 0.88s`. Reproduced independently.
- **Disclosed command correction — LEGITIMATE wrong-target false-negative fix, NOT
  goalpost-moving.** The entire phase-47.8 `verification` object is a `+` diff in the
  working tree (no `-` counterpart → no prior committed criteria weakened). Original
  command ast-checked `backend/services/autonomous_loop.py` (the 110 KB 47.7 learn-loop
  file — confirmed it exists and was NOT edited this cycle: `git diff --name-only` shows
  only `backend/autonomous_loop.py`). ast-checking an untouched file would pass vacuously
  while missing the real `AutonomousLoopOrchestrator.planner_model` edit — a textbook
  false-negative. Correction added the right path + `streaming_integration.py` (9 files).
  The immutable **success_criteria** are unchanged in substance. Same class as the
  phase-47.1 precedent (which a prior Q/A also accepted).
- **Independent grep — every remaining 4-7 classified legit; NO operative DEFAULT
  reads 4-7.** `grep -rn "claude-opus-4-7" backend/ --include="*.py" | grep -v test`:
  - `main.py:157,184` historical comments; `model_tiers.py:170` comment; `:185`
    EFFORT_SUPPORTED_MODELS (4-8 co-present :184); `:235` MODEL_EFFORT_FALLBACK xhigh
    (4-8 :234); `settings.py:30` deep_think historical note (default is Gemini now);
    `cost_tracker.py:27` legacy pricing (4-8 added 47.3); `harness_memory.py:53` legacy
    window (4-8 :52); `llm_client.py:472/585/1385/1404/1444/1478/1980/1981`
    accept-lists/provider-map/comments (4-8 everywhere); `app_home.py:21` legacy dropdown
    option (4-8 :20); `settings_api.py:31,215` accept-list + legacy pricing (4-8 :214).
  - `multi_agent_orchestrator.py:1061` is the WIDENED predicate with 4-8 listed first.
  - **Zero operative DEFAULT pins still read 4-7.** Neither under-edited (no missed
    operative default) nor over-edited (no purged compat entry).
- **CRITICAL :1061 fix — VERIFIED.** `backend/agents/multi_agent_orchestrator.py:1061`
  reads `if agent_config.model.startswith(("claude-opus-4-8", "claude-opus-4-7")):`.
  Inspected the surrounding block (`:1059-1081`): the IF branch sets
  `_thinking_arg={"type":"adaptive"}`, `_extra_kwargs={}` (no sampling params); the ELSE
  branch sets `{"type":"enabled","budget_tokens":2048}` + `temperature=1` — exactly the
  manual-budget+sampling combo Opus 4.8 rejects with a 400. A 4-8 pin now takes the IF
  (adaptive-only) path. Fix is real and load-bearing.
- **Behavioral spot-checks (independent, my own probes, not the test's) — all genuine:**
  - `get_context_window("claude-opus-4-8") == 1_000_000`;
    `get_context_window("zzz-nonexistent") == 128_000` — confirms the 4-8=1M assertion is
    load-bearing (unknown model → 128K default), NOT tautological.
  - `inspect.signature(...).default == "claude-opus-4-8"` independently confirmed for
    `PlannerAgent.__init__`, `get_planner_agent`, `AutonomousLoopOrchestrator.planner_model`,
    `MultiAgentOrchestrator.should_reset_context`, `multimodal_index_claude`.
  - `AGENT_MODEL_OVERRIDES["main"]/["qa"] == "anthropic/claude-opus-4-8"`;
    `["research"] == "anthropic/claude-sonnet-4-6"` (deliberately cost-efficient, kept).
  - `ticket_queue_processor.py:165-171` `agent_model_map` main/q-and-a == 4-8, `.get`
    default == 4-8, research == sonnet-4-6.
  - Compat: `MODEL_PRICING["claude-opus-4-7"] == ["claude-opus-4-8"] == (5.0, 25.0)`.
- **All 9 edited modules import cleanly — 9/9.**

### STEP 2b — Frontend gate: N/A. `git diff --name-only` shows 0 `frontend/**` files;
ESLint/tsc gate correctly not triggered.

### Code-review heuristics (5 dimensions evaluated)

- **Security (Dim 1):** No `secret-in-diff`. The `OPENCLAW_GATEWAY_TOKEN` literal in
  `openclaw_client.py:34` is PRE-EXISTING (NOT in the `+` diff — verified) and was
  honestly disclosed as an out-of-scope follow-up. No prompt-injection / command-injection /
  insecure-output-handling surface (model-id string constants only). No dep-pin removal.
- **Trading-domain (Dim 2):** No execution-path / kill-switch / stop-loss / perf-metrics /
  position-sizing / crypto-asset touch. Diff is model-pin constants + a thinking-branch
  predicate + a Slack dropdown list. No BLOCK.
- **Code quality (Dim 3):** Targeted small edits (`git diff --stat`: 9 code files, +/- a
  handful of lines each). No new broad-except, print, or unicode-logger introduced.
- **Anti-rubber-stamp (Dim 4):** The change is config/model-pin + a routing predicate, not
  Sharpe/drawdown/sizing math — `financial-logic-without-behavioral-test` does NOT apply.
  Nonetheless a behavioral test exists and exercises real lookups/signatures. No
  tautological/over-mocked/rename-as-refactor patterns. The disclosed pre-existing emoji in
  `app_home.py` AGENT_DISPLAY is outside the diff (verified — only `+"claude-opus-4-8",` added).
- **LLM-evaluator anti-patterns (Dim 5):** First Q/A, fresh evidence → no
  sycophancy-under-rebuttal / second-opinion-shopping. This critique carries file:line +
  command-output citations throughout (no missing-chain-of-thought).

### STEP 3 — LLM judgment

- **Contract alignment — all 4 immutable success_criteria MET:**
  1. CRITICAL :1061 widened to include `claude-opus-4-8` (4-8 takes adaptive-only path,
     no 400) — MET (verified in source + ELSE-branch inspection).
  2. `harness_memory.MODEL_CONTEXT_WINDOWS['claude-opus-4-8']=1_000_000` + `app_home`
     AVAILABLE_MODELS includes 4-8, 4-7 kept — MET (behavioral 1M + 128K-default contrast).
  3. Operative 4-7 defaults bumped (ticket_queue main/q-and-a/default, rag vision,
     planner + autonomous_loop, openclaw main/qa); compat (MODEL_EFFORT_FALLBACK,
     cost_tracker pricing, llm_client accept-lists) PRESERVED — MET (5 signature guards +
     AST map guard + 2 compat guards, all independently reproduced).
  4. pytest guard + ast clean + green — MET (`ast OK 9 files`, `11 passed`).
- **Mutation-resistance — confirmed real guards (simulated each revert):**
  - Revert :1061 widening → narrow-form-absent assert fails AND wide-form-present assert
    fails. CAUGHT.
  - Drop 4-8 context-window entry → `get_context_window` returns 128K != 1M. CAUGHT.
  - Revert any signature default to 4-7 → `inspect` default mismatch. CAUGHT.
  - Purge 4-7 compat pricing → `MODEL_PRICING["claude-opus-4-7"]` assert fails. CAUGHT
    (over-edit guard works in the other direction too).
- **Anti-rubber-stamp:** Nothing claimed-MET is actually unmet. Not over-edited (compat
  4-7 entries verified preserved with 4-8 co-present), not under-edited (zero operative
  4-7 defaults remain).
- **Scope honesty — HONEST.** experiment_results `:52-53` marks the live 400-fix
  confirmation as DEFERRED (no live LLM call this cycle, $0 spend), and discloses the
  pre-existing `OPENCLAW_GATEWAY_TOKEN` secret + pre-existing app_home emoji +
  `run_autonomous_loop.py:73 claude-opus-4-6` as out-of-scope follow-ups rather than
  silently fixing or hiding them. No operator-gated flag/spend triggered (verified: no
  live Anthropic call; pure static/structural + unit test).
- **Research-gate compliance:** researcher output present, gate_passed true, cited in
  contract.

### Verdict

All 4 immutable success_criteria independently MET; immutable command exit 0; CRITICAL
:1061 fix verified at source with ELSE-branch inspection; behavioral guards proven
load-bearing (128K-default contrast); mutation-resistance confirmed in all four directions;
zero operative 4-7 defaults remain and zero legit compat entries purged; scope honestly
disclosed with $0 spend. No BLOCK or WARN from any code-review dimension.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable success_criteria MET and independently reproduced. Immutable command exit=0 (ast OK 9 files; 11 passed). CRITICAL multi_agent_orchestrator.py:1061 widened to startswith(('claude-opus-4-8','claude-opus-4-7')) — 4-8 now takes the adaptive-only/no-sampling IF branch instead of the manual budget_tokens+temperature=1 ELSE branch Opus 4.8 rejects with 400; verified at source incl. ELSE branch. harness_memory 4-8=1M added (4-7 kept), proven load-bearing via get_context_window('zzz')==128_000 contrast. All 5 operative signature defaults read claude-opus-4-8 (independent inspect probes). Compat 4-7 (cost_tracker pricing, MODEL_EFFORT_FALLBACK, llm_client accept-lists) preserved. Independent grep: zero operative DEFAULT pins still read 4-7; every remaining 4-7 is a comment/co-present accept-list/legacy-fallback row with 4-8 alongside. Mutation-resistance confirmed in all 4 directions. Disclosed verification.command correction is a legit wrong-target false-negative fix (backend/services/autonomous_loop.py was the untouched 47.7 file; backend/autonomous_loop.py is the file actually edited) on an entirely-new (uncommitted) phase block — success_criteria unchanged. Scope honest: live 400-fix confirmation deferred to next 4-8 cycle, $0 spend, pre-existing secret+emoji disclosed not hidden. No frontend files (ESLint/tsc N/A). 5/5 harness-compliance. No code-review BLOCK/WARN.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "independent_grep_4_7_classification", "critical_1061_source_verification", "behavioral_signature_probes", "context_window_load_bearing_check", "mutation_resistance_simulation", "import_check_9_modules", "code_review_heuristics", "research_gate_envelope", "evaluator_critique"]
}
```
