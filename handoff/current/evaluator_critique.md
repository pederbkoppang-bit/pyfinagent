# Q/A Evaluator Critique — phase-47.10: generate_content max_tokens floor

**Verdict: PASS** | Cycle 11 | FIRST Q/A pass on 47.10 (no prior CONDITIONAL) |
Single merged Q/A (deterministic-first, then LLM judgment). Self-evaluation by
orchestrator forbidden — independent verification performed.

---

## STEP 1 — Harness-compliance audit (5 items, evidence-cited)

1. **Researcher gate — PASS.** `handoff/current/research_brief_phase_47_10_generate_content_floor.md`
   exists; END JSON envelope reports `gate_passed: true`,
   `external_sources_read_in_full: 5`, `recency_scan_performed: true`,
   `urls_collected: 14`, `internal_files_inspected: 7`. Contract `:5-6` cites
   the brief by name + researcher id `a2073408b08340a8d`. Brief includes the
   mandatory "Recency scan (2024-2026)" section (brief `:49-60`) and 3-variant
   query discipline (`:44-47`).
2. **Contract before generate — PASS.** `contract.md` IS the 47.10 contract:
   step id (`:1`), research summary (`:5-11`), verbatim immutable criteria
   (`:16-20`, matched against masterplan), hypothesis (`:13-14`), plan steps
   (`:22-26`), references (`:32-35`). mtime ordering confirmed by stat:
   brief `04:45:37` < contract `04:47:11` < experiment_results `04:48:51`.
   Correct research -> contract -> generate order.
3. **experiment_results.md present — PASS.** Edits + file list (`:31-32`),
   verbatim immutable-command output (`:13-19`), success-criteria mapping
   (`:22-26`), scope-honesty / FLAGGED follow-ups (`:28-29`), reachability
   disclosure (`:5-6`).
4. **Log-last — PASS (correctly absent).** `grep "phase=47.10" handoff/harness_log.md`
   exit=1 (no match). The 47.10 block is correctly NOT yet appended — it is
   added only after this PASS. Not a defect.
5. **No verdict-shopping — PASS.** First Q/A on 47.10, fresh evidence; no prior
   verdict to shop. Diff scope confirmed via `git status`: only
   `backend/agents/llm_client.py` (modified) + new test file
   `tests/agents/test_phase_47_10_generate_content_floor.py`. (The pre-existing
   file content here was the 47.9 verdict, now replaced.)

---

## STEP 2 — Deterministic checks (reproduced, not trusted)

| Check | Command | Exit | Result |
|-------|---------|------|--------|
| ast + 4 helper asserts (IMMUTABLE) | `python -c "import ast; ast.parse(...llm_client.py...); ...4 asserts..."` | **0** | `ast+helper OK` |
| pytest 47.10 suite (IMMUTABLE) | `pytest tests/agents/test_phase_47_10_generate_content_floor.py -q` | **0** | `6 passed in 0.17s` |
| import-cycle check | `python -c "import backend.agents.llm_client"` | **0** | `import llm_client OK` (orchestrator imports llm_client; a reverse import would cycle — none) |
| symmetry check | import both floors | **0** | llm_client=16384, orchestrator=16384, **equal=True** |
| regression scan | `pytest tests/agents/ -k "llm_client or phase_47"` | **0** | `25 passed, 21 deselected` |

(One pre-existing urllib3 RequestsDependencyWarning — environmental noise.)

**Floor is real + correctly gated** (`llm_client.py:1189-1203`):
- const `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384` (`:1189`).
- helper floors thinking+Opus (2048->16384, 1024->16384), no-ops thinking-off
  (2048->2048), no-ops non-Opus (sonnet/haiku/gemini/""->2048), respects large
  budget (30000->30000), boundary (16384->16384). All asserted by tests.

**Wired into generate_content correctly** (`llm_client.py:1479-1481`):
`kwargs["max_tokens"] = _opus_adaptive_max_tokens(kwargs["max_tokens"], model_id, thinking_requested)`
placed AFTER thinking resolution (`:1406-1416`) and AFTER effort resolution
(`:1449-1473`). In-scope locals verified: `model_id` set `:1405`
(`self.model_name or ""`), `thinking_requested` set `:1404`. **Adversarial
KeyError check:** `kwargs["max_tokens"]` is set unconditionally in the dict
literal at `:1352-1358` (`"max_tokens": max_tokens`, where `max_tokens` is bound
at `:1307`); no code path between `:1354` and `:1479` deletes the key. **No
KeyError possible.**

**Symmetry claim verified:** floor value == `multi_agent_orchestrator._OPUS_ADAPTIVE_MIN_MAX_TOKENS`
(both 16384), confirmed by live import + `test_floor_value_matches_orchestrator`.

**Effort-without-thinking decision — CORRECT, not an under-fix.** The helper
gates on `thinking_requested`, NOT effort. Anthropic effort doc (cited brief
`:26,67-76`): "Set `thinking: {type: 'adaptive'}` to enable thinking; without
it, requests run without thinking." Effort raises text/tool tokens but creates
ZERO thinking tokens absent an explicit `thinking` block — so `max_tokens` on an
effort-only call is pure visible output the caller's `max_output_tokens`
intentionally bounds. Flooring it would override the caller's deliberate budget
for no safety benefit. Confirmed in code: `generate_content` only sets
`thinking:{type:adaptive}` when `thinking_requested` is true (`:1406-1409`); it
never auto-enables thinking from effort. `thinking_requested` is the exact-right
gate. **This is NOT an under-fix.**

---

## STEP 2b — Code-review heuristics (5 dimensions evaluated)

Scanned the diff (added lines only). **No findings.** No `secret-in-diff`
(grep exit=1), no `broad-except`, no `print`, no command/SQL/path/SSRF sink, no
LLM-output-to-execution path (`llm-output-to-execution-without-validation` N/A).
`financial-logic-without-behavioral-test` does NOT fire: the token-budget guard
has a dedicated test exercising all four branches + a structural wiring assert.
Diff touches `frontend/` = 0 files -> **§1b ESLint/tsc gate N/A** (correctly
skipped). Helper is fully type-hinted with docstring (`no-type-hints` N/A).
Append `code_review_heuristics` to checks_run.

---

## STEP 3 — LLM judgment

**Contract alignment — all 4 immutable criteria MET:**
1. Floors to 16384 ONLY on thinking+Opus via a pure unit-tested helper; no-op
   thinking-off / non-Opus (effort-without-thinking NOT floored). MET —
   `test_floors_opus_with_thinking` + `test_noop_when_thinking_off` +
   `test_noop_when_not_opus`.
2. Large budgets respected (max(), never lowers); no import cycle (local def).
   MET — `test_respects_larger_caller_budget`; import succeeded.
3. pytest guard + ast clean + green + import clean. MET — `ast+helper OK`;
   6 passed; import OK.
4. Silent text-tail swallow NOT in scope, flagged as follow-up. MET — flagged
   in contract `:29`, experiment_results `:29`; I confirmed the swallow exists
   (`llm_client.py:1623` `stop_reason=max_tokens on text; partial output`,
   logs + returns partial, no retry) and is genuinely deferred (same call 47.9
   made). Honest scope bound, not a silent drop.

**Mutation-resistance — REAL guards (verified by simulation):**
- Regressed helper that floors UNCONDITIONALLY (drops the gate) returns 16384
  for the thinking-off case (test expects 2048) AND the non-Opus case (test
  expects 2048) -> both tests FAIL. Gate is a real guard, not tautological.
- If the gate dropped the Opus check (fired for sonnet): `test_noop_when_not_opus`
  asserts `(2048,"claude-sonnet-4-6",True)==2048` -> FAILS. Real guard.
- If generate_content stopped calling the helper:
  `test_generate_content_applies_the_floor_at_source` asserts the literal wiring
  line `kwargs["max_tokens"] = _opus_adaptive_max_tokens(` AND
  `kwargs["max_tokens"], model_id, thinking_requested` are present in source
  -> FAILS. Structural guard against silent removal.

**Anti-rubber-stamp / severity honesty — severity NOT understated (independently
re-derived).** The claim is "operator-override-only" reachability (ENABLE_THINKING=true
AND DEEP_THINK_MODEL=opus, both non-default). Verified against ACTUAL defaults:
- `settings.py:35` `enable_thinking: bool = Field(False, ...)` — DEFAULT FALSE.
- `settings.py:30` `deep_think_model = Field("gemini-2.5-pro", ...)` — DEFAULT
  GEMINI (explicitly reverted off Opus in phase-37.2 to stop a credit regression).
- `orchestrator.py:613` `self.enable_thinking = settings.enable_thinking`.

I enumerated ALL thinking-config injection sites that reach `generate_content`:
`risk_debate.py:62` (Claude-capable: `getattr(model,"supports_thinking")` AND
`thinking_budget>0`), `debate.py:66` (Gemini-gated: `isinstance(model,GeminiClient)`
— CANNOT route thinking to Claude, confirms brief), and a path the brief's caller
table did NOT explicitly enumerate: **`orchestrator.py:703-715`** — the deep-think
judge injection `if ...supports_thinking AND self.enable_thinking AND is_deep_think
AND agent in thinking_budgets`. Note the static `_THINKING_*` configs
(`orchestrator.py:95,102,107,119`) are DEFINED-BUT-UNUSED (grep shows no consumers);
the live injection is `:703`. **This minor incompleteness in the brief's caller
enumeration does NOT understate the severity** — the `:703` path is gated on the
SAME `self.enable_thinking` (default False) AND only floors for Opus (needs
DEEP_THINK_MODEL=opus, default gemini). Every live thinking-on-Claude path through
generate_content requires the SAME two non-default operator flips. **No
default-config path passes `thinking` with budget>0 to an Opus model.** Severity =
LOW / operator-override-only is correct (the brief slightly UNDER-counted live
paths; it never OVER-stated safety). NOTE-level observation; does not degrade the
verdict.

**Scope honesty — PASS.** Live confirmation honestly DEFERRED
(`experiment_results.md:28-29`): $0 static change, no LLM spend; the only live
exercise requires the operator-override config + a RiskJudge cycle. The text-tail
swallow is flagged out-of-scope (confirmed real at `:1623`), not silently dropped.
Unrelated pre-existing items (COMMUNICATION router effort, `:987` emoji, openclaw
token literal) correctly scoped out.

**Research-gate compliance — PASS.** Researcher output present
(`research_brief_phase_47_10_generate_content_floor.md`, gate_passed:true) and
cited in the contract references section (`:32-35`).

---

## checks_run
syntax, verification_command (ast+helper exit 0), pytest (6 passed),
import_cycle, symmetry_check, generate_content_wiring, keyerror_adversarial,
reachability_severity_audit, mutation_test, code_review_heuristics,
frontend_scope_check, regression_scan, evaluator_critique, masterplan_success_criteria,
scope_honesty, research_gate.

## violated_criteria
(none)

## VERDICT: PASS
All 4 immutable success criteria met and independently reproduced. Deterministic:
immutable ast+helper command exit=0 (`ast+helper OK`), immutable pytest exit=0
(6 passed), import clean, symmetry confirmed (both floors=16384), 25/25 regression
tests green. Mutation-resistance real (gate-drop, opus-check-drop, and
helper-removal each fail a test). Severity honestly stated and independently
re-derived against actual settings.py defaults — operator-override-only confirmed;
no default-config path reaches the floor. Effort-without-thinking correctly NOT
floored (per Anthropic effort doc) — not an under-fix. Out-of-scope text-tail
swallow honestly flagged (confirmed real at `:1623`). Wiring placement after
thinking+effort with in-scope locals and no KeyError exposure. One NOTE: the
brief's caller table omitted the `orchestrator.py:703` deep-think injection as a
generate_content thinking path, but that path shares the identical two-flip gating,
so the severity conclusion is unaffected. PASS-with-NOTE; verdict not degraded.
