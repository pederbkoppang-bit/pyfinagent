# Contract -- 67.6 Layer-2 API modernization (Fable-5/Sonnet-5 request guards + SDK pin)

Step: masterplan phase-67 / 67.6 (P0; must precede 67.4's Sonnet-5 option). Research
gate: PASSED (moderate; research_brief_67_6.md -- 7 read in full incl. 5 official
Anthropic docs, 28 URLs, recency scan, 12 internal files).

## Research-gate summary

- Traps confirmed from official docs: fable-5 400s on ANY explicit thinking config
  (incl. {"type":"disabled"}) -- the only valid shape is omitting the key; sonnet-5
  400s on enabled+budget_tokens and on non-default sampling; adaptive accepted.
- SDK: 0.96.0 installed and probed (messages.create has output_config + thinking
  kwargs; OutputConfigParam effort low..max; advisor/usage.iterations are BETA-
  namespace only -- consistent with the code's beta usage). The ==0.87.0 pin is a
  DOWNGRADE TIME BOMB: any `pip install -r requirements.txt` removes the
  output_config kwarg -> TypeError on every effort call. 0.87.0=2026-03-31 vs
  0.96.0=2026-04-16.
- Bonus real bug: the xhigh guard (llm_client.py:1507-1512) spuriously downgrades
  fable-5's xhigh fallback (effort doc: xhigh GA on fable-5 + sonnet-5).
- SECOND TRAP (declared in-scope, see Design): multi_agent_orchestrator.py:1089-1116
  unconditionally sends enabled+budget_tokens 2048 + temperature=1 per tool-loop turn
  for any non-opus-4-8/4-7 model -- fires the instant a mas_* role pins sonnet-5/fable.
- CLI rail clean (claude_code_client.py argv carries no sampling/thinking flags);
  fable refusal stop_reason already handled (llm_client.py:1666-1674).
- Test seams identified: patch ClaudeClient._get_client (client built inside
  generate_content); fake needs .messages.create AND .beta.messages.create;
  hermeticity via COST_BUDGET_HARD_BLOCK_DISABLED=1 + patching
  backend.services.observability.log_llm_call.
- Sonnet-5 tokenizer ~+30% tokens -> handoff note for 67.4 (max_tokens revisit),
  recorded in the LOG phase.

## Hypothesis (falsifiable)

Extending the model-family guards (thinking allowlist, sampling strip, xhigh
allow, structured-output eligibility, effort tables) to claude-fable-5 /
claude-sonnet-5 and reconciling the SDK pin makes every reachable Claude request
shape valid for those families with zero change to currently-pinned families --
provable by builder-level request-shape tests that capture the exact kwargs passed
to the (patched) SDK client.

## Success criteria (verbatim from .claude/masterplan.json 67.6 -- IMMUTABLE)

1. "backend/agents/llm_client.py sends NO sampling params (temperature/top_p/top_k)
   in request shapes for claude-fable-5, claude-sonnet-5, claude-opus-4-8,
   claude-opus-4-7 (strip-list extended; behavioral test proves it)"
2. "llm_client.py never sends legacy {type:enabled, budget_tokens} thinking to
   claude-fable-5 or claude-sonnet-5: fable-5 omits the thinking param entirely
   (always-on per Anthropic docs); sonnet-5 receives adaptive; behavioral tests
   prove both"
3. "backend/requirements.txt anthropic pin matches the installed-and-required SDK
   (exact pin 0.96.0, supply-chain-hardening rationale comment preserved); the
   documented-vs-pinned mismatch is gone"
4. "No behavior change for currently-pinned models: opus-4-8/4-7/4-6, sonnet-4-6,
   haiku-4-5 request shapes preserved and asserted by the same test module"
5. "Fresh Q/A PASS with the 67.1 gates applied to this diff (lint + runtime smoke
   over the changed files, output in the critique)"

## Design (minimal diff per the brief; house idiom = inline startswith tuples)

1. `backend/agents/llm_client.py`:
   (a) thinking dispatch (~:1444-1457): fable-5 branch FIRST -- omit the thinking key
       entirely; add "claude-sonnet-5" to the adaptive tuple; legacy else preserved
       for older models.
   (b) sampling strip (~:1467-1470): tuple += "claude-fable-5", "claude-sonnet-5".
   (c) xhigh allow (~:1507-1512): tuple += "claude-fable-5", "claude-sonnet-5"
       (fixes the spurious fable xhigh->high downgrade).
   (d) structured-output eligibility (~:1548-1551): _fmt-eligible tuple += both
       (official structured-outputs support list includes both).
2. `backend/config/model_tiers.py`: EFFORT_SUPPORTED_MODELS += "claude-sonnet-5"
   (:218 block); MODEL_EFFORT_FALLBACK += ("claude-sonnet-5", "high") (:273 block).
   Load-bearing for the live_check payload dump (fable entries already exist).
3. `backend/agents/multi_agent_orchestrator.py` (~:1089-1116) -- DECLARED ADDITIONAL
   SCOPE (same trap class, same step purpose; disclosed here so scope honesty holds):
   apply the same family guard so fable-5/sonnet-5 get adaptive-or-omitted thinking
   and no sampling params in the tool-loop; current models unchanged.
4. `backend/requirements.txt:39`: `anthropic==0.96.0` with the hardening rationale
   comment preserved + phase-67.6 reconciliation note.
5. NEW `backend/tests/test_claude_request_shapes.py`: monkeypatch
   ClaudeClient._get_client; capture kwargs; assert per-family shapes -- fable-5 (no
   thinking key, no temperature, effort in output_config), sonnet-5 (adaptive
   thinking, no temperature, effort present), opus-4-8/4-7 (no temperature, adaptive
   when requested), opus-4-6 / sonnet-4-6 / haiku-4-5 (existing shapes preserved,
   temperature present as today), legacy enabled-branch survival for older models;
   plus an orchestrator tool-loop shape test for the declared additional scope.
   Hermetic: COST_BUDGET_HARD_BLOCK_DISABLED=1, observability patched, zero network.

## Anti-patterns guarded

- consumer-contract-break: guards are additive family branches; every existing family
  asserted unchanged by the same test module (criterion 4 IS the consumer check).
- over-mocked-test: only the SDK client boundary is faked (the seam), the entire
  request-construction path under test runs for real.
- criteria-erosion / scope creep: the orchestrator fix is declared here in the
  contract, not smuggled; nothing else in the orchestrator changes.
- Downgrade time bomb regression: the pin bump is asserted by the immutable grep.

## Out of scope

Streaming adoption for the 16384 adaptive path (registered); always-adaptive default
(registered); Gemini/OpenAI paths; any model PIN change (67.4's job); CLI rail.

## Risk

- A future SDK bump changes kwargs shape -> tests pin the CURRENT contract; the exact
  pin prevents silent drift either direction.
- Orchestrator tool-loop has its own response handling -> the added guard only alters
  request kwargs for not-yet-pinned families; asserted by test.
