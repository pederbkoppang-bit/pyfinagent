---
name: fable-sonnet5-request-shapes-67-6
description: phase-67.6 API-guard research — fable omits thinking ENTIRELY (disabled also 400s), sonnet-5 adaptive-by-default + new tokenizer +30%, xhigh now GA on fable/sonnet-5, SDK 0.96.0 advisor/iterations are BETA-namespace-only, pin-downgrade time bomb, second ungated trap in mas orchestrator
metadata:
  type: project
---

Facts from the 67.6 research gate (2026-07-09/10) that outlive the step:

- **Fable 5 thinking**: ANY explicit thinking config 400s, including `{type:"disabled"}` — the only valid shape is OMITTING the key (adaptive always-on server-side). Do not map budget_tokens->adaptive for fable.
- **Sonnet 5**: adaptive thinking ON by default (no thinking field -> thinks; `disabled` allowed, unlike fable); manual enabled+budget_tokens 400s; NON-default sampling params 400 (explicit temperature=1 = default = accepted, but strip anyway); **new tokenizer ~+30% tokens for same text** -> any 67.4 Sonnet-5 exercise must revisit max_tokens budgets; effort default = high; intro pricing $2/$10 through 2026-08-31 then $3/$15; 1M ctx / 128K out; NO Priority Tier.
- **xhigh effort is GA on fable-5 AND sonnet-5** (plus opus-4-8/4-7) per the effort doc — llm_client's old xhigh guard (opus-only) and any "xhigh is opus-only" comment is stale.
- **Structured outputs (output_config.format) GA on fable-5 + sonnet-5** (official verbatim list).
- **SDK 0.96.0 (released 2026-04-16)**: output_config{effort,format} + ThinkingConfigAdaptiveParam in STABLE types; advisor_20260301 (BetaAdvisorTool20260301Param) and usage.iterations (BetaUsage only) live in the BETA namespace — stable Usage has no iterations field. Typed model literals for sonnet-5 landed 0.114.0 (2026-06-30), fable ~0.108.0 — irrelevant functionally (model is str pass-through).
- **Pin-downgrade time bomb pattern**: requirements pin (==0.87.0) older than installed (0.96.0) means any `pip install -r` (incl. nightly maintenance install) silently DOWNGRADES and removes the output_config kwarg -> TypeError on every effort call. When code comments say ">=X" but requirements pin "<X", treat as P0 not doc-drift.
- **Second ungated trap site**: multi_agent_orchestrator.py:1089-1116 sends `thinking={"type":"enabled","budget_tokens":2048}, temperature=1` UNCONDITIONALLY per tool-loop turn for any non-opus-4-8/4-7 model (no ENABLE_THINKING gate) — fires the instant a mas_* role repins to sonnet-5/fable-5. 67.6's immutable criteria cover llm_client.py only; this site needs its own fix/step.
- **Test seam for request-shape tests**: monkeypatch `ClaudeClient._get_client` (client is constructed INSIDE generate_content) + `COST_BUDGET_HARD_BLOCK_DISABLED=1` env (existing escape hatch llm_client.py:411) + patch backend.services.observability.log_llm_call. Fake needs BOTH `.messages.create` and `.beta.messages.create`.

**Why:** these are external-doc + venv-probe facts not derivable from the repo until 67.6 ships, and 67.4 (Sonnet-5 option) depends on the tokenizer/max_tokens and mas-orchestrator-trap facts.
**How to apply:** cite when researching 67.4, any mas_* repin, or future SDK bumps past 0.96.0. Related: [[project_fable5_adoption]], [[project_cost_pricing_tables_inventory]].
