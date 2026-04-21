# Sprint Contract — phase-4.14.3
Generated: 2026-04-18
Step: 4.14.3 — [T2] Add output_config.effort pass-through to ClaudeClient (xhigh/high/medium/low per agent class)
Related gap: MF-28

## Research-gate summary (MODERATE tier — gate_passed: true)

Researcher returned 13 citations (9 external Anthropic-first-party
+ 4 internal file:line refs). Key findings:

1. **`output_config.effort` is independent of thinking** — docs state
   "It doesn't require thinking to be enabled in order to use it."
   Effort controls ALL tokens (text + tool calls + thinking when on).
2. **Valid values**: low / medium / high / xhigh / max. `xhigh` is
   **Opus 4.7 ONLY** — sending it to Sonnet 4.6 or Opus 4.6 returns
   a 400. `max` is supported on Opus 4.7, 4.6, 4.5 and Sonnet 4.6.
3. **Supported models** (per docs): opus-4-7, opus-4-6, opus-4-5,
   sonnet-4-6. Haiku 4.5 is NOT in the docs' supported list — do
   not inject `output_config` for Haiku routes.
4. **Anthropic recommended defaults**:
   - Opus 4.7 coding/agentic → `xhigh` (explicitly documented)
   - Sonnet 4.6 → `medium` (docs: "explicitly set … to avoid
     unexpected latency")
   - API implicit default when omitted = `high`.

Authoritative sources:
- https://platform.claude.com/docs/en/build-with-claude/effort
- https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
- https://platform.claude.com/docs/en/build-with-claude/extended-thinking

Internal refs:
- backend/agents/llm_client.py:638-653 — current wiring BUG: effort
  only applied inside `if thinking_requested:` branch
- backend/config/model_tiers.py:42-62 — role → model mapping (home
  for the new effort defaults)
- backend/agents/multi_agent_orchestrator.py:159-165 — direct
  anthropic.Anthropic() bypass (out of scope; MF-35)
- backend/agents/planner_agent.py:35 — direct client bypass (MF-35)

## Hypothesis

Adding a role-keyed `EFFORT_DEFAULTS` table in `model_tiers.py` +
a `resolve_effort(role)` helper, and hoisting the effort injection
in `llm_client.py` OUTSIDE the `thinking_requested` branch, will:

1. Cause Sonnet 4.6 callers to start at `medium` instead of the
   API implicit `high` — cuts latency + cost on the highest-volume
   MAS call path (mas_communication routing).
2. Cause Opus 4.7 callers (forward-looking; no build-tier usage
   today) to auto-start at `xhigh` when the `_LIVE_TIER` is
   populated at May launch.
3. Preserve the existing kwarg > config > default precedence so
   explicit callers still win.

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. effort_passthrough_wired
2. sonnet_4_6_default_medium_not_high
3. opus_4_7_starts_xhigh_on_coding_paths

### Verification command (immutable)

```
source .venv/bin/activate && python -c "import inspect, backend.agents.llm_client as c; assert 'output_config' in inspect.getsource(c) and 'effort' in inspect.getsource(c)"
```

(Q/A noted in phase-4.14-T1 close that this command is weak. We
are not empowered to change it; MF-51 tracks hardening. We will,
however, supplement with additional deterministic checks in the
experiment_results command log so Q/A can audit semantic
correctness.)

## Plan steps

1. RESEARCH — done (gate_passed: true)
2. PLAN — this contract
3. GENERATE:
   a. Add `EFFORT_DEFAULTS: dict[str, str | None]` in
      `backend/config/model_tiers.py` keyed by role, with values:
      - `mas_communication` → `"low"` (routing only)
      - `mas_main` → `"high"`
      - `mas_qa` → `"high"`
      - `mas_research` → `"medium"`
      - `autoresearch_fast` → `None` (Haiku unsupported)
      - `autoresearch_smart` → `"medium"`
      - `autoresearch_strategic` → `"high"`
   b. Add `resolve_effort(role, tier=None) -> str | None` that
      returns `EFFORT_DEFAULTS[role]` (raises KeyError for unknown
      roles, returns None if role deliberately unsupported).
   c. Add a `MODEL_EFFORT_FALLBACK: dict[str, str | None]` keyed
      by model-ID prefix for callers that supply no role:
      - `claude-opus-4-7` → `"xhigh"`
      - `claude-opus-4-6`, `claude-opus-4-5`, `claude-opus-4-1`
        → `"high"`
      - `claude-sonnet-4-6`, `claude-sonnet-4-5` → `"medium"`
      - `claude-haiku-4-5` → `None`
   d. Add a `resolve_effort_by_model(model_id) -> str | None`
      helper that walks the prefix map; returns None if no match.
   e. Add `EFFORT_SUPPORTED_MODELS` allowlist for the 400-guard:
      opus-4-7, opus-4-6, opus-4-5, sonnet-4-6. Omit effort
      entirely for any other model.
   f. In `backend/agents/llm_client.py` `_call_claude()`:
      - Hoist effort resolution OUT of the thinking branch:
        effort = (explicit kwarg) or config["effort"] or
                 thinking_cfg["effort"] or
                 resolve_effort_by_model(model_id)
      - Apply `xhigh` downgrade-with-log guard: if effort ==
        "xhigh" and model is not opus-4-7 → log WARNING and drop
        to "high".
      - Apply model-support guard: if model_id doesn't start
        with any EFFORT_SUPPORTED_MODELS prefix → drop effort to
        None (Haiku, Gemini, etc.).
      - If effort is not None → `kwargs["output_config"] =
        {"effort": effort}`.
   g. Remove the now-redundant `effort` handling inside the
      thinking_requested block (it becomes a no-op since the
      outer hoist already set output_config).
4. VERIFY — run the immutable cmd + supplementary semantic checks
   (grep for EFFORT_DEFAULTS keys, import model_tiers and call
   resolve_effort, import llm_client and grep for the hoist).
5. EVALUATE — spawn Q/A. If CONDITIONAL/FAIL → fix and
   SendMessage back to SAME Q/A.
6. LOG — append cycle block; flip masterplan status; move MF-28
   to FIXED in GAP_REPORT.md.

## Out of scope

- planner_agent.py / evaluator_agent.py direct-client refactors
  → tracked as MF-35
- multi_agent_orchestrator.py direct-client effort threading →
  same (MF-35 consolidation is the lever; effort wiring follows)
- Hardening the weak immutable verification command (MF-51)

## References

- Masterplan: .claude/masterplan.json step 4.14.3
- Researcher report: inline in this session (agentId a8493df077b16b9f5)
- Anthropic effort docs: https://platform.claude.com/docs/en/build-with-claude/effort
- .claude/rules/backend-agents.md (cost controls section)
