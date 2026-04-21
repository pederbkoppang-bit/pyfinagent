# Sprint Contract — Phase 4.14 Tier 1 Hotfix Batch
Generated: 2026-04-18 (combined cycle for 4.14.0, 4.14.1, 4.14.2)
Author: Main (orchestrator) after Researcher gate PASS.

## Steps covered

- **4.14.0** — [T1 HOTFIX] Opus 4.7 thinking-API gate (MF-29)
- **4.14.1** — [T1 HOTFIX] Fix MODEL_PRICING for Opus 4.7/4.6/4.5/4.1 + Haiku 4.5 + Sonnet 4.5 (MF-1)
- **4.14.2** — [T1 HOTFIX] MCP Alpaca write-tool deny + settings.local.json allowlist contradiction (MF-2)

These three Tier 1 hotfixes are batched because (a) they originate from the
same phase-4.15 compliance audit, (b) they were already implemented in a prior
session's uncommitted diff, and (c) each has an independent deterministic
verification command that can be asserted separately. The batch does NOT
merge their immutable success criteria — each step retains its own.

## Research-gate summary (MODERATE tier)

Researcher ran 2026-04-18, returned `gate_passed: true`.

Authoritative external sources (≥3):
1. https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
2. https://platform.claude.com/docs/en/build-with-claude/extended-thinking
3. https://platform.claude.com/docs/en/about-claude/pricing
4. https://code.claude.com/docs/en/settings
5. https://code.claude.com/docs/en/mcp
6. https://github.com/block/goose/issues/7293 (third-party corroboration)
7. https://github.com/anthropics/claude-code/issues/24657 (third-party corroboration)

Internal code references (≥7):
8. backend/agents/llm_client.py:631-653 — MF-29 gate
9. backend/agents/multi_agent_orchestrator.py:944-966 — MAS tool-loop adaptive branch
10. backend/agents/cost_tracker.py:20-76 — MODEL_PRICING table
11. .claude/settings.json:101-112 — deny list
12. .claude/settings.local.json — contradiction removed
13. docs/audits/compliance-mcp-permissions.md:20-46 — P-01/P-02 findings
14. docs/audits/GAP_REPORT.md:24-31 — Tier 1 table

### Key facts from research

- **Opus 4.7 requires `thinking={"type":"adaptive"}` only.** Passing
  `{"type":"enabled","budget_tokens":N}` returns 400. Older models
  (Opus 4.5 and earlier, Sonnet 4.5, Haiku 4.5, Sonnet 3.7) require the
  opposite: enabled+budget_tokens; adaptive is not supported there.
  Sonnet 4.6 and Opus 4.6 accept both; adaptive is now recommended.
  Source: Anthropic adaptive-thinking docs (URL 1).
- **Current per-MTok pricing** (verified 2026-04-18):
  - opus-4-7, opus-4-6, opus-4-5: $5 in / $25 out
  - opus-4-1: $15 in / $75 out
  - sonnet-4-6, sonnet-4-5: $3 in / $15 out
  - haiku-4-5: $1 in / $5 out
  5-min cache-write premium: 1.25× base input. Hit: 0.1× input.
  Source: Anthropic pricing page (URL 3).
- **Claude Code deny list semantics**: deny evaluated first, then ask,
  then allow; first match wins. `mcp__<server>__<tool>` is the
  correct tool-level deny key. `enableAllProjectMcpServers: true`
  supersedes `enabledMcpjsonServers`, silently nullifying the
  allowlist. Source: Claude Code settings + MCP docs (URLs 4-5).

## Hypothesis

The three hotfixes already present on disk close three distinct production
risks documented in `docs/audits/GAP_REPORT.md`:

1. **MF-29 risk**: any call that routes a thinking-enabled request to
   `claude-opus-4-7` through `ClaudeClient` would 400. With the gate
   in `llm_client.py:638-650`, Opus 4.7 calls are rewritten to
   `{"type":"adaptive"}` while legacy models continue to receive
   `{"type":"enabled","budget_tokens":…}`.
2. **MF-1 risk**: pre-fix `MODEL_PRICING` missed all Opus 4-x and
   Haiku 4.5 keys, so cost_tracker fell through to `DEFAULT_PRICING`
   and under-reported spend up to 187×. The new table matches the
   Anthropic-published per-MTok rates to the cent.
3. **MF-2 risk**: pre-fix, `enableAllProjectMcpServers: true` coexisted
   with `enabledMcpjsonServers`, making the allowlist a dead letter;
   and there was no deny entry for `alpaca__place_order / cancel_order
   / replace_order` nor for `bigquery__execute_sql`. The hotfix adds
   those denies and removes the contradiction.

Once Q/A confirms PASS, all three steps transition `status: done` in
`.claude/masterplan.json` and `docs/audits/GAP_REPORT.md` moves MF-29,
MF-1, MF-2 into the "FIXED" tier.

## Immutable success criteria (verbatim from masterplan.json)

### 4.14.0
- opus_4_7_uses_adaptive_not_enabled
- legacy_models_still_work_on_enabled_budget
- no_400_on_claude_opus_4_7_in_live_calls

Verification command:
```
source .venv/bin/activate && python -c "import backend.agents.llm_client as c; import inspect; src = inspect.getsource(c); assert 'claude-opus-4-7' not in src or 'adaptive' in src, 'Opus 4.7 path must use adaptive'"
```

### 4.14.1
- all_current_models_priced
- cost_dashboard_reports_actual_opus_spend
- no_fallback_to_DEFAULT_PRICING_in_production_calls

Verification command:
```
source .venv/bin/activate && python -c "from backend.agents.cost_tracker import MODEL_PRICING; needed = {'claude-opus-4-7','claude-opus-4-6','claude-haiku-4-5','claude-sonnet-4-5','claude-opus-4-5','claude-opus-4-1'}; assert needed <= set(MODEL_PRICING.keys()), f'missing: {needed - set(MODEL_PRICING.keys())}'"
```

### 4.14.2
- no_enableAll_vs_allowlist_contradiction
- alpaca_place_order_cancel_replace_all_in_deny
- bigquery_execute_sql_in_deny

Verification command:
```
python -c "import json, re; s=json.load(open('.claude/settings.local.json')); assert not (s.get('enableAllProjectMcpServers') is True and 'enabledMcpjsonServers' in s), 'contradiction persists'; deny=json.load(open('.claude/settings.json')).get('permissions',{}).get('deny',[]); assert any('alpaca__place_order' in r for r in deny), 'no alpaca write-tool deny'"
```

## Plan steps

1. RESEARCH — done (Researcher returned gate_passed: true; see summary above)
2. PLAN — this contract
3. GENERATE — already on disk in the uncommitted diff:
   - `backend/agents/llm_client.py` (MF-29 gate at 638-650)
   - `backend/agents/cost_tracker.py` (MODEL_PRICING rows at 20-76)
   - `.claude/settings.json` (deny list at 101-112)
   - `.claude/settings.local.json` (contradiction removed)
4. EVALUATE — spawn Q/A for deterministic + LLM judgment pass
5. LOG — append cycle block to `handoff/harness_log.md`; flip
   masterplan statuses to `done`; move the three MF-# entries into
   GAP_REPORT.md's "FIXED THIS CYCLE" section

## Open nuances (for Q/A attention)

- `compliance-mcp-permissions.md:P-02` recommended a wildcard
  `mcp__alpaca__*` instead of named tools. The hotfix uses named
  tools (more conservative). If a future Alpaca MCP upgrade adds a
  new write tool, deny must be updated. Not a blocker; logged.
- Six callsites (`debate.py:59`, `risk_debate.py:56`,
  `orchestrator.py:91/96/101/108/400/406`) still build config dicts
  with `{"type":"enabled","budget_tokens":N}`. These are safe only
  because `ClaudeClient._call_claude` rewrites them at dispatch.
  If any path bypasses `ClaudeClient` (the MF-35 consolidation gap),
  a direct Opus-4.7 call via those configs would 400. Tracked as
  MF-35; NOT in scope for this batch.
- `cost_tracker.py` currently implements 1.25× 5-min cache-write
  premium but not the 2.0× 1-hour premium. Deferred to MF-48.

## References

- Masterplan: `.claude/masterplan.json` steps 4.14.0 / 4.14.1 / 4.14.2
- Gap report: `docs/audits/GAP_REPORT.md` Tier 1 table (lines 24-31)
- Compliance audit trail: `docs/audits/compliance-thinking.md`,
  `compliance-models-pricing.md`, `compliance-mcp-permissions.md`
- Researcher report: saved inline above; agentId a72916d9355ceefa6
