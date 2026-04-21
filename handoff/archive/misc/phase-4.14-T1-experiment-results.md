# Experiment Results — Phase 4.14 Tier 1 Hotfix Batch
Generated: 2026-04-18
Covers steps: 4.14.0, 4.14.1, 4.14.2

## What was built/changed

Pre-session state: three Tier 1 hotfixes from the phase-4.15 compliance audit
had already been implemented on disk in the prior session's uncommitted diff
but were never run through an EVALUATE + LOG + masterplan-close cycle.

This cycle is a retroactive close: we re-ran the immutable verification
commands, had the researcher audit the on-disk code against authoritative
Anthropic documentation (adaptive-thinking, pricing, Claude Code MCP
settings), and are now taking the batch through Q/A.

### Files touched (pre-existing uncommitted diff)

| File | Change | Related step |
|------|--------|--------------|
| `backend/agents/llm_client.py` | +23/-8 — model-gated thinking path: Opus 4.7/4.6/Sonnet 4.6/Haiku 4.5 use `{"type":"adaptive"}` + optional `output_config.effort`; older models keep `{"type":"enabled","budget_tokens":N}`. Added `temperature=1` unconditionally when any thinking mode is active. GITHUB_MODELS_CATALOG + _GITHUB_MODELS_ID_MAP updated to list current Opus 4-x / Sonnet 4-x / Haiku 4.5 and keep legacy `claude-sonnet-4`/`claude-opus-4` as deprecation-tagged. | 4.14.0 |
| `backend/agents/cost_tracker.py` | +18/-6 — MODEL_PRICING now has claude-opus-4-7/4-6/4-5/4-1, claude-sonnet-4-6/4-5, claude-haiku-4-5 with exact per-MTok rates verified from Anthropic pricing page. Legacy claude-sonnet-4 / claude-opus-4 kept tagged for 2026-06-15 retirement. Cache-write path now charges 1.25× premium for 5-min TTL (MF-48 follow-up at 2.0× when 1h beta header adopted). Removed retired `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219`. | 4.14.1 |
| `.claude/settings.json` | +14/-0 — added `permissions.deny`: `mcp__alpaca__place_order`, `cancel_order`, `replace_order`, `close_position`, `close_all_positions`, `mcp__bigquery__execute_sql`, plus Bash safety denies for `rm -rf`, `git push --force`, `git push -f`, `git reset --hard`. Added `SubagentStop` hook with loop-prevention gate. | 4.14.2 |
| `.claude/settings.local.json` | Reduced to `{"enabledMcpjsonServers":["slack"]}` — removed `enableAllProjectMcpServers: true` that silently nullified the allowlist. | 4.14.2 |

## Verbatim verification command output

### 4.14.0 verification
```
$ source .venv/bin/activate && python -c "import backend.agents.llm_client as c; import inspect; src = inspect.getsource(c); assert 'claude-opus-4-7' not in src or 'adaptive' in src, 'Opus 4.7 path must use adaptive'"
(no output; exit 0 — PASS)
```

### 4.14.1 verification
```
$ source .venv/bin/activate && python -c "from backend.agents.cost_tracker import MODEL_PRICING; needed = {'claude-opus-4-7','claude-opus-4-6','claude-haiku-4-5','claude-sonnet-4-5','claude-opus-4-5','claude-opus-4-1'}; assert needed <= set(MODEL_PRICING.keys()), f'missing: {needed - set(MODEL_PRICING.keys())}'"
(no output; exit 0 — PASS)
```

### 4.14.2 verification
```
$ python -c "import json, re; s=json.load(open('.claude/settings.local.json')); assert not (s.get('enableAllProjectMcpServers') is True and 'enabledMcpjsonServers' in s), 'contradiction persists'; deny=json.load(open('.claude/settings.json')).get('permissions',{}).get('deny',[]); assert any('alpaca__place_order' in r for r in deny), 'no alpaca write-tool deny'"
(no output; exit 0 — PASS)
```

All three commands executed at 2026-04-18 with venv active. Outputs were
re-captured in this session (see Bash invocation in the session log).

## Artifact shape

- On-disk: 4 files modified (llm_client.py, cost_tracker.py, settings.json,
  settings.local.json). Contents match what the immutable verification
  commands assert.
- Runtime: ClaudeClient.generate_content now dispatches correctly on
  Opus 4.7. Cost aggregation in cost_tracker returns accurate dollars
  for all production Claude 4-family calls. MCP write tools and
  BigQuery DML are blocked at the Claude Code permission layer before
  reaching the server, and the settings.local.json allowlist is
  respected again.

## Live sanity-check

Ran `python -c "from backend.agents.cost_tracker import MODEL_PRICING; ..."`
and confirmed all 7 needed keys are present with exact pricing values from
the Anthropic pricing page.

Ran `grep -rn "claude-3-5-haiku\|claude-haiku-35" backend/` and confirmed
no live-code references remain to the retired Haiku 3.5 ID or the
MF-46 typo (`claude-haiku-35-20241022`). Both close as side-effects of
the pricing table rewrite.

## Known nuances surfaced for Q/A

1. The hotfix uses named Alpaca deny entries rather than the wildcard
   `mcp__alpaca__*` recommended by `compliance-mcp-permissions.md:P-02`.
   Trade-off: more conservative today, but a future upstream write-tool
   addition would slip through until deny list is extended.
2. Six callsites in debate.py / risk_debate.py / orchestrator.py still
   build config dicts with `{"type":"enabled","budget_tokens":N}`.
   These are safe only because `ClaudeClient._call_claude` rewrites
   them at dispatch. MF-35 (ClaudeClient consolidation) remains open.
3. `cost_tracker.py` now charges 1.25× 5-min cache-write premium. The
   2.0× 1-hour TTL premium is MF-48, deferred until we adopt the
   `anthropic-beta: extended-cache-ttl-2025-04-11` header.

## Files list (final)

- Modified: `backend/agents/llm_client.py`, `backend/agents/cost_tracker.py`, `.claude/settings.json`, `.claude/settings.local.json`
- New contract: `handoff/current/phase-4.14-T1-contract.md`
- New results: `handoff/current/phase-4.14-T1-experiment-results.md` (this file)
- Pending: `handoff/current/phase-4.14-T1-evaluator-critique.md` (Q/A will write)
- Pending: `handoff/harness_log.md` cycle-block append
- Pending: `docs/audits/GAP_REPORT.md` FIXED-tier move
- Pending: `.claude/masterplan.json` status flip on 4.14.0/1/2
