# Contract -- Full-App End-to-End UAT masterplan phase (task #48)

## Research gate

- Researcher spawn: 2026-04-24. Brief at `handoff/current/full-app-uat-research-brief.md`.
- JSON envelope: tier=moderate, external_sources_read_in_full=6 (floor 5), urls_collected=16, recency_scan_performed=true, internal_files_inspected=47, gate_passed=true.
- Brief produced a complete inventory (18 subsystem categories, file-anchored) + external patterns (Anthropic Harness Design, Google SRE Pre-Launch Checklist, Exactpro Algo Test Harness, XUAT-Copilot multi-agent UAT, Galileo AI-agent readiness).
- Phase id `phase-12` is already taken (Rainbow Deploys, done). Using `phase-16` (unused).

## Top 3 gotchas the plan MUST handle

1. `cache.preload_macro()` before any backtest step (else silent hang ~40min).
2. Assert `ALPACA_PAPER_TRADE=true` + `execution_router._refuse_live_keys()` DID NOT short-circuit to live fills before any paper-cycle exercise.
3. `phase-16.15` (Go/No-Go verdict) MUST have "Q/A spawn and PASS verdict" as an immutable success criterion. Self-evaluation is forbidden.

## Planned change

Write ONE new masterplan entry `phase-16` with 15 sub-steps and immutable
success-criteria. The JSON envelope-shape must match sibling phases
(phase-4.17 and phase-12 reviewed). Each sub-step has:
- `id` (e.g., `16.1`), `name`, `status=pending`
- `harness_required: true` on steps that need MAS oversight (mid-to-late
  steps; inventory/infra checks can be lighter)
- `verification.command` (copy-pasteable shell/python one-liner)
- `verification.success_criteria` (immutable — list of pass conditions)
- `contract: null`, `retry_count: 0` (to match sibling shape)

Sub-step skeleton (titles + the critical verification check each):

| Sub-step | Name | Critical check |
|---|---|---|
| 16.1 | Infrastructure readiness | launchctl list for backend/frontend/mas-harness all active; `curl /api/health` 200; BQ `SELECT 1` round-trip; disk free > 5GB |
| 16.2 | Analysis pipeline (Layer 1) | POST /api/analysis/start + poll /status until done; assert report.json written + BQ `analysis_results` row inserted |
| 16.3 | MAS Orchestrator (Layer 2) live round-trip | Spawn planner->evaluator round; assert reflection loop produced >=1 iteration; DecisionTrace written |
| 16.4 | Autonomous paper-trading dry-run | Force one `run_autonomous_cycle()` NOW; assert: loop refused live keys, `paper_portfolio_snapshots` row appended, observability logs (BLOCKER-1 + task #47 patterns) visible |
| 16.5 | Self-improving loops | MetaCoordinator.decide() round + 1-iteration of skill_optimizer + 1-iteration of perf_optimizer; assert each wrote its TSV/BQ result |
| 16.6 | Kill switch + risk guards | fire pause via API -> assert all cycles skip -> resume -> run zero_orders_drill -> PASS; confirm execution_router lockout intact |
| 16.7 | HITL C/C gate end-to-end | re-run hitl_gate_drill.py + real-BQ row verification (SELECT FROM strategy_deployments_log WHERE strategy_id LIKE 'UAT-%') |
| 16.8 | Slack bot + scheduled jobs | Post a `[UAT-16.8]` test message; assert it lands in the UAT channel; enumerate APScheduler jobs + assert each has a valid next_run_time |
| 16.9 | Backtest + quant optimizer | `cache.preload_macro()` first (critical); 2-iter walkforward run; assert Sharpe > 0 AND experiments TSV appended |
| 16.10 | Frontend full-page sweep | `curl /` then `curl /backtest /paper-trading /reports /sovereign /signals /performance /settings /agents /login` — all 200 + contain non-empty HTML body |
| 16.11 | Auth + OWASP | JWE session roundtrip via `/api/auth/session`; 401 on `/api/paper/*` without token; `X-Frame-Options: DENY` header on all GET responses |
| 16.12 | Observability | `cycle_health.py::gather_health()` returns fresh < 86400s; perf_tracker has rows; harness_log tail non-empty |
| 16.13 | Drills aggregate gate | `python scripts/go_live_drills/aggregate_gate_check.py` exit 0 AND zero_orders + revert_hygiene + hitl_gate all green |
| 16.14 | Harness MAS full cycle dry-run | `python scripts/harness/run_harness.py --cycles 1 --iterations-per-cycle 3 --dry-run` exits 0; all 5 handoff files produced |
| 16.15 | **Go/No-Go verdict** | spawn `qa` subagent with full UAT evidence bundle; Q/A returns PASS/CONDITIONAL/FAIL; PASS required to flip phase-16 done. **Q/A spawn + PASS verdict is an immutable criterion.** |

## NOT in scope this cycle

- Executing the UAT. This cycle PLANS it — the UAT itself runs as a future cycle against the added masterplan entries.
- Changing any application code.
- Flipping any existing masterplan statuses.
- Renumbering or consolidating existing phases.

## Immutable success criteria (for THIS planning cycle)

1. `.claude/masterplan.json` contains a top-level entry with `id: "phase-16"` whose `status` is `pending`.
2. That entry has exactly 15 sub-steps with ids `16.1` through `16.15`.
3. Every sub-step has a non-null `verification.command` string.
4. Every sub-step has a `verification.success_criteria` list of length >= 2.
5. Sub-step `16.9` verification.command contains the literal string `preload_macro` (gotcha #1).
6. Sub-step `16.4` success_criteria contains the literal string `paper` AND the literal string `live keys` or `lockout` (gotcha #2).
7. Sub-step `16.15` success_criteria contains the literal string `Q/A` or `qa` and the literal string `PASS` (gotcha #3).
8. JSON validity: `python -c "import json; json.loads(open('.claude/masterplan.json').read())"` exits 0.
9. Sibling-shape compatibility: the new phase entry has the same top-level keys as sibling `phase-12` (id, status, name, description, created_at, steps).
10. `handoff/current/uat-runbook.md` exists with a one-page operator summary of phase-16.

## Verification command (Q/A reproduces)

```bash
source .venv/bin/activate
python3 -c "
import json
mp = json.loads(open('.claude/masterplan.json').read())
def walk(n):
    if isinstance(n, dict):
        if n.get('id') == 'phase-16': return n
        for v in n.values():
            r = walk(v)
            if r: return r
    elif isinstance(n, list):
        for i in n:
            r = walk(i)
            if r: return r
p16 = walk(mp)
assert p16 is not None, 'phase-16 missing'
assert p16.get('status') == 'pending', f'status={p16.get(\"status\")}'
steps = p16.get('steps', [])
ids = [s.get('id') for s in steps]
assert ids == [f'16.{i}' for i in range(1,16)], f'ids={ids}'
for s in steps:
    v = s.get('verification') or {}
    assert v.get('command'), f'{s[\"id\"]} missing verification.command'
    assert isinstance(v.get('success_criteria'), list) and len(v['success_criteria']) >= 2, f'{s[\"id\"]} criteria<2'
# gotcha checks
s9 = next(s for s in steps if s['id']=='16.9')
assert 'preload_macro' in s9['verification']['command'], '16.9 missing preload_macro'
s4 = next(s for s in steps if s['id']=='16.4')
s4_text = ' '.join(s4['verification']['success_criteria']).lower()
assert 'paper' in s4_text and ('live keys' in s4_text or 'lockout' in s4_text), '16.4 missing paper/lockout'
s15 = next(s for s in steps if s['id']=='16.15')
s15_text = ' '.join(s15['verification']['success_criteria']).lower()
assert ('q/a' in s15_text or 'qa' in s15_text) and 'pass' in s15_text, '16.15 missing qa/pass'
print('ALL_ASSERTS_OK')
"
test -f handoff/current/uat-runbook.md
```

All must succeed for PASS.

## References

- `handoff/current/full-app-uat-research-brief.md` (research deliverable)
- `.claude/masterplan.json` (target file)
- `handoff/current/uat-runbook.md` (new deliverable — operator summary)
- Sibling reference for shape: phase-12 (Rainbow Deploys, done)
