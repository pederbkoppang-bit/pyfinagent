---
step: phase-16.16
cycle_date: 2026-04-25
retrospective: false
forward_cycle: true
---

# Experiment Results -- phase-16.16

## What was done this cycle

Read-only verification re-run of the backend correctness suite. No code
changes. The four-stage verification command was executed verbatim;
each stage's stdout + exit code captured below.

### Files touched this cycle
| Path | Action |
|------|--------|
| `handoff/current/contract.md` | overwrite (rolling) |
| `handoff/current/experiment_results.md` | overwrite (this file) |
| `handoff/current/phase-16.16-research-brief.md` | created by researcher subagent |

No backend / frontend / migration / config code modified.

## Verification command (run verbatim)

```
source .venv/bin/activate \
  && python -m pytest backend/tests/ -q \
  && python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('backend/**/*.py', recursive=True) if '__pycache__' not in f]; print('syntax_ok')" \
  && python scripts/migrations/create_strategy_deployments_view.py --verify \
  && curl -sS http://127.0.0.1:8000/api/health
```

### Stage 1: pytest

```
........................................................................ [ 40%]
........................................................................ [ 80%]
..............s...................                                       [100%]
177 passed, 1 skipped, 1 warning in 15.97s
```

**Result: PASS** (177/177 actively tested; 1 skipped; 0 failed; 0 errors)

### Stage 2: AST audit

```
syntax_ok: 258 files parsed
```

**Result: PASS** (258 `.py` files under `backend/`, all parseable)

### Stage 3: BQ migration --verify

```
[verify] view_exists: PASS (sunny-might-477607-p8.pyfinagent_pms.strategy_deployments)
[verify] at_least_one_champion_row: PASS (1 champion rows)
[verify] ALL CHECKS PASS
```

**Result: PASS**

### Stage 4: /api/health

```
HTTP 200
{
    "status": "ok",
    "service": "pyfinagent-backend",
    "version": "6.5.85",
    "mcp_servers": {
        "data": {"status": "ok"},
        "backtest": {"status": "ok"},
        "signals": {"status": "ok"}
    },
    "limits_digest": "edf822591bb17c9d8f62f4f50a8fca72f11690b21884b7cd2f0988e0e2c9bad4"
}
```

**Result: PASS** -- 200 + status="ok" + 3 of 3 in-app MCP servers report "ok" + immutable limits digest visible.

## Success criteria assessment (verbatim from masterplan)

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | pytest_all_pass | PASS | 177 passed, 0 failed |
| 2 | ast_clean_all_backend_py | PASS | 258 files parsed clean |
| 3 | bq_migration_verify_pass | PASS | view_exists + 1 champion row + ALL CHECKS PASS |
| 4 | backend_health_200 | PASS | HTTP 200 + status=ok + mcp_servers all ok |

## Honest disclosures / caveats

1. **One test was skipped** in pytest. The pytest output shows `1 skipped`. Likely a `@pytest.mark.skip` or `pytest.importorskip` on something with an environmental gate. Not flagged as a failure by pytest, so not a blocker. Q/A can drill if it wants.

2. **One DeprecationWarning** from `google.genai.types.py` about `_UnionGenericAlias` in Python 3.17 -- 3rd-party SDK warning, not our code. Not a blocker.

3. **One RequestsDependencyWarning** about urllib3 / chardet version skew. Pre-existing in `.venv`, fires on every Python invocation that touches `requests`. Cosmetic.

4. **The `cd backend && pytest` shadow bug remains** -- pytest from repo root works, but if anyone runs the masterplan 10.5.0 verification command literally (which says `cd backend && pytest`) it will still fail. This is a known follow-up (task #9 in the session task bar) and orthogonal to 16.16's deliverable.

5. **Backend was already running on PID 8301** before this cycle started (earlier-this-session restart for FRED key load). Not restarted this cycle. The /api/health response confirms it's live and healthy.

6. **No code changes** -- verification cycle only. No git diff this cycle.

## No-regressions

`git status` shows handoff/ artifacts modified (expected per protocol) plus `.claude/masterplan.json` (the 16.16-16.23 insertion done at cycle start). No backend/frontend/script code modified by Main this cycle.

## Next

Spawn Q/A to audit. If PASS: append harness_log.md, flip 16.16, hook auto-archives. Then proceed to 16.17 (frontend re-verification).
