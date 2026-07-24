# live_check 67.1 -- lint gate + runtime smoke, executed live 2026-07-09

Required shape (masterplan 67.1): "(a) verbatim output of the new lint gate executed
from the venv over a real diff and (b) the runtime-smoke output (module import +
exercised path), plus the fresh Q/A verdict JSON".

## (a) Lint gate, verbatim (uvx ruff, run live this session)

Over this step's own changed fileset (gate semantics -- markdown-only diff):
```
$ CHANGED_PY=$(git diff --name-only HEAD -- '*.py'); echo "changed .py files: '${CHANGED_PY:-NONE}'"
changed .py files: 'NONE'
```

Teeth demonstration on the known-buggy real file (67.2's fix target):
```
$ uvx ruff check --select F821,F401,F811 backend/agents/agent_definitions.py; echo "exit=$?"
F401 [*] `typing.Optional` imported but unused
  --> backend/agents/agent_definitions.py:25:20
F821 Undefined name `json`
   --> backend/agents/agent_definitions.py:396:13
Found 2 errors.
[*] 1 fixable with the `--fix` option.
exit=1
```

Clean-file control:
```
$ uvx ruff check --select F821,F401,F811 backend/config/model_tiers.py; echo "exit=$?"
All checks passed!
exit=0
```

## (b) Runtime smoke, verbatim (live backend)

```
$ curl -s -o /dev/null -w "backend :8000 -> HTTP %{http_code} in %{time_total}s\n" --max-time 5 http://localhost:8000/api/health
backend :8000 -> HTTP 200 in 0.004779s
```
Module-import leg: N/A for this diff (no backend module changed); the leg is exercised
for real in 67.2 (whose diff touches backend/agents/agent_definitions.py).

## (c) Fresh Q/A verdict JSON

Returned by qa-67-1 (pre-change snapshot) 2026-07-09: `verdict: PASS, ok: true,
violated_criteria: [], certified_fallback: false`, 10 checks_run including independent
reproduction of the immutable command (exit=0), the ruff teeth demo (F821 :396 +
F401 :25, exit=1) and clean control (exit=0), pytest-timeout presence, gate-integrity
greps (1b/1c/3rd-CONDITIONAL/certified_fallback/read-only all intact), and the
settings.json-untouched check. Full JSON: evaluator_critique_67.1.md.
