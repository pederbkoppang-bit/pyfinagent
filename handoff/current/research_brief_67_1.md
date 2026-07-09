# Research Brief -- Step 67.1 (Q/A verification-depth upgrade)

Tier: moderate. Researcher: sole (external literature + internal code audit).
Status: IN PROGRESS (WRITE-FIRST incremental brief).

## Step 67.1 immutable success criteria (verbatim from .claude/masterplan.json)

1. qa.md contains NO global 55-second runtime cap; verification budget is
   tiered and explicitly permits full pytest + runtime smoke for backend-
   touching diffs.
2. qa.md defines a deterministic backend gate REQUIRED for diffs touching
   `backend/**`: (a) undefined-name-class Python lint (ruff or pyflakes)
   over changed .py files with verbatim exit code, and (b) a runtime smoke
   that imports each changed module inside .venv and, when the diff touches
   a live API/service path, exercises it (endpoint or command) with output
   captured.
3. The `stop_hook_active` escape hatch no longer returns `ok:true`; loop-
   prevention exits are verdict-neutral (`ok:false` + explicit no-evaluation-
   performed reason) so no auto-PASS path remains.
4. CONDITIONAL/FAIL recovery guidance is consistent across qa.md,
   docs/runbooks/per-step-protocol.md, and CLAUDE.md: fix blockers -> update
   handoff evidence -> spawn a FRESH Q/A; respawn on UNCHANGED evidence stays
   forbidden; no artifact still mandates SendMessage-to-the-SAME-agent.
5. A fresh Q/A returns PASS on this step's diff, and the new lint gate is
   proven live with verbatim ruff/pyflakes output over the step's own
   changed files.

---

## Internal code inventory (file:line anchors)

| File | Lines | Role | Status / finding |
|------|-------|------|------------------|
| `.claude/agents/qa.md` | 244 | `**Maximum runtime: 55 seconds** (leave buffer for hook timeout)` | DEAD CAP. The "hook timeout" it buffers for is the Stop hook's `timeout: 55` (settings.json:119). Q/A now spawns via the Agent tool (`maxTurns: 30`), NOT inside that hook. Cap is a relic. |
| `.claude/agents/qa.md` | 247-248 | `If stop_hook_active is true ... return {"ok": true, "reason": "loop prevention"}` | AUTO-PASS BACKDOOR. In Q/A's output schema `ok:true` == PASS verdict. Copied the loop-prevention idiom from the Stop hook, but in the evaluator schema it emits a PASS. |
| `.claude/agents/qa.md` | 250-251 | `Never second-opinion-shop ... SendMessage back to the SAME agent, not spawn a new one` | CONTRADICTS CLAUDE.md canonical fresh-respawn recovery. |
| `.claude/agents/qa.md` | 83-93 | Deterministic-checks example block; line 92 `python -m pytest tests/ -v --timeout=30` | TWO BROKEN COMMANDS: (a) `--timeout=30` errors `unrecognized arguments` because pytest-timeout is NOT installed; (b) `tests/` (repo-root) currently has 7 collection errors (725 collected, 7 err) and is a different tree than `backend/tests/` (991 collected clean). Example is aspirational, not runnable as written. |
| `.claude/agents/qa.md` | 79-93 | `### 1. Deterministic checks` | Covers syntax + file-existence + verification cmd + pytest, but NO Python lint (undefined-name class). Frontend has ESLint gate (§1b, 95-120) but backend has no analogue -- exactly the gap 67.1 fills (mirror of the phase-23.2.24 ESLint precedent). |
| `.claude/settings.json` | 113-124 | Stop hook (`type: agent`, inline prompt, `timeout: 55`, `stop_hook_active -> {"ok": true, "reason": "loop prevention"}`) | SEPARATE, LEGITIMATE. This is a Stop-hook, NOT the qa.md subagent. Here `ok:true` correctly means "allow Claude to stop." 67.1 verification command does NOT touch settings.json -- leave it. Only qa.md's evaluator copy is the bug. |
| `.claude/settings.json` | 125-134 | SubagentStop hook, also emits `{"ok": true, "reason": "loop prevention"}` | Command hook, not qa.md. Out of 67.1 scope. |
| `docs/runbooks/per-step-protocol.md` | 235-237 | Anti-pattern #5: `fix the blockers then SendMessage back to the SAME agent. Do NOT spawn a fresh Q/A` | CONTRADICTS §4 Retry-on-FAIL (fresh spawn) AND CLAUDE.md. Must be reconciled -- 67.1 verification greps `! grep -q "back to the SAME agent"` here too. |
| `docs/runbooks/per-step-protocol.md` | 255-256 | Drift-mode fix: `require SendMessage-to-same-agent after any fix` | Same contradiction, second occurrence in the same file. |
| `CLAUDE.md` | "canonical cycle-2 flow" (Harness Protocol section) | fix blockers -> update handoff files -> spawn a FRESH Q/A | THE CANONICAL RULE. qa.md + runbook must be brought into line with this, not vice-versa. |
| `backend/agents/agent_definitions.py` | 396 | `except (json.JSONDecodeError, KeyError, TypeError) as e:` -- `json` never imported | LIVE NameError (67.2's fix). Used here as PROOF the lint gate has real value: ruff F821 catches it exactly (see verbatim below). Also F401 flags unused `typing.Optional` (line 25). |
| `backend/main.py` | 512-547 | `@app.get("/api/health")` returns `{status, service, version, mcp_servers, limits_digest}` | The runtime-smoke target. Backend is LIVE on :8000 now (`curl /api/health` -> HTTP 200). Auth middleware skips `/api/health` (security.md) so smoke needs no token. |
| `pytest.ini` | 1-9 | `[pytest]` + `requires_live` marker only | No `--timeout` addopts; pytest-timeout plugin NOT installed. No `testpaths` set. |
| repo root | -- | NO `pyproject.toml`, `setup.cfg`, `.ruff.toml`, `ruff.toml`, `tox.ini`, `.flake8` | Confirmed: zero Python linter config. ruff + pyflakes both ABSENT from `.venv` and from all `requirements*.txt`. Green field. |

### Internal-half question answers (from the caller)

**Q1 -- Does any hook spawn Q/A or depend on its runtime?** NO.
`grep` of `.claude/settings.json` + `.claude/hooks/` shows the only
agent-type hook is the Stop hook (settings.json:113, its own inline
prompt, NOT the qa.md subagent). No hook spawns the qa.md subagent; no
hook consumes Q/A's JSON. The TaskCompleted hook that once backstopped
Q/A was retired in phase-23.8.2 (per-step-protocol.md:227-229,
251-254; audit R-2 `docs/audits/dev-mas-2026-05-11/04-remediation.md`).
CONSEQUENCE: retiring the 55s cap and neutralizing `stop_hook_active`
in qa.md is SAFE -- nothing bounds Q/A's runtime except the Agent-tool
`maxTurns: 30`, and nothing depends on the auto-PASS return.
`git log -S "55 seconds" -- .claude/agents/qa.md` -> single commit
`b3507436` ("agents: add SendMessage to qa + researcher tool lists"),
i.e. the phrase has been carried forward untouched; no live consumer.

**Q4 -- stop_hook_active semantics.** It is a field Claude Code injects
into Stop / SubagentStop *hook* JSON input to signal a hook is already
re-firing (loop guard); a hook returns `{"ok": true, ...}` to LET Claude
stop. It is NOT normally present in an Agent-tool subagent's context, so
qa.md's clause is near-dead in practice -- but it is a latent auto-PASS:
the day a subagent DOES see the flag, Q/A emits a PASS with no
evaluation. The verdict-neutral fix: `{"ok": false, "verdict": null,
"reason": "loop-prevention exit; no evaluation performed"}` -- ok:false
so it can never be read as a passing verdict, verdict null/absent so it
is not a FAIL either. (Claude Code Stop-hook docs:
https://code.claude.com/docs/en/hooks .)

**Q3 -- What a backend runtime smoke can rely on.** Backend is expected
running on :8000 ("Backend (8000) + Frontend (3000) must always be
running" -- CLAUDE.md Critical Rules); `/api/health` returns 200 live
now and is auth-exempt. venv convention: `source .venv/bin/activate`
first (CLAUDE.md). pytest layout: `backend/tests/` = 991 tests, clean
collect in ~4.9s; repo-root `tests/` = 725 collected + 7 collection
errors (messier, older tree). Scoped runs (single file / `-k`) collect
in ~2-5s. No timeout plugin, so any `--timeout=N` flag ERRORS -- the
smoke must NOT use it. Module-import smoke = `python -c "import
backend.x.y"` inside the venv (cheap, catches import-time NameError /
circular-import / missing-dep).

**Q5 -- Lint tooling.** No config, no installed linter -> green field.
RECOMMEND **ruff** over pyflakes: single static binary, ~10-100x faster,
runs via `uvx ruff` with ZERO venv mutation (ephemeral, mirrors the
project's uvx MCP pattern), and its F-rules are a superset of pyflakes
(F821 undefined-name, F401 unused-import, F811 redefinition, F-series ==
Pyflakes-parity by design). Dev-time only -- per the researcher-rules
precedent (`.claude/rules/research-gate.md` "pdfplumber ... NOT in
backend/requirements.txt ... a research-time convenience, not a project
dependency"), ruff stays OUT of `backend/requirements.txt`. Invocation
for changed-files-only undefined-name checking:
`uvx ruff check --select F821,F401 <changed .py files>`
(add F811 for redefinition). PROVEN LIVE on the known-bad file:

```
$ uvx ruff@latest check --select F821,F401 backend/agents/agent_definitions.py
F401 [*] `typing.Optional` imported but unused
  --> backend/agents/agent_definitions.py:25:20
F821 Undefined name `json`
  --> backend/agents/agent_definitions.py:396:13
Found 2 errors.
$ echo exit=$?   # -> exit=1  (clean file -> exit=0)
```

GATE-IMPLEMENTATION CAVEAT (found live): `ruff ... | tail` MASKS ruff's
non-zero exit (the pipe's last stage wins). The gate MUST capture ruff's
true exit via `${PIPESTATUS[0]}` or run without a pipe, else a failing
lint reads as pass. This is the exact class of silent-green bug the
step is meant to kill, so the gate text should call it out.

---

(external research + recommendations below)
