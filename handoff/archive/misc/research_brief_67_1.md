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

## External research

### Read in full (7; clears the moderate >=5 floor)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents | 2026-07-09 | Official (Anthropic) | WebFetch full | "We recommend choosing **deterministic graders where possible**, LLM graders where necessary." Code-based = "Fast, Cheap, Objective, Reproducible, Easy to debug." "does the code run and do the tests pass? ... SWE-bench Verified and Terminal-Bench." "we do not take eval scores at face value until someone digs into the details." |
| 2 | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 2026-07-09 | Official (Anthropic, 2026 -- NEWER than the 2025 harness-design piece we cite) | WebFetch full | Deterministic baseline check: "read `init.sh` ... run a basic test on the development server to catch any undocumented bugs" (= the runtime-smoke gate). "It is unacceptable to remove or edit tests." Progress file = handoff artifacts. |
| 3 | https://docs.astral.sh/ruff/rules/undefined-name/ | 2026-07-09 | Official (Astral/Ruff) | WebFetch full | F821 "Checks for uses of undefined names ... **Derived from the Pyflakes linter** ... An undefined name is likely to raise `NameError` at runtime." = the exact bug class at agent_definitions.py:396. |
| 4 | https://dev.to/saurav_bhattacharya/deterministic-checks-vs-model-as-judge-a-tiered-approach-to-agent-evaluation-3217 | 2026-07-09 | Practitioner (2026) | WebFetch full | Three-tier cascade: Tier-1 deterministic (~60% of failures, <1ms), Tier-2 heuristic (~20%, no LLM), Tier-3 model-judge (~20%, last). "Only now -- after deterministic and heuristic checks pass -- do you invoke an LLM judge." Tiered = "10x cheaper" ($5-15/day vs $50-150). |
| 5 | https://www.emergentmind.com/topics/swe-bench-verified-47773414-8319-4e96-b867-a5a13ef278a7 | 2026-07-09 | Reference | WebFetch full | Execution-based: "apply the proposed patch, and confirm all tests transition (FAIL->PASS and no regressions in PASS->PASS)." Verified removed cases where "12.5%-22% of successful patches were logically wrong ... unflagged due to insufficient tests." |
| 6 | https://qaskills.sh/blog/swe-bench-explained-guide-2026 | 2026-07-09 | Practitioner (2026) | WebFetch full | "a task counts as resolved only if the failing tests now pass and the passing tests still pass." Docker sandbox "fixed a major early source of noise, where flaky local dependency setups produced inconsistent scores." |
| 7 | https://arxiv.org/html/2507.21504v1 (Evaluation and Benchmarking of LLM Agents: A Survey) | 2026-07-09 | Peer-review-adjacent (arXiv, 2025) | WebFetch full | "code-based method is the most deterministic and objective approach"; cost "estimated based on ... input and output tokens." Notably SILENT on wall-clock vs step budgets -- a literature gap (see below). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | Official | The canonical piece we already cite project-wide; superseded-context by #2. |
| https://openai.com/index/introducing-swe-bench-verified/ | Official | HTTP 403 (bot-blocked); covered by #5/#6. |
| https://realpython.com/ruff-python/ | Practitioner | HTTP 403; ruff facts covered by #3. |
| https://arxiv.org/pdf/2605.12270 (Characterizing Failure Modes of LLMs on GitHub issues) | arXiv 2026 | Snippet: eval inflation up to 6.2 pts, ~7.8% from un-run tests. |
| https://arxiv.org/html/2606.22737v2 (GroundEval: deterministic replacement for LLM-as-judge) | arXiv 2026 | Recency hit; corroborates deterministic-first. |
| https://waxell.ai/blog/ai-agent-output-validation-production | Practitioner 2026 | "Layer 1 = deterministic pre-emission checks ... run on 100% of outputs." |
| https://futureagi.com/blog/evaluating-llm-self-reflection-loops-2026/ | Practitioner 2026 | Reflection "pays for verification twice" when a separate judge stack exists. |
| https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/ | Press | Summary of the 3-agent harness. |
| https://openreview.net/pdf?id=VTF8yNQM66 | Peer-reviewed (ICLR 2024) | SWE-bench founding paper; canonical prior-art (year-less hit). |
| https://cogniswitch.ai/guides/llm-as-a-judge-vs-deterministic-verification | Practitioner | Deterministic-vs-judge overview. |

### Recency scan (2024-2026)

Performed. Query variants run per topic: current-frontier (`...2026`),
last-2-year (`...2025 2026`), and year-less canonical (`deterministic gate
before LLM judge`, `undefined name linter`). **New findings that
complement/supersede the canonical sources:** (1) Anthropic published a
NEWER "Effective harnesses for long-running agents" (2026) that adds the
`init.sh` + dev-server smoke as an explicit deterministic baseline check --
direct support for the runtime-smoke gate, superseding the 2025 piece we
cite. (2) The 2026 tiered/cascade practitioner consensus quantifies the
"deterministic-first" ordering (Tier-1 catches ~60% of failures at <1ms;
10x cheaper) -- this is NEW quantitative backing for retiring a wall-clock
cap in favor of a cascade. (3) 2026 SWE-bench work shows eval inflation of
up to 6.2 pts with ~7.8% attributable to tests that never actually ran --
direct evidence that "the code must actually execute" is load-bearing, not
ceremonial. No finding contradicts the step's design.

### Key findings (per-claim)

1. **Deterministic graders first, LLM judge last is the documented
   Anthropic position.** "We recommend choosing deterministic graders where
   possible, LLM graders where necessary." (Source: Anthropic, Demystifying
   evals, #1) -- validates criterion 2's deterministic backend gate and the
   tiered ordering, and validates KEEPING qa.md's existing "LLM judgment
   (last resort)" section (§4, lines 159-171).
2. **Execution/"does it run" is the coding-agent ground truth.** "does the
   code run and do the tests pass?" (#1); SWE-bench "resolved only if the
   failing tests now pass and the passing tests still pass" (#6). ~7.8% of
   eval inflation traced to un-run tests (snippet, 2605.12270). -> the
   runtime-smoke gate (import + exercise the live path) is the correct
   mechanism; a green that never executed the changed module is a false green.
3. **F821 == the NameError bug class, from Pyflakes.** (#3) -> ruff F821 is
   the right guard; proven live catching agent_definitions.py:396.
4. **Bound WORK, not wall-clock; cascade so expensive checks run only on
   what cheap checks couldn't decide.** Tiered: Tier-1 <1ms/~60%, judge last
   (#4). The agent-eval survey is silent on wall-clock caps (#7) -- the
   literature bounds compute/turns, not seconds. -> replace the flat 55s cap
   with a tiered WORK budget anchored on `maxTurns: 30`.
5. **Don't rubber-stamp; read transcripts.** "we do not take eval scores at
   face value" (#1) -> keep qa.md's anti-rubber-stamp/mutation-resistance
   leg; the stop_hook auto-PASS is the antithesis of this and must go.

### Consensus vs debate

Consensus (strong, cross-domain -- Anthropic official + 3 practitioner 2026
sources + arXiv survey): deterministic/execution checks run FIRST and are
cheaper/more reliable than LLM judgment; tests must actually execute. Debate:
none found against deterministic-first. The only tension is that pure static
gates "fail" for subjective/semantic correctness (waxell 2026) -- which is
why the tiered design KEEPS the LLM-judgment layer rather than replacing it.
That matches 67.1's intent (add deterministic gates, don't remove judgment).

### Pitfalls (from literature + live probes)

- **Un-run tests / false green.** A smoke that doesn't actually import+exercise
  the changed path proves nothing (SWE-bench inflation, #5/#6). Capture verbatim output.
- **Pipe masks exit code (live-found).** `ruff ... | tail` returns tail's exit
  (0), hiding ruff's non-zero. Use `${PIPESTATUS[0]}` or no pipe.
- **Broken example commands already in qa.md.** Line 92 `pytest tests/ -v
  --timeout=30`: `--timeout` ERRORS (no pytest-timeout installed) and `tests/`
  (root) has 7 collection errors. Fix or drop when rewriting the block.
- **Over-prescription.** The goal explicitly wants NO over-prescription; the
  gate should name the bug CLASS and the invocation, not enumerate every flag.

---

## Application to pyfinagent -- concrete recommendations

### R1. Lint tool + invocation
Use **ruff via uvx** (zero venv mutation, mirrors the project's uvx/MCP
pattern; F-rules are Pyflakes-parity). Do NOT add to `backend/requirements.txt`
(research-gate.md dev-dep precedent). Changed-files-only undefined-name check:
```
uvx ruff check --select F821,F401,F811 <changed .py files>; echo "ruff exit=$?"
```
Non-zero = FAIL. NEVER pipe ruff to `tail`/`head` (masks exit). Proven live:
F821 catches `Undefined name json` at agent_definitions.py:396 (exit=1);
clean file exits 0.

### R2. New qa.md deterministic backend gate (sketch -- concise, mirror of §1b ESLint)
```
### 1a. Backend Python lint + runtime smoke (REQUIRED if the diff touches backend/**)
No Python linter ships in .venv; run ruff ephemerally via uvx.
(a) Undefined-name class over CHANGED files (F821 NameError, F401 unused, F811 redef).
    Capture ruff's REAL exit -- never pipe to tail:
      uvx ruff check --select F821,F401,F811 <changed .py>; echo "exit=$?"
    Non-zero exit = FAIL (name file:line).
(b) Runtime smoke: import each changed module in the venv, and when the diff
    touches a live API/service path, exercise it and capture output:
      source .venv/bin/activate && python -c "import backend.<changed.module>"
      curl -s -m5 http://localhost:8000/api/health   # or the changed endpoint/command
    Import error or non-200 on the exercised path = FAIL.
```

### R3. Tiered verification budget (replaces "Maximum runtime: 55 seconds")
```
- Verification budget (tiered; bound WORK, not wall-clock). Deterministic
  syntax/lint/existence checks first, fast (<60s). Scoped tests -- the affected
  backend/tests/* files or -k <pattern>, NOT the full 991-test suite -- up to
  ~5 min. Runtime smoke (import + exercised path) up to ~2 min. LLM judgment
  last, only on what the deterministic gates couldn't decide. The real bound is
  maxTurns: 30 (observed 20-26 tool uses/eval), not a wall-clock cap.
```

### R4. stop_hook_active -> verdict-neutral (replaces the ok:true clause)
```
- If stop_hook_active is true in your context, return
  {"ok": false, "verdict": null, "reason": "loop-prevention exit; no evaluation performed"}.
  NEVER return ok:true here -- in this schema ok:true is a PASS, so an
  auto-ok:true is an unaudited auto-PASS backdoor.
```
(Leave settings.json Stop/SubagentStop hooks untouched -- there `ok:true`
correctly means "allow Claude to stop"; the 67.1 verification command
scopes the grep to qa.md only, which is correct.)

### R5. Reconciled CONDITIONAL/FAIL recovery rule (identical text in qa.md + per-step-protocol.md, aligned to CLAUDE.md)
```
- CONDITIONAL/FAIL recovery -- fresh spawn on CHANGED evidence. If the first
  Q/A returned CONDITIONAL or FAIL, the orchestrator fixes the named blockers,
  UPDATES the handoff evidence (experiment_results.md + evaluator_critique.md
  follow-up + any flagged code), THEN spawns a FRESH Q/A that reads the updated
  files. The new verdict reflects the fix, not a second opinion on the same
  evidence. FORBIDDEN: respawning on UNCHANGED evidence hoping for PASS
  (verdict-shopping). Dormant subagents do not wake on SendMessage (Cycle-76);
  fresh-spawn-on-changed-evidence is both the documented Anthropic pattern and
  the empirically reliable one.
```

### R6. GOTCHAS -- verification command is necessary but NOT sufficient (watermelon risk)
The immutable `verification.command` uses exact-string greps that miss
several real references. Main must fix these for HONEST criterion compliance
even though the grep stays green:
- **Second "55" reference:** qa.md:120 "well within the **55s** Q/A budget"
  says `55s`, not `55 seconds` -> the `grep -q "55 seconds"` will NOT catch it,
  but criterion 1 ("no global 55-second cap") requires removing it too.
- **Second SendMessage reference:** per-step-protocol.md:255-256 "require
  **SendMessage-to-same-agent** after any fix" -> not the literal string
  "back to the SAME agent", so the grep misses it; criterion 4 ("no artifact
  still mandates SendMessage-to-the-SAME-agent") requires fixing it.
- **CLAUDE.md** already states the canonical fresh-respawn flow correctly; no
  change needed there beyond confirming consistency (criterion 4 lists it as a
  reconciliation target, not necessarily an edit target).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (7 read + 10 snippet-only = 17)
- [x] Recency scan (2024-2026) performed + reported (3 new findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md, settings.json,
      per-step-protocol.md, CLAUDE.md, backend/main.py, pytest.ini, agent_definitions.py)
- [x] Contradictions/consensus noted (deterministic-first consensus; static-gates-alone tension)
- [x] Claims cited per-claim
- [x] Three-query-variant discipline visible (current / last-2-year / year-less per topic)

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
