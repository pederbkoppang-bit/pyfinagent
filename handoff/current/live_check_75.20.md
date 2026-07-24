# live_check_75.20 — Q/A live-UI gate enforceability + read-only primary path

All output verbatim from live runs 2026-07-24 (Main session, repo root
`/Users/ford/.openclaw/workspace/pyfinagent`). Scratchpad artifacts referenced:
`mutation_matrix_75_20.txt`, `concurrency_demo_75_20.txt`.

## 1. Immutable verification command — exit 0

```
$ python3 -c "import json; qa=open('.claude/agents/qa.md').read(); assert 'mcp__playwright__browser_navigate' in qa and 'mcp__playwright__browser_snapshot' in qa, ...; assert 'browser_run_code_unsafe' not in qa.split('tools:')[1].split(chr(10))[0], ...; m=json.load(open('.mcp.json')); pw=m['mcpServers']['playwright']; assert '--isolated' in pw['args'] or any('user-data-dir' in str(a) and 'profile' not in str(a) for a in pw['args']), ...; s=json.load(open('.claude/settings.json')); deny=' '.join(str(d) for d in s['permissions'].get('deny',[])); assert 'browser_run_code_unsafe' in deny and 'browser_evaluate' in deny, ..."
immutable-verification-exit=0
```

**R11 DISCLOSURE (research-gate finding, empirically proven):** the command's assert #3
is VACUOUS — it also passes on the OLD hazardous args, because the flag TOKEN
`--user-data-dir` satisfies `'user-data-dir' in a and 'profile' not in a`:

```
$ python3 -c "... args = git show HEAD:.mcp.json playwright args ... print(vac)"
R11 proof -- immutable assert #3 on the OLD (hazardous) args evaluates: True
```

The command is immutable and stays; C4's real evidence is §4 (the demonstration) and the
NON-vacuous test `test_playwright_server_is_isolated_and_sheds_the_profile_pin`
(`'--isolated' in args AND no arg contains 'user-data-dir'`), whose mutations N3a AND N3b
both kill (§5) — N3b (hazard restored ALONGSIDE --isolated) passes the immutable assert
but fails the test.

## 2. Step test suite — exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_20_qa_browser_grant.py -q
.............                                                            [100%]
13 passed in 0.02s
$ uvx ruff check backend/tests/test_phase_75_20_qa_browser_grant.py
All checks passed!
```

## 3. Tool-surface probes — BEFORE/AFTER, proving what binds (C3)

### 3a. BEFORE — the primary path as shipped (Workflow agentType 'general-purpose', probe wf_9277ada4-390)

```json
{"tools_listed":["Artifact","Bash","Edit","Read","ReportFindings","Skill","ToolSearch","Write",
  "mcp__pyfinagent-data__ping","mcp__pyfinagent-risk__evaluate_candidate","mcp__pyfinagent-risk__factor_exposure",
  "mcp__pyfinagent-risk__kill_switch","mcp__pyfinagent-risk__pbo_check","mcp__pyfinagent-risk__ping",
  "mcp__pyfinagent-risk__portfolio_cvar","StructuredOutput"],
 "has_edit":true,"has_write":true,"deferred_tools_mentioned":true,
 "notes":"Seven MCP tools loaded ...; hundreds more deferred via ToolSearch including ... playwright ..."}
```

### 3b. AFTER-mechanism — agentType 'qa' on the same Workflow path (probe wf_9277ada4-390)

```json
{"tools_listed":["Read","Bash","SendMessage","Write","Edit","StructuredOutput"],
 "has_edit":true,"has_write":true,"deferred_tools_mentioned":true}
```

The switch REMOVES: Artifact, Skill, ReportFindings, all 7 loaded MCP tools, and the
full-MCP deferral surface. **DISCLOSED RESIDUAL:** Write/Edit are injected by the loader
past the frontmatter allowlist (`tools:` line has excluded them for ≥5 commits — verified
via `git show 207da6b4/3a7942cf/515c35e1/0e003e6e/de7e8270:.claude/agents/qa.md | grep tools:`),
and a `disallowedTools` option is SILENTLY IGNORED (probe wf_78b46633-fdd: no error,
Write/Edit present). The residual is covered by qa.md prose + Main's post-verdict
`git status` check and is queued as masterplan step **75.20.1** (research-gated).

### 3c. Deny rules bind LIVE and session-wide (C2)

The instant `.claude/settings.json` gained the two deny entries, the harness stripped both
tools from this session's own deferred surface (system notice, verbatim):

```
The following deferred tools are no longer available (their MCP server disconnected).
Do not search for them -- ToolSearch will return no match:
mcp__playwright__browser_evaluate
mcp__playwright__browser_run_code_unsafe
```

Per the official permissions doc (research brief): deny rules are deny-FIRST and are
honored even under `defaultMode: bypassPermissions`, and bind subagents and
Workflow-spawned agents — defense in depth for the primary path.

## 4. Concurrency demonstration (C4) — vendor citation + live two-client run

Vendor: @playwright/mcp documents that a persistent profile serves one browser instance
at a time; the error text itself names the fix. Live demonstration
(`concurrency_demo_75_20.py`, two real stdio MCP clients driven via JSON-RPC
initialize → tools/call browser_navigate):

```
===== SCENARIO: BASELINE shared --user-data-dir (pre-75.20 shape) =====
client1 navigate ok=True
client2 navigate ok=False  detail: ### Error
Error: Browser is already in use for /tmp/pw-mcp-demo-shared-profile, use --isolated to run multiple instances of the same browser

===== SCENARIO: FIXED --isolated (post-75.20 shape) =====
client1 navigate ok=True
client2 navigate ok=True

===== VERDICT =====
baseline shared-profile: client1=True client2=False  (expected: second contends)
isolated:                client1=True client2=True  (expected: both succeed)
CONTENTION DEMONSTRATED: True; ISOLATION FIXES IT: True
demo-exit=0
```

Note: the connected stdio server does not respawn on a `.mcp.json` edit — the new args
bind on /mcp reconnect or next session (playwright is `alwaysLoad:false`, so next session
is the normal path).

## 5. Mutation matrix (qa.md §4c) — 7 mutations, 7 killed, 0 survivors

Runner: `run_mutations_75_20.py`; full verbatim log `mutation_matrix_75_20.txt`.

```
SUMMARY: 7 mutations, 7 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

| # | Mutation (applied to the real config/test, executed) | Killed by |
|---|---|---|
| N1 | browser_click ADDED to the qa.md grant line | `test_tools_line_grants_no_mutation_tool[browser_click]` + `test_browser_grant_is_within_the_allowed_superset` |
| N2 | browser_evaluate deny entry REMOVED | `test_settings_denies_the_rce_tools_exactly` |
| N3a | `--isolated` REMOVED | `test_playwright_server_is_isolated_and_sheds_the_profile_pin` |
| N3b | fixed `--user-data-dir` RESTORED alongside `--isolated` (the immutable assert would PASS this) | same test — the non-vacuous guard catches what R11 cannot |
| N4 | qa-verdict.js reverted to `general-purpose` | `test_qa_verdict_launches_the_restricted_agent_type` |
| N5 | `select:` instruction stripped from §1c | `test_1c_instructs_the_deterministic_select_form` |
| N6 | **STUB/harness**: suite's `_tools_line` extractor neutered (returns "") | `test_tools_line_grants_the_read_only_browser_subset` |

## 6. git diff --stat (working tree at capture time, this step's edits)

```
 .claude/agents/qa.md            (tools line + §1c amendment)
 .claude/settings.json           (+2 deny entries)
 .mcp.json                       (--user-data-dir pin -> --isolated)
 .claude/workflows/qa-verdict.js (agentType general-purpose -> qa + rationale comment)
 backend/tests/test_phase_75_20_qa_browser_grant.py (new, 13 tests)
 .claude/masterplan.json         (75.20 pending->in_progress; +75.20.1 queued step)
```

## 7. Roster-live confirmation after restart — NEXT-SESSION OWED (structural)

The Agent-tool roster snapshots at session start, so the new qa.md tools grant CANNOT be
confirmed live from the authoring session. NEXT SESSION (after operator review of the
qa.md diff): run `scripts/qa/verify_qa_roster_live.sh`, then spawn a fresh Agent-tool qa
probe asking it to self-disclose its tools list (expect the browser subset present,
loaded schemas or ToolSearch-resolvable), and APPEND the verbatim result here. Until that
confirmation lands, 75.20's status stays `in_progress` (flip HELD — 75.18 precedent +
separation of duties).
