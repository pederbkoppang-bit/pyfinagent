# live_check 75.20.1 — verbatim probe + enforcement evidence (2026-07-24, Claude Code 2.1.218)

## 1. Injection reproduction (criterion 1 — BEFORE this step's hook)

Earlier this session (before the hook existed), a fresh Agent-tool qa probe
self-disclosed Write+Edit present on the live roster (recorded verbatim in
handoff/archive/misc/live_check_75.20.md §7a). Version: Claude Code 2.1.218
(`claude --version`). Root cause: qa.md:25 `memory: project` → documented upstream
auto-enable (sub-agents doc verbatim quote in the brief + experiment_results).

## 2. Hook smoke matrix (deterministic, REAL hook via stdin JSON — verbatim)

```
qa Write outside       -> rc=2 + "qa-write-guard: BLOCKED -- the qa evaluator is read-only on file contents"
qa Write inside memory -> rc=0
traversal escape (.claude/agent-memory/qa/../../../etc/x) -> rc=2
Main (no agent_type)   -> rc=0
other agent_type       -> rc=0
malformed payload      -> rc=0 (fail-open)
```

## 3. LIVE behavioral probe (wf_90a3a61d-4a4, agentType:'qa', haiku) — criterion 1 + 3 block arm

Per-tool attempts (verbatim from the structured return):

- **Write** → attempted, FAILED: `PreToolUse:Write hook error [qa-write-guard]: BLOCKED -- the qa evaluator is read-only on file contents (Write/Edit allowed only under .claude/agent-memory/qa/; see per-step-protocol.md section 4 and phase-75.20.1)`
- **Edit** → attempted, failed: file never existed (Write was blocked upstream); the
  hook's Edit branch is deterministically proven by t2 (rc=2) — disclosed precisely,
  not over-claimed.
- **Glob** → attempted, failed: "tool not available in runtime surface" — the Glob/Grep
  drop is REAL at execution level on the Workflow path (not a self-report artifact).
- **Grep** → attempted, failed: "tool not available in runtime surface"
- **Read** → attempted, SUCCEEDED (read the hook's own header lines)

**THE HOOK BLOCKED A REAL qa-AGENT WRITE, LIVE, THIS SESSION** — criterion 3's block
arm proven without a restart gate.

## 4. Guard log — runtime field-shape confirmation on 2.1.218 (verbatim tail)

```
LOG {"ts": "2026-07-24T11:59:29.148420+00:00", "agent_type": "qa", "tool_name": "Write", "file_path": "/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/525d6185-0dc7-45ec-aacf-da3233ef9c70/scratchpad/qa_probe_75_20_1.md"}
```

That line was populated by the Claude Code runtime itself (the probe agent's real
Write call), NOT by a test fixture — `agent_type: "qa"` is confirmed present in the
PreToolUse input exactly as the hooks doc states. (Earlier same-log lines 11:57-11:59
are the manual smoke + pytest fixtures, distinguishable by their /tmp fixture paths.)

## 5. Criterion 3 Main-allow arm — measured, honestly stated

Main's Write/Edit calls after registration ALL SUCCEEDED (this GENERATE's runbook
edit, test-file writes, and this file were authored with the hook registered; none
were blocked). Mechanism measured from the guard log: the hook fires immediately for
NEWLY-SPAWNED subagents (the probe's line above) but produced NO lines for Main's own
calls — the main session's hook set snapshots at session start, so the hook simply is
not yet invoked on Main's calls this session. Either way "Main is not blocked" holds
today; the allow BRANCH (empty agent_type → exit 0) is deterministically proven by t4
and the smoke matrix. NEXT-SESSION EXPECTATION: Main's Write/Edit calls will start
appearing in qa_write_guard.log with `agent_type:""` and rc continue 0 — a one-glance
confirmation, noted for the next cycle, NOT a gate on this step (both criterion-3
arms are recorded above).

## 6. Mutation matrix — 6 mutations, 6 killed, 0 survivors (verbatim results)

| # | Mutation (applied to the REAL file, executed) | Killed by |
|---|---|---|
| M1 | qa condition removed (hook guards everyone incl. Main) | `test_t4_main_no_agent_type_allowed` — 1 failed |
| M2 | memory-dir exception removed (`if True:`) | `test_t3_qa_write_inside_memory_allowed` — 1 failed |
| M3 | block → allow (exit 2 → exit 0) | `test_t1_qa_write_outside_memory_blocks` — 1 failed |
| M4 | hook unregistered from settings.json | `test_t7_hook_registered_in_settings` — 1 failed |
| M5 | **FIXTURE**: t1 payload field renamed `agent_type`→`agentType` | t1 — 1 failed (fail-open on a wrong-shape fixture proves the fixtures speak the real schema) |
| M6 | **STUB**: `_run_hook` neutered to constant rc=2 (no subprocess) | t3 + t4 — 2 failed |

Pre/post SHA-256 identical for all three mutated files (`shasum -c`: OK OK OK); full
suite re-run post-restore: `11 passed`.

## 7. git diff --stat (step-scoped)

```
 .claude/hooks/qa-write-guard.sh                       (NEW)
 .claude/settings.json                                 (+1 PreToolUse entry)
 .claude/workflows/probe-qa-tool-surface.js            (NEW)
 docs/runbooks/per-step-protocol.md                    (+19 lines, section 4)
 backend/tests/test_phase_75_20_1_qa_write_injection.py (NEW, 11 tests)
```

## 8. POST-CLOSE upgrade (2026-07-24 ~15:00 UTC): Main-allow arm now proven LIVE

handoff/logs/qa_write_guard.log began carrying agent_type:"" lines for MAIN's own
Write/Edit calls later in the session (e.g. 14:58:56 Edit run_nightly.sh, 15:00:48
Write test_phase_76_9_2_max_bridge.py, 15:42:36 Write experiment_results.md — all
ALLOWED, all succeeded). The main-session hook binding happened mid-session after
all (later than the spawn-time snapshot suggested — hooks evidently refresh on some
settings/goal reload). Criterion 3's Main-allow arm is therefore now LIVE-proven
first-hand, upgrading section 5's deterministic-only proof. The next-session
one-glance check is already satisfied.
