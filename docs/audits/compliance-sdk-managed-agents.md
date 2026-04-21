# Compliance Audit: Agent SDK + Managed Agents Adoption
**Phase:** 4.15.14
**Date:** 2026-04-18
**Auditor:** researcher agent (merged, confirmation audit)
**Prior audits:** phase-4.11 agent_sdk.md, managed_agents.md; phase-4.12 managed_agents_gaps.md

---

## Live check results

```
grep claude_agent_sdk -r backend/ scripts/ --include='*.py'   → 0
grep @anthropic-ai/claude-agent-sdk frontend/                  → 0
grep claude-agent-sdk backend/requirements.txt                 → 0
grep managed_agents|/v1/agents|/v1/sessions backend/ -r        → 0
grep ant-cli|bedrock-mantle backend/ scripts/ -r               → 0
find docs/ handoff/ -name '*pilot*' -o -name '*managed*'       → 0 files in docs/; phase-4.11/managed_agents.md + phase-4.12/managed_agents_gaps.md only (audit records, not pilot trackers)
```

All zeros. No surprises.

---

## Pattern status table

| # | Pattern | Status | Evidence | Notes |
|---|---|---|---|---|
| 1 | `claude-agent-sdk` / `claude_agent_sdk` import | ABSENT — correct | grep → 0 hits (excl. prior audit files) | Intentional per hybrid strategy |
| 2 | `@anthropic-ai/claude-agent-sdk` in frontend | ABSENT | grep → 0 | Not applicable |
| 3 | `claude-agent-sdk` in `backend/requirements.txt` | ABSENT | grep → 0 | Raw `anthropic` SDK used instead |
| 4 | Managed Agents endpoints (`/v1/agents`, `/v1/sessions`, `/v1/environments`) | ABSENT | grep → 0 | Pilot form not submitted; no access |
| 5 | `ant-cli` / `bedrock-mantle` tooling | ABSENT | grep → 0 | Not installed; not relevant to stack |
| 6 | Managed Agents pilot form submitted | NOT SUBMITTED | No pilot tracker in `docs/` or `handoff/`; prior audit (phase-4.12) said "CONDITIONAL GO — submit form"; no evidence of submission | Blocker: FastMCP public HTTPS, VPC peering, no gcloud/bq/gh in container remain unresolved |
| 7 | `excludeDynamicSections` on Claude Code harness invocation | NOT USED | No flag in `.claude/settings.json` or harness call; phase-4.11 flagged as NICE-TO-HAVE (~10-30% token cost reduction) | Uncaptured SDK lever |
| 8 | OTel / `CLAUDE_CODE_ENABLE_TELEMETRY=1` | NOT USED | No env var set in harness; no collector configured | Uncaptured SDK lever; harness metrics are markdown-only |
| 9 | File checkpointing (`enable_file_checkpointing`) | NOT USED | Manual `.bak-phase4.5` / `.bak-harness-ABCD` files used instead | Uncaptured SDK lever; `rewind_files` would replace hand-rolled backups |
| 10 | Tool search (`ENABLE_TOOL_SEARCH`) | NOT USED | No env var set on Claude Code harness | Flagged NICE-TO-HAVE in phase-4.11 given 28 skill prompts + 4 MCP servers |
| 11 | Managed Agents `define_outcome` (rubric grader) | NOT USED | Research Preview; requires pilot access | Strong structural fit for immutable success criteria — grader is a separate context window, matching anti-self-eval rule |
| 12 | Managed Agents multiagent / `callable_agents` | NOT USED | Research Preview | Direct analogue of dual-evaluator rule (parallel threads, isolated context); adoption requires pilot access |
| 13 | Managed Agents memory stores | NOT USED | Research Preview | Would replace `pyfinagent_data.harness_learning_log` BQ table; not submitted |

---

## Agent SDK: zero adoption summary

Correct and intentional. The hybrid strategy (phase-4.11 cluster E1) holds:

- **Gemini is Layer 1.** 28 agents on Vertex AI. Agent SDK is Claude-only for sampling. Full migration would multiply inference cost ~10x. No.
- **HMAC capability tokens are stronger than SDK permission modes.** `backend/agents/mcp_capabilities.py` binds `(session_id, role, scopes, expires_at)` with HMAC-SHA256 TTL 1800s. SDK `permissionMode` is advisory and inherited by subagents silently — a documented security hazard for a trade-deciding system.
- **Markdown handoff files are load-bearing.** The five-file protocol (contract, experiment_results, evaluator_critique, harness_log, masterplan flip) is human-read by Peder and the evaluator pair. SDK `sessions` jsonl transcripts are machine-parseable but not suited to the `violated_criteria` / PASS/CONDITIONAL/FAIL structured review contract.

**Remaining migration candidate (phase-4.11 cluster E1):** Port `planner_agent.py` + `evaluator_agent.py` (Layer 3, Claude-only, 2-3 day estimate) to Agent SDK for built-in tools, structured-output schema enforcement, and sessions for planner/evaluator continuity. This is additive and reversible. Not yet scheduled.

---

## Managed Agents: pilot not submitted

Phase-4.12 recommended "CONDITIONAL GO — submit access form, scope pilot to one parameter-optimization step." That submission has not occurred. Blockers from phase-4.12 remain unresolved:

1. **FastMCP servers are in-process.** Managed Agents accepts remote streamable-HTTP MCP only. Each of the 4 FastMCP servers (`backtest_server.py`, `data_server.py`, `risk_server.py`, `signals_server.py`) would need a public HTTPS deployment (e.g., Cloud Run shim) before attachment.
2. **FastAPI backend on `:8000` is unreachable from a managed container.** No VPC peering; `networking: limited` with `allowed_hosts` is egress-only; no private inbound.
3. **No `gcloud` / `bq` / `gh` pre-installed in the container.** Ubuntu 22.04 base includes Python 3.12+, Node 20+, git — but not GCP CLI tools. Vaults are MCP-auth-only (not a generic secret manager), so injecting a GCP service-account JSON has no documented path.
4. **Permission policy is too coarse.** Managed Agents exposes only `always_allow` / `always_ask` at toolset granularity. The HMAC capability model in `mcp_capabilities.py` is strictly more expressive and cannot be mapped 1-to-1; fine-grained enforcement would have to move into the host app's `user.tool_confirmation` handler.

A narrow pilot (one parameter-optimization step, no FastAPI call-outs, GitHub MCP + `agent_toolset_20260401` only) remains viable whenever the form is submitted. The pilot would not require resolving blockers 1-3 if scoped to a step with no backend or BQ dependency.

---

## Uncaptured Claude Code SDK levers (unchanged from phase-4.11)

Three flags on the Claude Code harness invocation that remain unused and would provide immediate value without any migration:

- **`excludeDynamicSections`** — moves cwd/git-status out of the system prompt so prompt cache hits across harness cycles. Estimated 10-30% token cost reduction. Zero risk.
- **`CLAUDE_CODE_ENABLE_TELEMETRY=1`** — OTel export for per-tool-call spans. Complements `harness_log.md`; does not replace it. Requires a local or cloud OTel collector.
- **`enable_file_checkpointing=True`** — programmatic `rewind_files(uuid)` for Write/Edit changes. Would replace `.bak-phase4.5` and `.bak-harness-ABCD` hand-rolled backup files and protect `optimizer_best.json` during harness cycles.

---

## Novel findings

None. This is a confirmation audit. All patterns match phase-4.11 and phase-4.12 conclusions. No new SDK pages, no new endpoints, no unexpected imports discovered.

---

## References

- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/phase-4.11/agent_sdk.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/phase-4.11/managed_agents.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/phase-4.12/managed_agents_gaps.md`
- https://code.claude.com/docs/en/agent-sdk/overview
- https://code.claude.com/docs/en/agent-sdk/python
- https://platform.claude.com/docs/en/managed-agents/overview
- https://platform.claude.com/docs/en/managed-agents/quickstart
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_capabilities.py` (HMAC capability model)
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/requirements.txt`
- `/Users/ford/.openclaw/workspace/pyfinagent/CLAUDE.md` (hybrid strategy, dual-evaluator rule)
