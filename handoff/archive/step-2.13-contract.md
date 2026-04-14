# Step 2.13: Claude Code Configuration Audit — Sprint Contract

## Hypothesis
Auditing all Claude Code configuration files against official documentation ensures the MAS harness system works correctly for autonomous operation. Misconfigurations (wrong field names, missing timeouts, invalid schemas) silently break hooks and agent spawning.

## Success Criteria (Research-Backed)

1. **All hooks valid schema** — per https://code.claude.com/docs/en/hooks
   - No invalid fields (e.g., `agent` field in agent-type hooks)
   - All agent hooks have explicit `timeout`
   - Stop hook returns `{ok: bool}` not `{decision: string}`
   - All command hooks reference existing executable scripts

2. **All agents have required frontmatter** — per https://code.claude.com/docs/en/sub-agents
   - `name`, `description`, `tools`, `model` present
   - `maxTurns` set to prevent infinite loops
   - `memory: project` for cross-session learning

3. **MCP config exists and valid** — per https://code.claude.com/docs/en/mcp
   - `.mcp.json` at project root (project-scoped, version controlled)
   - Slack server configured with `${VAR}` env expansion

4. **Permissions allow autonomous operation** — per settings.json docs
   - `Agent(researcher)`, `Agent(qa-evaluator)`, `Agent(harness-verifier)` allowed

5. **4-tier memory wired** — per CoALA (Princeton 2023)
   - Tier 2 Episodic: `.claude/context/sessions/*.md`
   - Tier 3 Semantic: `.claude/context/*.md`
   - Tier 4 Procedural: CLAUDE.md + agents + skills + rules

6. **Remote trigger prompt has full MAS protocol**
   - 4-tier memory instructions, subagent spawn templates, 12 critical rules

## Fail Conditions
- Any hook with invalid schema that would crash at runtime
- Any agent missing `name` or `description`
- Missing `.mcp.json` (remote agent has no Slack)
- Remote trigger missing `Agent` in allowed_tools
