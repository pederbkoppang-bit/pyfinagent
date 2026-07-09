---
name: cc-runtime-adoption-67-5
description: Claude Code v2.1.196+ MCP approval topology (where approvals actually live), fallbackModel trigger exclusions, SessionStart fail-open contract — phase-67.5 research
metadata:
  type: project
---

Phase-67.5 research (2026-07-09/10, local Claude Code 2.1.205):

- **MCP approvals live in FOUR places, none of them .mcp.json**: per-project
  `~/.claude.json` -> `projects[<path>].enabledMcpjsonServers` (interactive prompt
  writes here), user `~/.claude/settings.json`, `--settings`, and an UNTRACKED
  `.claude/settings.local.json`. Since v2.1.196 a repo-committed
  `.claude/settings.json` approval is distrusted in untrusted folders ("a cloned
  repository can't approve its own servers"). `disabledMcpjsonServers` in ANY file
  rejects. Diagnostic surface: `claude mcp list` shows `⏸ Pending approval` —
  token-free, works headless.
- **Empirical (pre-67.5 fix)**: ALL pyfinagent .mcp.json servers were Pending
  approval despite hasTrustDialogAccepted:true — headless away sessions had ZERO
  stdio servers. `.mcp.json` has 8 servers (not 9); settings.local.json had a stale
  `"slack"` enable entry (no such server) and an intentional `"alpaca"` reject
  (keep — defense-in-depth over the settings.json order-tool denies).
- **fallbackModel** (v2.1.166): array, max 3 after dedup, NO cross-file merge
  (highest-precedence file supplies the whole chain — settings.local.json would
  SHADOW a project chain). Triggers ONLY on overloaded/unavailable/non-retryable
  server errors; "Authentication, billing, rate-limit, request-size, and transport
  errors never trigger a switch"; switch lasts the current turn. Rate-limit subagent
  cutoffs are instead covered by v2.1.199 partial-work retention. Separate from the
  Fable cyber/bio content-classifier fallback (headless: flagged request = refusal).
- **SessionStart**: cannot block startup (exit 2 = notice only); plain stdout AND
  `hookSpecificOutput.additionalContext` both become context (double fail-open
  pattern: JSON emit `|| printf` plaintext); stdin `model` field may be ABSENT
  after /clear.
- **Nesting trap**: in subagent frontmatter, `Agent(type-list)` parentheses are
  IGNORED — plain `Agent` in tools is what enables self-nesting (5 levels,
  v2.1.172); type allowlists only work for `claude --agent` main-thread.

**Why:** cost a full research cycle to establish; the approval topology is invisible
from repo files alone. **How to apply:** any future MCP-attach question — check
`claude mcp list` + `~/.claude.json` project entry FIRST; any fallback/retry claim —
verify against the trigger exclusion list before attributing a rescue to
fallbackModel. See [[research-gate-discipline]].
