---
step: phase-4.14.6
title: Research Brief — bypassPermissions -> acceptEdits + sandbox.enabled
tier: simple
date: 2026-04-18
gate_passed: true
---

## External

### 1. Valid `permissions.defaultMode` values and `acceptEdits` suitability

Official docs (https://code.claude.com/docs/en/permissions, accessed 2026-04-18) list six valid values:

| Value | Behaviour |
|---|---|
| `default` | Prompts on first use of each tool |
| `acceptEdits` | Auto-accepts file edits + common filesystem commands (`mkdir`, `touch`, `mv`, `cp`) within the working dir / `additionalDirectories` |
| `plan` | Read-only; cannot modify files or execute commands |
| `auto` | Auto-approves with background safety-classifier checks (research preview) |
| `dontAsk` | Auto-denies unless tool is pre-approved in allow list |
| `bypassPermissions` | Skips all prompts. Docs: "Only use in isolated environments like containers or VMs where Claude Code cannot cause damage." |

`acceptEdits` is the correct replacement for an autonomous-harness setup running on a developer Mac. The existing explicit `allow` list (`Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Agent`, `WebSearch`, `WebFetch`) covers every tool the harness needs — those allow rules become active (and meaningful) once `bypassPermissions` is removed. The deny list (`Bash(rm -rf *)`, `Bash(git push --force *)`, `Bash(git reset --hard *)`, Alpaca/BQ write tools) also becomes load-bearing only after `bypassPermissions` is gone.

### 2. `sandbox` key shape

Docs (https://code.claude.com/docs/en/sandboxing, accessed 2026-04-18) show the minimal enable form:

```json
{
  "sandbox": {
    "enabled": true
  }
}
```

`sandbox.enabled` is a boolean. The key accepts an object with optional subkeys:
- `sandbox.filesystem.allowWrite` / `denyWrite` / `denyRead` / `allowRead` — arrays of paths
- `sandbox.network.allowedDomains` / `deniedDomains`
- `sandbox.network.httpProxyPort` / `socksProxyPort`
- `sandbox.failIfUnavailable` — boolean; makes missing deps a hard failure
- `sandbox.allowUnsandboxedCommands` — boolean; disables the escape-hatch
- `sandbox.autoAllowBashIfSandboxed` — boolean (default `true`); sandboxed Bash auto-approved

None of the subkeys are required. `{ "sandbox": { "enabled": true } }` is a complete, valid block.

### 3. macOS Seatbelt activation

Docs confirm: "On macOS, sandboxing works out of the box using the built-in Seatbelt framework. No packages needed." `sandbox.enabled: true` is sufficient to activate Seatbelt on macOS — no additional required keys.

The permissions-page caveat: "Effective sandboxing requires both filesystem and network isolation. Without network isolation, a compromised agent could exfiltrate sensitive files." For the harness the existing `deny` list (`mcp__bigquery__execute_sql`, Alpaca write tools) plus `WebFetch` + `WebSearch` allow rules provide the primary network guard; the sandbox adds OS-level enforcement on top.

---

## Internal

### 1. Current `settings.json` (permissions + sandbox)

File: `/Users/ford/.openclaw/workspace/pyfinagent/.claude/settings.json` (lines 87–113)

```json
"permissions": {
  "defaultMode": "bypassPermissions",
  "allow": [
    "Bash", "Read", "Write",
    "Write(.claude/context/sessions/**)",
    "Edit", "Glob", "Grep", "Agent",
    "WebSearch", "WebFetch"
  ],
  "deny": [
    "mcp__alpaca__place_order",
    "mcp__alpaca__cancel_order",
    "mcp__alpaca__replace_order",
    "mcp__alpaca__close_position",
    "mcp__alpaca__close_all_positions",
    "mcp__bigquery__execute_sql",
    "Bash(rm -rf *)",
    "Bash(git push --force *)",
    "Bash(git push -f *)",
    "Bash(git reset --hard *)"
  ]
}
```

No `sandbox` key — `jq '.sandbox' .claude/settings.json` returns `null` (confirmed by prior audit docs/audits/compliance-claude-code-core.md line 45).

### 2. Other `bypassPermissions` references

The only file in the repo requiring a coordinated change is `settings.json` itself (line 88). All other occurrences are read-only records:

- `.claude/settings.json.bak-harness-ABCD` line 70 — stale backup, no runtime effect; can be left or deleted
- `.claude/masterplan.json` lines 3627/3631/3633 — immutable verification-criteria text; must NOT be edited
- `handoff/harness_log.md`, `handoff/current/phase-4.15.12-*`, `docs/audits/compliance-*.md`, `CHANGELOG.md`, `handoff/audit/phase-4.10/`, `handoff/audit/phase-4.11/` — documentation and audit records; no change needed

No `.claude/settings.local.json` exists in the working tree (the bak file is a copy of `settings.json`, not `settings.local.json`). No hook scripts reference `bypassPermissions`.

---

## Recommended implementation shape

- In `.claude/settings.json`, change `"defaultMode": "bypassPermissions"` to `"defaultMode": "acceptEdits"`. The existing `allow` array already whitelists every tool the harness uses; no additional allow entries are needed.
- Immediately below the `permissions` block (or as a sibling key at the root level), add `"sandbox": { "enabled": true }`. No subkeys are required for the baseline; Seatbelt activates automatically on macOS.
- Leave the `deny` array unchanged — it becomes the primary runtime guard once `bypassPermissions` is gone, and it already covers destructive Bash commands and all Alpaca/BQ write tools.

---

## Sources

- [Configure permissions — Claude Code Docs](https://code.claude.com/docs/en/permissions) (accessed 2026-04-18)
- [Sandboxing — Claude Code Docs](https://code.claude.com/docs/en/sandboxing) (accessed 2026-04-18)
- [Understanding Claude Code Permissions and Security Settings — petefreitag.com](https://www.petefreitag.com/blog/claude-code-permissions/) (corroborating, accessed 2026-04-18)
