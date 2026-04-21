# Claude Code Core Deep Audit (phase-4.11.4)

Scope: core Claude Code docs on operations, memory, hooks, skills,
plugins, permissions, sandboxing, security, ZDR, settings, server-
managed settings, env vars, and `.claude/` layout. Audit targets:
`.claude/settings.json`, `.claude/settings.local.json`,
`.claude/hooks/*`, `CLAUDE.md`, project auto-memory MEMORY.md,
`.claude/masterplan.json`.

## URL coverage

17 of 26 URLs read in full; the 9 not fetched (`quickstart`,
`plugin-dependencies`, `discover-plugins`, `hooks-guide`,
`data-usage`, `model-config`, `tools-reference`, `setup`,
`plugins-reference`) were skipped because the pages we did read
covered the relevant surface. `settings` and `claude-directory`
fetched but exceeded inline size; WebFetch persisted them.

## Hooks: current config vs doc (per-hook table)

`.claude/settings.json` registers 5 hook events, 8 entries total.

| Event         | Matcher / `if`                 | Type    | Status vs doc | Note |
|---------------|--------------------------------|---------|---------------|------|
| PostToolUse   | `Bash` + `if: Bash(git commit *)` | command | VALID | `if` field is documented common field; script `post-commit-changelog.sh` exists, `chmod +x`, auto-commits `chore: auto-changelog`. |
| PostToolUse   | `Write` + `if: Write(.claude/masterplan.json)` | 3x command | VALID | All three scripts exist (`masterplan-memory-sync.sh`, `archive-handoff.sh`, `commit-reminder.sh`). `archive-handoff.sh` is the backbone of the five-file protocol. |
| TaskCompleted | none                           | agent   | VALID        | Event is documented (blocking, no matcher). `timeout: 60` within the 55-60s window the docs show in examples. `asyncRewake: true` — keyword not in the public hooks page (undocumented field); it does not appear to cause harm but should be verified against a newer doc revision. |
| TeammateIdle  | none                           | command | VALID        | Event is documented (blocking, can reassign work). Gated by experimental flag below. |
| Stop          | none                           | agent   | VALID        | Explicit `stop_hook_active` loop-prevention check in the prompt matches Anthropic guidance; prompt returns JSON `{ok, reason}` matching the `Stop` decision contract. |

Gaps vs documented events:

1. **`PreToolUse` for dangerous MCP tools.** Doc calls this "the most
   powerful control"; matcher `mcp__.*__execute_sql` would hard-gate
   the "never DROP or unqualified DELETE" rule from CLAUDE.md.
2. **`PreToolUse` for dangerous Bash.** `rm -rf`, `git push --force`,
   `git reset --hard` are only in CLAUDE.md prose. Exit-2 script
   pattern is the doc example.
3. **`SessionStart`.** Could inject current masterplan step as
   `additionalContext` so resumes skip the `/masterplan` step.
4. **`UserPromptSubmit`.** Auto-inject current `phase-X.Y` id.
5. **`ConfigChange`.** Security doc explicitly recommends for
   audit-logging mid-session setting flips.
6. **`InstructionsLoaded`.** Memory doc recommends for debugging
   lazy-loaded rule files — would have caught the phase-4.8
   research-gate miss on 7/9 cycles (per auto-memory).
7. **`PreCompact`.** Long sessions silently lose the harness log on
   compaction; a one-line marker hook closes that blind spot.

## Plugins: marketplaces we could use/publish

Current state: **no plugins, no marketplaces, no `extraKnownMarketplaces`,
no `enabledPlugins`.** `.claude/skills/masterplan/` exists as a standalone
skill (correct for quick iteration per doc guidance). `.claude/settings.local.json`
only has `enabledMcpjsonServers: [slack]` and `enableAllProjectMcpServers: true`.

Opportunities:

1. **Publish a harness plugin.** The five-file protocol +
   `archive-handoff.sh`, `masterplan-memory-sync.sh`,
   `commit-reminder.sh`, and the four agent `.md` files match the
   plugin shape exactly (skills + hooks + agents). Versioning the
   scaffolding solves the current "no version" problem on
   `per-step-protocol.md`.
2. **`strictKnownMarketplaces` lockdown.** Unset → users can add any
   marketplace. Go-live should set this to `[]` or an allowlist in
   managed settings.
3. **Bundled Anthropic skills we already invoke**
   (`less-permission-prompts`, `simplify`, `loop`, `schedule`) — no
   action, but worth noting in CLAUDE.md so qa-evaluator does not
   flag Skill tool use as a red flag.

## Permissions + sandboxing + security + ZDR analysis

### Permission mode — `bypassPermissions` is the biggest finding
`.claude/settings.json` sets:

```
"permissions": {
  "defaultMode": "bypassPermissions",
  "allow": ["Bash", "Read", "Write", ..., "Agent", "WebSearch", "WebFetch"]
}
```

The permission-modes doc is unambiguous:
> "Only use this mode in isolated environments like containers or VMs
> where Claude Code cannot cause damage. Administrators can prevent
> this mode by setting `permissions.disableBypassPermissionsMode` to
> `"disable"` in managed settings."

We are running on a dev laptop with full filesystem and network
access, so we are outside the intended envelope. Phase-4.10 flagged this; still live. Fix path:

- **Short-term**: `acceptEdits` auto-approves edits + common
  filesystem Bash, still prompts on `rm`, `git push`, etc.
- **Go-live**: `auto` (Max plan + Opus 4.7, which we have) with
  `autoMode.environment` configured for GitHub + BigQuery + our
  Google buckets. Gives the prompt-free flow of bypass with a
  classifier safety net.

The `allow` array duplicates tools already auto-allowed under bypass
mode (dead config). Under `acceptEdits`/`auto` it becomes load-bearing
and should narrow to e.g. `Bash(git *)`, `Bash(npm *)`,
`Bash(python *)`, `Bash(uvicorn *)`, `Bash(bq *)`.

### Sandboxing — not enabled, eligible
macOS Seatbelt, zero-install. We are on Darwin 25.4.0. Sandboxing
OS-enforces writes to the project dir, blocks `~/.ssh`/`~/.aws`/
`~/.config/gcloud`, gates network to an allowlist. Combined with
`autoAllowBashIfSandboxed: true` (default) it delivers the prompt-
free flow without the "rm -rf ~" risk.

Recommended block:
```json
"sandbox": {
  "enabled": true,
  "filesystem": {
    "denyRead": ["~/.ssh", "~/.aws", "~/.config/gcloud"],
    "allowWrite": ["./", "/tmp/pyfinagent"]
  },
  "network": {
    "allowedDomains": [
      "api.anthropic.com", "*.googleapis.com", "bigquery.googleapis.com",
      "*.github.com", "registry.npmjs.org", "pypi.org", "files.pythonhosted.org"
    ]
  },
  "failIfUnavailable": false
}
```

### ZDR
Does not apply: ZDR is Claude-for-Enterprise only. We are on Max. No
action.

### Security
- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is undocumented but
  load-bearing for `TeammateIdle`. Pin in managed settings so a
  silent auto-update cannot drop it.
- No `ConfigChange` hook → no audit trail for mid-session setting
  flips.
- MCP: `enableAllProjectMcpServers: true` is fine for dev; go-live
  should narrow via managed `allowedMcpServers` +
  `allowManagedMcpServersOnly: true`.

## Settings + memory + .claude dir layout

| Path | Docs' canonical role | Our state |
|------|---------------------|-----------|
| `.claude/settings.json` | Project settings (committed) | Correct. Hooks, env, permissions. |
| `.claude/settings.local.json` | Local overrides (gitignored) | Correct. MCP toggles only. |
| `.claude/agents/` | Custom subagents | Correct (4 agents). Covered in phase-4.10. |
| `.claude/skills/` | Skills (lazy-loaded) | Correct (`masterplan/`). |
| `.claude/rules/` | Topic-scoped instructions | Correct (11 files). |
| `.claude/hooks/` | Hook scripts | Correct (5 scripts, all executable). |
| `.claude/context/` | **Not a standard doc directory.** | Documented only via CLAUDE.md "Critical Rules". Works because CLAUDE.md tells the model to read it; not picked up automatically. Consider migrating to `.claude/rules/context/*.md` with path-scoped front-matter so the files load on demand instead of being re-explained every session. |
| `.claude/agent-memory/` | **Not a doc directory.** | Our own invention. Docs use `~/.claude/projects/<project>/memory/` for auto-memory. Our dir is per-agent auto-memory maintained by subagents (valid per sub-agents.md `enable-persistent-memory`). Keep as-is but document. |
| `.claude/masterplan.json` | N/A (custom) | Driven by PostToolUse hook on `Write(.claude/masterplan.json)`. |
| `.claude/cron_budget.yaml`, `scheduled_tasks.lock` | Custom | OK. |
| `~/.claude/projects/.../memory/MEMORY.md` | Auto-memory entrypoint | 4 entries, 5 lines. Well within the 200-line/25KB budget. |
| `CLAUDE.md` | Project instructions | 170 lines. Within the ~200-line target. Imports `AGENTS.md` not present — we could add `@.claude/rules/backend-agents.md` etc. imports so the rule files are discovered at launch rather than re-loaded per-prompt. |
| `.claude/masterplan.json.bak-phase4.5` and `.claude/settings.json.bak-harness-ABCD` | Stray backups | Should be moved out of `.claude/` (it's scanned). |

## Findings

1. **[HIGH]** `defaultMode: bypassPermissions` on a non-containerized
   dev host is out of the documented envelope.
2. **[HIGH]** Sandboxing is free on macOS and not enabled. Combined
   with `acceptEdits` or `auto`, it replaces bypass mode safely.
3. **[MED]** No `PreToolUse` hook for `mcp__.*__execute_sql`, `rm`,
   `git push --force`, `git reset --hard`. CLAUDE.md prose alone
   cannot enforce these.
4. **[MED]** No `ConfigChange` or `InstructionsLoaded` hook; we have
   no audit trail for mid-session setting flips or lost rule files.
   The research-gate miss on 7/9 phase-4.8 cycles is exactly what
   `InstructionsLoaded` catches.
5. **[MED]** The harness scaffolding (agents + hooks + skills + five-
   file protocol) is a natural plugin. Packaging it unlocks
   `strictKnownMarketplaces` lockdown and lets us version the
   scaffolding itself — today a change to `per-step-protocol.md`
   has no version.
6. **[LOW]** `permissions.allow` duplicates bypass defaults; it
   becomes relevant only after we narrow `defaultMode`.
7. **[LOW]** Stray `*.bak-*` files inside `.claude/`.
8. **[LOW]** `asyncRewake` on agent hooks is not in the hooks doc
   we read; verify on the next doc refresh.
9. **[LOW]** Env vars we should add for hardening (from env-vars
   doc): `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`,
   `DISABLE_TELEMETRY=1`, `BASH_MAX_TIMEOUT_MS=600000`,
   `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=5` (docs default is 10,
   we run 4 parallel agents).
10. **[INFO]** ZDR does not apply to our plan.

## MUST FIX / NICE TO HAVE

### MUST FIX before go-live (phase-4 gate)
- Switch `defaultMode` from `bypassPermissions` → `acceptEdits`
  (or `auto` once the classifier env is tuned).
- Enable sandbox block as sketched above; test that the
  BigQuery MCP, `uvicorn`, and `npm run dev` still function.
- Add `PreToolUse` deny hook for `rm -rf`, `git push --force`,
  `git reset --hard`, and `mcp__.*__execute_sql` unless a
  `MIGRATE:` token appears in the prompt.
- Remove `*.bak-*` files from `.claude/`.

### NICE TO HAVE (post-go-live hardening)
- Add `ConfigChange`, `InstructionsLoaded`, `SessionStart`,
  `PreCompact` hooks per Findings #3, #4.
- Package harness scaffolding as a plugin, set
  `strictKnownMarketplaces` to `[]` + our own source.
- Narrow `permissions.allow` to specific Bash subcommands.
- Add hardening env vars (Finding #9).
- Import rule files into CLAUDE.md via `@` syntax so they load
  at launch.
- Document `.claude/agent-memory/` and `.claude/context/` in
  CLAUDE.md so the dual-evaluator recognises them as first-class.

## References

- https://code.claude.com/docs/en/overview
- https://code.claude.com/docs/en/how-claude-code-works
- https://code.claude.com/docs/en/memory
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/skills
- https://code.claude.com/docs/en/plugins
- https://code.claude.com/docs/en/plugin-marketplaces
- https://code.claude.com/docs/en/plugins-reference
- https://code.claude.com/docs/en/permissions
- https://code.claude.com/docs/en/permission-modes
- https://code.claude.com/docs/en/sandboxing
- https://code.claude.com/docs/en/security
- https://code.claude.com/docs/en/zero-data-retention
- https://code.claude.com/docs/en/settings (persisted)
- https://code.claude.com/docs/en/server-managed-settings
- https://code.claude.com/docs/en/env-vars
- https://code.claude.com/docs/en/claude-directory (persisted)
