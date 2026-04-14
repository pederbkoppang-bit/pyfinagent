# Git Push Credential Diagnosis (2026-04-14)

## Status
**BLOCKED** since at least 2026-04-13. Every Ford remote session has reported
identical 403s on `git push origin main`. The same PAT works for git fetch
(reads) but fails for all writes.

## Verified root cause

The bootstrap PAT in the Ford remote-trigger prompt
(`github_pat_11BZJ24UY0C0xRtpAYJnL8_...`) is a **fine-grained PAT that lacks
`Contents: Read and write` permission** on `pederbkoppang-bit/pyfinagent`.

### Diagnostic proof (2026-04-14 05:15 UTC)

```
$ curl -H "Authorization: Bearer $TOKEN" https://api.github.com/user
  -> {"login": "pederbkoppang-bit", ...}                     # PAT authenticates OK

$ curl -H "Authorization: Bearer $TOKEN" \
       https://api.github.com/repos/pederbkoppang-bit/pyfinagent
  -> {"permissions": {"admin": true, "push": true, ...}}     # but this is the
                                                              # user's repo role,
                                                              # not the token's
                                                              # granted permissions

$ curl -X PUT -H "Authorization: Bearer $TOKEN" \
       https://api.github.com/repos/pederbkoppang-bit/pyfinagent/contents/probe.txt \
       -d '{"message":"probe","content":"UEFUIHByb2JlCg=="}'
  -> {"message": "Resource not accessible by personal access token"}

$ git push origin main
  -> remote: Permission to pederbkoppang-bit/pyfinagent.git denied to
     pederbkoppang-bit.
  -> fatal: ... 403
```

### Why this is confusing

Classic PATs have broad OAuth scopes (`repo`, `write:repo`) and are grant-all.
**Fine-grained PATs have per-permission grants** that are independent of the
user's role in the repo. A user can be an admin on a repo (which shows up in
`repo.permissions.push: true`) but their PAT can still be denied writes if the
PAT was not explicitly granted `Contents: Read and write` at creation time.

The repo permissions endpoint returns the **user's** role, not the **token's**
permissions. This is why the diagnostic gave a false "push: true" signal.

### Why MCP github still works

The `mcp__github__*` tools use a **separate credential** (likely a GitHub App
installation token minted by the Claude Code harness). GitHub App tokens have
their own permission set, and this one has Contents: write on pyfinagent. That
is why `mcp__github__create_or_update_file` and `mcp__github__push_files`
succeed while `git push` and direct Contents API calls with the bootstrap PAT
fail.

## The fix (requires Peder)

Only Peder can fix the bootstrap PAT because only he can mint tokens on his
GitHub account.

### Option A -- Regenerate the fine-grained PAT with Contents: write

1. Go to https://github.com/settings/personal-access-tokens
2. Find the existing Ford bootstrap PAT
3. Click **Edit** -> **Repository access** -> confirm `pederbkoppang-bit/pyfinagent`
4. Under **Repository permissions**, set:
   - **Contents**: Read and write   (currently Read only -- this is the bug)
   - **Metadata**: Read-only        (keep, required by all fine-grained PATs)
   - **Pull requests**: Read and write (optional, for `gh pr create`)
5. Save
6. Update the token string in the Ford remote-trigger bootstrap message
   (the `git remote set-url origin https://pederbkoppang-bit:github_pat_...@...`
   line at the top of the remote-trigger prompt)

### Option B -- Generate a new classic PAT

Less preferred (broader scope) but simpler:
1. https://github.com/settings/tokens/new
2. Scopes: `repo` (the full `repo` scope grants Contents: write)
3. No expiration or 90 days
4. Replace the PAT in the Ford bootstrap

### Option C -- Use MCP github for everything (current workaround)

This session proved that MCP github writes work. The Ford bootstrap's
4-tier memory protocol could be rewritten to use `mcp__github__create_or_update_file`
and `mcp__github__push_files` instead of `git push`. The cost is that every
file push requires inlining the file content in a tool call (~15k tokens for
a 60 KB file). That's acceptable for small files and docs but prohibitive for
routine commits. Not recommended as the primary path.

## Current local state (this session's uncommitted work)

As of 2026-04-14 05:15 UTC, the local `main` worktree is **3 commits ahead**
of origin/main:

- `fa2b696` Phase 2.12: logger ASCII hardening on multi_agent_orchestrator
  - Fixes all 21 residual non-ASCII chars in `backend/agents/multi_agent_orchestrator.py`
    `logger.*()` calls (emoji + em-dash + right-arrow -> bracketed ASCII tags + `--`/`->`)
  - Pure string substitution, 22+/22-
  - AST scan 0 violations, py_compile clean, qa-evaluator subagent PASS
- `17a932f` Session log 2026-04-14 05:00: Phase 2.12 logger ASCII v5
- `bb69b3a` CHANGELOG: auto-update for 17a932f session log commit

Once the PAT is fixed, a simple `git push origin main` from this worktree (or
any new Ford session that re-attaches to main) will publish all three commits.

## Lessons for CLAUDE.md

Add to `.claude/rules/` or CLAUDE.md:

> **Fine-grained PAT gotcha**: the `repo.permissions.push` field from
> `GET /repos/.../pyfinagent` reflects the **user's role**, not the **token's
> granted scopes**. Do not use it to diagnose a fine-grained PAT. Instead test
> with `PUT /repos/.../contents/probe` -- if it returns "Resource not
> accessible by personal access token" while the user is authenticated,
> the PAT lacks Contents: Read and write.
