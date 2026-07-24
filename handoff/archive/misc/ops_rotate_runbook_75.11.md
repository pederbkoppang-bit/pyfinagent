# Runbook -- Step 75.11 SRE hardening (log rotation, service authority, timeouts)

This step shipped repo files only (scripts + plist TEMPLATES + this
runbook). Nothing was bootstrapped on the machine and no live launchd
agent or plist was touched -- both of those are OPERATOR actions, gated
behind the two tokens below. `backend/.env` was not edited either; its
one known defect (an unbalanced quote at line 81) is drafted as its own
token, separate from the launchd bootstrap.

## What changed (repo-only, zero live-system mutation)

- `scripts/ops/rotate_logs.sh` + `scripts/ops/com.pyfinagent.logrotate.plist.template`
  -- cp+truncate rotation for `backend.log`, `frontend.log`,
  `handoff/logs/slack_bot.log`, `handoff/logs/auto-push.log` (size cap +
  gzip archives, keep-N retention), plus a watchdog-liveness alarm that
  pages if `handoff/away_ops/health.jsonl` goes >2h stale while this agent
  is loaded.
- `scripts/start_services.sh` -- primary path is `launchctl kickstart -k`;
  the old unsupervised `pkill -9` + `nohup` + `> backend.log` design is
  gone. A legacy direct-launch path survives ONLY behind `LEGACY_DIRECT=1`,
  using a scoped SIGTERM-then-wait `pkill -f 'uvicorn backend.main'`
  (never `-9`).
- `scripts/ops/frontend_start.sh` + `scripts/ops/com.pyfinagent.frontend.plist.template`
  -- single frontend authority running production `next start` behind a
  pre-start build wrapper. **Live `~/Library/LaunchAgents/com.pyfinagent.frontend.plist`
  still runs `next dev` today -- this template REPLACES it on bootstrap.**
- `scripts/ops/run_ablation.sh` + `scripts/ops/com.pyfinagent.ablation.plist.template`
  -- replaces the live ablation plist's raw `. backend/.env` sourcing
  (which crash-failed ~37 nights on the line-81 quote defect) with the
  sanitized-grep sourcing block already proven in
  `scripts/autoresearch/run_nightly.sh`, plus a paging seam after N
  consecutive failures.
- `scripts/autoresearch/run_nightly.sh` -- gained the same paging seam
  (it already logged FAIL rc; only the page was missing).
- `.claude/hooks/pre-tool-use-danger.sh` -- new pkill/killall rail
  (targets `python|uvicorn|next|slack_bot`, `CLAUDE_ALLOW_DANGER=1` escape
  inherited).
- `scripts/away_ops/run_away_session.sh`, `scripts/slack_mention_checker.sh`,
  `scripts/mas_harness/run_cycle.sh` -- gtimeout / `-m 15` caps on the
  three previously-unbounded unattended calls.
- `backend/main.py` -- `setup_logging()`'s formatter branch order
  corrected (`debug` -> `CompactFormatter`, default -> `JsonFormatter`,
  matching the pre-existing comment's stated intent).

## Operator token 1: OPS-ROTATE-BOOTSTRAP

Bootstraps the THREE new/replacement launchd agents on the operator's
machine. Each step is independently reversible (`launchctl bootout` +
restore the prior plist from the paths noted).

```
OPS-ROTATE-BOOTSTRAP: AUTHORIZE
```

When authorized, run (as the operator, on the machine -- NOT by an
unattended agent session per this step's boundary):

```bash
REPO=/Users/ford/.openclaw/workspace/pyfinagent
UID_N=$(id -u)

# 1. Log rotation agent (NEW -- no live equivalent today).
sed "s#__REPO_ROOT__#$REPO#g" "$REPO/scripts/ops/com.pyfinagent.logrotate.plist.template" \
    > ~/Library/LaunchAgents/com.pyfinagent.logrotate.plist
launchctl bootstrap "gui/$UID_N" ~/Library/LaunchAgents/com.pyfinagent.logrotate.plist

# 2. Frontend authority (REPLACES the live `next dev` plist).
launchctl bootout "gui/$UID_N/com.pyfinagent.frontend" 2>/dev/null || true
cp ~/Library/LaunchAgents/com.pyfinagent.frontend.plist \
   ~/Library/LaunchAgents/com.pyfinagent.frontend.plist.pre-75.11.bak
sed "s#__REPO_ROOT__#$REPO#g" "$REPO/scripts/ops/com.pyfinagent.frontend.plist.template" \
    > ~/Library/LaunchAgents/com.pyfinagent.frontend.plist
launchctl bootstrap "gui/$UID_N" ~/Library/LaunchAgents/com.pyfinagent.frontend.plist

# 3. Ablation wrapper (REPLACES the raw `. backend/.env` sourcing).
launchctl bootout "gui/$UID_N/com.pyfinagent.ablation" 2>/dev/null || true
cp ~/Library/LaunchAgents/com.pyfinagent.ablation.plist \
   ~/Library/LaunchAgents/com.pyfinagent.ablation.plist.pre-75.11.bak
sed "s#__REPO_ROOT__#$REPO#g" "$REPO/scripts/ops/com.pyfinagent.ablation.plist.template" \
    > ~/Library/LaunchAgents/com.pyfinagent.ablation.plist
launchctl bootstrap "gui/$UID_N" ~/Library/LaunchAgents/com.pyfinagent.ablation.plist
```

Rollback: `launchctl bootout gui/$UID_N/<label>` then restore the
`.pre-75.11.bak` copy and re-bootstrap it.

## Operator token 2: backend/.env:81 quote repair

`backend/.env` line 81 carries an unbalanced quote (introduced by an
operator paste on 2026-06-12) that breaks any RAW `. backend/.env`
dot-sourcing (this is exactly why `run_ablation.sh` above uses the
sanitized-grep block instead, which tolerates the malformed line by
dropping it). The file itself is untouched by this step -- fixing line 81
is optional cleanup, not a blocker for anything shipped here, and is
gated behind its own token because it is a direct `backend/.env` edit
(this step's boundary forbids editing `backend/.env`).

```
ENV-QUOTE-REPAIR-81: AUTHORIZE
```

When authorized, the operator (not an unattended session) should open
`backend/.env` at line 81, locate the unbalanced quote in the comment
text, and either close the quote or remove the offending comment line,
then re-run `bash -n` against a throwaway copy sourced the old raw way to
confirm the fix (do not commit `.env`; it is gitignored/untracked).

## Not in scope / explicitly deferred

- No live launchd agent was bootstrapped, unloaded, or modified.
- No live log was rotated, truncated, or deleted.
- No service was restarted.
- `backend/.env` was not read for content and was not edited (its sha256
  hash before/after this step is recorded identical in
  `handoff/current/experiment_results_75.11_draft.md`).
