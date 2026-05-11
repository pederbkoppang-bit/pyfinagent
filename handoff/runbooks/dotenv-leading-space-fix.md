# Operator runbook: fix `backend/.env` leading-space bug (phase-23.6.0)

**Symptom:** `com.pyfinagent.autoresearch` (and previously
`com.pyfinagent.ablation`) failing nightly. `launchctl print` shows
`last exit code = 127` (originally) or `last exit code = 1` (after partial
fix). Per phase-23.3.5 + phase-23.5.19 audits, the root cause is one or
more `KEY= value` lines in `backend/.env` (note the space after `=`):

- `set -a; . backend/.env; set +a` parses `KEY= value` as `KEY=""` followed by
  `value` as a shell command.
- Bash tries to execute `value` and fails with "command not found" → exit 127.
- After lines 24/25 were repaired (per phase-23.3.7), residual exit 1 likely
  originates from line 56 (ANTHROPIC_API_KEY) OR a python entrypoint error.

This runbook is the doctrine-respecting fix path. The Claude Code session is
sandbox-blocked from `backend/.env` directly; **the operator must run these
commands** at a Mac terminal.

## Step 0 — pre-fix scan (always safe)

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python3 scripts/validators/check_dotenv_syntax.py backend/.env
```

If the validator exits 0, the file is clean and the residual exit-1 lives
elsewhere (likely the python autoresearch entrypoint — see "If still
failing" below). If it exits 1, the validator output names the offending
line numbers.

A separate manual scan (no Python deps):

```bash
grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env
```

## Step 1 — apply the fix (idempotent)

```bash
sed -i '' 's/^\([A-Z_][A-Z0-9_]*\)=  *\([^ ]\)/\1=\2/' backend/.env
```

This pattern:

- Matches a key (uppercase letters / digits / underscore) followed by `=`
  AND one-or-more spaces AND a non-space character.
- Replaces with `KEY=<first non-space char>`, removing the leading-space.
- Is a no-op on already-clean lines (`KEY=value` does not match the pattern).
- Safe to re-run as many times as you like.

## Step 2 — verify clean

```bash
# Manual:
grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env && echo "STILL BROKEN" || echo "CLEAN"

# Or via the validator:
python3 scripts/validators/check_dotenv_syntax.py backend/.env
```

## Step 3 — restart autoresearch

```bash
launchctl bootout gui/$(id -u)/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl kickstart gui/$(id -u)/com.pyfinagent.autoresearch
```

If the bootstrap fails silently (a known 2025-2026 macOS quirk), prepend:

```bash
launchctl enable gui/$(id -u)/com.pyfinagent.autoresearch
```

## Step 4 — confirm via dashboard

```bash
curl -s http://localhost:8000/api/jobs/all | python3 -c '
import json, sys
job = next(j for j in json.load(sys.stdin)["jobs"] if j["id"] == "com.pyfinagent.autoresearch")
print(f"status={job[\"status\"]}")
'
```

After the next 02:00 ET fire, status should transition from `failed` to
`ok`. Status takes up to 30 seconds to refresh due to the launchd-bridge
cache.

## If still failing after Step 3

If the validator exits 0 but autoresearch still shows `failed`, the
residual exit 1 lives in the python entrypoint (the autoresearch script
itself — `scripts/autoresearch/run_nightly.sh:6` has `set -euo pipefail`,
so anything that exits non-zero kills the wrapper).

Tail the script's own log for the python error:

```bash
tail -50 handoff/autoresearch.log
```

That's a separate fix from the `.env` syntax issue and would need its
own masterplan step.

## Pre-commit hook (prevention)

A hook installed by phase-23.6.0 runs the validator on every staged
`.env` file before committing. To re-install if it disappears:

```bash
# Verify the hook is active:
grep -l "check_dotenv_syntax" .git/hooks/pre-commit

# If not present, the hook is documented in the phase-23.6.0 archive:
cat handoff/archive/phase-23.6.0/contract.md
```

## Why this bug bit hard

Per phase-23.6.0 research:

- `python-dotenv`'s `dotenv_values()` SILENTLY strips the leading space
  and returns `KEY=value`. So Python code that loads the file works fine
  while bash crashes — masking the problem during local development.
- `set -a` + bash sourcing is intolerant of `KEY= value`; some other
  shells (zsh, fish) handle it differently. macOS launchd uses
  `/bin/sh -> bash` for `ProgramArguments`, so the bug hits hard there.

## Cross-references

- Phase-23.3.5 audit: `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md`.
- Phase-23.5.19 verification (current `failed` state):
  `handoff/archive/phase-23.5.19/`.
- Validator: `scripts/validators/check_dotenv_syntax.py`.
- Pytest fixtures: `tests/services/test_dotenv_syntax.py`.
