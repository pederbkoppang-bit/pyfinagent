---
name: away-ops-session-gitignore-86-67
description: 86.67 away_ops session_*.json -- sole consumer reads from DISK so gitignore is safe; session.log is the live precedent; and `sk-[A-Za-z0-9]{20,}` CANNOT match sk-ant-oat01- tokens (hyphens)
metadata:
  type: project
---

Step 86.67 (`handoff/away_ops/session_*.json` not gitignored, `git add -A` can
publish a credential). Researched 2026-08-14.

**The consumer question resolves cleanly: DISK, not repo.**
`scripts/away_ops/auth_state.py:67` is the ONLY consumer --
`glob.glob(os.path.join(ops, "session_*.json"))` sorted by `os.path.getmtime`,
newest only, looking for `"api_error_status": 401` in a 36h window
(`DEFAULT_WINDOW_S=129600` at `:51`). `ops` is a REQUIRED CLI arg (`:113`),
passed by `healthcheck.sh:106` as `$REPO/handoff/away_ops`. A repo-path fallback
is *forbidden by test* (`test_phase_85_3_auth_latch_freshness.py:270`). Writer is
`run_away_session.sh:135`. The Slack away-digest (`scheduler.py:519/533/539/548`)
reads `pending_tokens.json` / `health.jsonl` / `session.log` / `defect_register.md`
-- **never** a session JSON.

**Why:** criterion 1 turned entirely on disk-vs-repo, and a consumer keyed on
`getmtime` can't be reading a git-materialised artifact (git doesn't store mtime).

**How to apply:** for any "can we gitignore X" question, find the consumer's PATH
SOURCE (hardcoded repo path vs injected param) and its FRESHNESS KEY (mtime vs
content) -- those two settle it faster than reading the whole rail.

## The live precedent nobody had cited
`handoff/away_ops/session.log` is ALREADY gitignored by `.gitignore:28 *.log`,
is untracked, and `scheduler.py:539` reads it from disk every digest run. Also
`run_away_session.sh:99` already excludes `handoff/away_ops/` from its own
dirty-tree gate ("perpetually dirty by design"). Look for an existing
ignored-and-still-consumed sibling before arguing from first principles.

## Credential-regex class trap (the highest-value lesson)
A caller handed me "ZERO credential-shaped values, positive-controlled" as GIVEN.
It was FALSE: 5 TRACKED files, all on `origin/main`, contain
`sk-ant-oat01-sk-ant-oat01-...` (92 chars) inside `.result`, from an API error
echoing `Authorization: Bearer ...`.

Root cause, proven with a discriminating probe:
`sk-[A-Za-z0-9]{20,}` returns **0** on `sk-ant-oat01-OvM72Xwg...` because the
class **cannot cross the hyphens** in `ant-oat01-`; `sk-ant-[A-Za-z0-9_-]{20,}`
returns 1. Their positive control used a hyphen-free `sk-` token, so it passed on
a shape that does not resemble a real Anthropic token.

**Why:** this is the "control and fail-safe answer coincide" probe failure --
the control proved the harness ran, not that the branch that mattered could fire.
See [[feedback_mutation_probe_must_discriminate]] and [[feedback_suspect_the_clean_check]].

**How to apply:** (1) always include `_` and `-` in credential character classes;
(2) build the positive control from a REAL token's literal prefix
(`sk-ant-oat01-`, `xoxb-`, `AIza`), never a stylised stand-in; (3) note these
files are single-line compact JSON with **zero** `": "` occurrences -- any regex
requiring a literal space after the colon can never fire on them.

## git check-ignore mechanics worth reusing
- Test a candidate rule WITHOUT editing `.gitignore`:
  `git -c core.excludesFile=<tmpfile> check-ignore -v --no-index <path>`.
  Faithful because `core.excludesFile` is the *lowest* precedence and is anchored
  at the worktree root -- matching there implies matching in `.gitignore`.
- **`--no-index` is REQUIRED for TRACKED paths**: without it a tracked file
  returns rc=1 even when the rule matches (tracked files are exempt). Measured
  both ways. Same trap as
  `.claude/agent-memory/qa/project_committed_criterion_gitignore_check.md:21-23`.
- Rule shape is load-bearing: `handoff/away_ops/session_*.json` is correct;
  `handoff/away_ops/session*` also swallows the TRACKED, rail-critical
  `session_notes.md` (written by `prompt_am.md:67`, `prompt_pm.md:22,41,44,48`).

## Doctrine short of history rewriting (criterion 4 / ask 06-2)
GitHub Docs: rotate/revoke is the FIRST step and rewriting *"may not be
warranted"* once rotated; rewriting risks recontamination from stale clones --
acute here given concurrent sessions + hook auto-push. OWASP orders
Revocation -> Rotation and says secrets must *"never be logged"*. GitGuardian
2026: **>64% of credentials leaked in 2022 were still valid in Jan 2026** -- never
assume a published token is dead, even a malformed-looking one.
