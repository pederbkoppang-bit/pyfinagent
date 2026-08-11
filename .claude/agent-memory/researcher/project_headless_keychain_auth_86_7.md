---
name: headless-keychain-auth-86-7
description: Step 86.7 -- the dead token is COMMITTED in 5 tracked files (rotation now mandatory); all pyfinagent launchd jobs are AGENTS not daemons; login keychain measures no-timeout so screen lock does NOT lock it; every auth alert is a one-shot latch and the skip path exits 0
metadata:
  type: project
---

Step 86.7 (headless/unattended auth for a keychain-backed CLI). Measured 2026-08-11.
Five findings that change the shape of the work, each of which refutes something a
reader would otherwise assume.

**1. The plist removal did NOT close 62.1.1 or 85.3.3 -- the token is committed to git.**
The 08-09 fix removed `CLAUDE_CODE_OAUTH_TOKEN` from all four live plists (verified: their
`EnvironmentVariables` key sets no longer contain it). But **12 `.bak` siblings under
`~/Library/LaunchAgents/` still carry it**, and the 92-char malformed value appears
**verbatim in 5 git-TRACKED files** under `handoff/away_ops/session_{am,pm}_*.json`. Proven by
fingerprint overlap, never by printing a value: backup plists hold three distinct tokens
(len 79 / sha `15dc02093816`, 79 / `35558d206b8c`, 92 / `32fd30514637`) and the tree's tracked
hits are all the 92-char `32fd30514637`. A fourth, 108-char value (`d773f475c399`) sits in the
untracked `backend/.env.env.bak-20260417-224659`.

**Why:** 62.1.1's criterion 1 explicitly covers "including the `*.bak` siblings", and 85.3.3
says a positive git-history result means "the token must be rotated". Both resolve against
closing. **How to apply:** the discriminating test is a **full-shape** regex
`sk-ant-oat01-[A-Za-z0-9_-]{20,}` cross-checked by hash against the known values -- NOT
`git log -S CLAUDE_CODE_OAUTH_TOKEN` (45 commits) or `-S "sk-ant-oat01-"` (40), both of which
are dominated by artifacts that merely *discuss* the name and the prefix in prose. See
[[count-the-class-not-your-list]] -- the raw `-S` count is not the exposure.

**2. All pyfinagent launchd jobs are AGENTS; there are ZERO LaunchDaemons.**
`/Library/LaunchDaemons/com.pyfinagent.*` does not exist. Per Apple TN2083 an agent runs "on
behalf of a particular user" and can "reliably access the user's home directory", which is
*why* the keychain is reachable at all. The daemon-can't-reach-the-login-keychain literature
does not apply here. **How to apply:** do not import daemon-context advice into this design;
and if anything is ever moved to `/Library/LaunchDaemons`, the credential path breaks by
construction.

**3. `claude auth status` green is NOT evidence the rail authenticates.**
`healthcheck.sh:86-89` says so in its own comment: it proves "LOCAL credential presence",
not authorization. The away-watchdog (a LaunchAgent, `runs=654`, `last exit code = 0`) has
been writing `auth_ok:"true"` every 30 min post-removal -- real, but the weaker test.
**How to apply:** measure with the real probe shape from `run_away_session.sh:147-149`
(`claude -p --max-turns 1 --output-format json`) and assert on `is_error` / `subtype` /
`duration_api_ms`. A 401 exits **0** and is caught by the *subtype* check at
`claude_code_client.py:474-483`, never the returncode branch; `duration_api_ms=0` is the
auth fingerprint.

**4. The login keychain measures `no-timeout` -- so screen lock does NOT lock it.**
`security show-keychain-info` returns `no-timeout`: no idle timeout, no lock-on-sleep. This
**partly refutes** 86.7's own audit_basis phrase "can be locked by reboot, logout, or screen
lock". The realistic unattended trigger is a **reboot**, not a screensaver. **How to apply:**
to test the keychain-unavailable branch you must force it with `security lock-keychain`; and
the setting is one `set-keychain-settings -l` away from getting worse, so pin and assert it.

**5. Every alert on this path is a one-shot latch, and the skip path exits 0.**
Three of them: `_RAIL_GUARD.paged` (`claude_code_client.py:97`, once per *cycle*, and the
message says `breaker_open`, never `auth`); `auth_page_state.json` `incident_open` ("paged
ONCE per incident"); and `run_away_session.sh:157` which logs `result=auth-dead-skip` then
**`exit 0`**, so launchd records a healthy run for a session that did nothing. **Why:** SRE
Workbook's *recall* attribute ("100% if every significant event results in an alert") is what
transition-only alerting fails. **How to apply:** the missing piece is a **staleness /
positive-heartbeat** alarm on the newest successful run -- both Google SRE chapters explicitly
omit the alert-on-absent-signal pattern, so it has to be built locally. Related:
[[fail-open-guards-hide-their-own-breakage]].

**Fallback shapes, and why three of four are unavailable** (useful next time this recurs):
`setup-token` is broken for this account AND revocable anyway; `apiKeyHelper` sits at
precedence **slot 4, ABOVE the keychain at slot 6** -- adopting it re-arms the exact override
that caused the outage, and it also suppresses the 3-day expiry warning (which fires only for
a claude.ai/Console login credential); the GitHub-CI ephemeral-keychain pattern needs you to
*possess* the secret, which nobody does here; and `security unlock-keychain -p` is flagged
insecure by Apple's own man page and "might succeed when an incorrect password is presented".

Brief: `handoff/current/research_brief_86.7.md`.
