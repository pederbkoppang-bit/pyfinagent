---
name: credential-expiry-66-4
description: phase-66.4 empirics — keychain claudeAiOauth.expiresAt is the 8h ACCESS token (refresh-token expiry invisible -> pre-expiry warning infeasible); auth status = LOCAL presence check (loggedIn:true with dead creds); 401 session JSON has subtype:"success" + api_error_status:401 + rc=1; healthcheck tail-1 dedupe re-pages every other run
metadata:
  type: project
---

phase-66.4 research facts (2026-07-07), credential-expiry resilience:

- **Credential surface (this machine, v2.1.201)**: ~/.claude/.credentials.json
  ABSENT on macOS; storage = login keychain service `Claude Code-credentials`
  acct `ford` (NOT "claude-code" as some 2026 guides claim). Payload keys:
  claudeAiOauth.{accessToken,refreshToken,expiresAt,scopes,subscriptionType,
  rateLimitTier}. **expiresAt = ACCESS-token expiry, exactly mdat+8h** —
  auto-refreshed; refresh-token lifetime UNPUBLISHED and unexposed anywhere
  -> ">=24h pre-expiry warning" is infeasible; only lagging staleness proxy
  (expiresAt older than ~16-24h) works. `security find-generic-password -w`
  read no prompt from user session; pipe straight to parser, never log.
- **`claude auth status`**: JSON by default (loggedIn/authMethod/apiProvider/
  email/orgId/orgName/subscriptionType — NO expiry field); documented exit
  contract 0=logged-in/1=not (cli-reference). It is a LOCAL presence check:
  returned loggedIn:true throughout the 17-day dead-credential outage class.
  Auth-dead detection MUST also scan newest session_*.json.
- **401 session signature** (handoff/away_ops/session_am_20260621T053010Z.json):
  `subtype:"success"` BUT `is_error:true, api_error_status:401, num_turns:1,
  duration_api_ms:0, total_cost_usd:0`, result "Failed to authenticate...";
  CLI exits rc=1 (session.log 06-21). Detector = rc!=0 AND
  grep '"api_error_status":401'. Never key on subtype. Dead-auth run still
  burns ~4.4 min (duration_ms 265138).
- **healthcheck.sh dedupe defect (do not copy)**: tail-1 health.jsonl replay
  (:117-129) re-pages every OTHER run during sustained failure (~1/hr), not
  once-per-incident. Multi-week incidents need a latch file or date-keyed
  dedupe. Drill isolation precedent: HEALTHCHECK_TEST_P1 runs never write
  p1_raised (:150-156). Bot-token fallback :139-148 greps backend/.env
  python-free, channel default C0ANTGNNK8D. JSON line = 18 fields (:182-186).
- **Wrapper hooks**: run_away_session.sh rc-case :144-149 logged the 401 as
  "crash or limit" then exit 0 -> 34-slot burn; dirty-tree filter :97
  excludes handoff/{audit,away_ops,logs}/ -> state file
  handoff/away_ops/auth_page_state.json is recovery-safe; *.json under
  away_ops NOT gitignored (only *.log is, .gitignore:24).
- **Recovery path**: `claude setup-token` = 1-year inference-scoped token,
  printed once, NOT saved; consumed via CLAUDE_CODE_OAUTH_TOKEN env at
  precedence slot 5, which OUTRANKS keychain subscription OAuth (slot 6);
  `--bare` does not read it (authentication docs 2026).

**Why:** all empirical (live keychain/CLI inspection + dead-window artifacts);
none derivable from docs alone, and one 2026 guide is factually wrong on the
service name.
**How to apply:** any auth-probe/page-once design in 66.4+ builds on these;
see [[away-watchdog-p1-path]] and [[cc-rail-guard-66-1]].
