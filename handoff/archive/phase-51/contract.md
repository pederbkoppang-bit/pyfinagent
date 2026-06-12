# Contract -- phase-62.2: Inbound operator-token handler (Socket-Mode bot)

Date: 2026-06-12. Goal: goal-away-ops. (Rolling slot reclaimed from the harness sprint
contract; durable per-step copy of the brief: research_brief_62.2.md.)

## Research-gate summary

Brief: handoff/current/research_brief_62.2.md (gate_passed: true, 5 in full -- Bolt
listener-middleware docs, Slack Events API, Socket Mode docs, OWASP logging, Slack
security best practices; recency: Bolt 1.27.0 source verified in-venv, 2026 Slack DM
authz-bypass CVE noted). KEY FINDINGS:
- Bolt dispatch is FIRST-MATCH-WINS in registration order with fall-through on non-match
  (async_app.py:565+). Register the token handler ABOVE the catch-all (commands.py:184;
  register_commands is the first registrar per app.py:32). Implement the allowlist as a
  listener MATCHER returning bool -- matcher-False falls through so non-operator
  lookalikes still become tickets (never swallowed).
- Operator identity: NEW setting slack_operator_user_id; resolved live via the session
  Slack connector: U0A078KP4FQ (Peder, peder.bkoppang@hotmail.no -- the email-lookup API
  was scope-blocked; connector identity + user-search cross-confirmed). Hardcoded default
  per the _APPROVAL_CHANNEL="C0ANTGNNK8D" precedent (commands.py:25; identity constants
  are not secrets; .env writes are gate-blocked by design). Allowlist channels: digest
  channel (settings.slack_channel_id) + _APPROVAL_CHANNEL.
- Append/dedupe: reuse the kill_switch.py:109-119 append shape (open "a" + single
  json.dumps line, atomic under PIPE_BUF) + module asyncio.Lock. Bolt 1.27.0's Socket
  Mode adapter DROPS retry headers (async_internals.py:18) -- dedupe on body event_id +
  (channel, ts). Correct handler order: dedupe-check -> append -> threaded ACK (envelope
  acks after dispatch; crash-mid-handler = redelivery; redelivery then hits the dedupe).
- @app.message structurally cannot match message_changed (subtype constraints,
  async_app.py:877-925) -- edit double-recording impossible. ^...$ without MULTILINE =
  single-line tokens; uppercase keys = deliberate friction (lowercase falls to tickets).
- Bare reserved words (HALT-DEV, RESUME-DEV) carry no "KEY: value" -- the registration
  keyword needs an alternation; re-parse with re.match in-handler (context matches are
  lossy).
- operator_tokens.jsonl is TRACKED in git (kill_switch_audit precedent; tamper evidence;
  17.4 lesson applied: verified no ignore rule matches -- no .log suffix).
- FO-2 (from 62.0, binding): cursor = JSON {applied_line, token_sha256(raw), step, key,
  value, applied_at}; SESSIONS validate the specific token via an explicit KEY->ENV_VAR
  map before any .env touch; temp+rename advance refreshes mtime (opens the 62.0 hook 6h
  window). Hook stays cheap/mtime-based; semantics live session-side; 62.4 sentinel
  reconciles as backstop.
- Tests: pure-function pattern (test_phase_slack_digest_71.py precedent); the file name
  test_phase_62_2_operator_tokens.py satisfies the immutable -k filter.
- Verification trap: the command tails handoff/operator_tokens.jsonl -- a real line must
  exist at close (the live round-trip provides it).

## Immutable success criteria (verbatim from masterplan 62.2)

1. "a message handler registered BEFORE the catch-all @app.message at
   backend/slack_bot/commands.py:184 parses
   ^(?:(?P<step>[0-9][0-9.]*)\\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\\s*(?P<value>.+)$ plus
   reserved words and appends the structured line to handoff/operator_tokens.jsonl"
2. "only the operator's Slack user ID in the configured channel is accepted; unit tests
   assert other users/bots/channels are ignored and malformed lines are NOT written"
3. "live round-trip: operator sent a real test token (e.g. 'TEST TOKEN: PING') and the
   jsonl line + the bot's threaded ACK are pasted verbatim in live_check_62.2.md"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && source
.venv/bin/activate && python -m pytest backend/tests -k 'operator_token or 62_2' -q &&
tail -3 handoff/operator_tokens.jsonl

## Plan

1. settings.py: slack_operator_user_id Field(default "U0A078KP4FQ").
2. NEW backend/slack_bot/operator_tokens.py: TOKEN_RE + RESERVED (bare HALT-DEV /
   RESUME-DEV + generic "KEY: value" incl. KILL SWITCH: RESUME), parse_operator_token,
   async append_operator_token (lock + event_id/(channel,ts) dedupe + append-then-ACK),
   FO-2 cursor read/advance helpers (sessions consume them; bot only appends).
3. commands.py: matcher (user == operator AND channel allowlisted AND parseable) +
   handler (dedupe -> append -> threaded ACK quoting the recorded line number),
   registered FIRST inside register_commands (above the :184 catch-all).
4. Tests: grammar matrix, allowlist matrix (wrong user/bot/channel/malformed never
   written), dedupe, cursor semantics.
5. launchctl kickstart -k the bot; verify handler registration line in slack_bot.log.
6. LIVE ROUND-TRIP: operator sends "TEST TOKEN: PING" in the bot channel; paste the jsonl
   line + threaded ACK verbatim into live_check_62.2.md.
7. ONE fresh Q/A -> harness_log -> flip (auto-commit; manual fallback if stalled).

## Out of scope

Session-side token APPLICATION procedure (62.3 prompts encode it); digest sections
(62.8); any .env write.
