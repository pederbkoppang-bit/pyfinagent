# Research Brief — phase-62.2 (goal-away-ops): Inbound operator-token handler in the Socket-Mode bot

Tier: moderate (caller-stated). Date: 2026-06-12. Agent: researcher (Layer-3, merged Explore).
62.1 brief preserved at `handoff/current/research_brief_62.1.md`. Tool calls ~13 of 18 budget.

## Read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.slack.dev/tools/bolt-python/concepts/listener-middleware/ | 2026-06-12 | Official docs | WebFetch (301 from tools.slack.dev followed) | "Listener middleware is only run for the listener in which it's passed"; if `next()` isn't called the listener won't run; matchers "return `bool` value (`True` for proceeding)". Doc is silent on cross-listener dispatch order — resolved from installed source (see findings 1-2). |
| 2 | https://docs.slack.dev/apis/events-api/ | 2026-06-12 | Official docs | WebFetch, full page | At-least-once: retries "up to 3 times" (immediately / 1 min / 5 min) with `x-slack-retry-num` 1-3 + `x-slack-retry-reason`; 3-second 2xx ack window; `event_id` = "a unique identifier for this specific event, globally unique across all workspaces"; dedup is explicitly app-layer responsibility. |
| 3 | https://docs.slack.dev/apis/events-api/using-socket-mode/ | 2026-06-12 | Official docs | WebFetch, full page | Ack each envelope by `envelope_id` (`{"envelope_id": <id>, "payload": <optional>}`); up to 10 simultaneous connections, "each payload may be sent to any of the connections"; unacked envelopes are redelivered. |
| 4 | https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html | 2026-06-12 | Official-grade (OWASP) | WebFetch, full page | Each event must record "when, where, who and what"; "Perform sanitization on all event data to prevent log injection attacks e.g. carriage return (CR), line feed (LF) and delimiter characters" (CWE-117); build in tamper detection; strict directory permissions. |
| 5 | https://docs.slack.dev/authentication/best-practices-for-security | 2026-06-12 | Official docs | WebFetch (302 from api.slack.com followed) | Least privilege; verify authenticity of inbound requests; "Never expose tokens... in error messages or by echoing them back"; validate message sources before acting on them. |

Also read in full locally (code, not WebFetch — does not count toward floor): installed `slack_bolt` 1.27.0 source — `app/async_app.py:565+` (dispatch), `:877-925` (`message()`), `middleware/message_listener_matches/async_message_listener_matches.py`, `listener_matcher/builtins.py:143-151`, `adapter/socket_mode/async_internals.py`.

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://github.com/slackapi/bolt-python/issues/561 | Community | Matchers run before the `@app.message` keyword middleware — confirmed in source instead |
| https://static.usenix.org/event/sec09/tech/full_papers/crosby.pdf | Peer-reviewed (Crosby & Wallach, USENIX Sec '09) | Canonical tamper-evident-logging prior art; hash-chain concept extracted from snippet; PDF, beyond moderate budget |
| https://mattermost.com/blog/compliance-by-design-18-tips-to-implement-tamper-proof-audit-logs/ | Industry | Append-only + sequence-gap detection; corroborates OWASP |
| https://dev.to/robertatkinson3570/the-architecture-behind-tamper-proof-audit-logs-56ek | Blog | "Each entry's SHA-256 covers the one before it" — hash-chain pattern for FO-2 echo |
| https://dailycve.com/openclaw-slack-dm-authorization-bypass-improper-access-control-cve-2026-xxx-moderate/ | Industry (CVE) | 2026 Slack-bot authz bypass: handler "incorrectly treats any DM sender as command-authorized" — exact failure mode 62.2 must avoid |
| https://docs.openclaw.ai/gateway/security | Official docs | "Allowlists gate triggers and command authorization"; strict DM policy |
| https://docs.stackstorm.com/chatops/chatops.html | Official docs | ChatOps RBAC/ACL precedent |
| https://pkg.go.dev/github.com/slack-go/slack/socketmode | Official SDK docs | Socket-Mode requests carry RetryAttempt/RetryReason fields |
| https://github.com/slackapi/node-slack-sdk/issues/2141 | Community | Duplicate message events observed WITHOUT retry header — dedupe must not rely on retry-num |
| https://docs.slack.dev/tools/bolt-python/reference/listener/index.html | Official docs | Listener API reference (matchers list, primary_matcher position 0) |
| https://www.sonarsource.com/resources/library/audit-logging/ | Industry | Audit-log completeness/access-control checklist |
| https://docs.nautobot.com/projects/chatops/en/latest/admin/platforms/slack/ | Official docs | ChatOps per-user access-grant precedent |

Queries run (3-variant discipline): year-less — "Slack Bolt Python listener matchers middleware...", "append-only audit log integrity tamper-evident..."; 2025 — "Slack Events API retry semantics x-slack-retry-num... 2025"; 2026 — "Slack Socket Mode envelope_id... 2026", "chatops slack bot command authorization allowlist... 2026".

## Recency scan (2024-2026)

Searched 2025/2026-scoped variants. Findings: (a) 2026 CVE — Slack DM authorization bypass in a chat-ops gateway (sender not checked against allowlist on the message path) — directly informs the allowlist design; (b) Slack docs migrated to docs.slack.dev (2025) — old api.slack.com/tools.slack.dev URLs redirect (301/302), cite new host; (c) hash-chained tamper-evident JSONL write-ups (2025-2026) complement the canonical Crosby & Wallach 2009; (d) installed Bolt 1.27.0 (current line) behavior verified from source. No findings that supersede the official Slack docs.

## Key findings (external)

1. **Bolt dispatch is first-match-wins in registration order, with fall-through on non-match.** `async_dispatch` iterates `self._async_listeners` (registration order); the first listener whose constraints+matchers pass AND whose listener middleware calls `next()` runs, then dispatch returns. If the `@app.message(keyword)` regex middleware does NOT match, Bolt hits the explicit `continue` ("This means the listener is not for this incoming request") and tries the next listener. (Source: installed slack_bolt 1.27.0 `app/async_app.py` async_dispatch; doc #1 confirms middleware semantics.) → Registering the token listener ABOVE the catch-all inside `register_commands` gives: token messages consumed exactly once, everything else falls through to the catch-all. No double-processing.
2. **`@app.message` structurally cannot see edits/deletes.** `message()` registers constraints `{"type": "message", "subtype": (None, "bot_message", "thread_broadcast", "file_share")}` (`async_app.py:877-925`) — `message_changed`/`message_deleted` subtypes never match, so edit-double-recording is impossible via this decorator. Keyword matching is `re.findall(keyword, event["text"])` on the raw text; empty text skips (`async_message_listener_matches.py`).
3. **Delivery is at-least-once; dedupe is ours; retry headers are unavailable on this path.** Events API retries 3x (#2); unacked Socket-Mode envelopes are redelivered (#3); and Bolt 1.27.0's Socket-Mode adapter builds `AsyncBoltRequest(mode="socket_mode", body=req.payload)` WITHOUT mapping `retry_attempt`/`retry_reason` (`adapter/socket_mode/async_internals.py:18`) — handlers can't see x-slack-retry-num. Dedupe on `body["event_id"]` (globally unique per event, #2) with `(channel, ts)` as belt-and-braces. Envelope ack is sent only AFTER dispatch returns (`send_async_response`), so a crash mid-handler → redelivery → dedupe required.
4. **Allowlist must gate the message path itself** (2026 CVE: any-sender-authorized bug). Implement operator-user+channel as a listener **matcher** (bool): matcher-False → dispatch falls through to the catch-all, so non-operator ALL-CAPS messages still become tickets instead of being swallowed. Least-privilege + never echo secrets in the ACK (#5).
5. **Audit-log shape:** record when/where/who/what; JSON-encode raw text (json.dumps escapes CR/LF → CWE-117 log injection neutralized); tamper-evidence via git-tracked file + cursor hash echo (#4, Crosby & Wallach snippet).

## Internal code inventory

| File | Lines | Role / status |
|------|-------|---------------|
| backend/slack_bot/commands.py | 25; 88; 175-182; 184-273; 191; 268-270 | `_APPROVAL_CHANNEL="C0ANTGNNK8D"` hardcoded (:25); `register_commands(app)` (:88) is the FIRST registrar called (app.py:32) → insert token handler inside it, above :184. Catch-all `@app.message("")` (:184) ingests tickets; bot filter `message.get("bot_id")` (:191); thread-ACK idiom `thread_ts = message.get("thread_ts") or message.get("ts")` + `say(text=..., thread_ts=...)` (:268-270) — reuse verbatim. |
| backend/slack_bot/app.py | 27-36; 32 | `create_app()` registers commands → assistant_lifecycle → governance; registration order = dispatch priority. |
| backend/config/settings.py | 527-529 | `slack_bot_token`/`slack_app_token` (SecretStr), `slack_channel_id` (digest channel). **NO operator-user-id field exists** — 62.2 must add e.g. `slack_operator_user_id: str = Field("")`. |
| .claude/hooks/pre-tool-use-danger.sh | 176-199; 215-249; 160-171 | 62.0 gate: `handoff/away_ops/tokens_cursor` mtime < 21600s authorizes backend/.env writes (Bash + Edit/Write paths). Purely mtime today — FO-2 target. Force-push block at :160-171 protects git history of the jsonl. |
| backend/services/kill_switch.py | 36; 109-119 | Reusable append idiom: `_AUDIT_PATH.open("a", encoding="utf-8")` + single `f.write(json.dumps(row)+"\n")` (atomic under PIPE_BUF), swallow-and-warn. No flock anywhere in repo; single-process bot + `asyncio.Lock` suffices. |
| backend/tests/test_phase_slack_digest_71.py | 1-80 | Only slack test; pattern = pure-function tests on formatters, no Bolt app/network mocking. 62.2 tests: extract `parse_operator_token(text)` + append/dedupe helper as pure functions; name file `test_phase_62_2_operator_tokens.py` (matches `-k 'operator_token or 62_2'`). |
| .gitignore + `git check-ignore` | 71-76 | Only `handoff/logs/`, `handoff/*.log`, quarantine ignored. `handoff/operator_tokens.jsonl` NOT ignored → **tracked** (decision below). |
| handoff/away_ops/ | dir listing | Contains only `approved_plan_2026-06-12.md`; no tokens_cursor (gate closed-by-default), no pending_tokens.json, no operator_tokens.jsonl yet. |
| .claude/masterplan.json | 14621-14630; 14661; 15063 | 62.2 spec + verification (`pytest -k 'operator_token or 62_2'` AND `tail -3 handoff/operator_tokens.jsonl` — file must exist with lines at close); 62.4 sentinel reconciles .env vs jsonl (FO-2 backstop); 65.2 expects token timestamps in the jsonl. |
| slack_bolt 1.27.0 (venv) | per findings 1-3 | Built-in `AsyncIgnoringSelfEvents` ON by default (drops this bot's own ACK replies); `subtype` tuple includes `bot_message` → keep an explicit `bot_id` guard for OTHER bots. |

**Internal-audit answers:** (1) First-match-wins + fall-through (finding 1) — register above :184, keep matcher/middleware non-match = fall-through. (2) No operator-id exists; channel = `settings.slack_channel_id` (:529) vs hardcoded `_APPROVAL_CHANNEL` (:25) — gate on operator user id (new setting) + channel set {both, non-empty}; ACK via the :268-270 idiom. (3) Reuse kill_switch append shape (:109-119) + module `asyncio.Lock` + in-memory seen-set on `event_id`/`(channel,ts)` (finding 3); on restart, rehydrate seen-set from the file tail. (4) No operator_tokens.jsonl exists; cursor design below. (5) Pure-function tests per digest-71 pattern; for handler-level coverage call the inner coroutine with a dict message + `AsyncMock` say (no Socket-Mode needed).

## FO-2 semantic-cursor design (binding from 62.0)

Tokens file (bot-written, append-only, **git-tracked**): one JSON line `{ts, user, channel, raw, step, key, value}` + recommended `event_id`, `recorded_at`.
Cursor `handoff/away_ops/tokens_cursor` becomes JSON (hook only stats mtime, content-agnostic — backward compatible):
`{"applied_line": <1-based line no>, "token_sha256": <sha256 of raw jsonl line>, "step": ..., "key": ..., "value": ..., "applied_at": <iso>}`.
Session procedure (encoded in 62.3 away prompts): read jsonl → next unapplied line = `applied_line`+1 → validate THE SPECIFIC token via an explicit KEY→ENV_VAR map (e.g. `DATA INTEGRITY: ON` → `PAPER_DATA_INTEGRITY_GATE=true`; unknown key = refuse) → atomically rewrite cursor (temp+rename; rename refreshes mtime → opens the 6h hook window) → write backend/.env → 62.4 sentinel reconciles .env flag lines against jsonl (masterplan:14661) as the detective backstop. Line-number = ordering/at-most-once application; sha256 echo = tamper-evidence tying cursor to the exact line (hash-chain-lite, finding 5); key/value echo lets a future hook harden from mtime to content-grep without jsonl parsing. Hook stays the cheap mtime layer (62.0 design intact).

## Risks & gotchas

- **Grammar misses bare reserved words**: `HALT-DEV`/`RESUME-DEV` have no `: value` — the Bolt keyword must be an alternation: `^(?:KILL SWITCH: RESUME|HALT-DEV|RESUME-DEV)$|^(?:(?P<step>[0-9][0-9.]*)\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\s*(?P<value>.+)$`. `KILL SWITCH: RESUME` already parses under the generic grammar (key may contain spaces). Re-parse with `re.match` inside the handler; ignore Bolt's `context["matches"]` (findall+groups is lossy).
- **`^...$` without re.MULTILINE** → only single-line messages match (multi-line falls through to ticketing — acceptable; document it). Keep CASE-SENSITIVE uppercase keys (deliberate friction; lowercase falls through to tickets harmlessly).
- **False-positive capture**: operator typing `TODO: fix x` in-channel matches grammar → recorded as token, NOT ticketed. Mitigation: threaded ACK makes capture visible ("recorded operator token KEY=...; reply NOT-A-TOKEN to void" optional); allowlist-as-matcher means only the operator can ever trigger this.
- **Own-bot loop**: ACK replies are dropped by built-in `AsyncIgnoringSelfEvents`; still keep `bot_id`/`user` guards (subtype tuple admits other bots' `bot_message`).
- **Slack text munging**: URLs/channels arrive as `<...>`; smart quotes possible on mobile — store `raw` verbatim, parse leniently on value.
- **Duplicate delivery**: dedupe on `event_id` (+`channel,ts`); append BEFORE `say()` so a slow ACK can't widen the redelivery window (3s ack budget, finding 3).
- **Verification trap**: immutable command tails the REAL `handoff/operator_tokens.jsonl` — Main must produce >=1 real/synthetic appended line (live token test) before close; tests use tmp_path, never the real file.
- **Tracked-file decision**: track it (matches `kill_switch_audit.jsonl` + `handoff/audit/*` precedent; git history = tamper evidence; force-push blocked by 62.0 hook:160-171). Appends during git ops are safe (O_APPEND single write); never `git stash` (hook-append conflict, standing rule).
- **Channel ambiguity**: digest channel (`slack_channel_id`) vs hardcoded approval channel (:25) may differ — allowlist both, primary control is the user id.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (≈40 collected across 5 searches)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered commands/app/settings/hook/kill_switch/tests/gitignore/masterplan + installed Bolt source
- [x] Contradictions noted (doc silent on dispatch order → resolved from source; doc gap on event_id stability across retries → mitigated with dual dedupe key)
- [x] Per-claim citations

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
