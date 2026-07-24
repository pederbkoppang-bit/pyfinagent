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

---

## Revalidation 2026-06-14 (post-implementation drift check)

Tier: simple (caller-stated, post-implementation revalidation). Agent: researcher (Layer-3). Session: AM away-ops (operator away). The 62.2 code shipped 2026-06-12. This section is a DRIFT CHECK: does the shipped implementation satisfy the design + masterplan criteria, and has the external literature moved since 2026-06-12. NOT changing trading behavior; `commands.py`/`operator_tokens.py` are bot-side (not rail-6 trading-behavior files).

### External re-confirmation (read in full this session via WebFetch; >=5 floor)

| # | URL | Accessed | Kind | Fetched how | Drift finding |
|---|-----|----------|------|-------------|---------------|
| 1 | https://docs.slack.dev/tools/bolt-python/concepts/listener-middleware/ | 2026-06-14 | Official docs (Slack) | WebFetch, full | STILL RESOLVES (no 301 this time — already on docs.slack.dev). Re-confirms: a matcher "returns `bool` value (`True` for proceeding) instead of requiring `next()` method call." Doc STILL silent on cross-listener dispatch order + fall-through — unchanged since 2026-06-12; resolved from SDK source (see below). No drift. |
| 2 | https://docs.slack.dev/tools/bolt-python/reference/listener_matcher/async_listener_matcher.html | 2026-06-14 | Official docs (Slack) | WebFetch, full | `AsyncListenerMatcher.async_matches(req, resp) -> bool`; "Matches against the request and returns True if matched." Confirms the bool contract the shipped `_operator_token_matcher` implements. Doc does not state AND-combination — that is in SDK source (anchored below). No drift. |
| 3 | https://cwe.mitre.org/data/definitions/117.html | 2026-06-14 | Official-grade (MITRE) | WebFetch, full | CWE-117 mitigations: (1) accept-known-good input validation, (2) "Use and specify an output encoding that can be handled by the downstream component that is reading the output," (3) canonicalization. Example exploits CRLF (`%0a`). Note: MITRE frames it as "output encoding" generically; `json.dumps` (which escapes `\n`/`\r` to `\\n`/`\\r`) IS a valid output-encoding instance — the shipped `append_operator_token` uses `json.dumps(record, ensure_ascii=False)` so raw operator text cannot inject a fake JSONL line. No drift. |
| 4 | https://docs.slack.dev/concepts/security/ | 2026-06-14 | Official docs (Slack) | WebFetch, full | "Never expose tokens (or other customer secrets) to the end user, especially in error messages or by echoing them back to the UI." + "validating message sources before processing (particularly important for AI-integrated apps)." Directly validates the shipped allowlist-as-matcher + the ACK that echoes only the raw token line, never any secret. No drift. |
| 5 | https://moldstud.com/articles/p-comprehensive-guide-to-auditing-slack-bot-user-permissions-for-compliance | 2026-06-14 | Industry practitioner (2025) | WebFetch, full | RECENCY HIT: "45% of security incidents [in collaboration platforms] stem from unchecked or excessive access granted to automation tools"; 2025 Snyk survey: 40% of enterprise Slack apps request a scope beyond documented need; "less is more in permission management." Corroborates (does NOT supersede) the 62.0 design's fail-closed single-operator allowlist. No drift to the design. |

Also re-read in full locally (installed `slack_bolt` 1.27.0 source — code, not WebFetch; does NOT count toward floor, anchors the load-bearing CRIT-1 fact the docs won't confirm):
- `slack_bolt/app/async_app.py:614` — `for listener in self._async_listeners:` (registration order); `:617` `if await listener.async_matches(...)` runs the first matching listener; `:634-635` `# This means the listener is not for this incoming request.` / `continue` — fall-through to the next listener when the `@app.message(keyword)` regex middleware does NOT match.
- `slack_bolt/listener/async_listener.py:27-31` — matcher AND-combination with short-circuit: `for matcher in self.matchers: is_matched = await matcher.async_matches(...); if not is_matched: return is_matched`. So `_operator_token_matcher` returning False = the whole listener doesn't match = Bolt falls through to the catch-all. Empirically re-confirmed this session (version still 1.27.0; behavior byte-identical to the 2026-06-12 brief).

### Recency scan (last 2 years) — revalidation pass

Three-variant queries run this session: current-year — "Slack Bolt Python listener matchers message dispatch order 2026"; last-2-year — "chatops slack bot command authorization allowlist security 2025"; year-less canonical — "OWASP logging cheat sheet log injection CWE-117 sanitization". Findings:
- **No semantic change to Slack Bolt-Python matcher/dispatch behavior** in 2025-2026. Bolt remains on the 1.27.0 line (installed); matchers still return bool, still AND-combine with short-circuit, dispatch still first-match-wins-with-fall-through. The official docs completed their migration to `docs.slack.dev` (2025) — the prior brief's 5 URLs all resolve there now (#1 no longer even 301s).
- **One NEW corroborating practitioner data point (2025):** Moldstud/Snyk 45%/40% over-permissioned-automation statistics (#5) — reinforces, does not change, the fail-closed single-operator allowlist already shipped. CWE-117 mitigation guidance unchanged (output encoding / accept-known-good).
- **No new finding supersedes the 62.0 design.** The shipped allowlist-as-matcher + json.dumps append + dual-key dedupe + append-before-ACK remain best-practice as of 2026-06-14.

### Internal drift audit (file:line evidence — the load-bearing part)

**Settings field NOW EXISTS** (prior brief flagged it missing): `backend/config/settings.py:530-538` `slack_operator_user_id: str = Field("U0A078KP4FQ", ...)` — default is the operator's real Slack uid (identity constant, not a secret; same class as the hardcoded approval channel). Empty string = fail-closed. The shipped code is MORE complete than the prior design (which only recommended adding it).

**File-absence confirmed:** `handoff/operator_tokens.jsonl` does NOT exist; `handoff/away_ops/tokens_cursor` does NOT exist (verified via `ls` 2026-06-14). No operator token has been sent yet -> criterion 3 (live round-trip) is structurally unsatisfiable headless this session.

#### Drift Q1 — CRIT-1: token handler ABOVE catch-all; non-match falls through (NOT swallowed)?  PASS
- Token listener registered at `commands.py:115` (`@app.message(_TOKEN_KEYWORD, matchers=[_operator_token_matcher])`) INSIDE `register_commands` (`commands.py:88`), which is the FIRST registrar called (`app.py:32`, before `register_assistant_lifecycle`/`register_governance`). Catch-all is `@app.message("")` at `commands.py:237`. So the token listener is registered ABOVE the catch-all in the same registrar -> earlier in `self._async_listeners` order.
- Fall-through proven from SDK source: `async_app.py:614/617/634-635`. A non-token message (or non-operator/wrong-channel, via the matcher) fails the token listener's match -> Bolt `continue`s -> reaches the catch-all at `:237` -> ticket ingestion at `:260`. Not swallowed.
- Tests assert the fall-through INVARIANT at the matcher level: `test_matcher_rejects` (`:64-71`) asserts non-operator/bot/wrong-channel/non-token all return False (-> would fall through), and `test_malformed_never_written` (`:112-116`) asserts a non-token never writes the jsonl.

#### Drift Q2 — CRIT-2: allowlist on operator user id AND channel; others/bots/wrong-channel IGNORED; malformed NOT written?  PASS
- Allowlist is on the MESSAGE PATH as a matcher: `is_operator_token_message` (`operator_tokens.py:79-95`) enforces, in order: fail-closed if `operator_user_id` unset (`:87-88`), reject `bot_id` (`:89-90`), reject `user != operator_user_id` (`:91-92`), reject `channel not in allowed_channels` (`:93-94`), then require parseability (`:95`). Channel set built at `commands.py:102-104` = `{slack_channel_id, _APPROVAL_CHANNEL}` (both non-empty), wired with `slack_operator_user_id` at `:108`.
- Tests (names): `test_matcher_accepts_operator` (`:60-61`); `test_matcher_rejects` parametrized over `user="U_SOMEONE_ELSE"`, `bot_id="B123"`, `channel="C_RANDOM"`, `text="not a token"` (`:64-71`) — all asserted False; `test_matcher_fail_closed_when_unconfigured` (`:74-75`) asserts empty operator id => False; `test_malformed_never_written` (`:112-116`) asserts a non-token append returns None AND `not TOKENS_PATH.exists()` (malformed never written).

#### Drift Q3 — Grammar: bare HALT-DEV/RESUME-DEV (no `: value`) AND `KILL SWITCH: RESUME`; regex an alternation?  PASS
- `parse_operator_token` (`operator_tokens.py:67-76`): bare reserved words handled FIRST via `RESERVED_BARE = {"HALT-DEV", "RESUME-DEV"}` set membership (`:46`, `:70-71`) -> `{step:None, key:<word>, value:""}`; otherwise `TOKEN_RE` (`:43-45`) `^(?:(?P<step>[0-9][0-9.]*)\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\s*(?P<value>.+)$`. `KILL SWITCH: RESUME` parses under the generic rule (key allows spaces) -> `{step:None, key:"KILL SWITCH", value:"RESUME"}`.
- The Bolt KEYWORD at `commands.py:111-113` IS an explicit alternation: `^(?:[0-9][0-9.]*\s+)?[A-Z][A-Z0-9 _-]+:\s*.+$|^(?:HALT-DEV|RESUME-DEV)$` — matches the prior brief's recommendation. NOTE: this keyword is a coarse pre-filter; the authoritative parse is re-done inside the handler via `parse_operator_token` (handler calls `append_operator_token` which re-parses at `:107`), exactly as the prior brief advised (ignore Bolt's lossy `context["matches"]`).
- Tests: `test_grammar_accepts` (`:25-37`) covers `KILL SWITCH: RESUME`, `HALT-DEV`, `RESUME-DEV`, stepped tokens, whitespace-trim; `test_grammar_rejects` (`:39-49`) covers lowercase, prose, no-value, reserved-word-with-trailing-prose, multiline (no MULTILINE flag), empty/None.

#### Drift Q4 — Safety: secrets never echoed in ACK; raw JSON-encoded on write (CWE-117); append-before-ACK + dedup?  PASS
- **No secret in ACK:** the success ACK (`commands.py:133-140`) echoes only `record['raw']` (the operator's own token text) + the line number + a static pointer to `pending_tokens.json`. No token/secret is in scope in the handler. Matches Slack security doc (#4) "never echo tokens back."
- **CWE-117 neutralized on write:** `append_operator_token` writes `json.dumps(record, ensure_ascii=False)` (`operator_tokens.py:125-127`); `raw` is stored as a JSON string value, so any CR/LF in operator text is escaped to `\\n`/`\\r` and cannot forge a new JSONL line. Matches CWE-117 "output encoding" mitigation (#3).
- **Append-before-ACK:** the handler `await append_operator_token(...)` FIRST (`commands.py:119-125`), THEN `say(...)` (`:133`). Append happens before the ACK round-trip, narrowing the 3s-ack redelivery window (prior-brief finding 3).
- **Dedup:** dual-key `{event_id, (channel, ts)}` (`operator_tokens.py:110`), checked under `_append_lock` against process-lifetime `_seen_events` (`:111-114`), updated only after a successful write (`:128`). `event_id` threaded from `body.get("event_id")` at `commands.py:124`. Tests: `test_duplicate_event_id_not_rewritten` (`:94-100`), `test_duplicate_channel_ts_not_rewritten` (`:103-109`). Documented limitation (`operator_tokens.py:18-19` docstring): a redelivery straddling a bot restart can double-append — ACCEPTED by design (sessions treat identical raw+slack_ts as one token; 62.4 sentinel is the detective backstop).

#### Drift Q5 — CRIT-3 (live round-trip) is operator-gated.  CONFIRMED OPERATOR-GATED
- Masterplan verification for 62.2 tails the REAL `handoff/operator_tokens.jsonl` and requires >=1 line at close. That file does not exist yet (confirmed `ls`, 2026-06-14) and CAN ONLY be created by a real operator message from `slack_operator_user_id` in an allowlisted channel reaching the running Socket-Mode bot. It is NOT satisfiable headless in this AM away session (operator away). The pytest leg (`pytest -k 'operator_token or 62_2'`) IS satisfiable headless and exercises the same append helper against `tmp_path`. State plainly: **criterion 3 is operator-gated and cannot be closed by Main alone this session.**

#### Drift Q6 — GAP between shipped code and criteria / prior-brief recs?  NO BLOCKING GAP (3 non-blocking notes)
- **No missing allowlist leg, no grammar miss, no secret echo, no non-atomic append.** All four safety properties + both CRIT-1/CRIT-2 are present with test coverage. The shipped code is a faithful (and more complete) implementation of the 62.0 design.
- Non-blocking note (a): the catch-all at `commands.py:241` gates on `channel != _APPROVAL_CHANNEL` ONLY (single channel), while the token allowlist accepts `{slack_channel_id, _APPROVAL_CHANNEL}`. CONSEQUENCE: an operator token sent in the digest channel (`slack_channel_id`, if different from the approval channel) IS recorded as a token but would NOT have fallen through to ticketing anyway (catch-all early-returns on non-approval channels). No swallow risk, no behavior bug — just an asymmetry to be aware of. Not a blocker.
- Non-blocking note (b): the handler's success-ACK references `handoff/away_ops/pending_tokens.json` (`commands.py:137`) which does not exist yet and is produced by a later step. Cosmetic only (a forward-looking pointer in a Slack message); not a code defect.
- Non-blocking note (c) — **rail-6 check:** this handler/parser touches NEITHER `kill_switch.py` logic NOR any trading-behavior file. `KILL SWITCH: RESUME` is merely RECORDED as a token line; no kill-switch state is mutated by 62.2 code (the FO-2 cursor + a later session applies effects). So nothing here is rail-6 (no dark+token gating needed for the 62.2 code itself). `operator_tokens.py` + `commands.py` are bot-side, outside the rail-6 trading-behavior file list — consistent with the away-ops rules.

### Revalidation Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 this session; all 5 prior-brief sources re-confirmed resolving on docs.slack.dev)
- [x] 10+ unique URLs (5 full + 8 snippet-only this session; ~40 cumulative with prior brief)
- [x] Recency scan (2024-2026) performed + reported (Bolt unchanged; one new 2025 Snyk/Moldstud corroboration; no supersession)
- [x] file:line anchors for every internal claim (drift Q1-Q6)
- [x] All six drift questions answered with file:line evidence

### Snippet-only this session (does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://docs.slack.dev/tools/bolt-python/reference/listener_matcher/index.html | Official docs | Matcher base-class ref; AND-combination not in doc -> resolved from SDK source |
| https://github.com/slackapi/bolt-python/issues/284 | Community | Event-matching-specific-reaction; matcher predicate precedent |
| https://docs.slack.dev/tools/bolt-python/concepts/authorization/ | Official docs | Per-installation authorization (OAuth scope), not message-path allowlist |
| https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html | Official-grade (OWASP) | Re-confirmed via CWE-117 (#3); CR/LF sanitization guidance unchanged |
| https://cwe.mitre.org/data/definitions/117.html (dup of #3) | — | (read in full as #3) |
| https://www.rapid7.com/fundamentals/chatops/ | Industry | ChatOps conversational-security overview; no new allowlist mechanics |
| https://www.tines.com/blog/chatbots-for-security-and-it-teams-part-3-creating-a-slack-chatbot/ | Blog | Slack security-bot build walkthrough; corroborates least-privilege |
| https://docs.slack.dev/concepts/security/ (dup of #4) | — | (read in full as #4) |
| https://docs.nautobot.com/projects/chatops/en/latest/admin/platforms/slack/ | Official docs | Per-user ChatOps access-grant precedent (carried from prior brief) |

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

### Internal drift audit (file:line evidence — the load-bearing part)

**Settings field NOW EXISTS** (prior brief flagged it missing): `backend/config/settings.py:530-538` `slack_operator_user_id: str = Field("U0A078KP4FQ", ...)` — default is the operator's real Slack uid (identity constant, not a secret; same class as the hardcoded approval channel). Empty string = fail-closed. So the shipped code is MORE complete than the prior design (which only recommended adding it).

**File-absence confirmed:** `handoff/operator_tokens.jsonl` does NOT exist; `handoff/away_ops/tokens_cursor` does NOT exist (verified via `ls` 2026-06-14). No operator token has been sent yet -> criterion 3 (live round-trip) is structurally unsatisfiable headless this session.

(drift questions 1-6 answered below as the section is filled)

