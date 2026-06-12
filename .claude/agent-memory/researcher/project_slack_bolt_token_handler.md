---
name: slack-bolt-token-handler-62-2
description: Bolt 1.27.0 dispatch mechanics (first-match + fall-through), subtype/retry facts, and FO-2 semantic-cursor design for the 62.2 operator-token handler
metadata:
  type: project
---

Phase-62.2 (goal-away-ops) facts, researched 2026-06-12 from installed slack_bolt 1.27.0 source + official docs.

**Why:** the away-ops token plane (62.2-65.2) hangs off Bolt dispatch subtleties that are NOT in the docs (doc is silent on cross-listener order) and were pinned by reading the venv package source.

**How to apply:** any work on backend/slack_bot message handlers, operator_tokens.jsonl, or the tokens_cursor gate.

- **Dispatch = first-match-wins in registration order, WITH fall-through**: `async_dispatch` (async_app.py:565+) runs the first listener whose constraints+matchers pass and whose listener middleware calls next(); a non-matching `@app.message(keyword)` regex middleware hits `continue` -> next listener. So: token handler registered ABOVE the catch-all `@app.message("")` at commands.py:184 (inside register_commands, which app.py:32 calls first) consumes tokens exactly once; everything else falls through to ticket ingestion. Implement operator-user+channel allowlist as a **matcher** (bool) so non-operator lookalike messages fall through to tickets instead of being swallowed (2026 OpenClaw CVE lesson: allowlist must gate the message path).
- **`@app.message` cannot see edits**: constraints subtype tuple = (None, bot_message, thread_broadcast, file_share) (async_app.py:877-925) -> message_changed/deleted never match. Keyword match = `re.findall(keyword, event.text)`; `^...$` without MULTILINE = single-line messages only; bare reserved words (HALT-DEV) need an alternation since the grammar requires `KEY: value`.
- **Retry headers are DROPPED on Socket Mode in Bolt 1.27.0**: adapter builds `AsyncBoltRequest(mode="socket_mode", body=req.payload)` without mapping retry_attempt/reason (adapter/socket_mode/async_internals.py:18). Delivery is at-least-once (envelope acked only AFTER dispatch returns) -> dedupe on body["event_id"] + (channel, ts), append before say().
- **No operator Slack user-id setting existed** (settings.py:527-529 has only tokens + slack_channel_id); commands.py:25 hardcodes approval channel C0ANTGNNK8D; thread-ACK idiom at commands.py:268-270. JSONL append idiom to copy: kill_switch.py:109-119 (open "a" utf-8, single json.dumps write) + asyncio.Lock. operator_tokens.jsonl is NOT gitignored -> tracked (auditability; force-push blocked by 62.0 hook).
- **FO-2 semantic cursor**: tokens_cursor (mtime-gated by pre-tool-use-danger.sh:176-199/:215-249, 6h window) becomes JSON {applied_line, token_sha256, step, key, value, applied_at}; session validates the SPECIFIC token via KEY->ENV_VAR map before .env write, advances cursor via temp+rename (refreshes mtime); 62.4 sentinel reconciles .env vs jsonl as backstop. Hook stays content-agnostic/mtime (cheap layer).
- **Verification trap**: 62.2's immutable command tails the REAL handoff/operator_tokens.jsonl -> a live/synthetic appended line must exist at step close; tests (test_phase_62_2_operator_tokens.py, pure-function style per test_phase_slack_digest_71.py) use tmp_path.
