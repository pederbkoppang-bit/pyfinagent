# Contract -- phase-62.2: Inbound operator-token handler in the Socket-Mode bot (VERIFY-ONLY)

Date: 2026-06-14 (AM away session, ~07:30 CEST). Goal: goal-away-ops.
Binding: docs/runbooks/away-ops-rules.md (read first this session).

## Why this is a verify-only execution (state delta)

The 62.2 handler code ALREADY SHIPPED on 2026-06-12 as part of the 62.0 away-ops install
(commit batch around the away-goal install / `1be98e83` I-4 cursor-advance rule). It was
never run through the per-step harness loop and remains `status: pending`. This session
runs the loop to verify the shipped implementation against the immutable criteria and to
consolidate the evidence -- NO code change is planned (research found no blocking gap).

Shipped surface (confirmed by Main + researcher, file:line):
- `backend/slack_bot/commands.py:115` -- `@app.message(_TOKEN_KEYWORD, matchers=[_operator_token_matcher])`
  registered inside `register_commands` (`:88`, the FIRST registrar per app.py:32), ABOVE
  the catch-all `@app.message("")` at `commands.py:237`.
- `backend/slack_bot/operator_tokens.py` -- `is_operator_token_message` (:79-95 allowlist,
  fail-closed), `append_operator_token` (:110-128 append-before-ACK + dual-key dedup under
  `_append_lock`, `json.dumps(..., ensure_ascii=False)` CWE-117 neutralization),
  `TOKEN_RE` (:43-45) + `RESERVED_BARE` (:46) grammar, `KNOWN_TOKEN_ENV_MAP`,
  `unapplied_tokens`, `advance_cursor`.
- `backend/config/settings.py:530` -- `slack_operator_user_id` (default `U0A078KP4FQ`;
  fail-closed on empty). (The prior brief flagged this as missing-to-add; it now exists.)
- `backend/tests/test_phase_62_2_operator_tokens.py` -- 29 tests (grammar accept/reject,
  matcher rejects other-user/bot/wrong-channel/non-token, fail-closed-when-unconfigured,
  malformed-never-written, duplicate event_id / (channel,ts) dedup).

## Research-gate summary

Brief: `handoff/current/research_brief_62.2.md` (original design brief 2026-06-12 +
`## Revalidation 2026-06-14 (post-implementation drift check)` section appended this
session). gate_passed: **true** -- 5 external sources read in full (all 5 prior-brief URLs
re-confirmed resolving on docs.slack.dev; SDK source re-anchored), recency scan performed
(no movement: Bolt still 1.27.0, same matcher/dispatch semantics), 13 URLs, 5 internal
files audited with file:line evidence. Drift verdict: **NO BLOCKING GAP** -- shipped code
is a faithful, more-complete implementation of the 62.0 design. Three non-blocking notes:
(a) catch-all gates on `_APPROVAL_CHANNEL` only while the allowlist accepts
`{slack_channel_id, _APPROVAL_CHANNEL}` -- asymmetry, no swallow risk; (b) ACK references a
not-yet-existing `pending_tokens.json` (cosmetic forward pointer); (c) rail-6 CLEAN -- the
handler only RECORDS `KILL SWITCH: RESUME` as a jsonl line and mutates NO `kill_switch.py`
state; `operator_tokens.py`/`commands.py` are bot-side, outside the rail-6
trading-behavior file list -> no dark+token gating needed for the 62.2 code.

## Immutable success criteria (verbatim from masterplan 62.2)

1. "a message handler registered BEFORE the catch-all @app.message at
   backend/slack_bot/commands.py:184 parses
   `^(?:(?P<step>[0-9][0-9.]*)\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\s*(?P<value>.+)$` plus
   reserved words and appends the structured line to handoff/operator_tokens.jsonl"
2. "only the operator's Slack user ID in the configured channel is accepted; unit tests
   assert other users/bots/channels are ignored and malformed lines are NOT written"
3. "live round-trip: operator sent a real test token (e.g. 'TEST TOKEN: PING') and the
   jsonl line + the bot's threaded ACK are pasted verbatim in live_check_62.2.md"

verification.command (verbatim): `cd /Users/ford/.openclaw/workspace/pyfinagent && source
.venv/bin/activate && python -m pytest backend/tests -k 'operator_token or 62_2' -q &&
tail -3 handoff/operator_tokens.jsonl`

verification.live_check (verbatim): "live_check_62.2.md with the verbatim jsonl line and
ACK permalink from the live round-trip"

## Hypothesis

The shipped handler satisfies criteria 1 + 2 deterministically: the pytest leg of the
verification command passes (29 tests) and code inspection confirms handler-above-catch-all
ordering, operator-user+channel allowlist, and malformed-not-written. Criterion 3 requires
a REAL operator-sent Slack token -- `handoff/operator_tokens.jsonl` does not yet exist, so
the `tail -3` leg of the verification command FAILS (no such file) until the operator sends
one. That round-trip is owned by the 62.7 dress rehearsal (operator watching) or any real
operator token; it cannot be produced headless. **A synthetic jsonl line must NOT be
written** -- it would fabricate operator evidence and (per the I-4 rule) a stale
`KILL SWITCH: RESUME` could re-fire on a future real breach. Honest expected verdict this
AM: **CONDITIONAL** on criterion 3; 62.2 is NOT flipped to done this session.

## Plan (verify-only sequence; will NOT diverge in Generate)

1. Run the pytest leg verbatim: `python -m pytest backend/tests -k 'operator_token or 62_2'
   -q` -- expect 29 passed. Paste output.
2. Run the FULL immutable verification command verbatim -- expect pytest pass + `tail`
   failure on the missing operator_tokens.jsonl. Paste both, document the operator-gate.
3. Code-evidence for criterion 1: paste `grep -n '@app.message'` (115 token handler < 237
   catch-all) + the `_TOKEN_KEYWORD` alternation (commands.py:111-113) + `TOKEN_RE`/
   `RESERVED_BARE` (operator_tokens.py:43-46). Confirm the masterplan-quoted regex is the
   in-handler authoritative re-parse.
4. Code-evidence for criterion 2: paste `is_operator_token_message` (operator_tokens.py
   :79-95) + the test names that assert other-user/bot/wrong-channel ignored
   (`test_matcher_rejects`) and malformed-never-written (`test_malformed_never_written`).
5. Write experiment_results_62.2.md (verbatim command output + file list + criteria 1+2
   evidence + criterion-3 operator-gate statement).
6. Write live_check_62.2.md: criteria 1+2 PASS evidence + criterion 3 = the EXACT operator
   reply string needed (`TEST TOKEN: PING` in the approvals channel) and where the jsonl
   line + ACK permalink will be pasted on round-trip.
7. ONE fresh Q/A (expect CONDITIONAL on criterion 3) -> harness_log append
   (result=CONDITIONAL) -> leave 62.2 status=pending. Add the exact reply string to
   pending_tokens.json + session_notes.md.

## Handoff-file convention (this session)

Running in SUFFIXED files (`*_62.2.md`) -- same convention as the 62.1 CONDITIONAL no-flip
session. No status flip this AM -> the archive hook does not fire -> rolling slots stay
untouched for the parked step 62.6 (PM-owned).

## Scope honesty / out of scope

- NO code change (researcher: no blocking gap). If a pure bug surfaced in
  commands.py/operator_tokens.py (non-trading-behavior files) I would fix it under rail-6
  live-fix authority (failing->passing test + fresh Q/A + live evidence); none expected.
- Do NOT write a synthetic operator_tokens.jsonl line (criterion 3 needs a REAL token).
- Do NOT touch kill_switch.py / any trading-behavior file (rail 6).
- Criterion-3 live round-trip is owned by 62.7 / a real operator token -- not produced
  this AM (operator away).
- The three non-blocking researcher notes (catch-all/allowlist channel asymmetry; ACK's
  pending_tokens.json forward pointer; ) are cosmetic and out of scope for this verify.

## References

- `handoff/current/research_brief_62.2.md` (design 2026-06-12 + revalidation 2026-06-14).
- docs.slack.dev Bolt listener-middleware + async listener-matcher; CWE-117; Slack
  security best practices.
- Prior: `contract_62.1.md` (the CONDITIONAL no-flip away-session precedent this follows);
  masterplan 62.2 spec + verification block.
