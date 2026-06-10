# Research Brief — Step 55.2: Ops incidents + agent-quality audit

**Tier:** complex
**Away window:** 2026-06-01 → 2026-06-10 (autonomous paper-trading)
**Scope:** review-only (NO fixes), $0 (bounded BQ reads, 30s timeout, no LLM cycle spend)
**Researcher:** sole research agent (external lit + internal code audit, one session)

This brief feeds the contract for step 55.2. It carries (A) an internal
code audit with file:line anchors and a live BQ agent-label→skill mapping,
and (B) external literature read in full. Written incrementally.

---

## PART A — INTERNAL CODE AUDIT (file:line)

### A(a) — Slack "Approve" → "Missing API key for provider anthropic"

**Render site (the button):** `backend/slack_bot/governance.py:165-168`
```
"text": {"type": "plain_text", "text": "Approve"},
"value": "approve",
"action_id": "approval_approve"
```

**FINDING F-A1 (preliminary):** the string `Missing API key for provider`
does NOT appear anywhere in the repo source (only in handoff/masterplan
audit-basis text describing the incident). The litellm SDK string at
`.venv/.../litellm/main.py:5601` reads "Missing API key for **Volcengine**..."
— NOT the observed "...for provider \"anthropic\"". So the exact error
phrasing comes from a *different* code path (an Anthropic SDK wrapper or a
provider-router), to be pinned below.

**Correct llm client path:** `backend/agents/llm_client.py` (NOT
backend/services/llm_client.py — that file does not exist).

**Path the operator's "Approve" message takes:**
1. `commands.py:185 @app.message("") handle_any_message` — every non-bot
   message in `#ford-approvals` (`_APPROVAL_CHANNEL`) is ingested as a
   ticket (`ingest_slack_message`) and acknowledged. The literal word
   "Approve" matches NO special branch (`clear queue`/`status` only) — so
   it falls through to ticket creation + the assistant/orchestrator answer
   path.
2. The `approval_approve` / `approval_deny` **button** action_ids are
   RENDERED at `governance.py:166-175` but have **NO `@app.action()`
   handler** anywhere (grep of all `@app.action(...)` registrations shows
   only `app_home_*`, `agent_model_change_*`, `agent_feedback_*`). So a
   button click is a dead control; the operator typed the word instead.
3. The message-answer path routes through the assistant handler /
   orchestrator, which for non-DIRECT classifications calls the LLM via
   `make_client()` → because `paper_use_claude_code_route=True` (confirmed
   live) this constructs `ClaudeCodeClient` (the `claude` CLI subprocess
   rail), NOT direct Anthropic. ("Approve" is not in the
   `direct_responder.can_handle_directly` trigger list — `commands.py`/
   `direct_responder.py:33-46` — so it does not get the local no-LLM
   fast-path.)

**ROOT CAUSE (FINDING F-A1, HIGH confidence):** The error string
`Missing API key for provider "anthropic"` is NOT emitted by any repo or
venv code (exhaustive grep: only litellm's "Missing API key for
**Volcengine**" exists, and the ticket processor raises a *different*
string "ANTHROPIC_API_KEY not found in settings or environment"). It is
emitted by the **`claude` CLI binary itself**, invoked by
`claude_code_client.py:claude_code_invoke()`. That function deliberately
**scrubs `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the
subprocess env** (`claude_code_client.py:163-170`, phase-38.13.1) so the
CLI authenticates via `~/.claude/` OAuth (the Max-subscription flat-fee
rail) instead of billing api.anthropic.com. **When the `~/.claude/` OAuth
session is expired/absent (plausible during a 10-day unattended away
week), the CLI has neither an env key (scrubbed by design) nor a valid
OAuth token → it errors with a provider-named missing-key message.** The
CLI's non-zero exit is wrapped into `ClaudeCodeError` at
`claude_code_client.py:188-197`; whichever caller surfaced it to Slack
relayed the CLI's own stderr text (hence the exact phrasing not being in
our source).

Live state confirmed via settings introspection (no secret values
printed): `paper_use_claude_code_route = True`, `anthropic_api_key
is_set = True` (≥20 chars). So the env key EXISTS but is intentionally
withheld from the CLI subprocess; the failure is OAuth-session, not a
missing `.env` var. **GENERATE must verify `~/.claude/` token validity
and the CLI's current auth state as the primary check, and only secondarily
confirm the .env var name is present.** Do NOT "fix" by un-scrubbing
ANTHROPIC_API_KEY — that silently re-routes billing to the exhausted
direct-API account (the exact failure phase-38.13.1 guarded against; see
the hard-fail at `llm_client.py:1967-1976`).

**FAIL-OPEN vs FAIL-CLOSED determination (FINDING F-A2):** The approval
path **FAILS CLOSED** with respect to *trade execution* — and this is the
correct posture, though arrived at by accident rather than design:
- The thing the operator was approving is answered by an LLM. When the LLM
  rail is down, the assistant handler hits its `except Exception` →
  "Deterministic fallback" (`assistant_handler.py:~344+`), so the operator
  got an error message, NOT a silent success.
- Critically, typing "Approve" does **not** itself execute any trade. The
  autonomous trading loop (`autonomous_loop.py`) runs on its own scheduler
  and does its own buy/sell; the Slack approve flow is a
  human-acknowledgement/Q&A surface, not a trade-gating interlock. So the
  broken approve flow did **not** cause unintended trades — it caused
  *loss of operator oversight* (the human-on-the-loop could not get
  answers), which is a fail-closed-on-action / fail-open-on-observability
  split.
- SECURITY NOTE for the contract: there is a **latent fail-open** risk in
  the dead `approval_approve` button. Because no handler is registered, if
  a future feature wires the button to a privileged action without
  re-checking auth, a click would either no-op (current) or, worse, be
  added without the LLM-down guard. Flag as finding for phase-56, severity
  LOW (currently inert), but note the dead control explicitly.

---

_(continued: watchdog, scoring, BQ mapping below)_
