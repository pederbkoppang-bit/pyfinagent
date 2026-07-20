# Research Brief -- masterplan step 75.2

**Step:** Audit75 S2 -- Slack control-plane authorization + dead-plane removal
**Tier:** moderate (caller-specified)
**Researcher:** Layer-3 Researcher (merged external + internal)
**Started:** 2026-07-20
**Status:** COMPLETE -- gate_passed: true (8 sources read in full, 32 URLs, recency scan done)

## Topic

Slack bot control-plane authorization and safe dead-code removal:
- Slack Bolt `reaction_added` event authorization (verify `event['user']`, bind approval to a bot-posted message `ts`, fail-closed when operator id unset)
- Blocking subprocess (`git push`) off the asyncio event loop via `asyncio.to_thread` in Bolt async apps
- Pre-LLM deterministic intent refusal routing (model can never hallucinate a deploy)
- Per-user sliding-window rate limiting for chat assistants
- Append-only JSONL audit logging for privileged interactions
- Slack App Home block-action authorization
- Confused-deputy / fail-closed authorization at the SINK, not only at the matcher

## Search queries PLANNED (write-first skeleton; superseded by the next section)

| # | Variant | Query |
|---|---------|-------|
| 1 | current-year (2026) | slack bolt reaction_added authorization 2026 |
| 2 | last-2-year (2025) | slack bolt python async subprocess blocking event loop 2025 |
| 3 | year-less canonical | slack bolt reaction_added event payload user |
| 4 | year-less canonical | asyncio.to_thread blocking call event loop |
| 5 | year-less canonical | confused deputy problem authorization |
| 6 | current-year | sliding window rate limiting algorithm |
| 7 | year-less canonical | append-only audit log JSONL tamper evident |
| 8 | current-year | prompt injection deterministic guardrail pre-LLM routing 2026 |

## Search queries actually run (3-variant discipline)

| Variant | Query | Result |
|---------|-------|--------|
| current-year (2026) | `slack bolt reaction_added event authorization verify user 2026` | 10 URLs; surfaced Bolt authorization docs + Events API |
| year-less canonical | `confused deputy problem authorization check at sink not entry point` | 10 URLs; surfaced Wikipedia + AWS IAM canonical prior art AND the 2026 arXiv frontier paper |
| last-2-year (2025) | `slack bot security operator allowlist audit log privileged action 2025` | 10 URLs; Slack Audit Logs API + 2025 BotScope over-permissioning stats |

## Read in full (8; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.slack.dev/reference/events/reaction_added | 2026-07-20 | Official doc (tier 2) | WebFetch, full | Payload carries **`user`** (who reacted) vs **`item_user`** (who authored the reacted-to message) -- two different fields, and `item` carries `{type, channel, ts}`. Verbatim example: `{"type":"reaction_added","user":"U123ABC456","reaction":"thumbsup","item_user":"U222222222","item":{"type":"message","channel":"C123ABC456","ts":"1360782400.498405"},"event_ts":"..."}`. Scope: `reactions:read`. |
| 2 | https://docs.python.org/3/library/asyncio-task.html | 2026-07-20 | Official doc (tier 2) | WebFetch, full | `asyncio.to_thread(func, /, *args, **kwargs)`, added 3.9. "This coroutine function is primarily intended to be used for executing IO-bound functions/methods that would otherwise block the event loop if they were run in the main thread." `contextvars.Context` is propagated. GIL caveat applies to CPU-bound only. `loop.run_in_executor()` is the lower-level alternative. |
| 3 | https://docs.slack.dev/tools/bolt-python/concepts/authorization/ | 2026-07-20 | Official doc (tier 2) | WebFetch, full | **Bolt's `authorize` is installation-level, NOT per-user.** It answers "does this app have valid credentials for this workspace", never "may this user perform action X". `AuthorizeResult` carries `bot_token`/`user_token`, `bot_user_id`, `enterprise_id`, `team_id`. Per-user authorization "NOT handled by this mechanism; developers must implement custom logic separately". |
| 4 | https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html | 2026-07-20 | Standards (tier 1/2) | WebFetch, full | Four load-bearing rules: **deny by default**; **validate permissions on every request** ("an attacker only needs to find one way in"; validating "just the majority of requests is insufficient"); never rely on client-supplied data; log authorization decisions in a "consistent, well-defined, readily parseable format". |
| 5 | https://arxiv.org/html/2606.28679 | 2026-07-20 | Preprint / peer-review tier (tier 1) | WebFetch, full | *Capability Gates Are Not Authorization*. **Capability gating** ("does this tool exist in the menu?") is NOT **per-call authorization** ("is THIS call with THESE values allowed?"). Requires **complete mediation at the tool-call boundary** with a fail-closed PDP/PEP; policy must live "outside the model's context window, write path, and influence". "The implementation treats errors as denial. A policy engine that fails open on malformed input recreates the vulnerability." Audited LangChain/LlamaIndex/Stripe Agent Toolkit: all capability-gate, none per-call-authorize by default. |
| 6 | https://genai.owasp.org/llmrisk/llm01-prompt-injection/ | 2026-07-20 | Standards (tier 1/2) | WebFetch, full | LLM01:2025. "Provide the application with its own API tokens for extensible functionality, and **handle these functions in code rather than providing them to the model**." "Implement human-in-the-loop controls for privileged operations." Security gates must "operate outside the model through code-based controls rather than trusting model outputs". |
| 7 | https://blog.cloudflare.com/counting-things-a-lot-of-different-things/ | 2026-07-20 | Authoritative eng blog (tier 3) | WebFetch, full | Sliding-window approximation: `rate = previous_count * ((period - elapsed) / period) + current_count`. Worked example: `42 * ((60-15)/60) + 18 = 49.5`. Costs **two integers per counter** vs a sliding log's per-request timestamps. Measured over 400M requests: "0.003% of requests have been wrongly allowed or rate limited", ~6% mean deviation from true rate. |
| 8 | https://docs.slack.dev/surfaces/app-home/ | 2026-07-20 | Official doc (tier 2) | WebFetch, full | App Home is "a private, one-to-one space in Slack shared by a user and an app" -- per-user. Block-action payloads carry the acting user's id. **No built-in restriction on which workspace users may open App Home or fire its block actions.** `views.publish` targets a `user_id`. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://docs.slack.dev/apis/events-api/ | Official doc | Superseded by #1 for the payload question |
| https://api.slack.com/events-api | Official doc (legacy host) | Redirect of the above |
| https://tools.slack.dev/bolt-python/api-docs/slack_bolt/authorization/index.html | API reference | Autogenerated stubs; #3 covers the concept |
| https://github.com/slackapi/bolt-js/issues/680 | Community issue | JS-specific; project is Bolt-Python |
| https://tools.slack.dev/bolt-js/concepts/authorization/ | Official doc | JS variant of #3 |
| https://en.wikipedia.org/wiki/Confused_deputy_problem | Encyclopedia | Canonical prior-art marker (Hardy 1988); #5 supersedes with agent-era specifics |
| https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html | Official doc | Cross-service-role framing, not applicable to a Slack event sink |
| https://arxiv.org/pdf/2603.19469 | Preprint | *A Framework for Formalizing LLM Agent Security*; overlaps #5, budget |
| https://blog.quarkslab.com/agentic-ai-the-confused-deputy-problem.html | Industry blog | Tier 4; #5 covers with measurements |
| https://safeguard.sh/resources/blog/ai-agent-tool-confused-deputy-problem-2026 | Vendor blog | Tier 4/5 marketing-adjacent |
| https://kurums.com/confused-deputy-ai-security/ | Community blog | Tier 5 |
| https://earezki.com/ai-news/2026-06-15-ai-agents-expose-the-security-checks-you-never-actually-wrote/ | Community blog | Tier 5 |
| https://authfyre.com/blog/a-quick-primer-on-the-confused-deputy-problem-in-cybersecurity | Vendor blog | Tier 5 |
| https://docs.slack.dev/admins/audit-logs-api/ | Official doc | Enterprise Grid only -- **not available on this workspace tier**; recorded as a rejected alternative to local JSONL |
| https://slack.com/help/articles/360000394286-Audit-logs-in-Slack | Help centre | Same Enterprise-Grid limitation |
| https://api.slack.com/admins/audit-logs-call | API reference | Same |
| https://slack.engineering/slack-audit-logs-and-anomalies/ | Vendor eng blog | Anomaly detection at Slack's scale; not applicable to a single-operator local bot |
| https://slack.com/help/articles/360001670528-Security-recommendations-for-approving-apps | Help centre | Workspace-admin guidance, not app-code guidance |
| https://www.reco.ai/hub/securing-slack-enterprise-communication-compliance | Vendor | Tier 5 |
| https://moldstud.com/articles/p-comprehensive-guide-to-auditing-slack-bot-user-permissions-for-compliance | Community | Tier 5; source of the 2025 BotScope over-permissioning stat |
| https://github.com/jeremylongshore/claude-code-slack-channel | GitHub | Hash-chained Ed25519 audit journal for Slack+Claude governance -- interesting prior art, over-engineered for this step |
| https://www.questionbase.com/resources/blog/slack-events-api-acknowledgement-requirements-what-every-developer-needs-to-know | Vendor blog | Tier 5 |
| https://github.com/slackapi/python-slack-events-api | GitHub | Deprecated Flask adapter, not Bolt |
| https://docs.python.org/3/library/asyncio-eventloop.html | Official doc | `run_in_executor` reference; #2 covers the recommended API |

**URLs collected: 32 unique** (8 read in full + 24 snippet-only).

## Recency scan (last 2 years, 2024-2026)

**Performed.** Dedicated 2025- and 2026-scoped passes were run (queries 1 and 3
above) and the year-less canonical pass (query 2) was cross-checked for
frontier hits. Result: **3 new findings that supersede or materially extend the
canonical sources.**

1. **arXiv:2606.28679 (2026) supersedes the canonical Hardy-1988 confused-deputy
   framing for this exact problem class.** The classical formulation is about
   ambient authority in compilers/OS capabilities; the 2026 paper reframes it for
   event- and model-driven sinks and supplies the operative rule for 75.2:
   a *matcher* is a capability gate, not an authorization decision, so the check
   must be repeated **at the sink**. This is precisely gap1-11's thesis and
   independently validates it.
2. **OWASP LLM01 moved to the 2025 revision** and hardened the language from
   "sanitize input" toward "handle these functions in code rather than providing
   them to the model" plus explicit human-in-the-loop for privileged operations.
   This is the literature basis for 75.2(b)'s pre-LLM deterministic refusal:
   the 2025 text says the control belongs in code *outside* the model, which is
   stronger than a prompt-level instruction.
3. **2025 Slack-ecosystem telemetry** reports >18% of bots with excessive
   message-read rights in mid-sized orgs and >45% of collaboration-platform
   incidents tracing to unchecked automation access (BotScope 2025, via
   MoldStud). Directional/vendor-sourced, so weighted low, but it corroborates
   that an ungated bot-side privileged sink is a realistic rather than
   theoretical exposure.

No new finding contradicts the older canonical guidance; `asyncio.to_thread`
(3.9, 2020) and the Cloudflare sliding-window formula (2017) remain current
best practice with no 2024-2026 supersession found.

## Key findings

**F1 -- Slack gives you the acting user for free; the handler must simply read
it.** `event['user']` is the reactor, `event['item_user']` is the message
author (Source: Slack `reaction_added` reference, accessed 2026-07-20). The
current handler reads neither. Using `item_user` instead of `user` would be a
subtle and severe inversion -- the implementer must gate on **`event['user']`**.

**F2 -- Bolt will not do per-user authorization for you.** "Authorization
operates entirely at the app installation level... developers must implement
custom logic separately to check individual user permissions" (Bolt-Python
authorization concept doc, accessed 2026-07-20). There is no framework-level
`require_operator` to lean on; the check is hand-rolled by definition.

**F3 -- A matcher is a capability gate; the sink still needs its own check.**
"Capability gating is static and asks: does this tool exist in the agent's
menu... per-call authorization is dynamic and asks: is *this specific call*
with *these argument values* allowed in this context?" (arXiv:2606.28679).
This is the direct literature warrant for gap1-11: `is_operator_token_message`
is the gate, `append_operator_token` is the sink, and today the sink trusts
its caller.

**F4 -- Fail-closed must include the error path, not just the unset path.**
"The implementation treats errors as denial. A policy engine that fails open on
malformed input recreates the vulnerability" (arXiv:2606.28679). Applies to
75.2 concretely: a missing `event['user']`, a `None` channel, or an exception
while resolving the pending-approval ts must all deny, never proceed.

**F5 -- Deterministic controls must sit outside the model.** OWASP LLM01:2025:
"handle these functions in code rather than providing them to the model" +
"implement human-in-the-loop controls for privileged operations". A prompt
telling the assistant "never deploy" is not a control; a branch that returns
before `_classify_via_llm` is.

**F6 -- Validate on every request, deny by default.** "An attacker only needs to
find one way in"; validating "just the majority of requests is insufficient"
(OWASP Authorization Cheat Sheet). Supports gating *all four*
`agent_model_change_*` actions rather than the one the audit happened to anchor.

**F7 -- Sliding window costs two integers and is accurate to ~0.003%.**
`rate = previous_count * ((period - elapsed) / period) + current_count`
(Cloudflare, 400M-request measurement). No Redis needed for a single-operator
bot -- an in-process dict of two counters per user is sufficient and matches
the measured accuracy.

**F8 -- App Home has no built-in access restriction.** "No built-in restrictions
preventing users... from accessing the App Home or triggering block actions"
(Slack App Home surface doc). Any workspace member who can see the app can open
its Home tab and fire `agent_model_change_*`, which today mutates
`AGENT_CONFIGS[agent_type].model` process-wide.

## Internal code inventory

### A. The vulnerable sink -- `reaction_added` (gap1-01)

`backend/slack_bot/commands.py:328-355` (verbatim structure):

```python
@app.event("reaction_added")                       # :328
async def handle_reaction(event, say):             # :329
    reaction = event.get("reaction", "")           # :331
    item = event.get("item", {})                   # :332
    channel = item.get("channel", "")              # :333
    if channel != _APPROVAL_CHANNEL:               # :335  <-- ONLY gate
        return
    if reaction == "white_check_mark":             # :338
        result = subprocess.check_output(          # :342  BLOCKING, 30s timeout
            ["git", "push", "origin", "main"],     # :343
            cwd=str(_PROJECT_ROOT), text=True, timeout=30,
            stderr=subprocess.STDOUT)
```

Line-anchored defects:

| # | Anchor | Defect |
|---|--------|--------|
| 1 | `commands.py:331-336` | `event['user']` is **never read**. ANY member of `C0ANTGNNK8D` (`_APPROVAL_CHANNEL`, `commands.py:25`) who adds `:white_check_mark:` triggers `git push origin main`. |
| 2 | `commands.py:332-336` | `item.ts` is never read. The reaction may be on ANY message in the channel -- an old ticket, a bot ack, the reactor's own message. There is no "pending push approval" message concept anywhere in the module. |
| 3 | `commands.py:342-346` | `subprocess.check_output(..., timeout=30)` is a **blocking** call inside an `async def` Bolt handler -> stalls the whole Socket-Mode event loop for up to 30 s (all other handlers, scheduler jobs, ticket acks). |
| 4 | `commands.py:338` | No fail-closed branch when `slack_operator_user_id` is unset -- contrast `operator_tokens.py:87-88` which returns `False` on an empty operator id. |
| 5 | `commands.py:347,349,351,355` | `say()` with no `thread_ts` -- the push result lands as a channel-level message, not threaded to the approval. |

### B. The correct in-repo precedent -- `operator_tokens.py`

`backend/slack_bot/operator_tokens.py:79-95` `is_operator_token_message()` is the
pattern the reaction handler must mirror:

```python
if not operator_user_id:          # :87
    return False                  # :88  <-- fail-closed on unset operator
if message.get("bot_id"):         # :89-90
    return False
if message.get("user") != operator_user_id:   # :91-92
    return False
if message.get("channel") not in allowed_channels:  # :93-94
    return False
```

`append_operator_token()` at `operator_tokens.py:98-132` is the **sink** and it
does NOT repeat any of those checks (gap1-11): its signature takes
`text, user, channel, ts, event_id` but validates only `parse_operator_token(text)`
(`:107-108`). The docstring at `:104-106` even claims "append is the last line of
defense for file integrity" -- it is a defense for *parseability*, not *authorization*.
A future caller that forgets the matcher writes an unauthenticated record.

Wiring: `commands.py:106-115` -- `_operator_token_matcher` closes over
`_settings.slack_operator_user_id` and `_token_channels` (`commands.py:102-104`,
built from `slack_channel_id` + `_APPROVAL_CHANNEL`).

JSONL writer convention to copy (`operator_tokens.py:111-132`): `asyncio.Lock`
(`:63`, `:111`), single `json.dumps(..., ensure_ascii=False)` line (`:125`),
`mkdir(parents=True, exist_ok=True)` (`:124`), append-mode `open(..., "a")` (`:126`),
process-lifetime dedupe set (`:64`, `:110`, `:128`). Record keys: `ts` (UTC
isoformat), `user`, `channel`, `slack_ts`, `event_id`, `raw` (`:116-122`).

### C. The LIVE assistant path -- where the pre-LLM refusal must sit

Registration chain proving what is live (`app.py:32-34`):
`register_commands` (commands.py) + `register_assistant_lifecycle`
(assistant_lifecycle.py) + `register_governance` (app_home.py). **Nothing else.**

`assistant_lifecycle.py:193-198` `@assistant.user_message` ->
`handle_user_message` (`:115`) -> `:151-153` imports and awaits
`streaming_integration.handle_user_message_with_streaming`. That is the only
live assistant entry point.

Inside `streaming_integration.py::handle_user_message_with_streaming` (`:71-148`):

| Anchor | What happens |
|--------|--------------|
| `:92-96` | `message`/`channel_id`/`thread_ts`/`user_id`/`user_text` extracted |
| `:98-99` | empty-text early return -- **the natural insertion point for the deterministic refusal is immediately after this** |
| `:104` | `orchestrator = get_orchestrator()` |
| `:105` | `classification = classify_trivial(user_text)` -- first classification |
| `:108` | `await orchestrator._classify_via_llm(user_text)` -- **LLM classification**; the refusal branch MUST precede this line |
| `:122-124` | DIRECT branch |
| `:128-133` | COMPLEX branch |
| `:136-139` | SIMPLE/MODERATE branch |

Absent on this path (gap1-05): **no rate limiting of any kind** and **no audit
log**. `grep -n 'rate\|limit\|budget\|audit' streaming_integration.py` returns
only unrelated hits. Existing defense on this path is output-side only:
`scrub_leaks` (`:411-425`) + `detect_llm_leak` (`:473-497`) +
`apply_leak_defenses` (`:500-517`) -- and note `apply_leak_defenses` is defined
but **never called** anywhere in the live flow (`_stream_simple_response` at
`:151-190` streams `full` raw at `:184-185`). That is an independent latent gap
worth flagging to the implementer, though it is outside 75.2's scope.

### D. The dead control plane (gap1-03/04/08/09/10) -- importer proof

`grep -rn 'slack_bot import <m>|slack_bot\.<m>|from \.<m> import'` across the
whole repo, excluding `__pycache__`, `.git/`, and the module's own file:

| Module | Lines | Live importers | Only references |
|--------|-------|----------------|-----------------|
| `self_update.py` | 467 | **0** | `assistant_handler.py:238` (itself dead) + audit JSON |
| `assistant_handler.py` | 785 | **0** | `handoff/harness_log.md:6353` + `.claude/.masterplan.json.bak.*` (stale backups, an inspect-source check on a since-superseded step) |
| `governance.py` | 315 | **0** | `assistant_handler.py:200,361,743,751` (itself dead) + audit JSON |
| `mcp_tools.py` | 247 | **0** | none |
| `streaming_handler.py` | 243 | **0** | none |
| `context_management.py` | 249 | **0** | none |
| **Total** | **2,306** | | |

Note the step spec estimates ~2,900 lines; the measured `wc -l` total is
**2,306**. The implementer should use the measured figure.

**Namespace hazard confirmed:** `backend/governance/` is a DIFFERENT package
from `backend/slack_bot/governance.py`. A careless
`grep -rn 'from backend.governance'` will hit the live package. Only the
`backend.slack_bot.governance` dotted path is dead.

**Dangling-import hazard (verified directly, stronger than the register claims):**
`assistant_handler.py:200-203` imports `AuditRecord, get_audit_logger,
get_token_tracker, classify_error, get_fallback_message` from
`slack_bot.governance`. A direct symbol scan of `governance.py` shows it defines
only `AuditLog` (`:23`), `AuditLogger` (`:58`), `HumanInTheLoopManager` (`:129`),
`ContentDisclaimer` (`:219`), `RateLimiter` (`:244`). **None of the five imported
names exist** -- so that import would raise `ImportError` if the branch were ever
reached. The dead code is not merely unused, it is non-functional. Deleting both
files together resolves it; do not attempt to repair it.

**Salvage note:** `governance.py:244` contains a `RateLimiter` class -- a dead
implementation of exactly what gap1-05 asks for on the live path. Worth reading
(via `git show HEAD:backend/slack_bot/governance.py`) as prior art before writing
the new sliding-window limiter, though the Cloudflare formula in F7 is the
recommended algorithm regardless.

**Audit-register locations:** `handoff/current/audit_phase75/register.md:38`
carries gap1-01 ("Any user's white_check_mark reaction on ANY message in
#ford-approvals executes git push origin main") and `:40` carries gap1-04
("Deploy plane has zero caller authorization"). The remaining gap1-* findings
(03/05/07/08/09/10/11) live in
`handoff/current/audit_phase75/confirmed_findings.json`, not in register.md --
the implementer should read the JSON for the full evidence strings.

**Stale verification-command hazard:** the `.masterplan.json.bak.*` files
reference `from backend.slack_bot import assistant_handler` inside an old
immutable verification command. Those are `.bak` backups, not the live
`.claude/masterplan.json` -- the implementer must confirm the LIVE masterplan
has no such command before deleting (see check list below).

## BLOCKERS the step spec does not mention (read this first)

### BLOCKER-1 -- deleting the six modules breaks THREE done steps' immutable verification commands

`.claude/masterplan.json` is scanned by dotted path below. CLAUDE.md forbids
amending immutable verification criteria ("Never edit verification criteria in
masterplan.json -- they are immutable"; "Amend a step's immutable verification
criteria" is in the Never-do list). These are all `status: done`:

| Path | step_id | status | Command fragment | Breaks how |
|------|---------|--------|------------------|-----------|
| `phases[26].steps[4].verification.command` | **4.14.4** | done | `from backend.slack_bot import assistant_handler` | `ImportError` -> non-zero exit |
| `phases[26].steps[23].verification.command` | **4.14.24** | done | `grep -n 'is_harmful\|harmlessness' backend/slack_bot/assistant_handler.py \| wc -l \| awk '{exit ($1<1)}'` | grep on a missing file -> count 0 -> `awk` exits 1 |
| `phases[29].steps[8].verification.*` | **4.17.9** | done | `python scripts/go_live_drills/self_update_audit_test.py` + `success_criteria[0]` mentions self_update | script named in the command **does not exist** (only `smoke_test_4_17_9.py` is on disk) -- so this command is **already broken today**, independent of 75.2 |

Recommended resolution (do NOT silently edit the criteria): treat these as
*historical* verifications of code that this step intentionally retires, and
record that fact verbatim in `experiment_results.md` + `live_check_75.2.md`
with the dotted paths above. 4.17.9 is already unrunnable pre-change, which is
itself worth reporting. Escalate to the operator rather than rewriting the
immutable fields.

### BLOCKER-2 -- a path-asserting smoke script hard-fails on deletion

`scripts/go_live_drills/smoke_test_4_17_9.py:33-34`:

```python
su = REPO_ROOT / "backend/slack_bot/self_update.py"
assert su.exists(), f"{su} missing"
```

This raises `AssertionError` the moment `self_update.py` is deleted. It must be
retired or guarded in the same commit.

**Contrast (safe, no action needed):** `scripts/qa/sweep_ascii_logger_v3.py`
lists `assistant_handler.py` (`:37`) and `self_update.py` (`:40`) in
`candidate_files`, but `:54-56` reads `for f in candidate_files: if not
f.exists(): continue` -- it **skips** missing files. Removing the two stale
entries is tidy but not required.

### BLOCKER-3 -- 75.2's own immutable command will fail on a "deleted in 75.2" comment

The step's own verification command concatenates **every** `backend/slack_bot/*.py`
and asserts:

```python
txt = ''.join(open(p).read() for p in glob.glob('backend/slack_bot/*.py'))
assert all(('slack_bot.%s' % d not in txt and 'import %s' % d not in txt) for d in dead)
```

`dead` includes the bare tokens `governance`, `mcp_tools`, `context_management`,
etc. So a perfectly natural tombstone comment such as
`# self_update.py deleted in phase-75.2` is fine (it contains neither
`slack_bot.self_update` nor `import self_update`), but
`# was: from backend.slack_bot.governance import ...` **fails the gate**.
Instruction to the implementer: **write no tombstone comment that contains a
dotted `slack_bot.<dead>` path or the literal `import <dead>`.** Put the
rationale in `experiment_results.md`, not in the source.

Verified-safe existing strings: `app.py:19` `from backend.slack_bot.app_home
import register_governance` and `app_home.py:342` `def register_governance` do
**not** contain `import governance` (they contain `import register_governance`)
or `slack_bot.governance`. No false positive today.

### BLOCKER-4 -- the deterministic command is weaker than the success criteria

The command only substring-checks for `slack_operator_user_id`, `to_thread`,
`deploy commands are disabled`, `assistant_audit`, `operator_user_id`. It does
**not** verify the `item.ts` binding, the fail-closed-when-unset behaviour, the
rate limiter, or that the refusal precedes the LLM call. Success criteria 1, 3,
4 and 6 therefore need **unit tests** as evidence, not just the command's exit
code. Q/A will look for them; `backend/tests/test_phase_62_2_operator_tokens.py`
is the pattern to copy (it already exercises the fail-closed-unset case at
`:75`).

### BLOCKER-5 -- gap1-11 is a breaking signature change with an existing test suite

`backend/tests/test_phase_62_2_operator_tokens.py` calls
`append_operator_token(...)` at `:81, :95, :97, :104, :106, :113, :121` -- seven
call sites, all keyword-style. Adding required `operator_user_id` +
`allowed_channels` parameters breaks all seven. Give them **defaults that
fail-closed** (e.g. `operator_user_id: str = ""`, `allowed_channels:
set[str] | None = None`) only if `""`/`None` means *deny*, matching
`operator_tokens.py:87-88`; otherwise update all seven call sites. Do not add a
permissive default.

## Application to pyfinagent

### (a) gap1-01 -- the reaction sink

Replace `commands.py:328-355`. Canonical shape, grounded in F1/F2/F4:

```python
# module scope, next to _APPROVAL_CHANNEL (commands.py:25)
_pending_push_ts: set[str] = set()      # bot-posted approval-request ts values

@app.event("reaction_added")
async def handle_reaction(event, say, logger):
    settings = get_settings()
    operator = settings.slack_operator_user_id
    if not operator:                                   # F4: unset -> deny
        logger.warning("reaction ignored: slack_operator_user_id unset (fail-closed)")
        return
    if event.get("user") != operator:                  # F1: `user`, NOT `item_user`
        logger.warning("reaction ignored: non-operator %s", event.get("user"))
        return
    item = event.get("item") or {}
    if item.get("channel") != _APPROVAL_CHANNEL:
        return
    ts = item.get("ts")
    if not ts or ts not in _pending_push_ts:           # bind to a bot-posted request
        logger.warning("reaction ignored: ts %s is not a pending push approval", ts)
        return
    if event.get("reaction") == "white_check_mark":
        _pending_push_ts.discard(ts)                   # single-use
        try:
            result = await asyncio.to_thread(          # F: doc source #2
                subprocess.check_output,
                ["git", "push", "origin", "main"],
                cwd=str(_PROJECT_ROOT), text=True, timeout=30,
                stderr=subprocess.STDOUT,
            )
            await say(text=f"Pushed to GitHub\n```{result.strip()}```", thread_ts=ts)
        except subprocess.CalledProcessError as e:
            await say(text=f"Push failed\n```{e.output[:500]}```", thread_ts=ts)
    elif event.get("reaction") == "x":
        _pending_push_ts.discard(ts)
        await say(text="Push rejected. Commits stay local.", thread_ts=ts)
```

Notes for the implementer:
- `asyncio` is **already imported** at `commands.py:7`; `subprocess` at `:9`.
  No new imports needed for the `to_thread` change.
- `asyncio.to_thread` passes `*args`/`**kwargs` straight through (doc source #2),
  so `subprocess.check_output` with its kwargs works unmodified. `timeout=30`
  is `check_output`'s own timeout and still applies inside the worker thread.
- `_pending_push_ts` must be **populated** wherever the bot posts a push-approval
  request. Today **no such poster exists** in `commands.py` -- the reaction
  handler was written against an imagined message. The implementer must either
  add the poster (capture `(await say(...))["ts"]`) or, minimally, expose an
  internal helper that registers a ts, and document that an empty
  `_pending_push_ts` means *every* reaction is denied (correct fail-closed
  default, and satisfies criterion 1 on day one).
- `_pending_push_ts` is process-local and resets on restart -- same caveat class
  as the App Home control in (d). Say so in the log/UX text.
- Single-use `discard` prevents replay of one approval into repeated pushes.

### (b) gap1-03/04/08/09/10 -- deletion + pre-LLM refusal

Delete the six files (measured **2,306** lines, not ~2,900). Insert the refusal
in `streaming_integration.py` **between `:99` and `:104`** -- after the
empty-text return, before `get_orchestrator()` and well before
`_classify_via_llm` at `:108`:

```python
_DEPLOY_VERBS = ("deploy update", "deploy rollback", "deploy status",
                 "deploy cleanup", "deploy pull", "update bot", "pull and restart")

low = user_text.lower()
if any(v in low for v in _DEPLOY_VERBS):
    await say("deploy commands are disabled -- see docs/runbooks/away-ops-rules.md")
    _audit(user_id, channel_id, user_text, outcome="refused_deploy")
    return
```

The literal `deploy commands are disabled` is asserted by the step's own
verification command -- keep it byte-exact and lowercase. The verb list should
cover the strings the deleted `self_update.handle_deploy_command` matched
(`self_update.py:436,444-445` per the audit register: `deploy update`,
`deploy pull`, `update bot`, `pull and restart`).

### (c) gap1-05 -- rate limit + audit on the live path

Sliding window per F7, two ints per user, no external store:

```python
_WINDOW_S, _MAX_PER_WINDOW = 60.0, 20
_rl: dict[str, tuple[float, int, int]] = {}   # user -> (window_start, prev, cur)

def _rate_ok(user_id: str, now: float) -> bool:
    start, prev, cur = _rl.get(user_id, (now, 0, 0))
    elapsed = now - start
    if elapsed >= _WINDOW_S:
        prev, cur, start = (cur if elapsed < 2 * _WINDOW_S else 0), 0, now
        elapsed = 0.0
    rate = prev * ((_WINDOW_S - elapsed) / _WINDOW_S) + cur      # Cloudflare formula
    if rate >= _MAX_PER_WINDOW:
        _rl[user_id] = (start, prev, cur)
        return False
    _rl[user_id] = (start, prev, cur + 1)
    return True
```

Audit writer -- copy the `operator_tokens.py:111-132` idiom exactly
(`asyncio.Lock`, `mkdir(parents=True, exist_ok=True)`, one
`json.dumps(..., ensure_ascii=False)` line, append mode). Record shape mirroring
`operator_tokens.py:116-122`:

```python
{"ts": <utc isoformat>, "writer": "assistant_audit", "user": user_id,
 "channel": channel_id, "slack_ts": thread_ts, "text_sha256": <hash>,
 "outcome": "ok"|"refused_deploy"|"rate_limited", "agent": <classification>}
```

**Design note the implementer must decide consciously:**
`handoff/logs/` is **gitignored** (`.gitignore:72`), so
`handoff/logs/assistant_audit.jsonl` is local-only and will not survive as a
git-tracked artifact -- unlike `handoff/operator_tokens.jsonl` (tracked; lives
at `handoff/` root) and `handoff/audit/*.jsonl` (tracked). The step spec
explicitly names the gitignored path and success criterion 4 says
"(gitignored logs dir)", so this is **intentional** -- presumably because
records contain raw user message text. Recommend hashing the message
(`text_sha256`) rather than storing it verbatim, which preserves the privacy
rationale while making a future promotion to a tracked path safe. Note that
Slack's own Audit Logs API is **Enterprise Grid only** and therefore not an
option here (see snippet-only table) -- local JSONL is the right call.

### (d) gap1-07 -- App Home actions

All four actions funnel through one helper, `_handle_model_change` at
`app_home.py:378-395`, and the mutation is `app_home.py:391`
(`AGENT_CONFIGS[agent_type].model = selected`). Gate **once, inside the
helper** -- that covers all four registrations (`:397, :401, :405, :409`) and
satisfies F6 without four copies:

```python
async def _handle_model_change(ack, body, client, agent_type_str):
    await ack()
    user_id = body["user"]["id"]
    operator = get_settings().slack_operator_user_id
    if not operator or user_id != operator:            # F4 + F8
        logger.warning("app_home model change denied for %s", user_id)
        _audit(user_id, None, agent_type_str, outcome="denied_model_change")
        await update_app_home(client=client, event={"user": user_id}, logger=logger)
        return
    ...
```

`await ack()` must stay **first** (Slack requires acknowledgement within 3 s
regardless of the authorization outcome) -- deny *after* ack, never by
withholding it. The UI label "process-local, resets on restart" is accurate:
`AGENT_CONFIGS` is an in-memory dict, and the select is rendered at
`app_home.py:202` (`action_id: f"agent_model_change_{agent_value}"`).
`get_settings` is not currently imported in `app_home.py` -- add it.

### (e) gap1-11 -- repeat the check at the sink

Per F3, add the identity/channel check inside `append_operator_token`
(`operator_tokens.py:98-132`), fail-closed, returning `None` + a warning, so a
future caller that skips `is_operator_token_message` cannot write. Reuse the
existing predicate rather than duplicating logic -- construct the same dict
shape the matcher expects, or factor the four checks in
`operator_tokens.py:87-95` into a small `_authorized(user, channel, operator,
allowed)` helper called by both. The matcher must keep returning `False`
(not raise) so Bolt still falls through to ticket ingestion
(`operator_tokens.py:84-86` documents this contract -- preserve it).

## Pitfalls (from literature + repo)

1. **`item_user` vs `user`** (source #1). Gating on `item_user` authorizes the
   *author* of the reacted-to message, not the reactor -- an attacker reacts to
   a message the operator wrote and passes. Use `event['user']`.
2. **Fail-open on error** (source #5, F4). Wrap the ts lookup and settings read
   so any exception denies. Do not `try/except: pass` around the gate.
3. **Assuming the framework authorizes** (source #3, F2). Bolt's `authorize`
   proves the *app* is installed, never that the *user* is the operator.
4. **Prompt-level "never deploy"** (source #6, F5). The refusal must be a code
   branch before `_classify_via_llm` (`streaming_integration.py:108`), not an
   instruction in a system prompt.
5. **Blocking the Socket-Mode loop.** `subprocess.check_output` in an `async def`
   stalls every other handler; `asyncio.to_thread` is the documented fix
   (source #2). Note `_read_status()` at `commands.py:60-63` has the **same
   defect** (a 5 s blocking `git log`) and `:74-76` a blocking
   `urllib.request.urlopen` -- out of 75.2's stated scope, but worth a one-line
   note to the operator since the fix is identical.
6. **Fixed-window burst** (source #7). A naive per-minute counter allows 2x the
   limit across a boundary; use the weighted formula.
7. **Deleting a module whose names are already broken.** `assistant_handler.py:200-203`
   imports names from `slack_bot.governance` that per the audit register do not
   all exist -- do not try to "fix" the dead code, delete both together.
8. **`backend/governance/` is a different, LIVE package.** Only
   `backend.slack_bot.governance` is dead. Scope every grep to the
   `slack_bot` path.
9. **Withholding `ack()` as a denial mechanism.** Slack retries unacknowledged
   interactions; always ack, then deny.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total -- **32** (8 full + 24 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, dedicated section
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] 3 search-query variants (current-year / last-2-year / year-less canonical) -- table above

Soft checks:
- [x] Internal exploration covered every module named in the spec, plus the
      consumer sweep (tests/, scripts/, masterplan) that surfaced BLOCKER-1/2/5
- [x] Contradictions / consensus noted (all 8 sources agree that authorization
      must be explicit, at the sink, and fail-closed; no adversarial source found
      -- this is settled guidance, not a live debate)
- [x] Claims cited per-claim with source number + access date

Source-quality mix: 1 preprint (tier 1), 2 standards bodies (tier 1/2),
4 official vendor docs (tier 2), 1 authoritative engineering blog (tier 3).
Zero community-tier sources in the read-in-full set.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 24,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "Eight sources read in full establish that Bolt performs no per-user authorization (installation-level only), that event['user'] not item_user identifies the reactor, that a matcher is a capability gate rather than an authorization decision so the check must repeat at the sink and fail closed on error (arXiv:2606.28679), that privileged operations must be gated by code outside the model (OWASP LLM01:2025), and that a two-integer Cloudflare sliding window is accurate to 0.003%. Internally, commands.py:328-355 reads neither event['user'] nor item.ts and blocks the event loop up to 30s on git push; all six delete-candidates have zero live importers (2,306 lines measured, not 2,900). Five blockers the step spec omits: deletion breaks three done steps' immutable verification commands (4.14.4, 4.14.24, 4.17.9 -- the last already broken), smoke_test_4_17_9.py:34 hard-asserts self_update.py exists, 75.2's own command fails on a tombstone comment containing a dotted dead path, the command is weaker than criteria 1/3/4/6 so unit tests are required, and gap1-11 breaks seven existing test call sites.",
  "brief_path": "handoff/current/research_brief_75.2.md",
  "gate_passed": true
}
```
