# Research Brief -- masterplan step 75.2.1

**Tier:** moderate (caller-specified)
**Date accessed (all URLs):** 2026-07-20
**Researcher:** Layer-3 Researcher (merged external-literature + internal-code)
**Status:** COMPLETE -- gate PASSED

## Topic

Human-in-the-loop approval for privileged git operations from chat: binding an
approval to a specific bot-posted message and showing the reviewer exactly what
they approve (WYSIWYS / informed consent); Slack Bolt async `@app.message` vs
slash commands (which requires app-config); running git subprocesses off the
asyncio event loop; single-use approval tokens + replay prevention; and
recording superseded/retired verification steps in an append-only plan without
rewriting immutable history.

## Search queries run (3-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | current-year (2026) | `Slack Bolt approval workflow reaction_added privileged action operator gate 2026` |
| 2 | current-year (2026) | `agent git push approval chat bot security replay confirmation binding message 2026` |
| 3 | last-2-year (2025) | `human-in-the-loop approval AI agent privileged action confirmation 2025` |
| 4 | year-less canonical | `what you see is what you sign transaction confirmation signing attack` |
| 5 | year-less canonical | `architecture decision record superseded status convention immutable append-only` |

Mix confirms the discipline: #1/#2 surfaced 2026 Bolt-v5 + live advisories, #3
the 2025 HITL practice literature, #4/#5 the canonical prior art (OWASP
Transaction Authorization, ADR supersession) that year-locked queries missed.

## Read in full (8; gate floor is 5)

| # | URL | Kind | Tier | Key finding |
|---|-----|------|------|-------------|
| 1 | https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html | official security standard | 2 | The canonical rule set for binding an approval to a transaction; rules 1.1, 1.5, 2.4, 2.6, 2.8, 2.9, 2.10 all bear directly on part (b). |
| 2 | https://docs.python.org/3/library/asyncio-task.html | official docs | 2 | `asyncio.to_thread` semantics, contextvar propagation, GIL + cancellation caveats. |
| 3 | https://docs.slack.dev/reference/events/reaction_added | official docs | 2 | Exact `reaction_added` payload; `reactions:read` scope; `item_user` may be absent. |
| 4 | https://learn.microsoft.com/en-us/azure/well-architected/architect-role/architecture-decision-record | official docs | 2 | Append-only ADR doctrine, verbatim: never edit accepted records; supersede + link. |
| 5 | https://docs.slack.dev/tools/bolt-python/concepts/message-listening/ | official docs | 2 | `@app.message` accepts `str` or `re.Pattern`; matcher mechanism. (Thin page -- scope/manifest details absent; noted as a gap.) |
| 6 | https://docs.slack.dev/interactivity/implementing-slash-commands | official docs | 2 | Slash commands must be declared **per command in the App Management dashboard** with a Request URL + 3000 ms ack. |
| 7 | https://github.com/NousResearch/hermes-agent/issues/36848 | industry security report | 4 | The exact anti-pattern: `if allowed_csv:` wrapping the auth block -> unset env var = fail-OPEN; any channel member can click Approve. |
| 8 | https://advisories.gitlab.com/npm/openclaw/GHSA-wv26-j37q-2g7p/ | advisory DB (CWE-863) | 2 | Approval-**scope** confusion: an exec-approver could resolve *plugin* approvals. Fixed 2026.5.12. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://link.springer.com/chapter/10.1007/978-3-030-29959-0_21 | peer-reviewed (WYSIWYS user confusion) | Paywalled chapter; abstract-level only -- would not count as read-in-full |
| https://docs.slack.dev/changelog/2026/07/15/bolt-js-release/ | official changelog | Bolt **JS** v5, not Python; no bearing on this step |
| https://github.com/NousResearch/hermes-agent/issues/10583 | feature request | Corroborates "reaction as approval is unenforceable without routing"; superseded by #36848 |
| https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html | official | Adjacent (nonce/replay) but not chat-approval specific |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/architectural-decision-records/adr-process.html | official docs | Duplicates #4's supersession doctrine |
| https://ctaverna.github.io/adr/ | community | Same doctrine, lower tier than #4 |
| https://csse6400.uqcloud.net/handouts/adr.pdf | lecture notes | "only the STATUS paragraph evolves" -- corroborates #4 |
| https://squiglos.com/architecture-ledger | vendor | Append-only decision ledger marketing |
| https://hidekazu-konishi.com/entry/architecture_decision_records_templates_and_operations.html | community | ADR templates |
| https://www.strata.io/blog/agentic-identity/practicing-the-human-in-the-loop/ | vendor blog | Source of the "confirmation fatigue" framing |
| https://truto.one/blog/implementing-human-in-the-loop-approval-workflows-for-consequential-saas-api-actions/ | vendor blog | HITL approval patterns for SaaS actions |
| https://hoop.dev/blog/how-to-keep-human-in-the-loop-ai-control-and-ai-privilege-escalation-prevention-secure-and-compliant-with-action-level-approvals | vendor blog | Action-level approval granularity |
| https://galileo.ai/blog/human-in-the-loop-agent-oversight | vendor blog | HITL oversight checkpoints |
| https://github.github.com/gh-aw/patterns/chat-ops/ | official (GitHub) | ChatOps approval patterns; not Slack-Bolt specific |
| https://github.blog/changelog/2026-06-11-bot-created-pull-requests-can-run-workflows-if-approved/ | official changelog | Bot PRs require human approval to run CI -- same doctrine, different surface |
| https://noma.security/blog/gitlost-how-we-tricked-githubs-ai-agent-into-leaking-private-repos/ | security research | Prompt-injection into an agent's git surface (2026-07) |
| https://www.theregister.com/security/2026/07/07/github-ai-agent-leaks-private-repos-when-asked-nicely/ | press | Coverage of the above |
| https://www.onespan.com/cybersecurity/solutions/transaction-authorization | vendor | WYSIWYS in banking |
| https://www.ledger.com/academy/cryptos-greatest-weakness-blind-signing-explained | vendor | "Blind signing" = the failure mode this step must avoid |

**URLs collected: 27** (8 read in full + 19 snippet-only).

## Recency scan (last 2 years, 2024-2026)

**Performed.** Result: **4 new findings** that complement -- and one that
sharpens -- the canonical sources.

1. **`GHSA-wv26-j37q-2g7p` (fixed 2026-05-12)** -- approval-*scope* confusion is
   a live CWE-863 class, not theory: a user authorized for one approval gate
   resolved a *different* gate's actions. Sharpens the canonical OWASP rule 2.2
   (no downgrade/substitution of the authorization method).
2. **hermes-agent #36848 (2026)** -- the fail-open-when-unconfigured pattern in a
   Slack Block-Kit approval handler. This is precisely the bug pyfinagent's
   phase-75.2 already fixed; it validates the fail-closed choice at
   `commands.py:352-357` rather than superseding it.
3. **2025 HITL practice literature** -- consensus on "hybrid autonomy" (classify
   actions by risk; synchronous human approval for irreversible actions) plus a
   named failure mode: **confirmation fatigue** -- "when users are bombarded with
   approval requests, they stop reading the payloads and blindly click Approve."
   Directly motivates keeping the push-approval message rare and information-dense.
4. **2026 agent-git-security work** (GitLost / GitHub bot-PR approval changelog)
   -- the industry is converging on exactly this step's shape: a confirmation
   that *shows a summary of what is being pushed and where*, plus binding the
   decision to the result with idempotency keys / external event IDs.

Nothing in the window supersedes OWASP Transaction Authorization or the ADR
append-only doctrine; both remain the canonical references.

## Key findings (external)

1. **WYSIWYS is a hard requirement, not a nicety.** OWASP 1.1: *"An authorization
   method must permit a user to identify and acknowledge the data that is
   significant to a given transaction."* For a push, the significant data is the
   commit list and the target ref. (Source: OWASP Transaction Authorization
   Cheat Sheet.)
2. **TOCTOU is the specific danger in an async approval.** OWASP 2.6:
   *"Developers must not allow attackers to modify transaction data when the user
   enters the data for the first time."* Modifications *"should trigger
   invalidation of authorization data."* Between posting the approval request and
   the operator reacting, `HEAD` can move -- the operator would then be approving
   a commit list that is no longer what gets pushed.
3. **Final-gate re-validation is mandatory.** OWASP 2.8: *"System should check each
   transaction execution and make sure it has been properly authorized"* -- *"A
   final control gate must verify proper authorization before execution."*
4. **Approval credentials must be unique per transaction and time-limited.** OWASP
   1.5 / 2.10 (*"Authorization credentials should be unique for every operation"*)
   and 2.9 (*"Authorization credentials should only be valid during a limited time
   period"*).
5. **Brute-force -> restart the whole flow.** OWASP 2.4: *"After a set number of
   failed authorization attempts, the entire transaction authorization process
   should be restarted."*
6. **Fail-open-when-unconfigured is the #1 real-world approval bug.** hermes-agent
   #36848: *"when the env var is unset (the default), the entire authorization
   block is skipped."* Recommended fix is explicit opt-in (`*` for unrestricted)
   and *"block handler registration when no allowlist is configured."*
7. **Append-only records: never edit, always supersede.** Microsoft Azure
   Well-Architected, verbatim: *"The ADR serves as an append-only log. Don't go
   back and edit accepted records. If a decision changes, write a new record that
   supersedes the original and link the two together. This approach preserves the
   history of your thinking and makes it clear when and why the direction
   shifted."* Corroborated by UQ CSSE6400 lecture notes: the only part of an
   accepted record that should evolve is the **status** paragraph.
8. **`asyncio.to_thread` is the correct primitive but does NOT cancel the thread.**
   Python docs: *"primarily intended to be used for executing IO-bound
   functions/methods that would otherwise block the event loop"*; the current
   `contextvars.Context` *"is propagated"*. Cancellation cancels the awaiting task,
   **not** the OS thread -- so the subprocess's own `timeout=` is the only real
   bound.
9. **Slash commands require app-config; message listeners do not.** Slack docs:
   each slash command must be registered in the App Management dashboard with a
   name, **Request URL**, description, and must ack within **3000 ms**.
   `@app.message` needs only a matcher in code (given the existing
   message-event subscription).

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/commands.py` | 395 | Slash + message + reaction handlers | LIVE; the step's primary edit target |
| `backend/slack_bot/operator_tokens.py` | 231 | `_authorized` shared predicate, token append sink | LIVE; the fail-closed convention to mirror |
| `backend/tests/test_phase_75_2_slack_control_plane.py` | 309 | phase-75.2 suite (parametrized) | LIVE; the new tests must extend its fixtures |
| `backend/slack_bot/app.py` | 71+ | `create_app()` wiring | LIVE; `register_commands(app)` at :32 (first registrar) |
| `backend/config/settings.py` | 580 | `slack_operator_user_id` Field at :580 | LIVE |
| `.claude/masterplan.json` | 99 phases / 837 steps | Plan of record | LIVE; the annotation target |
| `.venv/.../slack_bolt/request/internals.py` | 197-214 | Bolt channel resolution | 3rd-party; **explicitly handles `reaction_added`** |
| `scripts/go_live_drills/` | 29 files | Drill scripts | LIVE; naming drift documented below |

### Part (a): the immutable-criteria collision -- sweep results

**The three known collisions ARE the complete set for the 75.2 cause.** I swept
all 837 steps across all 99 phases, handling every `verification` shape found
(`dict` 674, `str` 126, `list` 13, `None` 24 -- a naive `.get('command')` sweep
crashes on the list-shaped ones, which is how an incomplete sweep would happen).
Only these `done` steps have a verification command referencing a module deleted
by `f55e6973`:

| Step | Path in JSON | Broken by | Failure mode |
|------|--------------|-----------|--------------|
| `4.14.4` | `phases[26].steps[4]` | `from backend.slack_bot import assistant_handler` | `ImportError` |
| `4.14.24` | `phases[26].steps[23]` | `grep ... backend/slack_bot/assistant_handler.py` | `grep` err -> `wc -l` = 0 -> `awk '{exit ($1<1)}'` exits 1 |
| `4.17.9` | `phases[29].steps[8]` | `python scripts/go_live_drills/self_update_audit_test.py` | file not found |

`75.2` itself (`phases[98].steps[2]`) also names all six modules, but **asserts
their absence** -- it is correct and must not be touched.

**Deleted by `f55e6973` (7 files, not 6):** `assistant_handler.py`,
`context_management.py`, `governance.py`, `mcp_tools.py`, `self_update.py`,
`streaming_handler.py`, **plus** `scripts/go_live_drills/smoke_test_4_17_9.py`.

**4.17.9 nuance the implementer must get right.**
`git log --all --diff-filter=A -- scripts/go_live_drills/self_update_audit_test.py`
returns **empty** -- that file **never existed in git history**, confirming the
step spec. But `smoke_test_4_17_9.py` *did* exist (added in `1122a021`, deleted in
`f55e6973`). So 4.17.9 carried a **name mismatch from day one** *and* 75.2 removed
the script that was plausibly its real target. The annotation should record both
facts, or it will read as if 75.2 alone caused the breakage.

**Wider pre-existing class (scope-excluded, but must be disclosed).** 15 `done`
steps have verification commands referencing a path absent from disk. Ten are
phase-29 `go_live_drills` steps (`4.17.2`-`4.17.12`) whose plan names
(`researcher_smoke_test.py`, `qa_smoke_test.py`, ...) never matched the on-disk
`smoke_test_4_17_N.py` convention. **4.17.9 is one member of a 10-member
pre-existing family**, all unrelated to 75.2. Others: `8.5.4`, `10.2`, `4.14.19`,
`16.50`. (Beware: a naive regex with `(?:py|sh|ts|tsx)` alternation mis-flags
`.tsx` files as missing `.ts` -- put `tsx` first. That artifact inflates the
count from 15 to 27.)

**The house annotation convention already exists -- mirror it, do not invent.**
Key census over all 837 steps shows an established sibling-key vocabulary:
`superseded_by` (6), `superseded_at` (1), `superseded_record` (1),
`dropped_reason` (5), `dropped_at` (5), `deferral_audit` (10), `advisories` (3),
`scope_reassignment_note` (1), `post_commit_note` (1), `former_id` (1),
`notes` (191), `audit_basis` (303).

The closest prior art is **`superseded_record`** on step `68.5`
(`phases[91].steps[5]`), an object with exactly the shape this step needs:

```json
"superseded_record": {
  "superseded_at": "2026-07-10",
  "authorized_by": "operator AskUserQuestion 2026-07-10: 'Restate 68.5 (Recommended)' (recorded in harness_log Cycle 83)",
  "reason": "...evidence...",
  "original_name": "...",
  "original_success_criteria": ["...", "..."]
}
```

Note the *direction*: 68.5 preserved the originals inside the sibling because the
live fields changed. **75.2.1 is the mirror image** -- `verification.command` and
`verification.success_criteria` stay byte-identical, and the sibling records that
they are now unrunnable. So the sibling should carry the *collision facts*, not
copies of the criteria.

**Schema safety.** `masterplan.json` `$schema` is the bare string `"masterplan-v1"`
-- not a resolvable JSON-Schema document; no `additionalProperties: false`
validator was found on disk. Adding a sibling key is therefore schema-safe, but
the implementer should still run `scripts/meta/preflight_verify_masterplan.py`
before and after, and confirm the `archive-handoff` / `auto-commit-and-push`
hooks tolerate the new key.

**Status field.** Do NOT flip these steps away from `done`. Per the ADR doctrine
(finding 7) the record of *what was decided and completed* is immutable; only the
status paragraph may evolve, and here the work genuinely *was* done -- it is the
*artifact* that is gone, not the history. Recording a `verification_unrunnable`
sibling preserves both truths.

### Part (b): the inert push approval -- what exists today

| Anchor | Fact |
|--------|------|
| `commands.py:25` | `_APPROVAL_CHANNEL = "C0ANTGNNK8D"` |
| `commands.py:28` | `_PROJECT_ROOT = Path(__file__).parent.parent.parent` |
| `commands.py:30-34` | `_pending_push_ts: set[str]` -- process-local, empty set denies everything |
| `commands.py:37-40` | `register_push_approval_request(ts)` -- **zero callers** (confirmed) |
| `commands.py:72-78` | `_read_status()` already runs `git log origin/main..HEAD --oneline`, `cwd=_PROJECT_ROOT`, `timeout=5` -- reuse this exact shape |
| `commands.py:101-104` | Comment: *"Bolt dispatch is first-match-wins in registration order; `register_commands` is the first registrar"* |
| `commands.py:118-127` | `_operator_token_matcher` + `_TOKEN_KEYWORD` -- the matcher-plus-regex idiom to copy |
| `commands.py:252` | `@app.message("")` catch-all -- anything registered after it is unreachable |
| `commands.py:343-394` | phase-75.2 reaction sink: operator check -> channel check -> ts-membership -> `discard` (single-use) -> `to_thread(check_output, ["git","push","origin","main"], timeout=30)` |
| `operator_tokens.py:79-97` | `_authorized()` -- shared predicate, fail-closed on unset operator, used by BOTH matcher and sink |
| `settings.py:580` | `slack_operator_user_id: str = Field(...)` |
| `app.py:32-34` | `register_commands(app)` runs first, then `register_assistant_lifecycle`, `register_governance` |

**Verified, not assumed:** `say()` *does* work inside a `reaction_added` handler.
Bolt 1.27.0 `slack_bolt/request/internals.py:208-210` contains an explicit branch
-- `if payload.get("item") is not None:  # reaction_added: body["event"]["item"]`
-- so the channel resolves from `item.channel`. The existing 75.2 `say(...,
thread_ts=ts)` calls are sound.

## Application to pyfinagent -- implementation guidance

### Trigger surface: use `@app.message`, not a slash command

A slash command would require registering `/push` in the App Management dashboard
with a Request URL (source 6) -- an out-of-band app-config change, which the
step's `$0 metered / control-plane only` boundary disfavours, and which is
awkward under Socket Mode. `@app.message` needs only code. Register it
**alongside the operator-token handler (`commands.py:127`), before the catch-all
at `:252`**, or it will never fire.

**Collision warning (real, verified).** `_TOKEN_KEYWORD` at `commands.py:123-127`
is `^(?:[0-9][0-9.]*\s+)?[A-Z][A-Z0-9 _-]+:\s*.+$`. A trigger phrased
`PUSH REQUEST: main` **matches that regex**, and the operator-token handler is
registered first -- so it would be silently recorded as a token and never reach
the push path. A colon-less uppercase trigger (e.g. bare `PUSH`) does **not**
match either alternative (the first branch requires `:\s*.+`; `RESERVED_BARE` is
only `HALT-DEV`/`RESUME-DEV`), so it falls through safely. Choose a colon-less
trigger, or register the push handler before the token handler -- and add a test
that pins whichever choice is made.

### Request path -- concrete shape

1. **Authorize first, using the existing predicate.** Reuse
   `operator_tokens._authorized(...)` (`operator_tokens.py:79-97`) in the matcher
   *and* re-check at the sink. Rationale is already written in that docstring:
   *"A matcher is a capability gate, not an authorization decision."* This is the
   documented countermeasure to hermes-agent #36848 and GHSA-wv26-j37q-2g7p.
   Never gate on `item_user` -- source 3 notes it can be absent entirely.
2. **Compute the payload off the loop.** `git log origin/main..HEAD --oneline`
   via `asyncio.to_thread` (source 8; `to_thread` propagates contextvars, and the
   subprocess `timeout=` is the only real bound because cancellation will not kill
   the thread). If the list is empty, say so and register nothing.
3. **Show exactly what will be pushed (OWASP 1.1).** Post the commit list, the
   count, the target ref (`origin/main`), and the resolved `HEAD` sha into
   **`_APPROVAL_CHANNEL`** -- not the triggering channel. `_token_channels`
   (`commands.py:114-116`) admits `slack_channel_id` too, but the reaction gate
   at `commands.py:365-366` only accepts reactions in `_APPROVAL_CHANNEL`, so a
   request posted elsewhere is un-approvable by construction.
4. **Register the ts of the bot's own posted message.** `say()` returns the
   `chat.postMessage` response; take `resp["ts"]` and pass it to
   `register_push_approval_request`. Registering the *operator's* request ts
   instead would let the operator self-approve their own message -- a different
   and weaker binding.
5. **Bind the approval to the commit set, not just to the ts (OWASP 2.6 / 2.8).**
   This is the one genuine gap in the current design: `_pending_push_ts` is a set
   of bare strings, so it records *that* an approval was requested but not *what*
   was shown. Store the `HEAD` sha alongside the ts (`dict[str, tuple[str, float]]`
   or a small dataclass) and, at push time, re-resolve `HEAD` and refuse if it
   moved. Otherwise a commit landing between request and reaction is pushed
   without ever having been shown -- textbook TOCTOU.
6. **Add a TTL (OWASP 2.9).** Nothing expires today. Store a monotonic timestamp
   with each entry and reject stale approvals (a few minutes is ample for an
   attended operator).
7. **Keep every 75.2 guarantee.** Order stays identity -> channel -> ts
   membership -> `discard()` before the push (single-use, `commands.py:375`).
   `discard` before rather than after the subprocess is correct: a crashed push
   must not leave a re-approvable ts.
8. **Consider `git fetch` staleness.** `origin/main` is a *local* ref;
   without a fetch the displayed "commits ahead" can overstate. Either fetch
   first (network -> also `to_thread`, and it lengthens the TOCTOU window) or
   state in the message that the comparison is against the last-known
   `origin/main`. Do not silently imply freshness.

### Test-shape constraint (load-bearing)

The `_App` stub at `test_phase_75_2_slack_control_plane.py:62-77` captures **only
`event` handlers** into `handlers`; `message()`, `command()` and `action()` return
`lambda fn: fn` and **discard the function**. If the request path is an
`@app.message`, the stub must be extended to capture message handlers (and their
matcher/`matchers=` kwargs) or the new tests cannot invoke it. The autouse
`_isolate_token_paths` fixture (`:271-275`) already redirects `TOKENS_PATH`, so
token-path bleed is handled.

Mutation-testable assertions to add (each must fail if the guard is removed):
non-operator request registers **nothing**; wrong-channel request registers
nothing; unset operator registers nothing; the registered ts equals the **bot's**
posted `ts`, not the operator's; the commit list in the posted message equals what
the push would send; a `HEAD` change between request and reaction **blocks** the
push; an expired entry blocks the push; request alone performs **no** push
(assert `push_calls == []`); and the existing single-use replay test still holds.

### Confirmation-fatigue note (recency finding 3)

Keep this surface rare and dense. An approval message the operator learns to
reflex-approve provides the audit trail of consent without the substance of it --
which is precisely the "blind signing" failure the WYSIWYS literature describes.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total -- **27**
- [x] Recency scan (last 2 years) performed + reported -- 4 findings
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the spawn prompt
- [x] Contradictions / consensus noted (fail-open vs fail-closed; ADR immutability)
- [x] Claims cited per-claim
- [ ] **Gap disclosed:** the Springer WYSIWYS user-confusion study (peer-reviewed,
      tier 1) is paywalled; its findings enter this brief only via the
      confirmation-fatigue framing from tier-3/4 sources. The OWASP cheat sheet
      (tier 2) carries the normative weight instead.
- [ ] **Gap disclosed:** the Bolt Python message-listening page does not document
      required scopes or manifest needs. The claim "`@app.message` needs no
      app-config change" rests on the *observed* live behaviour of the existing
      catch-all handler at `commands.py:252`, not on a doc statement. The
      implementer should confirm no new scope is needed before relying on it.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 19,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "The three known immutable-criteria collisions (4.14.4, 4.14.24, 4.17.9) are confirmed COMPLETE for the 75.2 cause by a full 837-step sweep across all four verification shapes; 75.2 itself names the dead modules but asserts their absence and must not be touched. Separately, 4.17.9's target never existed in git history while the real drill (smoke_test_4_17_9.py) was deleted by 75.2, and 4.17.9 belongs to a 10-member pre-existing phase-29 naming-drift family that is out of scope but must be disclosed. The masterplan already has a house annotation vocabulary; superseded_record on step 68.5 is near-exact prior art to mirror as a sibling key, and $schema is a bare string with no additionalProperties validator, so a sibling is schema-safe. For the push approval, OWASP Transaction Authorization supplies the normative rules: show the commit list (1.1), bind the approval to the commit set and re-validate at execution (2.6/2.8), unique per-operation credentials (1.5/2.10), and a TTL (2.9) which is currently absent. Use @app.message registered before the catch-all, beware the verified _TOKEN_KEYWORD regex collision, register the BOT's posted ts, and extend the _App test stub which currently discards message handlers.",
  "brief_path": "handoff/current/research_brief_75.2.1.md",
  "gate_passed": true
}
```
