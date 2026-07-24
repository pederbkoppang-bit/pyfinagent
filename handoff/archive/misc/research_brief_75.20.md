# Research Brief: Step 75.20 — Make the Q/A live-UI gate enforceable AND make the primary Q/A path read-only

Tier: **moderate** (caller-stated). Audit-class: **false** (caller-stated; bounded change, not "find every X").
Accessed dates: 2026-07-24 unless noted.
Status: **COMPLETE — gate_passed: true** (6 external sources read in full + primary-source probes; all mandatory internal reads done)

## Question (from step 75.20)

Two coupled defects in the Layer-3 Q/A live-UI gate:
1. `qa.md` §1c DEMANDS a live Playwright capture, but the Agent-tool `qa` subagent
   `tools:` line (Read, Bash, Glob, Grep, SendMessage, Write, Edit) grants NO browser
   tools — the gate is unenforceable on the fallback path.
2. The PRIMARY Workflow path (`.claude/workflows/qa-verdict.js`) launches
   `agentType:'general-purpose'` with NO tool restriction — full MCP + Edit/Write/Bash
   under bypassPermissions — so the "read-only evaluator" invariant is unenforced there.

The fix: grant qa.md a NAMED read-only browser subset; deny the mutation/RCE browser
tools in settings.json; constrain the primary path; add `--isolated` to the .mcp.json
playwright server; keep dev-server lifecycle with Main.

## This brief REVALIDATES the 2026-07-20 prior art

Prior art: `handoff/archive/misc/research_brief_playwright_qa.md` (complex tier,
gate_passed:true, 6 read-in-full + 1 primary-source probe). This step was QUEUED from
that assessment. My job: read it in full (DONE), re-verify its load-bearing claims
against CURRENT on-disk state, settle the remaining UNCERTAIN flags with citations,
and produce concrete PLAN recommendations.

---

## Sections (filled incrementally)

- [x] 1. Masterplan step 75.20 full text + 6 criteria (read from .claude/masterplan.json:17877)
- [x] 2. Revalidation table (prior claim -> current check -> verdict)
- [x] 3. Internal audit findings (file:line)
- [x] 4. Deny-rule syntax answer (with doc citations)
- [x] 5. External source table (read-in-full vs snippet)
- [x] 6. Recency scan
- [x] 7. Queries run
- [x] 8. PLAN recommendations (exact qa.md tools line, exact deny entries, qa-verdict.js approach)
- [x] 9. JSON envelope

---

## 1. Step 75.20 — the IMMUTABLE verification command + 6 criteria (verbatim anchors)

Verification command (`.claude/masterplan.json:17883`) asserts, in order:
1. `mcp__playwright__browser_navigate` AND `mcp__playwright__browser_snapshot` appear ANYWHERE in qa.md.
2. `browser_run_code_unsafe` NOT in `qa.split('tools:')[1].split(chr(10))[0]` (the frontmatter tools LINE only).
3. `.mcp.json` playwright args: `'--isolated' in args` OR `any('user-data-dir' in a and 'profile' not in a)`.
4. settings.json deny (space-joined) contains `browser_run_code_unsafe` AND `browser_evaluate`.

6 success criteria (`:17884-17891`): (1) tools line grants EXACTLY the read-only subset
{navigate, snapshot, take_screenshot, + at most console_messages/network_requests/resize},
NO mutation tool, and a test asserts run_code_unsafe/evaluate/click/type/fill_form are
absent; (2) settings.json deny rules for the playwright mutation/RCE tools (defense in
depth for the primary path); (3) the primary path is CONSTRAINED not expanded, with a
recorded probe proving the restriction binds; (4) .mcp.json no longer shares a fixed
persistent profile across concurrent clients (`--isolated` or per-client dir) + a
concurrency demonstration; (5) qa.md instructs the deterministic `select:` ToolSearch
form AND states dev-server lifecycle stays MAIN's; (6) §1c updated to say the capture
must be taken BY the evaluator when the path allows, Main-produced capture named the
degraded fallback; harness_log carries the operator-review request + next-session
roster-live verification per `scripts/qa/verify_qa_roster_live.sh`.

## 2. Revalidation table — prior brief claim -> CURRENT on-disk check -> verdict

| # | Prior-brief / step-text claim | Current-state check (this session, 2026-07-24) | Verdict |
|---|---|---|---|
| R1 | qa.md tools line = `Read, Bash, Glob, Grep, SendMessage` (NO MCP) | `qa.md:4` = `tools: Read, Bash, Glob, Grep, SendMessage` (grep confirms only ONE `tools:` at line 4) | **CONFIRMED** |
| R2 | **STEP TEXT says the tools line is `(Read, Bash, Glob, Grep, SendMessage, Write, Edit)`** | On-disk line 4 has **NO Write, NO Edit** — the step-prompt paraphrase is INACCURATE; the Agent-tool qa is already file-read-only via its allowlist AND via qa.md:464 "NEVER Edit or Write" | **STEP TEXT WRONG — corrected** |
| R3 | qa-verdict.js:111-118 runs `agentType:'general-purpose'`, no `tools` arg | `qa-verdict.js:111-118` verbatim: `agent(PROMPT,{label,phase:'QA',schema:VERDICT_SCHEMA,agentType:'general-purpose',model:'opus',effort:'max'})` — no `tools` key | **CONFIRMED** |
| R4 | @playwright/mcp pinned at 0.0.76 | `.mcp.json:84` = `"@playwright/mcp@0.0.76"` | **CONFIRMED (unchanged)** |
| R5 | .mcp.json pins a FIXED `--user-data-dir .../.playwright-mcp-profile`, NO `--isolated` | `.mcp.json:86-87` fixed `--user-data-dir .../.playwright-mcp-profile`; `--isolated` ABSENT; `--allowed-hosts localhost`, `--viewport-size 1440,900`, `alwaysLoad:false` | **CONFIRMED** |
| R6 | settings.json deny has alpaca mutators + bigquery execute-query, NO playwright | deny = **23 entries** (12 mcp: 11 alpaca + `mcp__bigquery__execute-query`; 11 Bash). ZERO `mcp__playwright__*` | **CONFIRMED — 23 exact count** |
| R7 | Step text "23" | The "23" = deny-list entry count (`among 23`). Also coincidentally 23 = prior-brief's playwright tool count (6 readOnlyHint). Both are 23; do NOT conflate | **CLARIFIED** |
| R8 | Claude Code v2.1.215 (hard-error on unresolved tools >= v2.1.208) | `claude --version` = **2.1.218** — newer; the v2.1.208 hard-error-on-typo'd-`tools:` behaviour applies (a bad MCP grant loud-fails, not silent-tool-less) | **CONFIRMED + newer** |
| R9 | §1c is a document-inspection gate (checks a capture is referenced, cannot tell correct from stale) | `qa.md:177-195` §1c verbatim: caps verdict at CONDITIONAL on missing/stale capture; "stale" asserted not measured | **CONFIRMED** |
| R10 | frontend.md steps 1/3/5 (start :3100 / kill :3100 / move screenshots) are Main-shaped mutations Q/A must not do | `frontend.md:79-97` steps 1/3/5 verbatim; all three are mutating Bash/filesystem ops forbidden by qa.md:464 | **CONFIRMED** |
| R11 | NEW: immutable verification-command assertion #3 (concurrency) | **VACUOUS** — empirically confirmed: disjunct2 `any('user-data-dir' in a and 'profile' not in a)` is TRUE today because the flag TOKEN `"--user-data-dir"` matches (`'user-data-dir' in '--user-data-dir'`=T, `'profile' not in '--user-data-dir'`=T). `--isolated` never needed for the command to pass assert #3 | **NEW DEFECT — flagged** |

**Bottom line:** every load-bearing prior-art claim REVALIDATES against current disk. The
only correction is cosmetic (R2: the step-prompt's tools-line paraphrase adds Write/Edit
that aren't there). One NEW finding (R11): the immutable command's concurrency assertion
is vacuous, so it cannot by itself evidence criterion #4 — the real `--isolated` add +
the live_check concurrency demo carry that criterion.

---

## 3. Internal audit findings (file:line)

| File | Anchor | Fact | Bearing on 75.20 |
|---|---|---|---|
| `.claude/agents/qa.md` | :4 | `tools: Read, Bash, Glob, Grep, SendMessage` (allowlist, NO MCP, NO Write/Edit) | The tools line to EDIT (criterion #1). Adding MCP names is safe: built-ins keep the list non-empty so no zero-tools hard-error even if playwright is cold (sub-agents doc :362) |
| `.claude/agents/qa.md` | :27 | `permissionMode: plan` | OVERRIDDEN by parent `bypassPermissions` (sub-agents doc :465) — so Q/A effectively runs bypass; only session-wide DENY constrains it |
| `.claude/agents/qa.md` | :177-195 | §1c live-UI gate (BINDING) — document-inspection only; caps at CONDITIONAL on missing/stale capture; "stale" asserted not measured | The gate to UPDATE (criterion #6): make the evaluator TAKE the capture; Main-capture = degraded fallback |
| `.claude/agents/qa.md` | :464 | "NEVER Edit or Write. Bash ONLY for non-mutating verification... no `>`/`>>`" | Aligns with keeping dev-server lifecycle (start/kill :3100, mv screenshots) OUT of Q/A |
| `.claude/workflows/qa-verdict.js` | :111-118 | `agent(PROMPT,{label,phase,schema:VERDICT_SCHEMA,agentType:'general-purpose',model:'opus',effort:'max'})` — NO tools/disallowedTools | The PRIMARY path. Inherits full MCP+Edit/Write/Bash. The criterion-#3 constraint site |
| `.claude/workflows/qa-verdict.js` | :44-47 | prompt makes the agent Read qa.md from disk at runtime | qa.md PROSE edits (§1c, select: form) are live on the primary path immediately; the `tools:` ALLOWLIST is NOT applied here unless agentType is switched to `qa` |
| `.mcp.json` | :79-92 | playwright server: `@0.0.76`, fixed `--user-data-dir .../.playwright-mcp-profile`, `--allowed-hosts localhost`, `--viewport-size 1440,900`, `alwaysLoad:false`; NO `--isolated` | Add `--isolated` (criterion #4). `alwaysLoad:false` => schemas deferred => Q/A needs ToolSearch `select:` to load them (criterion #5) |
| `.claude/settings.json` | :153 | `permissions.defaultMode: "bypassPermissions"` | Skips PROMPTS, not DENY. A deny still blocks (permissions doc :37 deny-first, non-bypassable) |
| `.claude/settings.json` | :166-190 | deny = 23 entries; MCP-tool form is exact `mcp__server__tool` (e.g. `mcp__bigquery__execute-query` — note hyphen). ZERO playwright | The deny-array to EXTEND (criterion #2). Session-wide => binds Main AND Q/A AND the Workflow general-purpose agent |
| `.claude/rules/frontend.md` | :79-97 | Live-UI verification steps: 1 start `:3100` skip-auth, 3 `lsof -ti tcp:3100 | xargs kill -9`, 5 mv screenshots to `handoff/current/captures_<step>/` | All mutating; STAY with Main (criterion #5). Q/A observes the already-running instance |
| `.claude/rules/frontend.md` | :99-102 | Config note: mid-session `.mcp.json` edit does NOT respawn a connected stdio server — reconnect via `/mcp` or disclose capture-time version | The `--isolated` add won't take effect on a live-connected server until `/mcp` reconnect or next session — disclose in live_check |
| `scripts/qa/verify_qa_roster_live.sh` | :17-37 | Checks the literal header `### 1b. Frontend lint + typecheck` is on disk + the phase-23.2.24 commit is on origin/main + embeds a self-disclosure prompt | The roster-live verifier 75.20 must reference (criterion #6). It is HEADER-SPECIFIC to 1b today; may be extended to also assert the browser-tools grant, but the binding requirement is to RUN it post-restart + log the operator-review request |

---

## 4. Deny-rule syntax answer (the load-bearing external question) — SETTLED with official-doc citations

**Q: exact matcher syntax to deny a single MCP tool; does deny bind under `bypassPermissions`; does it bind subagents AND the Workflow general-purpose agent?**

All answers from `https://code.claude.com/docs/en/permissions` (Anthropic official, read in full 2026-07-24):

1. **Single-tool deny syntax = `mcp__<server>__<tool>`.** Verbatim (§MCP): "`mcp__puppeteer__puppeteer_navigate` matches the `puppeteer_navigate` tool provided by the `puppeteer` server." Server-wildcard `mcp__<server>__*` and bare `mcp__<server>` ALSO match all tools of that server; `mcp__*` matches every MCP tool. **This RESOLVES the community-source contradiction** (one blog claimed `mcp__server__*` "not supported" — that is WRONG for deny/ask; the official doc line: "`mcp__puppeteer__*` uses wildcard syntax and also matches all tools from the `puppeteer` server"). Our existing deny uses the EXACT form (`mcp__bigquery__execute-query`) — match that style.
2. **Deny is deny-FIRST and cannot carry exceptions.** Verbatim: "Rules are evaluated in order: deny, then ask, then allow. The first match in that order determines the outcome... a deny rule can't carry allowlist exceptions."
3. **Deny survives `bypassPermissions`.** `bypassPermissions` "Skips permission PROMPTS" (permissions doc §modes) — a deny is a BLOCK, not a prompt, evaluated before allow. Corroborated by the hooks section: "a matching deny rule blocks the call... including deny rules set in managed settings." So a deny on `mcp__playwright__browser_run_code_unsafe` blocks it for every actor in the session.
4. **Binds subagents AND the Workflow general-purpose agent.** Subagents "inherit the permission context from the main conversation" and "If the parent uses `bypassPermissions`... this takes precedence and can't be overridden" (sub-agents doc :465) — but that is about the MODE, not deny. Deny rules are session-scoped and evaluated on every tool call regardless of which agent issues it. **This is the mechanism that makes criterion #2 correct**: the deny binds the primary Workflow path even though that path inherits the full MCP surface.
5. **CAVEAT — the typo warning EXEMPTS names with `_`.** Verbatim: "A deny or ask rule whose tool name matches no known tool produces a startup warning to catch typos. Tool names containing `_` or `*` are exempt from the check." Every `mcp__playwright__browser_*` name contains `_`, so a MISSPELLED deny entry gets **no warning** — the deny entries must be spelled EXACTLY. Cross-check each against the live `tools/list` (the prior brief's probe enumerated all 23).

**Sub-agents `tools:` grant (criterion #1) — SETTLED:** "Both fields accept MCP server-level patterns in addition to exact tool names: `mcp__<server>` or `mcp__<server>__*` grants or removes every tool from the named server" (sub-agents doc :364). Exact names work too (:279, :340 example). A background subagent "keeps every MCP tool" (:336) so the grant survives foreground/background. `alwaysLoad:false` defers only the SCHEMA (loaded via ToolSearch), not the permission.

---

## 5. External source table

### Read in full (>=5 required; counts toward the gate) — 6

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/permissions | 2026-07-24 | Official doc (Anthropic) | WebFetch (58.8KB persisted) + grep | deny-first non-bypassable; `mcp__server__tool` exact + `mcp__server__*` wildcard both valid in deny; typo-warning EXEMPTS `_`-names |
| https://code.claude.com/docs/en/sub-agents | 2026-07-24 | Official doc (Anthropic) | WebFetch (89KB persisted) + grep | `tools:` accepts exact MCP names + server patterns; inherits all MCP if omitted; background keeps every MCP tool; parent bypass overrides child permissionMode; `mcpServers` inline vs string-ref |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-24 | Vendor eng blog (project's canonical ref) | WebFetch (exact passages) | "I gave the evaluator the Playwright MCP, which let it interact with the live page directly before scoring each criterion"; "Separating the agent doing the work from the agent judging it"; 4-hour cost |
| https://raw.githubusercontent.com/microsoft/playwright-mcp/main/README.md | 2026-07-24 | Official vendor doc (Microsoft) | WebFetch (exact passages) | persistent-profile concurrency `[!IMPORTANT]`; `--isolated` keeps profile in memory; `--caps` additive-only (no read-only mode); `browser_run_code_unsafe` "RCE-equivalent"; `browser_evaluate` read-only:false |
| https://arxiv.org/html/2511.19477v1 | 2026-07-24 | Preprint (Nov 2025) | WebFetch (native HTML) | "Safety policies should be enforced by the execution layer code, not the LLM"; "specialized agents with least-privilege scopes"; read-only Assistant Agents "lack interaction tools (such as click or type)" = "high utility with minimal risk" |
| https://arxiv.org/html/2606.20023 | 2026-07-24 | Preprint (v2, July 2026) [RECENCY] | WebFetch (native HTML) | over-privileged tool selection common (6/11 models >30% OPUR); escalates after failures; "prompt-level controls provide only limited mitigation" — argues FOR mechanical deny over qa.md prose |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.claudedirectory.org/blog/claude-code-permissions-guide | Blog | Corroborating deny-vs-bypass; superseded by official doc |
| https://techtaek.com/claude-code-mcp-permissions-in-2026-... | Blog | Corroborating (allowedTools vs bypass) |
| https://dev.to/klement_gunndu/lock-down-claude-code-... | Community | 5 permission patterns; low tier |
| https://neurals.ca/tech/claude/settings-json/ | Community | precedence corroboration |
| https://www.developersdigest.tech/blog/claude-code-permissions-settings-guide | Blog | allow/deny/ask corroboration |
| https://github.com/anthropics/claude-code/issues/10906 | Issue | Plan-agent permission-inheritance edge case |
| https://github.com/anthropics/claude-code/issues/65783 | Issue | deny-glob doc-gap; confirms `*` / unknown-tool warning semantics |
| https://www.datacamp.com/tutorial/claude-code-security | Tutorial | MCP matcher-syntax corroboration |
| https://qaskills.sh/blog/playwright-mcp-profile-modes-guide-2026 | Blog | persistent/isolated/extension profile modes; corroborates `--isolated` for concurrency |
| https://www.npmjs.com/package/@playwright/mcp | Vendor registry | version/flags |
| https://github.com/microsoft/playwright/issues/40585 | Issue | sessionId multi-session isolation (future feature) |
| https://arxiv.org/abs/2509.08646 | Preprint | Plan-then-Execute secure agents; least-privilege |
| https://arxiv.org/html/2603.28166v2 | Preprint | Evaluating privilege usage w/ real tools |
| https://arxiv.org/pdf/2503.15547 | Preprint | Prompt Flow Integrity / privilege escalation |
| https://arxiv.org/pdf/2605.05868 | Preprint | SkillScope least-privilege for agent skills |
| https://arxiv.org/pdf/2512.11147 | Preprint | MiniScope least-privilege tool-calling framework |
| https://blog.alexewerlof.com/p/owasp-top-10-ai-llm-agents | Blog | OWASP Top-10 agents 2026 |
| https://www.morphllm.com/claude-code-dangerously-skip-permissions | Blog | bypass-mode safer setups |

---

## 6. Recency scan (last 2 years, 2024-2026) — PERFORMED

Three-variant discipline applied (see §7). Result: **2 in-window findings that reinforce
the step's mechanical-enforcement thesis; none supersede the canonical verifier-
independence + least-privilege principles.**

1. **arXiv:2606.20023 (v2, July 2026) — "When Lower Privileges Suffice."** Directly in
   the last-2-year window and DIRECTLY on-topic. Finding: LLM agents over-select
   privileged tools (6/11 models >30% OPUR; escalation spikes AFTER tool failures — and
   Q/A hits retries/failures), and critically **"general safety alignment does not
   reliably transfer to least-privilege tool choice, while prompt-level controls provide
   only limited mitigation."** This is a 2026 empirical argument AGAINST relying on
   qa.md prose ("please be read-only") and FOR the mechanical `tools:` allowlist +
   settings.json deny. Mild nuance/adversarial: the paper's OWN prescription is post-
   training (unavailable to us); it says mechanical controls "alone" are insufficient —
   but for our lever set, mechanical least-privilege is exactly the available and
   recommended-by-2511.19477 control.
2. **arXiv:2511.19477 (Nov 2025) — "Building Browser Agents."** Read-only Assistant
   Agents that "lack interaction tools (such as click or type)" give "high utility with
   minimal risk"; "Safety policies should be enforced by the execution layer code, not
   the LLM." Endorses the NAMED read-only subset over a wildcard, and the deny-rule
   (execution-layer) enforcement over prompt instruction.
3. Claude Code v2.1.218 (current) behaviours — hard-error on unresolved `tools:` entries
   (>=v2.1.208), `mcp__server__*` deny-glob, `_`-name typo-warning exemption — are all
   recent-months doc material and shape the failure modes above.

No 2024-2026 source contradicts verifier-independence (Anthropic harness-design) or
least-privilege; the 2026 literature has moved toward mechanical/out-of-band enforcement
(MiniScope, SkillScope, Progent), reinforcing the design.

---

## 7. Queries run (three-variant discipline per topic)

- **Deny-semantics topic:** frontier `Claude Code permissions deny MCP tool subagent bypassPermissions 2026`; last-2yr `Claude Code settings.json deny precedence allow subagent inherit 2025`; year-less canonical `Claude Code permissions deny MCP tool matcher syntax mcp__server__tool`.
- **Least-privilege evaluator topic:** frontier `least privilege tool grant LLM evaluator browser agent security 2026` (surfaced the year-less canonical prior-art too: MiniScope, SkillScope, OWASP).
- **Concurrency topic:** `playwright mcp isolated flag concurrent clients persistent profile conflict`.

Mix of current-year (2026), last-2-year (2025), and year-less canonical hits present in
the source tables (permissions/sub-agents docs are year-less canonical; 2606.20023 is
current-year; 2511.19477 is last-2-year).

---

## 8. PLAN recommendations (concrete, criterion-mapped)

### 8.1 Exact qa.md tools line (criterion #1)

Replace `.claude/agents/qa.md:4`. **Mandatory core** (satisfies the immutable command's
assert #1 + §1c): `browser_navigate, browser_snapshot, browser_take_screenshot`. The
criterion permits **at most** three more read-only additions (`browser_console_messages,
browser_network_requests, browser_resize`). Recommended full line (the criterion's
maximal read-only envelope — all six are genuinely read-only-useful for the gate:
console warnings, network/API state, viewport):

```
tools: Read, Bash, Glob, Grep, SendMessage, mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot, mcp__playwright__browser_take_screenshot, mcp__playwright__browser_console_messages, mcp__playwright__browser_network_requests, mcp__playwright__browser_resize
```

Do NOT put `browser_run_code_unsafe`/`browser_evaluate`/click/type/fill_form on this line
(assert #2 checks run_code_unsafe is absent from the tools LINE; criterion #1's test must
assert run_code_unsafe/evaluate/click/type/fill_form are ALL absent from the grant). Keep
the deny-rationale prose (which mentions `browser_run_code_unsafe`) in the BODY, never on
line 4. Note `browser_navigate` is `readOnlyHint:false` upstream but is REQUIRED to load a
page and is on the criterion's explicit read-only allowlist — that is a human judgment
call, not derivable from the annotation.

### 8.2 Exact settings.json deny entries (criterion #2)

Append to `permissions.deny` (`.claude/settings.json`), exact-name form matching the
existing `mcp__bigquery__execute-query` style. **Required minimum** (satisfies the
immutable command: run_code_unsafe + evaluate; and criterion #1's absent-set):

```json
"mcp__playwright__browser_run_code_unsafe",
"mcp__playwright__browser_evaluate",
"mcp__playwright__browser_click",
"mcp__playwright__browser_type",
"mcp__playwright__browser_fill_form"
```

**Do NOT deny** `browser_navigate/snapshot/take_screenshot/console_messages/network_requests/
resize` — the deny is SESSION-WIDE and would break BOTH Q/A's grant and Main's capture
workflow. Optional defense-in-depth extension (still session-safe — Main's observation-only
capture uses none of these): `browser_press_key, browser_hover, browser_drag, browser_drop,
browser_select_option, browser_file_upload, browser_handle_dialog`. Spell every entry
exactly — the typo warning is EXEMPT for `_`-names (§4 caveat).

### 8.3 qa-verdict.js primary-path constraint (criterion #3) — the hard part

The primary path (`agentType:'general-purpose'`, no tools arg) inherits full MCP +
Edit/Write/Bash. Two layers:
- **Layer A (definitely binds, no probe needed):** the §8.2 session-wide deny ALSO blocks
  the RCE/mutation-browser tools for the Workflow agent (deny is non-bypassable). This
  closes the RCE hole on the primary path — criterion #2 doing double duty.
- **Layer B (criterion #3 core — constrain Edit/Write/full-MCP WITHOUT breaking Main):**
  we CANNOT session-wide-deny Edit/Write (Main needs them). The constraint must live at
  the spawn site. **Preferred:** switch `qa-verdict.js:114` `agentType:'general-purpose'`
  → `agentType:'qa'` so the qa.md `tools:` allowlist (now = read-only+browser subset)
  binds the primary path — the qa.md file already carries the exact intended restriction.
  **Alternative:** pass an explicit `disallowedTools`/`tools` option to the `agent()`
  call. **BOTH require a GENERATE-time empirical probe** of the Workflow `agent()` API
  (does it accept a custom project `agentType`, and/or a tools/disallowedTools option, and
  still return the structured schema?) — I could not find docs for the Workflow `agent()`
  option surface, so this MUST be probed, not assumed. The live_check already mandates "a
  probe recorded... showing the Q/A path's tool surface BEFORE and AFTER" — that probe IS
  the criterion-#3 evidence. If neither A nor B's config lever is supported by the runtime,
  the honest fallback is Layer-A deny + documented degradation (NOT a silent PASS).

### 8.4 .mcp.json concurrency fix (criterion #4)

Add `"--isolated"` to `.mcp.json` playwright `args` (the `[!IMPORTANT]` vendor callout +
arXiv-corroborated). Because our capture workflow is auth-bypassed on :3100
(`LIGHTHOUSE_SKIP_AUTH=1`, frontend.md:82), `--isolated` costs nothing (no login state to
lose). **CRITICAL — do NOT treat the immutable command's assert #3 as evidence of this
fix:** it is VACUOUS (R11 — the `--user-data-dir` flag token satisfies it today with no
`--isolated`). The real evidence is the `--isolated` presence + the live_check concurrency
demonstration (two clients no longer contend). Disclose in the live_check that a mid-session
`.mcp.json` edit does NOT respawn a connected stdio server (frontend.md:99) — reconnect via
`/mcp` or next session; captures may have run on the pre-edit server.

### 8.5 qa.md §1c + select: form + dev-server lifecycle (criteria #5, #6)

- **§1c update (criterion #6):** add prose — when the launch path grants browser tools,
  the capture MUST be taken BY the evaluator (`browser_navigate` + `browser_snapshot`/
  `browser_take_screenshot` against the running app); reading a Main-produced capture is
  the explicitly-DEGRADED fallback (used only when the path cannot capture).
- **select: ToolSearch form (criterion #5):** instruct Q/A to load the browser schemas via
  the DETERMINISTIC `ToolSearch({query:"select:mcp__playwright__browser_navigate,mcp__playwright__browser_snapshot,mcp__playwright__browser_take_screenshot"})` form — NOT a keyword search (`alwaysLoad:false` defers the schema; a naive `"playwright browser"` keyword query surfaced run_code_unsafe + click in its top-5 and MISSED navigate/snapshot per the step text).
- **Dev-server lifecycle stays Main's (criterion #5):** §1c/qa.md must state Q/A OBSERVES an
  already-running instance and never starts/kills a server — frontend.md steps 1/3/5
  (start :3100, `kill -9`, mv screenshots) remain MAIN's (they are mutating; qa.md:464
  forbids them). Do NOT "solve" this by granting Q/A server-lifecycle Bash (re-creates the
  2026-07-17 :3000-outage class — auto-memory `feedback_second_next_dev_breaks_operator_3000`).

### 8.6 Separation of duties, restart, roster-live (criterion #6 + BOUNDARY)

- The authoring session must NOT self-evaluate (CLAUDE.md separation-of-duties on agent
  edits) — append an operator-review request to `handoff/harness_log.md`.
- The `tools:` line change binds the **Agent-tool** path only after a session RESTART
  (roster snapshot); the **Workflow** path reads qa.md PROSE live but the `tools:`
  allowlist applies to it only if §8.3 Layer-B switches agentType to `qa`. State this
  split explicitly in the handoff.
- Reference `scripts/qa/verify_qa_roster_live.sh` (criterion #6): run it post-restart to
  confirm the new tools line is in the Agent-tool snapshot. NOTE the script is currently
  header-specific to `### 1b. Frontend lint + typecheck`; extending its `SECTION_HEADER`
  check to also assert the browser-tools grant is a reasonable OPTIONAL hardening, but the
  binding requirement is to USE the script + log the operator-review request.

### 8.7 First-of-kind smoke + failure-safety

No pyfinagent agent file has ever carried an MCP `tools:` grant (prior-brief D9). On
v2.1.218 a bad entry hard-errors (safer than pre-2.1.208 silent tool-less), AND the
built-ins keep the list non-empty so a cold/unconnected playwright server silently drops
the browser entries rather than failing the launch (sub-agents doc :362) — Q/A degrades to
fallback-capture-reading, never to a false PASS. Smoke-test the Agent-tool qa launch after
restart.

---

## 9. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 2 official docs, 2 vendor docs/blogs, 2 preprints) + primary-source empirical probes
- [x] 10+ unique URLs total (6 read-in-full + 18 snippet-only = 24)
- [x] Recency scan (last 2 years) performed + reported; three-variant queries visible (§7)
- [x] Full pages read (not abstracts) for the read-in-full set (arXiv via native HTML per the PDF chain; large docs persisted + grepped for verbatim)
- [x] file:line anchors for every internal claim (§2, §3)

Soft checks:
- [x] Internal exploration covered every mandatory module (qa.md, qa-verdict.js, .mcp.json, settings.json, frontend.md, verify_qa_roster_live.sh, masterplan 75.20)
- [x] Contradictions noted + resolved (community "mcp__server__* not supported" REFUTED by official doc; step-text tools-line paraphrase corrected R2; vacuous assert #3 flagged R11)
- [x] Claims cited per-claim with URL + access date / file:line

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 18,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Revalidated the 2026-07-20 prior art against current disk: every load-bearing claim holds. Corrections: (R2) the step-prompt's qa.md tools-line paraphrase wrongly adds Write/Edit -- on-disk qa.md:4 is 'Read, Bash, Glob, Grep, SendMessage' (already file-read-only); (R11 NEW) the IMMUTABLE verification command's concurrency assert #3 is VACUOUS -- 'any(user-data-dir in a and profile not in a)' is True today because the flag TOKEN '--user-data-dir' matches, so --isolated never needs to be present for the command to pass; the real criterion-#4 evidence is the --isolated add + the live_check concurrency demo, empirically confirmed this session. Settled the load-bearing deny question with the official permissions doc: single-tool deny = exact 'mcp__server__tool'; deny is deny-first and NON-bypassable, so a settings.json deny binds the primary Workflow general-purpose agent too (this is what makes criterion #2 correct); the typo warning is EXEMPT for '_'-names so entries must be spelled exactly. Sub-agents doc confirms 'tools:' accepts exact MCP names and the grant survives background. Concrete PLAN: exact 6-tool read-only qa.md line; 5 exact deny entries (run_code_unsafe/evaluate/click/type/fill_form, NOT navigate/snapshot which the deny is session-wide and would break Main); criterion-#3 needs a GENERATE-time probe of the Workflow agent() API (switch agentType to 'qa' preferred, or pass disallowedTools) -- not assumable; add --isolated; keep :3100 lifecycle with Main; log operator-review + roster-live via verify_qa_roster_live.sh. Anthropic harness-design gives its evaluator Playwright verbatim; arXiv:2606.20023 (Jul 2026) says prompt-level controls give 'only limited mitigation' -- argues for mechanical deny over qa.md prose.",
  "brief_path": "handoff/current/research_brief_75.20.md",
  "gate_passed": true
}
```
