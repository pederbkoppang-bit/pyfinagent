# Research Brief: Granting Playwright MCP to the Layer-3 Q/A subagent

Tier: **complex** (assessment input; caller did not state tier explicitly -> assuming
`complex` because the question spans capability mechanics + concurrency semantics +
safety + internal audit; stating the assumption per protocol).
Audit-class: false (bounded question, not "find every X").
Status: COMPLETE -- gate_passed: true (6 sources read in full + 1 primary-source
probe of our pinned server; 2 findings explicitly flagged UNCERTAIN, see checklist)

Accessed dates: 2026-07-20 unless noted.

## Question

Should `.claude/agents/qa.md` (Layer-3 Q/A) be granted Playwright MCP tools so it
independently verifies UI, rather than reading a capture Main produced?

## Sections (filled incrementally)

- [ ] A. Capability / mechanics (subagent `tools:` + MCP; ToolSearch/alwaysLoad; Workflow agentType)
- [ ] B. Concurrency / blast radius (fixed --user-data-dir, singleton lock, --isolated, per-session servers)
- [ ] C. Safety (mutation-capable tools, --caps, verifier-independence literature)
- [ ] D. Internal repo inventory (file:line)
- [ ] Recency scan (2024-2026)
- [ ] Read-in-full table
- [ ] Snippet-only table
- [ ] Findings / recommendation
- [ ] JSON envelope

---

## A. Capability / mechanics

Source: https://code.claude.com/docs/en/sub-agents (accessed 2026-07-20, read in full
via WebFetch, 83.5KB persisted).

### A1 -- Can `tools:` grant MCP tools? YES, and server-level wildcards work.

Verbatim: "Both fields accept MCP server-level patterns in addition to exact tool
names: `mcp__<server>` or `mcp__<server>__*` grants or removes every tool from the
named server. In `disallowedTools`, `mcp__*` also removes every MCP tool from any
server."

So all three forms are valid in `tools:`:
- exact: `mcp__playwright__browser_navigate`
- server wildcard: `mcp__playwright__*`
- bare server: `mcp__playwright`

NEW (v2.1.208) failure mode worth knowing: "When nothing in the `tools` list resolves
to a tool, for example because every entry is misspelled or names a tool that isn't
available to subagents, Claude Code refuses to launch the subagent and the Agent tool
returns an error naming the unresolved entries. Before v2.1.208, that subagent
launched with no tools and could return an empty or confusing result." -> a typo'd
MCP grant used to silently produce a TOOL-LESS Q/A. On <2.1.208 that is a
silent-false-PASS risk vector; on >=2.1.208 it hard-errors (safer).

### A2 -- If `tools:` is OMITTED, does the subagent inherit MCP?

Verbatim: "Subagents inherit the internal tools and MCP tools available in the main
conversation by default." Excluded regardless: `AskUserQuestion`, `EnterPlanMode`,
`ExitPlanMode` (unless permissionMode: plan), `ScheduleWakeup`, `WaitForMcpServers`.

NOTE `WaitForMcpServers` being unavailable to subagents is relevant: a subagent
cannot explicitly wait for a lazily-connecting MCP server.

Our `qa.md:4` currently sets `tools: Read, Bash, Glob, Grep, SendMessage` -- an
ALLOWLIST, so today Q/A has NO MCP tools at all (not Playwright, not BigQuery).
The doc's own example makes this explicit: "This example uses `tools` to allow only
Read, Grep, Glob, and Bash. The subagent can't edit files, write files, or use any
MCP tools."

Alternative shape: `disallowedTools: Write, Edit` inherits everything else including
MCP ("The subagent keeps Bash, MCP tools, and everything else"). That is a MUCH
larger blast radius than adding one MCP pattern to the existing allowlist -- it would
also hand Q/A the alpaca (order-placing) and bigquery (execute-query) servers.
Resolution order if both set: "`disallowedTools` is applied first, then `tools` is
resolved against the remaining pool."

### A2b -- THIRD option the caller did not ask about: `mcpServers:` frontmatter

There is a dedicated `mcpServers` frontmatter field, and the doc's canonical example
is literally a Playwright browser-tester:

```yaml
name: browser-tester
description: Tests features in a real browser using Playwright
mcpServers:
  - playwright:            # inline definition: scoped to this subagent only
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
  - github                 # reference by name: reuses an already-configured server
```

**The load-bearing sentence for the concurrency question (B6):**
"Inline servers defined here are connected when the subagent starts and disconnected
when it finishes. String references share the parent session's connection."

This is the pivot of the whole assessment -- see section B.


### A3 -- Does the Workflow `agent(..., {agentType:'general-purpose'})` path get MCP tools?

**This is the single most consequential finding of the assessment, and it inverts
the framing of the question.**

`.claude/workflows/qa-verdict.js:111-118` launches Q/A as:

```js
const verdict = await agent(PROMPT, {
  label: 'qa-verdict:' + stepId,
  phase: 'QA',
  schema: VERDICT_SCHEMA,
  agentType: 'general-purpose',
  model: 'opus',
  effort: 'max',
})
```

There is **no `tools` restriction passed**. The `qa` subagent TYPE is never
instantiated on this path -- `qa.md` is consumed only as *prompt text* (the script's
`STEP 0 (binding): Read .claude/agents/qa.md IN FULL and follow it`, line 44).

Consequences:
- The `tools: Read, Bash, Glob, Grep, SendMessage` allowlist at `qa.md:4` binds ONLY
  the Agent-tool fallback path. On the **PRIMARY** (Workflow) path, Q/A runs as
  `general-purpose`, which per the sub-agents doc inherits "the internal tools and
  MCP tools available in the main conversation by default."
- So the read-only-ness of Q/A on its primary launch path is enforced by
  **instruction, not by the permission system** (`qa-verdict.js:50-52` tells it to be
  read-only; nothing stops it).
- Therefore the real question is NOT "should we grant Q/A Playwright?" but
  **"our primary Q/A path very likely ALREADY has Playwright (and alpaca, and
  bigquery execute-query) and we never noticed."** The proposal would, if done via
  the `tools:` allowlist on `qa.md`, only change the FALLBACK path.

**UNCERTAIN (flagged, not settled):** I could not find an official doc page that
states verbatim what tool set the built-in `general-purpose` agentType receives when
invoked from a Workflow `agent()` call, nor whether Workflow-spawned agents connect
to `.mcp.json` servers. The inference above rests on (a) the documented default
inheritance rule for subagents and (b) the absence of any `tools` argument in
`qa-verdict.js`. **This should be empirically probed before being relied on** -- a
one-line Workflow that returns the agent's own tool list would settle it. Treat as a
hypothesis with high prior, not an established fact.

### A2c -- `alwaysLoad: false` / ToolSearch interaction

`.mcp.json:77` sets `"alwaysLoad": false` for the playwright server. `alwaysLoad`
governs whether the server's tool DEFINITIONS are injected at session start or
deferred behind ToolSearch; it is a context-budget knob, not a permission boundary.
`.claude/agents/researcher.md:133` already documents the deferred-reach pattern for
THIS role: "tools `mcp__playwright__browser_*` via ToolSearch".

Caveat: `WaitForMcpServers` is on the doc's list of tools NOT available to subagents,
so a subagent cannot explicitly block on a lazily-connecting server. If a deferred
server has not connected, a subagent has no documented way to wait for it. **UNCERTAIN
whether ToolSearch inside a subagent triggers connection of a not-yet-connected
deferred stdio server** -- not documented either way; worth an empirical probe.

---

## B. Concurrency / blast radius

Source: https://github.com/microsoft/playwright-mcp README (accessed 2026-07-20;
fetched twice -- WebFetch + `curl` raw, 61,323 bytes, read in full).

### B4 -- Two instances against the SAME fixed `--user-data-dir`: DOCUMENTED CONFLICT

The README carries this as an `[!IMPORTANT]` callout (line 480 of raw README),
verbatim:

> "A persistent profile can only be used by one browser instance at a time, so
> concurrent MCP clients sharing the same workspace will conflict. To run several
> clients in parallel, start each additional client with `--isolated` or point it at
> a distinct `--user-data-dir`."

Our config is precisely the hazardous shape: `.mcp.json:73` pins a FIXED
`--user-data-dir /Users/ford/.openclaw/workspace/pyfinagent/.playwright-mcp-profile`.
Live evidence the profile is a real persistent Chrome profile (not ephemeral):

```
.playwright-mcp-profile/RunningChromeVersion -> 145.0.7632.6:1
.playwright-mcp-profile/Default/            (83 entries)
.playwright-mcp-profile/Local State         (modified 18 jul 06:45)
```

`RunningChromeVersion` is the Chromium singleton-family artifact; the `Singleton*`
lock files were ABSENT at inspection time, consistent with the browser being launched
LAZILY (the MCP server process is running -- PIDs 97203/97216 observed -- but no
`Chrome for Testing` browser process was). So the collision only materialises when
two clients both make a `browser_*` call, not merely when two servers are spawned.

**Exact failure mode -- UNCERTAIN, flagged.** The README states the constraint
("will conflict") but does NOT specify the observable behaviour of the loser.
Chromium's own singleton mechanism (`SingletonLock`/`SingletonSocket`/`SingletonCookie`
in the profile dir) normally makes a second launch either hand off to the first
instance or abort; Playwright's `launchPersistentContext` typically surfaces this as a
launch failure/timeout rather than silent data corruption. I did NOT find an
authoritative statement pinning which of {hard error, hang, silent attach} occurs for
this specific stack. **Do not present "it fails cleanly" as settled** -- the honest
statement is: the vendor documents it as unsupported and tells you to avoid it.

### B5 -- Is there an isolated mode? YES, and it is in our PINNED version.

Verified directly against our pin (`npx -y @playwright/mcp@0.0.76 --help`, run
2026-07-20, exit 0):

```
--isolated        keep the browser profile in memory, do not save it to disk.
--storage-state <path>   path to the storage state file for isolated sessions.
--user-data-dir <path>   path to the user data directory. If not specified,
                         a temporary directory will be created.
```

README on isolated mode: "In the isolated mode, each session is started in the
isolated profile. Every time you ask MCP to close the browser, the session is closed
and all the storage state for this session is lost. You can provide initial storage
state to the browser via the config's `contextOptions` or via the `--storage-state`
argument."

**Tradeoff, and why it is NEARLY FREE for us:** the cost of `--isolated` is losing
logged-in state. But the project's documented capture workflow
(`.claude/rules/frontend.md:73-102`, "Live-UI verification") deliberately bypasses
auth entirely -- it starts a second dev server with `LIGHTHOUSE_SKIP_AUTH=1` on
:3100 so no login is needed. **A Q/A-scoped Playwright server therefore has almost
nothing to lose from `--isolated`.** This is the clean way to give Q/A a browser
without touching Main's profile.

Minor doc inconsistency worth noting: the flag table says `--user-data-dir` defaults
to "a temporary directory", while the User-profile prose says the default is a
persistent `mcp-{channel}-{workspace-hash}` dir. Flagging as an unresolved doc
contradiction; irrelevant for us since we set the flag explicitly.

### B6 -- Same stdio process or two? -- DOCUMENTED, and it is a CHOICE.

From the sub-agents doc, `mcpServers` field: "Inline servers defined here are
connected when the subagent starts and disconnected when it finishes. **String
references share the parent session's connection.**"

This gives three distinct topologies, with materially different blast radius:

| Option | Mechanism | Server processes | Profile collision risk |
|---|---|---|---|
| 1. `tools: ..., mcp__playwright__*` in `qa.md` | Q/A reaches the session's existing server | ONE (shared with Main) | **None** -- single browser, but Main and Q/A share one browser state |
| 2. `mcpServers: [playwright]` (string ref) | Same as above, explicit | ONE (shared) | None |
| 3. `mcpServers: [ playwright: {inline...} ]` | Subagent-scoped server, spawned at start | TWO | **HIGH if the inline def reuses the same `--user-data-dir`; ZERO if it passes `--isolated`** |

Option 3 with `--isolated` is the doc's own canonical example shape (the README's
`browser-tester` agent) and is the only option that gives Q/A a genuinely independent
browser. Option 1/2 are cheaper but mean Main and Q/A drive the SAME browser -- which
partially defeats "independent evidence gathering" at the mechanism level (shared
cookies/session/page state), though the evidence is still independently *observed*.

**Caveat on option 3:** the doc says inline servers connect when the subagent starts.
Combined with the missing `WaitForMcpServers` for subagents, a cold `npx` spawn (which
can take seconds) at every Q/A launch is a latency + flakiness tax on EVERY evaluate
phase, including the ~74% of steps that make no UI claim at all.

---

## C. Safety

### C7 -- The mutation surface, measured directly against OUR pin

I did not rely on the README's tool list; I probed the pinned server itself
(JSON-RPC `tools/list` against `npx -y @playwright/mcp@0.0.76 --isolated`,
2026-07-20). Result: **23 tools**, of which only **6** carry `readOnlyHint: true`.

| readOnlyHint TRUE (6) | readOnlyHint FALSE / mutating (17) |
|---|---|
| `browser_snapshot` | `browser_navigate`, `browser_navigate_back`, `browser_click`, `browser_type`, `browser_fill_form`, `browser_press_key`, `browser_hover`, `browser_drag`, `browser_drop`, `browser_select_option`, `browser_file_upload`, `browser_handle_dialog`, `browser_tabs`, `browser_resize`, `browser_close`, `browser_evaluate`, **`browser_run_code_unsafe`** |
| `browser_take_screenshot` | |
| `browser_console_messages` | |
| `browser_network_requests` | |
| `browser_network_request` | |
| `browser_wait_for` | |

Two things fall out of this that materially change the design:

1. **`browser_navigate` is NOT annotated read-only.** So "just give it the
   read-only tools" does not mechanically yield a working capture agent -- you
   cannot even load a page. Any allowlist is a human judgment call about which
   mutations are acceptable, not a filter you can derive from the annotations.
2. **`browser_run_code_unsafe` exists in our pinned version.** Upstream describes
   it verbatim as: "Run a Playwright code snippet. Unsafe: executes arbitrary
   JavaScript in the Playwright server process and is **RCE-equivalent**."
   Granting `mcp__playwright__*` or `mcp__playwright` to Q/A hands the *evaluator*
   an RCE-equivalent primitive. An evaluator that can execute arbitrary code in a
   Node process running as the operator can, in principle, modify the very
   artifacts it is grading -- which defeats the read-only-evaluator invariant that
   `qa.md:345-348` exists to protect.

### C7b -- `--caps` does NOT provide a read-only mode (common misconception)

Verified against our pin: `--caps <caps>  comma-separated list of **additional**
capabilities to enable, possible values: vision, pdf, devtools.` It is
**additive**. There is no `--caps core-readonly`, no `--read-only` flag, and no
server-side way to drop the mutating tools in 0.0.76. **The only available
restriction mechanism is the Claude Code side: an explicit `tools:` allowlist of
individual `mcp__playwright__browser_*` names.**

### C7c -- Our permission posture makes this sharper than it looks

- `.claude/settings.json` -> `permissions.defaultMode = "bypassPermissions"`.
- The deny list (`settings.json:167-178`) covers `mcp__alpaca__place_stock_order`,
  the other order/position mutators, and `mcp__bigquery__execute-query`.
  **There is NO deny rule for ANY `mcp__playwright__*` tool.**
- `.claude/hooks/pre-tool-use-danger.sh:255` gates only `mcp__*__execute_sql`.

So today nothing in the permission layer or the danger hook would stop a
`browser_run_code_unsafe` call. A grant must therefore be an *allowlist of named
tools*, never a server wildcard -- the deny/hook layers will not catch the
difference.

### C7d -- Project-specific divergence from Anthropic's example (IMPORTANT)

Anthropic's harness evaluator **clicked** through the app (quote in C8). Our
cockpit is not a generated toy app: `.claude/rules/frontend.md:54` documents an
`OpsStatusBar` carrying "go-live gate, kill switch, cycle health, scheduler ...
with inline action buttons", and the project has run-now / trigger-cycle
endpoints. **A click in our UI can fire a real control-plane action against a
live paper-trading book.** That is a concrete, project-specific reason to adopt
Anthropic's *pattern* (evaluator gathers its own evidence) while rejecting their
*permission set* (evaluator may click). Observation-only is the defensible line
here, and it is a deliberate divergence, not an oversight.

### C7e -- Prompt-injection surface (mitigated but not zero)

The Claude Code MCP doc carries a standing warning: "Verify you trust each server
before connecting it. Servers that fetch external content can expose you to prompt
injection risk." A browser agent ingests page content into its context, and
arXiv:2511.19477 states plainly: "prompt injection attacks make general-purpose
autonomous operation fundamentally unsafe" and "Reducing risk by 90% is
insufficient for autonomous systems handling sensitive data."

**Our mitigation is already strong and mostly pre-existing:** `.mcp.json:74` pins
`--allowed-hosts localhost`, so the browser is confined to our own app. The
injected-content vector would require our own frontend to render attacker text --
plausible in principle (the cockpit renders LLM-generated agent output and news
text into the DOM) but far below the risk of a general web-browsing agent. Worth
noting rather than treating as decisive. The paper's core prescription is the same
one C7/C7b arrive at independently: "safety must be enforced through
deterministic, programmatic constraints instead of probabilistic reasoning" and
"use specialized agents with least-privilege scopes" -- i.e. an allowlist, not an
instruction in qa.md telling Q/A to please not click.

The paper's own taxonomy endorses precisely the shape proposed here: its
"Assistant Agents" have "read-only capabilities; it lacks interaction tools (such
as click or type) ... This pattern provides high utility with minimal risk."

### C8 -- Independent-verification literature: does a verifier need its OWN evidence?

**Yes -- and Anthropic's own canonical reference for this project already does
exactly what is being proposed.** This is the strongest single piece of evidence
in the assessment, and it is pro-change.

From https://www.anthropic.com/engineering/harness-design-long-running-apps
(accessed 2026-07-20, read in full) -- the doc CLAUDE.md names as the canonical
harness reference:

> "When asked to evaluate work they've produced, agents tend to respond by
> confidently praising the work—even when, to a human observer, the quality is
> obviously mediocre."

> "Separating the agent doing the work from the agent judging it proves to be a
> strong lever to address this issue."

> "I gave the evaluator the Playwright MCP, which let it interact with the live
> page directly before scoring each criterion and writing a detailed critique. In
> practice, the evaluator would navigate the page on its own, screenshotting and
> carefully studying the implementation before producing its assessment."

> "the evaluator used the Playwright MCP to click through the running application
> the way a user would, testing UI features, API endpoints, and database states."

Note the phrasing: the evaluator navigating "on its own", explicitly contrasted
with "scoring a static screenshot". Our current design is the static-screenshot
case, and it is the case Anthropic implicitly rejects.

**Counterweight from the same source** (cost, honestly reported):
> "Because the evaluator was actively navigating the page rather than scoring a
> static screenshot, each cycle took real wall-clock time. Full runs stretched up
> to four hours."

**Classical V&V corroboration (cross-domain triangulation).** IEEE 1012 (System,
Software and Hardware Verification & Validation; current edition 1012-2016)
formalises independence along **three axes -- technical, managerial, and
financial**. Technical independence specifically requires the V&V effort to use
its own personnel and analysis rather than inheriting the developer's, so that
the V&V team "is not part of the development team" and there is "less chance of
outside influence over the analysis and findings". Mapping to our case: Main
producing the artifact that Q/A grades is a *technical independence* violation in
the IEEE sense -- the evaluator's evidence base is selected by the developer.
(Sourced via ivvgroup.com and the HHS EPLC IV&V Practices Guide; the IEEE
standard itself is paywalled at ieeexplore.ieee.org/document/8055462 and was NOT
read in full -- flagged as secondary-sourced.)

From Anthropic's multi-agent research post (read in full): "Each subagent also
provides separation of concerns—distinct tools, prompts, and exploration
trajectories—which reduces path dependency and enables thorough, independent
investigations." Note "**distinct tools**" is named as part of what makes the
separation real.

---

## D. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/agents/qa.md` | 4 | `tools: Read, Bash, Glob, Grep, SendMessage` -- allowlist, NO MCP | Binds Agent-tool path only |
| `.claude/agents/qa.md` | 165-183 | §1c live UI capture gate (phase-59.2) | BINDING; consumes a capture, does not take one |
| `.claude/agents/qa.md` | 345-348 | "NEVER Edit or Write" + non-mutating-Bash constraint | Would be in tension with mutating browser tools |
| `.claude/agents/qa.md` | 20-21 | RESTART CAVEAT (roster snapshot) | Agent-tool path only |
| `.claude/workflows/qa-verdict.js` | 111-118 | `agentType:'general-purpose'`, no `tools` arg | **The real enforcement gap** |
| `.claude/rules/frontend.md` | 73-102 | "Live-UI verification" workflow | Agent-agnostic prose, but step 1 needs a server start |
| `.claude/rules/frontend.md` | 54 | OpsStatusBar w/ kill switch + inline action buttons | Why clicking is unsafe here |
| `.mcp.json` | 68-79 | playwright server: fixed `--user-data-dir`, `--allowed-hosts localhost`, `alwaysLoad:false` | Fixed profile = the collision hazard |
| `.claude/settings.json` | 167-178 | deny list: alpaca mutators + bigquery execute-query | **No playwright entry** |
| `.claude/settings.json` | (permissions) | `defaultMode: bypassPermissions` | No prompt would fire on browser tools |
| `.claude/hooks/pre-tool-use-danger.sh` | 253-257 | gates `mcp__*__execute_sql` only | No browser coverage |
| `.claude/agents/researcher.md` | 133 | documents `mcp__playwright__browser_*` via ToolSearch for THIS role | Prose only, not a `tools:` grant |

### D9 -- Does any agent file already list MCP tools in `tools:`? **NO.**

`grep -rn "mcp__\|mcpServers" .claude/agents/` returns only two PROSE mentions,
both in `researcher.md:133` and `:137`. Neither is a frontmatter grant.

**Consequence: the syntax is UNPROVEN in this repo.** Nobody has ever exercised
an MCP grant in a pyfinagent agent file. The docs say it works; we have no local
confirmation. Given the pre-v2.1.208 silent-tool-less-launch failure mode (we are
on **v2.1.215**, so we would get a hard error instead -- verified
`claude --version`), the risk is now "loud failure", which is acceptable. But this
should be treated as a first-of-its-kind change and smoke-tested, not assumed.

### D10 -- Is the frontend.md capture workflow Main-specific or agent-agnostic?

**Textually agent-agnostic; operationally Main-shaped.** `.claude/rules/frontend.md:79-97`
never names an agent -- it says "Start a second dev server", "Capture with the
Playwright MCP", "Kill the :3100 server after capture", "move them to
`handoff/current/captures_<step>/`".

But three of the five steps require capabilities Q/A does not and should not have:

- Step 1 starts a dev server (`npx next dev --port 3100`) -- a long-running
  mutating Bash process. `qa.md:345-348` forbids mutating Bash.
- Step 3 kills the server (`lsof -ti tcp:3100 | xargs kill -9`) -- explicitly a
  `kill`, forbidden.
- Step 5 MOVES screenshot files into `handoff/current/captures_<step>/` -- a
  filesystem mutation, forbidden.

**This is the deepest practical objection to the proposal.** Even with Playwright
granted, Q/A cannot independently stand up the environment under test. Main would
still start :3100, meaning Main still controls *what app instance Q/A looks at*.
The independence gain is therefore **partial**: Q/A would author its own
observations (real gain -- it chooses the URL, the viewport, what to look at, and
can go looking for what Main did NOT screenshot), but not its own environment.
That is still a meaningful improvement over grading Main's chosen PNG, but it
should not be sold as full independence.

### D11 -- `qa.md` §1c in full + RESTART CAVEAT (phase-59.2 origin)

`qa.md:165-183`. Origin: phase-59.2, 2026-06-11, operator-approved. Substance: any
step whose contract/criteria/diff makes UI claims **cannot receive PASS** unless
its live_check references a LIVE Playwright MCP capture (`browser_navigate` +
`browser_snapshot` for structure/text; `browser_take_screenshot` for
visual/color/layout). Rationale cited in-line: "the 345,968-NAV bug shipped through
all three [code reading, unit tests, build greens]; only the live capture caught
it -- 55.1 precedent". Missing/stale capture caps the verdict at CONDITIONAL with
`violated_criteria: ["Missing_Assumption: live UI capture"]`. Figma MCP explicitly
cannot satisfy it.

RESTART CAVEAT verbatim: "this section binds Q/A spawns from the session AFTER the
one that authored it (roster snapshot semantics)." Also `qa.md:20-21`: "the
Agent-tool roster snapshots at session start -- an edit here takes effect next
session; run scripts/qa/verify_qa_roster_live.sh. The Workflow qa-verdict.js path
reads this file from disk live."

**Key structural reading:** §1c is written as a *document-inspection* gate. Q/A
checks that a capture EXISTS and is referenced. It never verifies the capture is
of the right page, the right state, or the current build -- "stale" is asserted,
not measured. That is precisely the gap an independently-capturing Q/A closes.

### D12 -- The :3000 breakage memory: what actually caused it, and does it constrain this?

Source: `feedback_second_next_dev_breaks_operator_3000.md` (2026-07-17, phase-64.1).

**Cause -- and this matters: it was NOT the Playwright MCP.** It was the Playwright
**test runner's** `playwright.config.ts` `webServer` ARRAY, which included the :3000
`npm run dev` entry. Playwright's reuse-probe transiently missed the running :3000,
so it tried to START that entry, which ran `npm run dev`'s `predev: rm -rf .next`
against the operator's live build -> ":3000 served HTTP 500 then 404". Secondary
cause: two `next dev` on the same project dir clobber the shared
`frontend/.next/app-build-manifest.json`.

**Does it constrain giving Q/A browser control? Indirectly, and importantly:**
- The Playwright **MCP** does not start dev servers and has no `webServer` concept,
  so granting MCP browser tools does NOT reintroduce THIS bug.
- It does establish the hazard CLASS: anything that lets a harness agent spin up a
  second Next dev server can take down the operator's cockpit. So the correct
  reading is: **grant browser tools, do NOT grant server-lifecycle powers.** If
  someone "solves" D10 by giving Q/A Bash rights to run `next dev`, they walk
  straight back into the 2026-07-17 incident with a less careful agent.
- Corollary safety rail already in the memory: "Always `curl :3000/login` (expect
  200) BEFORE and AFTER -- `/ -> 302` is just the middleware redirect and MASKS a
  broken build."

### D13 -- How often would this gate fire?

Measured against `.claude/masterplan.json` (840 steps total; 729 done / 85 pending
/ 15 deferred / 5 dropped / 3 superseded / 2 merged / 1 blocked):

| Population | UI-touching | Share |
|---|---|---|
| All 840 steps (tight regex: `frontend/src`, `.tsx`, `page.tsx`, `playwright`) | 69 | ~8% |
| All 840 steps (loose regex incl. dashboard/cockpit/render/component) | 168 | ~20% |
| **Pending 85 steps (tight regex)** | **22** | **~26%** |

Pending UI-touching step ids: 44.3, 44.5, 44.7, 44.8, 44.10, 61.3, 61.4, 63.4,
75.4-75.17.

**Reading:** the historical base rate is modest (~8-20%), but the FORWARD-LOOKING
rate is high -- roughly **one in four remaining steps**, driven by the phase-75
code-quality remediation queue (75.4-75.17) and the phase-44 UX-adoption backlog.
So this is not a rarely-exercised gate; it is about to be exercised heavily. That
argues FOR investing in it now. It also means any per-spawn latency cost is paid on
the other ~74% for no benefit unless the grant is conditional/lazy.

---

## Recency scan (2024-2026) -- performed

Three-variant query discipline applied:
1. **Current-frontier (2026-scoped):** "browser agent MCP prompt injection risk
   evaluator agent autonomous browser control security 2026".
2. **Last-2-year window (2025/2026):** "'LLM judge' evaluator independent evidence
   gathering vs static artifact verification agent 2025 2026".
3. **Year-less canonical:** "IEEE 1012 verification and validation independence
   technical managerial financial independence IV&V" (deliberately unscoped to
   surface founding prior art).

**Result: 3 new in-window findings that MATERIALLY change the recommendation, none
that supersede the canonical V&V independence principle.**

1. **arXiv:2511.19477 (Nov 2025), "Building Browser Agents"** -- the strongest
   in-window source. Its central claim ("safety must be enforced through
   deterministic, programmatic constraints instead of probabilistic reasoning";
   "use specialized agents with least-privilege scopes"; read-only Assistant Agents
   "lack interaction tools (such as click or type) ... high utility with minimal
   risk") directly supports the *narrow allowlist* variant over the *server
   wildcard* variant. Quantitative: ~85% WebGames vs ~50% for prior general-purpose
   browser agents, i.e. specialization did not cost capability.
2. **Playwright MCP's own concurrency `[!IMPORTANT]` callout** is current-doc
   material and is the operative constraint on our fixed `--user-data-dir`.
3. **Claude Code v2.1.199 / v2.1.208 / v2.1.215 behaviours** (requiresUserInteraction
   annotation; hard-error on unresolved `tools:` entries) are all within the last
   months and change the failure modes described above.

Additional in-window items identified but NOT read in full: arXiv:2606.05233
(793-episode browser-agent safety benchmark, Jun 2026), arXiv:2506.17318 (context
manipulation / corrupted memory in web agents), arXiv:2603.21642 (are AI-assisted
dev tools immune to prompt injection), arXiv:2411.15594 (Survey on LLM-as-a-Judge).
No 2024-2026 source was found that *contradicts* the verifier-independence
principle; the 2026 LLM-judge literature has moved toward rubric calibration and
jury ensembles rather than away from independent evidence.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/sub-agents | 2026-07-20 | Official doc | WebFetch (83.5KB persisted) + grep | `tools:` accepts `mcp__<server>` / `mcp__<server>__*`; omitting `tools:` inherits ALL MCP; `mcpServers:` string-ref SHARES parent connection, inline spawns a separate one |
| https://github.com/microsoft/playwright-mcp (README) | 2026-07-20 | Official doc | WebFetch + `curl` raw 61,323 B | "A persistent profile can only be used by one browser instance at a time ... To run several clients in parallel, start each additional client with `--isolated` or point it at a distinct `--user-data-dir`." |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-20 | Vendor eng. blog (project's canonical ref) | WebFetch + `curl` + full-text extract | "I gave the evaluator the Playwright MCP, which let it interact with the live page directly before scoring each criterion" |
| https://www.anthropic.com/engineering/multi-agent-research-system | 2026-07-20 | Vendor eng. blog | WebFetch | "separation of concerns—**distinct tools**, prompts, and exploration trajectories—which reduces path dependency and enables thorough, independent investigations" |
| https://arxiv.org/html/2511.19477v1 | 2026-07-20 | Preprint | WebFetch (native HTML per research-gate PDF chain) | "The most effective defense is specialization ... use specialized agents with least-privilege scopes"; read-only agents lack click/type = "high utility with minimal risk" |
| https://code.claude.com/docs/en/mcp | 2026-07-20 | Official doc | WebFetch (71.6KB persisted) + grep | `anthropic/requiresUserInteraction` forces a prompt even under `bypassPermissions` (v2.1.199+); "Calls from subagents" are never backgrounded; trust/injection warning |
| PRIMARY-SOURCE EMPIRICAL: `npx -y @playwright/mcp@0.0.76 --help` + JSON-RPC `tools/list` probe | 2026-07-20 | Direct probe of OUR pin | Bash | 23 tools, only 6 `readOnlyHint:true`; `--isolated` present in 0.0.76; `--caps` is ADDITIVE only; `browser_run_code_unsafe` present |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://ieeexplore.ieee.org/document/8055462 | Standard (IEEE 1012-2016) | Paywalled |
| https://www.ivvgroup.com/the-new-ieee-std-1012-2016-is-available/ | Industry | Secondary summary; used for the 3-axis independence claim |
| https://www.hhs.gov/sites/default/files/ocio/eplc/EPLC%20Archive%20Documents/14%20-%20IVV/eplc_ivv_practices_guide.pdf | Gov. practice guide | Corroborating only |
| https://www.sei.cmu.edu/blog/incorporating-agile-principles-into-independent-verification-and-validation/ | Academic/industry | Corroborating only |
| https://ndia.dtic.mil/wp-content/uploads/2012/systemtutorial/14604.pdf | Gov. tutorial | Corroborating only |
| https://www.academia.edu/10483728/... | Preprint mirror | Low tier |
| https://arxiv.org/pdf/2606.05233 | Preprint (793-episode browser safety benchmark) | Time; in-window, worth a follow-up read |
| https://arxiv.org/pdf/2506.17318 | Preprint (context manipulation, web agents) | Time |
| https://arxiv.org/pdf/2603.21642 | Preprint (prompt injection in dev tools) | Time |
| https://arxiv.org/abs/2411.15594 | Survey (LLM-as-a-Judge) | Tangential to the mechanics question |
| https://arxiv.org/html/2606.01629v1 | Preprint (judge benchmark) | Tangential |
| https://www.practical-devsecops.com/mcp-security-vulnerabilities/ | Community | Low tier |
| https://obot.ai/blog/mcp-prompt-injection-ai-agent-security/ | Vendor blog | Low tier |
| https://www.buildmvpfast.com/blog/agentic-browser-security-agent-credentials-2026 | Blog | Low tier |
| https://blog.cyberdesserts.com/ai-agent-security-risks/ | Blog | Low tier |
| https://deepeval.com/blog/llm-as-a-judge | Vendor doc | Tangential |
| https://www.evidentlyai.com/llm-guide/llm-as-a-judge | Vendor doc | Tangential |
| https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method | Vendor blog | Tangential |
| https://www.sciencedirect.com/science/article/pii/S2666675825004564 | Survey | Paywall-ish; tangential |

---

## Key findings (ranked by decision-weight)

1. **Anthropic's own canonical harness reference already gives the evaluator the
   Playwright MCP.** "I gave the evaluator the Playwright MCP, which let it
   interact with the live page directly before scoring each criterion and writing a
   detailed critique" -- harness-design blog, the doc CLAUDE.md cites as canonical.
   The proposal is not a novel risk; it is closing a gap against the reference
   design. (STRONGEST PRO)
2. **The proposed change, done via `qa.md` `tools:`, would NOT affect the PRIMARY
   launch path.** `qa-verdict.js:111-118` runs `agentType:'general-purpose'` with no
   `tools` argument -- the qa.md allowlist is not applied there at all. Fix the
   enforcement point or the change is misdirected. (STRONGEST "STOP AND RETHINK";
   flagged UNCERTAIN on exactly what general-purpose inherits -- probe first.)
3. **Concurrency is a real but fully solvable hazard.** Fixed `--user-data-dir` +
   two clients = vendor-documented conflict. `--isolated` (present in our pin 0.0.76)
   removes it, and costs us nothing because our capture workflow is auth-bypassed
   (`LIGHTHOUSE_SKIP_AUTH=1` on :3100), so there is no login state to lose.
4. **A server-level wildcard grant would hand the evaluator an RCE-equivalent tool.**
   `browser_run_code_unsafe` is in the pinned tool list; `--caps` cannot remove it;
   `defaultMode: bypassPermissions` with no playwright deny rule means nothing
   prompts. Grant NAMED tools only.
5. **Q/A still cannot stand up the environment under test** (cannot start/kill the
   :3100 skip-auth server, cannot move screenshot files -- all forbidden by
   `qa.md:345-348`). Independence gained is over *observations*, not *environment*.
   Real but partial. Do not overclaim it.
6. **The gate is about to fire often:** ~26% of pending steps (22/85) are
   UI-touching, concentrated in the phase-75 remediation queue. Historical base rate
   was only ~8-20%.
7. **§1c today is a document-inspection gate, not a verification gate.** It checks a
   capture is referenced; it cannot tell a correct capture from a cherry-picked or
   stale one. That is the actual integrity hole.
8. **The 2026-07-17 :3000 incident does NOT indict the MCP** (it was the test
   runner's `webServer`), but it does define the boundary: browser tools yes,
   server-lifecycle powers no.
9. **Latency tax is real and paid on every spawn.** MCP calls from subagents are
   never backgrounded ("Calls from subagents; Claude Code backgrounds only
   main-conversation calls"), and Anthropic reports their navigating-evaluator runs
   "stretched up to four hours". Q/A runs at `maxTurns: 30`.

## Consensus vs debate

- **Consensus:** verifier independence is right (IEEE 1012 technical independence;
  Anthropic "separating the agent doing the work from the agent judging it");
  least-privilege tool scoping is right (arXiv:2511.19477; Anthropic "distinct
  tools").
- **Debate / genuine tension:** Anthropic's evaluator CLICKS; arXiv:2511.19477 says
  read-only observation agents are the "minimal risk / high utility" sweet spot and
  that clicking agents need programmatic element-level blocking. **Our cockpit's
  kill-switch and run-now buttons resolve this tension in favour of the paper, not
  the blog** -- Anthropic's example app had no live financial control plane.

## Pitfalls (from literature + this repo)

- Granting `mcp__playwright__*` instead of named tools (RCE-equivalent tool rides
  along; no deny rule catches it).
- Assuming `--caps` restricts (it only adds).
- Two clients on one persistent profile (vendor-documented conflict).
- Assuming the `qa.md` edit binds the Workflow path (it does not).
- "Solving" the server-start gap by giving Q/A Bash server-lifecycle rights
  (re-creates the 2026-07-17 :3000 outage class).
- Trusting `readOnlyHint` as a mechanical filter (`browser_navigate` is annotated
  mutating; a purely-readOnly allowlist cannot load a page).
- Roster-snapshot lag: an Agent-tool-path edit is inert until the next session
  (`qa.md:20-21`).

## Application to pyfinagent (mapping to file:line)

If the operator proceeds, the evidence supports this specific shape (presented as
options, not a decision -- the caller owns the call):

- **Enforcement point first:** decide whether the grant lives in
  `.claude/agents/qa.md:4` (binds Agent-tool fallback only) or in
  `.claude/workflows/qa-verdict.js:111-118` (binds the primary path). Probe what
  `agentType:'general-purpose'` actually receives BEFORE writing either.
- **Named-tool allowlist, minimum viable set:** `mcp__playwright__browser_navigate`,
  `browser_snapshot`, `browser_take_screenshot`, `browser_console_messages`,
  `browser_wait_for`, `browser_close`. Excluded by design: `browser_evaluate`,
  `browser_run_code_unsafe`, and every click/type/fill/drag/upload/dialog tool
  (rationale: `.claude/rules/frontend.md:54` OpsStatusBar kill-switch).
- **Isolation:** a subagent-scoped inline `mcpServers:` entry carrying `--isolated`
  (avoids the `.mcp.json:73` fixed-profile collision entirely), OR keep the shared
  string-ref connection and accept that Main and Q/A drive one browser.
- **Defence in depth:** add `mcp__playwright__browser_run_code_unsafe` (and
  `browser_evaluate`) to the `.claude/settings.json` deny list around line 167-178
  regardless of which path is chosen -- currently there is no playwright entry at
  all, and `defaultMode` is `bypassPermissions`.
- **Do NOT** grant server-lifecycle Bash. Keep `.claude/rules/frontend.md:79-97`
  steps 1/3/5 with Main; amend the rule to say who does what, since it is currently
  silent on the actor (D10).
- **Smoke-test as first-of-kind:** no agent file in this repo has ever carried an
  MCP grant (D9). On v2.1.215 a bad entry hard-errors rather than silently
  producing a tool-less Q/A -- verify with `scripts/qa/verify_qa_roster_live.sh`.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 + 1 primary-source probe)
- [x] 10+ unique URLs total (26 total: 6 read-in-full + 19 snippet-only + 1 direct probe)
- [x] Recency scan (last 2 years) performed + reported, 3-variant query discipline visible
- [x] Full pages read (not abstracts) for the read-in-full set (arXiv read via native HTML per the PDF chain, not the abstract page)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (agents, workflows, rules, .mcp.json, settings, hooks, masterplan)
- [x] Contradictions noted (Anthropic-clicks vs paper-says-read-only; README flag-table vs prose on `--user-data-dir` default)
- [x] Claims cited per-claim
- [ ] GAP: `agentType:'general-purpose'` tool inheritance is INFERRED, not documented -- flagged UNCERTAIN in A3, needs an empirical probe
- [ ] GAP: exact failure mode of two clients on one profile is vendor-discouraged but not vendor-specified -- flagged UNCERTAIN in B4

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 19,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 9,
    "dry": false
  },
  "summary": "Granting Playwright MCP to Q/A is directionally correct -- Anthropic's own canonical harness-design reference gives its evaluator the Playwright MCP so it navigates and screenshots the live page itself rather than scoring a static capture, and IEEE 1012 technical independence says the same. But three project-specific facts constrain the how. (1) The PRIMARY Q/A launch, qa-verdict.js:111-118, runs agentType general-purpose with NO tools argument, so a qa.md tools: edit binds only the fallback path -- and general-purpose likely already inherits MCP. Probe before building. (2) A wildcard grant hands the evaluator browser_run_code_unsafe, documented RCE-equivalent, with defaultMode bypassPermissions and no playwright deny rule; --caps cannot restrict, only add. Use a named allowlist. (3) Our fixed --user-data-dir is a vendor-documented conflict for concurrent clients; --isolated is in our pinned 0.0.76 and costs nothing because the capture workflow is auth-bypassed. Also: Q/A cannot start the :3100 server, so independence is over observations, not environment; and unlike Anthropic's toy app our cockpit has kill-switch buttons, so read-only, not click.",
  "brief_path": "handoff/current/research_brief_playwright_qa.md",
  "gate_passed": true
}
```
