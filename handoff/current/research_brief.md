# Research Brief — phase-59.2: MCP audit + integration (Playwright full, Figma frontend workflow)

Tier: moderate (caller-stated). Date: 2026-06-11. Researcher (Layer-3 MAS).
Note: caller's mandated brief shape (drafts + tables + audit) exceeds the 700-word
moderate ceiling; prose kept tight, sections kept as ordered. Tool-call budget
overran (~26 vs 18) due to 11 mandatory sub-questions — disclosed, not hidden.

## 1. Version-delta table: @playwright/mcp 0.0.75 → 0.0.76

| Item | 0.0.75 (pinned, `.mcp.json:84`) | 0.0.76 (npm `latest`) |
|---|---|---|
| Published | 2026-05-07T23:10Z | 2026-06-10T00:16Z (1 day old; `next`=0.0.76-alpha-2026-06-10) |
| Our flags | all valid | `--executable-path`, `--user-data-dir`, `--allowed-hosts`, `--viewport-size` ALL still in README flag list — **no breaking flag changes** |
| New tools | — | `browser_video_show_actions`/`browser_video_hide_actions` (devtools caps; irrelevant to us) |
| New flags | — | `--output-max-size` (caps tool-response size; optional, useful if snapshots bloat); `--browser moz-firefox` BiDi (irrelevant) |
| Bugfixes relevant to us | — | response closure when no open tabs; `waitUntil:'commit'` on back/forward nav; improved invalid-tool-arg reporting; regex validation; path-traversal checks in static serving; writable cache dir for user data (neutral — we pin `--user-data-dir`) |
| Tools we use | `browser_navigate`/`snapshot`/`take_screenshot`/`evaluate`/`click` | unchanged |

(Sources: github.com/microsoft/playwright-mcp/releases + README, read in full 2026-06-11; npm registry `time` query.)

**Recommendation: BUMP to 0.0.76 + smoke** (the criterion's preferred branch).
Delta is patch-level, bugfix-heavy, zero breaking changes for our flag set.
Risk = 1-day-old release; mitigation = the mandatory smoke; rollback = revert
the one-line pin + reconnect. **Execution note:** stdio MCP servers spawn at
session start — after editing `.mcp.json`, reconnect via `/mcp` (or note in
live_check which version served the capture; `npx -y @playwright/mcp@0.0.76 --help`
separately proves the new package resolves with our flags).

New capabilities NOT to adopt now: `browser_run_code_unsafe` is documented
RCE-equivalent ("executes arbitrary JavaScript in the Playwright server
process") — do not enable. `--caps testing` (`browser_verify_text_visible`
etc.) is genuinely useful for deterministic UI assertions — note as a future
option, out of 59.2 scope. Network-mock caps unneeded (we verify against the
live backend).

## 2. Best-practice config vs ours

README guidance: accessibility `browser_snapshot` is "better than screenshot"
for agent ACTIONS (structured, no vision model); `browser_take_screenshot`
is for visual evidence only ("can't perform actions based on the screenshot").
2026 ecosystem confirms: snapshots now incremental by default (token win);
verify-* assertion primitives exist behind `--caps testing`; vision caps only
for canvas-rendered apps. **Evidence doctrine for us: snapshot for text/value
claims, screenshot for color/layout claims — 55.1 used both, keep both
admissible in the qa rule.** Our config is already best-practice-aligned:
isolated-ish dedicated profile (`--user-data-dir` in-repo), `--allowed-hosts
localhost` (security), fixed `--viewport-size 1440,900` (deterministic).
Headed (default) vs `--headless`: keep headed — operator can watch (Glass
Box), and the 55.1/56.1-proven shape shouldn't churn; README recommends
headless only for CI/containers.

## 3. alwaysLoad — recommendation: `false` (KEEP current value; fix the docs)

**Stale-audit finding:** `.mcp.json:91` ALREADY has `"alwaysLoad": false` on
playwright — the masterplan audit_basis ("NO alwaysLoad key") is stale; the
verification command's python assert passes TODAY. The real gap is
**CLAUDE.md:64-68**: the discipline list names only the four pyfinagent-*
servers, and its line-ref sentence (".mcp.json:44,55,66,77") omits :91.

Rationale for `false` (official doc, code.claude.com/docs/en/mcp, read in
full): alwaysLoad is "for a small number of tools that Claude needs on every
turn"; it "blocks startup until the server connects, capped at the standard
5-second connect timeout"; requires v2.1.121+ (local 2.1.172, OK). Playwright
fires ~1-3x per UI-touching step (rare-event), ships ~22 core tool defs, and
is npx-spawned (cold-start) — alwaysLoad:true would tax every session start
for a rarely-used server. Mirrors `pyfinagent-backtest: false`.

CLAUDE.md bullet draft:
`- playwright -- alwaysLoad: false (fires ~1-3x per UI-touching step; ~22 tool defs; npx cold-start would block session startup if true). .mcp.json:91.`

## 4. qa.md BINDING rule — placement + wording

Placement: new `### 1c` under "Verification order", directly after `### 1b`
(frontend lint, the phase-23.2.24 precedent with identical "REQUIRED if diff
touches X" shape). qa.md current state verified post-59.1 (`model: fable`,
maxTurns 30). NOTE: the caller's "5-item harness-compliance audit" lives in
Main's SPAWN PROMPT (feedback memory), not in qa.md — 1c is the right home
for the deterministic leg; Main should also add one line to its spawn-prompt
template. Draft:

```
### 1c. Live UI capture gate (BINDING -- REQUIRED if the step makes UI claims)

phase-59.2: any step whose contract, success criteria, or diff makes UI
claims (a page renders X, component/layout/color/copy changes, anything
shipping pixels under frontend/src/) CANNOT receive PASS unless
handoff/current/live_check_<step_id>.md references a live Playwright MCP
capture against the RUNNING app: browser_navigate + browser_snapshot
(structure/text claims) and/or browser_take_screenshot (visual claims).
Code reading, eslint/tsc, unit tests, and greps are NOT UI evidence
(CLAUDE.md goal-post-away-review rule, 2026-06-10). Deterministic check:
grep the live_check for browser_navigate/browser_snapshot/screenshot
evidence; if auth was bypassed, the :3100 LIGHTHOUSE_SKIP_AUTH disclosure
(.claude/rules/frontend.md "Live UI verification") must be present.
Missing capture => verdict CONDITIONAL at best;
violation_type: Missing_Assumption (claimed UI state never observed).
```

## 5. frontend.md drafts (place both after "## Auth Flow", frontend.md:68-71)

### 5a. Live UI verification (Playwright MCP + :3100 skip-auth)

```
## Live UI verification (Playwright MCP + :3100 skip-auth)

Any step claiming UI behavior MUST verify against the RUNNING app via the
Playwright MCP: browser_navigate + browser_snapshot (structure/text) and/or
browser_take_screenshot (visual). Code reading is not UI evidence; qa.md
check 1c makes this binding.

NextAuth wall: NEVER touch the operator's :3000 instance. Start a second dev
server using the middleware bypass (frontend/src/middleware.ts:24):

    cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100

- capture against http://localhost:3100 (same code, same live backend :8000)
- kill the :3100 server after capture; verify :3000 still answers 302
- the live_check MUST disclose the method (second server, env flag, kill,
  :3000-untouched check) -- see handoff/archive/misc/live_check_55.1.md §A
- liveness-only smokes need no bypass: :3000 serves /login unauthenticated
```

### 5b. Figma MCP workflow (frontend design work)

```
## Figma MCP workflow (frontend design work)

The Figma MCP is a claude.ai session connector (tools mcp__claude_ai_Figma__*),
NOT pinned in .mcp.json. It exists only in operator-attached interactive
sessions and is ABSENT in headless/cron runs (run_harness.py, scheduler,
claude -p) -- never make a step's verification depend on it.

When present, use for:
- Design-to-code (new dashboard views): get_design_context on a Figma node
  URL ("copy link to selection") returns React + Tailwind by default --
  matches this stack. Reconcile output against this file's token rules
  (navy/slate palette, Phosphor icons, scrollbar-thin) before shipping.
  get_screenshot / get_metadata / get_variable_defs for layout + tokens.
- Code-to-design (design review): generate_figma_design captures the live
  cockpit UI into a Figma file; create_new_file starts mockups in drafts;
  use_figma edits files (load the /figma-use skill FIRST -- mandatory).
- Seats/cost: remote server works on all seats and plans, free during beta
  (will become usage-based paid); desktop server needs Dev/Full seat on a
  paid plan + the desktop app (selection-based workflows are desktop-only).
No Figma file is needed to start (create_new_file); design-to-code needs an
existing node URL.
```

### 5c. researcher.md awareness (placement: after "Both halves feed the same
output report.", researcher.md:~96)

```
UI-step tooling: for UI-related research, the project-pinned Playwright MCP
gives live-UI ground truth (browser_navigate + browser_snapshot of the
running app beats inferring UI state from code). The Figma MCP is a
claude.ai session connector -- available interactively, ABSENT in
headless/cron runs; treat designs it returns as session-only evidence.
```

qa.md one-line Figma awareness (constraints section): "Figma MCP evidence is
session-connector-dependent (absent headless) — never require it for PASS."

## 6. Figma capability audit vs the Next.js cockpit (for live_check)

CAN: (a) design-to-code emitting React+Tailwind (default output — direct fit
for Next.js 15 cockpit); (b) get_variable_defs → design tokens (maps to 47.5
tokens lib); (c) code-to-design: generate_figma_design captures live cockpit
pages into Figma for operator design review (remote-only, select clients —
present in this claude.ai session per server instructions); (d) Code Connect
mapping of cockpit components (ui/Button, StatusBadge) to a Figma library;
(e) FigJam diagrams (generate_diagram) for architecture docs; (f) whoami
confirms the operator's seat. CANNOT/caveats: absent in headless/cron; no
repo Figma file exists today (code-to-design would CREATE the first design
source of truth — that, not design-to-code, is the near-term value); generated
Tailwind may use generic palette → must be reconciled to navy/slate rules;
desktop-selection flows need the desktop app; beta → future usage-based
pricing (LLM-cost-approval rule applies when it goes paid).

## 7. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.mcp.json` | 79-92 | playwright block; pin at :84; `alwaysLoad: false` at :91 | bump :84 to 0.0.76; key already present |
| `CLAUDE.md` | 38-42 | session-level Playwright UI rule (2026-06-10) | exists; 59.2 adds qa-side enforcement |
| `CLAUDE.md` | 64-68 | alwaysLoad discipline list | STALE: add playwright bullet; line-ref sentence omits :91 |
| `.claude/agents/qa.md` | 52-118 | "Verification order"; 1b at :75-100 | insert 1c after :100; fable pin from 59.1 confirmed |
| `.claude/agents/researcher.md` | ~89-96 | internal-exploration protocol | insert UI-tooling note after :96 |
| `.claude/rules/frontend.md` | 68-71 | Auth Flow section | insert 5a+5b after :71; rule 5 at :40 (visual verification) is the precursor |
| `frontend/src/middleware.ts` | 24 | LIGHTHOUSE_SKIP_AUTH bypass | verified verbatim |
| `handoff/archive/misc/live_check_55.1.md` | §A | canonical skip-auth disclosure + captures | facts extracted |
| `handoff/archive/phase-56.1/contract.md` | 34 | second :3100 use | confirms repeatable pattern |

Smoke path verified live 2026-06-11: backend `/api/health` 200 (v6.43.0,
mcp_servers ok); frontend :3000 → 302 /login. A `browser_navigate
http://localhost:3000` + `browser_snapshot` (login page, no auth needed)
satisfies "live browser_navigate + snapshot against the running app".

## 8. Sources

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| github.com/microsoft/playwright-mcp/releases | 2026-06-11 | release notes (official) | WebFetch full | 0.0.76 full delta: 2 video tools, --output-max-size, 10 bugfixes, no breaking flags |
| github.com/microsoft/playwright-mcp (README) | 2026-06-11 | official doc | WebFetch full | flag list (ours all valid), snapshot-vs-screenshot doctrine, caps, run_code_unsafe=RCE, isolated/headless guidance |
| code.claude.com/docs/en/mcp | 2026-06-11 | official doc | WebFetch full (49KB persisted + grepped) | alwaysLoad exact semantics: "small number of tools... every turn"; blocks startup 5s cap; v2.1.121+; tool search default-on |
| developers.figma.com/docs/figma-mcp-server/ | 2026-06-11 | official doc | WebFetch full | remote vs desktop variants; free-during-beta → usage-based paid |
| help.figma.com/.../32132100833559 (MCP guide) | 2026-06-11 | official doc | WebFetch full | remote = ALL seats/plans; desktop = Dev/Full seat paid plans; code-to-design "from live UI"; link-to-selection input |
| developers.figma.com/docs/figma-mcp-server/tools-and-prompts/ | 2026-06-11 | official doc | WebFetch full | full 18-tool table; get_design_context default React+Tailwind; remote-only set; /figma-use skill mandatory for use_figma |

Plus non-WebFetch primary evidence: npm registry (`npm view @playwright/mcp
version|dist-tags|time`) — 0.0.76 latest, publish dates.

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched |
|---|---|---|
| playwright.dev/docs/test-agents | official doc | budget; verify-* primitives covered via search synthesis |
| playwright.dev/mcp/snapshots + /mcp/introduction | official doc | incremental-snapshot default noted from snippet |
| playwright.dev/docs/getting-started-mcp | official doc | duplicate of README content |
| testcollab.com/blog/playwright-mcp | blog 2026 | setup-level, below quality bar |
| testdino.com/blog/playwright-ai-ecosystem | blog 2026 | ecosystem overview only |
| mcp.directory/blog/playwright-browser-mcp-guide-2026 | community | low weight |
| morphllm.com/playwright-mcp | vendor blog | token-cost claim (114K vs 27K) noted as single-source |
| shipyard.build/blog/playwright-mcp-screenshots | blog | screenshot workflow anecdote |
| medium.com/@adnanmasood/... field guide | blog | tertiary |
| figma.com/blog/introducing-figma-mcp-server | vendor blog | superseded by current official docs |
| github.com/figma/mcp-server-guide | official repo | guide duplicates help-center content |

### Query log (3-variant discipline)

1. "Playwright MCP agent UI verification workflow 2026" (current-year frontier)
2. "Figma MCP server dev mode updates 2026" (current-year frontier)
3. "Playwright MCP best practices coding agent screenshots accessibility snapshot" (year-less canonical)
Plus direct registry/docs fetches (npm, GitHub releases, official docs).
Explicit note: no separate 2025-window query was run — both topics are
younger than 2 years (Playwright MCP launched 2025-03, Figma MCP 2025-06),
so the year-less canonical query IS the full-history query, and the
releases page read in full covers every version Feb-2026-back.

## 9. Recency scan (2024-2026)

Performed (queries 1-2 above, 2026-scoped). Findings: (a) 0.0.76 shipped
2026-06-10 — the version delta itself is the headline recency finding;
(b) 2026 ecosystem norm: accessibility-snapshot-first agents, incremental
snapshots by default, assertion primitives (`browser_verify_*`) behind
`--caps testing`, "MCP for discovery / codified .spec.ts for regression"
split, ~4x token premium of MCP-driven vs CLI-skill flows (morphllm,
single-source); (c) Figma 2026: remote server broadened (all seats/plans
during beta), code-to-design from live UI, deeper codebase integrations
announced. Nothing supersedes the chosen design; (b) mildly supports a
future `--caps testing` adoption.

## 10. Consensus vs debate

Consensus: snapshot-for-structure / screenshot-for-visuals; defer rarely-used
MCP servers (tool search); design-to-code needs a design-system source of
truth to be useful. Debate: MCP-driven browser flows vs codified test scripts
(token cost) — irrelevant at our ~1-3 captures/step volume; headed vs headless
for agent use — we keep headed (operator visibility, proven shape).

## 11. Pitfalls

1. Editing `.mcp.json` mid-session does not respawn the server — disclose
   capture-time version in live_check or `/mcp` reconnect.
2. qa.md/researcher.md edits snapshot at session start (CLAUDE.md rule) —
   the 1c rule is NOT live for this session's own Q/A; note in handoff +
   harness_log (same caveat 59.1 carries; verify_qa_roster_live.sh next session).
3. Don't enable `browser_run_code_unsafe` or broad `--caps` — RCE surface.
4. Figma connector absence in headless runs — any criterion requiring Figma
   evidence would be unverifiable in cron; keep Figma advisory-only.
5. The CLAUDE.md line-ref sentence (".mcp.json:44,55,66,77") goes staler with
   every .mcp.json edit — update it or de-brittle it while touching it.
6. Verification cmd greps `Playwright` in qa.md and `figma` (case-insensitive)
   in frontend.md — the drafts above satisfy both; keep capitalization.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (17)
- [x] Recency scan performed + reported (§9)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (§7)

Soft checks:
- [x] Internal exploration covered every relevant module (.mcp.json, qa.md, researcher.md, frontend.md, CLAUDE.md, middleware.ts, archives)
- [x] Contradictions/consensus noted (§10)
- [x] Claims cited per-claim
- [ ] Tool-call budget: overran moderate (~26 vs 18) — disclosed §0

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
