# live_check_59.2 — MCP audit + integration: evidence

**Step:** 59.2. **Date:** 2026-06-11. **Required shape (masterplan):** version-check evidence + Playwright smoke capture + Figma capability-audit notes + config diffs.

## A. Playwright version check + bump (criterion 1)

| | |
|---|---|
| Pinned (pre-step) | `@playwright/mcp@0.0.75` (published 2026-05-07) |
| npm latest | **`0.0.76`** (published 2026-06-10) |
| Delta | Patch-level: 2 video-annotation tools (devtools; irrelevant here), `--output-max-size` flag, ~10 bugfixes (nav `waitUntil`, response-closure-with-no-tabs, arg reporting). All four pinned flags (`--executable-path`, `--user-data-dir`, `--allowed-hosts`, `--viewport-size`) survive — **zero breaking changes** |
| Action | **Bumped to 0.0.76** in `.mcp.json` |
| alwaysLoad | **`false` — kept.** Honest stale-premise note: the key ALREADY existed at `.mcp.json:91` (the masterplan's audit_basis assumed it was missing); the real gap was the CLAUDE.md discipline list, now fixed. Rationale: alwaysLoad is for every-turn tools and blocks session startup until the npx server connects (5s cap); playwright fires ~1-3x per UI-touching step with ~22 tool defs — episodic, mirrors `pyfinagent-backtest` |

**Mid-session caveat (disclosed per the contract):** editing `.mcp.json` does not respawn a connected stdio server — the smoke below ran on the session's already-connected **0.0.75** instance; the **0.0.76** pin takes effect at the next session/`/mcp` reconnect. This caveat is now permanently documented in CLAUDE.md's discipline entry + frontend.md's workflow section.

## B. Playwright smoke — live browser_navigate + snapshot against the running app (criterion 1)

`browser_navigate http://localhost:3000/login` → page title "PyFinAgent — AI Financial Analyst"; `browser_snapshot` (verbatim excerpt):
```yaml
- heading "PyFinAgent" [level=1]
- paragraph: AI Financial Analyst
- button "Sign in with Google"
- button "Sign in with Passkey"
- paragraph: Access restricted to authorized users
```
(First navigate transiently rendered "Internal Server Error" — a dev-server compile-on-demand race after the day's hot reloads; curl + re-navigate confirmed healthy 200 with the full page. Operator :3000 untouched throughout — read-only navigation only.)

## C. Binding qa rule + researcher awareness + workflow docs (criterion 2)

- **`.claude/agents/qa.md` §1c "Live UI capture gate" (NEW, BINDING):** UI-claims steps CANNOT receive PASS without a live `browser_navigate` + `browser_snapshot`/`browser_take_screenshot` referenced in the live_check; snapshot admissible for structure/text claims, screenshot required for visual claims; absence caps the verdict at CONDITIONAL with `Missing_Assumption: live UI capture`; cites the 55.1 precedent (the 345,968-NAV bug shipped through code review + tests + builds and only the live capture caught it); Figma explicitly excluded from satisfying the gate. Restart caveat embedded (binds Q/A spawns from the next session).
- **`.claude/agents/researcher.md`:** MCP-awareness block in the internal-exploration section (prefer a live Playwright snapshot over code inference for "what does the page show"; Figma is design-advisory, absent headless, never a gate).
- **`.claude/rules/frontend.md` "Live-UI verification" (NEW):** the full :3100 `LIGHTHOUSE_SKIP_AUTH=1` workflow — start command, stale-port cleanup, snapshot-vs-screenshot guidance, kill-after-capture + :3000 health check, the 55.1 §A disclosure template, capture-relocation to `handoff/current/captures_<step>/`, and the mid-session version caveat.

## D. Figma capability audit vs the Next.js cockpit (criterion 3)

- **Connector reality:** claude.ai session connector (`mcp__claude_ai_Figma__*`) — NOT in `.mcp.json`, absent in headless/cron runs → classified **design-advisory only**, can never satisfy the Q/A capture gate. Availability is checked per-session via the deferred-tool list.
- **What it CAN do for this repo:** (a) **code-to-design** — `generate_figma_design`/`create_new_file` can capture the live cockpit into a Figma file for design review (no repo Figma file exists today; this is the highest-value first use); (b) **design-to-code** for NEW dashboard views — `get_design_context` emits React + Tailwind by default (direct stack fit), with `/figma-*` skills available in connected sessions.
- **What it CANNOT do / caveats:** output is NOT token-compliant by default (must be reconciled to the navy/slate palette, JIT-safe literal classes, Phosphor-icons, no-emoji rules before commit — "treat Figma output as a draft, not a diff"); desktop-server features need a Dev/Full seat on paid plans; the remote server is free during beta but moves to usage-based pricing — at which point Figma calls fall under the project's LLM-cost approval rule (documented in frontend.md).
- **Docs landed:** `.claude/rules/frontend.md` "Figma MCP workflow" + CLAUDE.md discipline-section note (connector, not pinned).

## E. Verification command (criterion 4, verbatim)

```
$ python -c "import json; cfg=json.load(open('.mcp.json')); assert 'alwaysLoad' in cfg['mcpServers']['playwright']"
alwaysLoad: False | version: @playwright/mcp@0.0.76
$ grep -q 'Playwright' .claude/agents/qa.md && echo qa-ok          -> qa-ok
$ grep -qi 'figma' .claude/rules/frontend.md && echo frontend-ok   -> frontend-ok
$ test -f handoff/current/live_check_59.2.md                       -> (this file)
```
No emojis introduced (doc sections use plain text + Phosphor-icon references only). Agent-file edits (qa.md §1c, researcher.md awareness) carry the restart caveat inline.
