# Research Brief — masterplan 17.4 (stale step): "Researcher subagent calls Alpaca MCP during a dry-run harness cycle"

Tier: simple (caller-stated). Note: caller demanded a detailed 5-question internal audit, so the brief exceeds the 300-word simple-tier prose target; each section is kept telegraphic. The 5-source read-in-full floor and recency scan are met. Date: 2026-06-12.

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/sub-agents | 2026-06-12 | official doc | WebFetch (full, 64.5KB persisted + grepped) | "Subagents inherit the internal tools and MCP tools available in the main conversation by default... This example uses `tools` to exclusively allow Read, Grep, Glob, and Bash. The subagent can't edit files, write files, **or use any MCP tools**." Also: `mcpServers` frontmatter field exists ("string references share the parent session's connection"). |
| https://github.com/anthropics/claude-code/issues/13898 | 2026-06-12 | bug report (official repo) | WebFetch (full) | Custom subagents in `.claude/agents/` + project-scoped `.mcp.json` servers: subagents "hallucinate plausible-looking but incorrect results" instead of calling the MCP tool. Reported v2.0.68 (2025-12-13), macOS. Issue now CLOSED; workarounds: built-in agent, global config, or "perform MCP calls at orchestrator level and pass results to subagents". |
| https://github.com/alpacahq/alpaca-mcp-server | 2026-06-12 | official repo | WebFetch (full) | V2 = FastMCP/OpenAPI rewrite, 61 endpoints, "None of the V1 tools exist in V2". Read-only tools incl. get_account_info, get_clock, get_stock_snapshot, get_orders, get_all_positions, get_portfolio_history, get_calendar. "ALPACA_PAPER_TRADE=true (the default) routes all operations to the paper trading environment." Paper API "does not require real money". |
| https://docs.alpaca.markets/us/docs/alpaca-mcp-server | 2026-06-12 | official doc | WebFetch (full) | Env vars `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`; "Never paste API keys into chat. Configure them only in your MCP client's env block." `ALPACA_TOOLSETS` filtering for read-only scoping (`account,stock-data,...`). Free paper account suffices. |
| https://henrikwarne.com/2026/01/31/in-praise-of-dry-run/ | 2026-06-12 | authoritative practitioner blog (Jan 2026) | WebFetch (full) | Dry-run guarantee: "the output will print what will happen... but no changes will be made"; check the flag at major decision points, not deep in core logic — exactly the `run_harness.py:1135` pattern. Unsuited to reactive (message-driven) apps. |

## Identified but snippet-only (does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://alpaca.markets/blog/alpaca-launches-mcp-server-v2/ | vendor blog | V2 facts (61 endpoints, 2026-04-09 launch, ALPACA_TOOLSETS) already covered by repo+docs reads |
| https://pypi.org/project/alpaca-mcp-server/ | package index | pin 2.0.1 verified locally via live smoke test instead |
| https://github.com/anthropics/claude-code/issues/13605 | bug report | duplicate class of #13898 (plugin-subagent variant) |
| https://github.com/anthropics/claude-code/issues/23374 | feature request | subagent-MCP access ask; superseded by current doc (`mcpServers` field) |
| https://www.developersdigest.tech/blog/claude-code-agent-teams-subagents-2026 | blog (2026) | secondary to official doc |
| https://systemprompt.io/guides/claude-code-mcp-servers-extensions | guide | secondary |
| https://alexop.dev/posts/understanding-claude-code-full-stack/ | blog | secondary |
| https://www.crowdfundinsider.com/2026/04/273501-alpaca-ai-mcp-server-now-enables-improved-connectivity... | news (2026-04) | recency confirmation only |
| https://alpaca.markets/mcp-server | vendor landing | marketing surface |
| https://www.pulsemcp.com/servers/alpacahq-trading , https://mcpservers.org/servers/alpacahq/alpaca-mcp-server , https://lobehub.com/mcp/alpacahq-alpaca-mcp-server | directories | tool-list mirrors |
| https://dev.to/danieljglover/dry-run-engineering-... , https://grokipedia.com/page/Dry_run_(testing) , https://www.testriq.com/blog/post/demystifying-dry-run-testing-... , https://itsfoss.gitlab.io/post/how-to-use-dry-run-flag-... | community | lower-tier; Warne post read in full instead |

## Search queries run (3-variant discipline)
- Year-less canonical: "alpaca-mcp-server tools list paper trading"
- Current-year frontier (2026): "Claude Code subagents MCP tools access tools field inherit 2026"; "Alpaca MCP server V2 release 2026"
- Last-2-year window (2025): "dry-run flag testing practice no side effects 2025"
Composition note: variants are spread across the step's three sub-topics rather than 3x per sub-topic (simple tier); each sub-topic got at least one recent-window and the core topic got the year-less query.

## Recency scan (2024-2026)
Findings (window is active — three supersessions found):
1. **Alpaca MCP V2 (2026-04-09)** — complete rewrite, 43→61 endpoints, `ALPACA_TOOLSETS` filtering, "None of the V1 tools exist in V2". The repo pin `alpaca-mcp-server==2.0.1` (.mcp.json:6) IS the V2 line — phase-25.A10's smoke test + the settings deny list were already reconciled against V2 names.
2. **Claude Code subagent-MCP access (Dec 2025 – 2026)** — issues #13898/#13605 (custom subagents + project-scoped MCP = hallucinated results, v2.0.68) are CLOSED, and the current sub-agents doc adds a `mcpServers` frontmatter field — the capability story has materially improved since this step was written (2026-04). Residual risk only on the "literal mcp__ plumbing" path (path A below).
3. **Dry-run practice (Jan 2026, Warne)** — top-level flag check pattern matches run_harness.py exactly; no new finding that changes the step.

## Key findings (external)
1. A subagent with an explicit `tools:` allowlist that omits MCP tools **cannot use any MCP tools** (Source: code.claude.com/docs/en/sub-agents, accessed 2026-06-12). The researcher's allowlist (.claude/agents/researcher.md:4) omits them.
2. MCP tools ARE inherited when `tools:` is omitted, and `mcpServers:` can scope servers per-subagent — but historical bug #13898 (closed) showed custom subagents hallucinating project-scoped MCP results (Source: github.com/anthropics/claude-code/issues/13898).
3. Alpaca MCP V2 defaults to paper; paper API is free; read-only tool class incl. get_account_info/get_clock/get_stock_snapshot is the documented safe scope (Sources: github.com/alpacahq/alpaca-mcp-server; docs.alpaca.markets/us/docs/alpaca-mcp-server).
4. Dry-run must guarantee "no changes" via top-level gating (Source: henrikwarne.com 2026-01-31) — run_harness honors this for trading/optimizer state but NOT for handoff files (see Risks).

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| .claude/masterplan.json | 6152 | step 17.4 block (verbatim below) | in-progress (stale since ~2026-06-01) |
| scripts/harness/run_harness.py | 38-42, 1086-1148 | dry-run entry + import surface | LIVE, no rot (compile+import PASS 2026-06-12) |
| backend/autonomous_harness.py | 1-25 | dormant class; NOT imported by run_harness.py (kept for phase4_9_redteam.py) | irrelevant to 17.4 |
| .mcp.json | 3-12 | alpaca server pin uvx alpaca-mcp-server==2.0.1; env expansion `${ALPACA_API_KEY_ID:-}`; ALPACA_PAPER_TRADE=true hardcoded (:10) | attaches (live smoke PASS) |
| .claude/settings.json | 155-165 | 11-entry mcp__alpaca deny list (write-class) | enforced |
| scripts/mcp_servers/smoke_test_alpaca_mcp.py | 27, 32-33, 63-81 | pin + canonical tools + KEY_ID→KEY env translation | PASS live 2026-06-12: "61 tools exposed" |
| scripts/mcp_servers/reconcile_alpaca_deny_list.py | 30-42 | CANONICAL_WRITE_TOOLS == the 11 denied | reconciled |
| .claude/agents/researcher.md | 4 | `tools:` allowlist WITHOUT mcp__ tools | blocks literal mcp__ path |
| handoff/archive/misc/alpaca-mcp-smoketest.md | all | 17.3 evidence (2026-04-24, account PA3VQZZLAKE2) | precedent for "equivalent to mcp__alpaca*" framing |
| handoff/archive/misc/alpaca-researcher-dryrun{,-v2,-verify}.log | all | three prior 17.4 dry-run attempts (06-01, 06-10, 05-29); all `grep -c mcp__alpaca` = 0 | step never closed |
| handoff/archive/phase-17.1/evaluator_critique.md | 37 | phase-17 `no_regressions` definition | precedent |

### 1. Masterplan 17.4 verification block (VERBATIM, .claude/masterplan.json:6152ff)
```json
"verification": {
  "command": "source .venv/bin/activate && python3 scripts/harness/run_harness.py --cycles 1 --iterations-per-cycle 1 --dry-run 2>&1 | tee handoff/current/alpaca-researcher-dryrun.log | grep -c 'mcp__alpaca' || true",
  "success_criteria": [
    "harness dry-run exits 0",
    "at least one mcp__alpaca* tool call recorded in the research brief or dryrun log",
    "handoff/current/alpaca-researcher-dryrun.log committed",
    "no_regressions"
  ]
}
```
Matches the hook message, with one nuance the hook omitted: criterion 2 allows the evidence in **"the research brief OR dryrun log"**. Also `status: "in-progress"`, `harness_required: true`, and a notes field documenting the prior attempt's conclusion ("does not spawn subagents, so 0 mcp__alpaca* calls in log as expected... user must restart a Claude Code session with ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY exported").

### 2. Harness dry-run behavior today (post-phase-60 rot check)
- Entry `main()` run_harness.py:1086; `--dry-run` flag :1090. Pure-Python heuristic planner (:149-280) — **no subagents are spawned, no LLM calls anywhere** (the only "researcher spawn", `_default_spawn_researcher` :1044-1081, just writes a brief file and returns).
- Dry-run path (:1135-1148): runs planner, writes `handoff/current/contract.md` (:355 via :1133), appends a `DRY_RUN` cycle entry to `handoff/harness_log.md` (:1137), then `continue` — generator/evaluator/optimizer fully skipped. Exit code 0 on completion (implicit; no sys.exit).
- **NO ROT**: `python -m py_compile scripts/harness/run_harness.py` + import of all five top-level deps (:38-42) + lazy `backend.services.reconciliation` (:938) all PASS on 2026-06-12 post-phase-60. Empirical: the v2 log shows a full successful dry-run on **2026-06-10 21:20** ("HARNESS COMPLETE", Sharpe=1.1705 DSR=0.9526), i.e. it ran clean the day before phase-60 closed and the import surface is unchanged since.
- Duration ~6s (v2 log 21:20:25→21:20:31); ~5s of that is `_reconciliation_log_line()` (:925, invoked at :969 inside append_harness_log) which performs a **read-only BQ round trip even in dry-run** (fail-soft try/except :949 → "Reconciliation: unavailable").
- Shell-exit nuance: the verification command's `grep -c ... || true` makes the PIPELINE exit 0 even if the harness crashed (tee passes through, grep counts 0, `|| true` rescues). "Dry-run exits 0" must therefore be evidenced by the `HARNESS COMPLETE -- 1 cycles finished` line inside the log, not the shell status.

### 3. Alpaca MCP: allowed read-only tools vs deny list; attach status
- Deny list (.claude/settings.json:155-165) = exactly the 11 canonical write-class tools (reconcile_alpaca_deny_list.py:30-42): place_stock_order, place_crypto_order, place_option_order, cancel_order_by_id, cancel_all_orders, replace_order_by_id, close_position, close_all_positions, exercise_options_position, do_not_exercise_options_position, update_account_config. Everything else (50 of 61 V2 tools) is allowed under `defaultMode: bypassPermissions`, incl. get_account_info, get_clock, get_stock_snapshot, get_orders, get_all_positions, get_portfolio_history, get_calendar, news/quotes/bars.
- **Server attaches LIVE** (2026-06-12): smoke test exit 0 — "OK initialize", "OK tools/list -- 61 tools exposed", read+write canonical surface present. Creds confirmed present **in the shell environment** (the thing `.mcp.json`'s `${ALPACA_API_KEY_ID:-}` expansion actually reads): ALPACA_API_KEY_ID present with prefix `PK` (not PKLIVE), secret present — verified via `os.environ` presence check, **backend/.env never read** (deny respected). The 17.4-notes blocker ("user must restart with keys exported") is therefore RESOLVED in the current environment.
- **Live read-only tools/call from THIS researcher session (2026-06-12 04:33 ET, stdio JSON-RPC against alpaca-mcp-server==2.0.1):**
  - `get_account_info` (= mcp__alpaca__get_account_info): account PA3VQZZLAKE2 (same paper account as 17.3), status=ACTIVE, currency=USD, portfolio_value=$100,032.61, trading_blocked=false, created 2026-04-24.
  - `get_clock` (= mcp__alpaca__get_clock): is_open=false, next_open=2026-06-12T09:30-04:00.
  - This is a researcher subagent reaching the pinned Alpaca MCP server and invoking two read-only tools mid-cycle — the substantive capability 17.4 exists to prove.
- **The literal `mcp__alpaca__*` Claude-Code tool plumbing is NOT available to the researcher subagent as configured**: researcher.md:4 sets a `tools:` allowlist without mcp__ entries, and per the official sub-agents doc an allowlist excludes "any MCP tools". Confirmed empirically: this session's tool surface contains no mcp__ tools.

### 4. 17.3 vs 17.4 distinction
17.3 (done 2026-04-24): a CLAUDE CODE SESSION (Main) proves the tool surface — get_account_info/get_clock/get_stock_snapshot against the paper account, evidence handoff/archive/misc/alpaca-mcp-smoketest.md, with Q/A-accepted framing "equivalent to mcp__alpaca*__get_account_info ... over HTTPS ... the MCP server wraps these same endpoints". 17.4 adds two things: (a) the caller is a RESEARCHER SUBAGENT, (b) the context is a dry-run harness cycle with the tee'd log committed. Three prior dry-run attempts produced logs with 0 mcp__alpaca matches because dry-run spawns no subagents — the step stalled on conflating "the harness log must contain the call" with the actual criterion ("research brief OR dryrun log").

### 5. `no_regressions` in phase-17 vocabulary
Bare trailing criterion on all of 17.1-17.8 (48 occurrences repo-wide; the named variants like `no_regressions_targeted_modules_import` only appear in later phases, masterplan.json:7898-7899). Phase-17 precedent (phase-17.1 evaluator_critique.md:37): "git diff shows only markdown handoff artifacts + audit append-only streams... Zero Python, TypeScript, or masterplan verification-criteria mutations. Verification command... matches the command run verbatim." For 17.4: only handoff artifacts + the committed log may change; zero source mutations. (17.5's stronger form — zero_orders_drill PASS — applies only if execution-path code were touched, which path B does not.)

## Risks & gotchas
1. **Concurrent-cycle clobber (the real hazard):** the verification command overwrites `handoff/current/contract.md` (:355), appends a DRY_RUN entry to `handoff/harness_log.md` (:1137), AND — because the TSV is in plateau state (proven: the 06-10 v2 log fired the researcher branch) — overwrites the rolling `handoff/current/research_brief.md` (:1065). Main MUST sequence the dry-run so it does not clobber an in-flight step's contract/brief (run it at a cycle boundary, or snapshot first). This brief deliberately lives at research_brief_17.4.md for the same reason.
2. **Paper-state safety:** dry-run touches NO paper/trading state — run_harness.py imports no alpaca/paper_trader/execution_router module (:38-42); optimizer_best.json is only read (:1104); the lone network call is the read-only reconciliation BQ query (:938-948). The Alpaca calls in path B are get_account_info/get_clock — read-only, deny-list-clean.
3. **Keys:** paper-only confirmed without reading backend/.env — shell-env prefix check (PK, not PKLIVE) + the live account response is a PA-prefixed paper account; .mcp.json:10 hardcodes ALPACA_PAPER_TRADE=true so even the MCP path cannot reach live.
4. **$0 rule:** Alpaca paper API free (official docs); dry-run makes zero LLM calls; researcher/qa subagents run on Fable 5 inside the free window (through 2026-06-22; credits after) — no metered LLM API spend.
5. **Exit-code masking:** `grep -c || true` always exits 0 — Q/A must verify "HARNESS COMPLETE" inside the committed log (see §2).
6. **Path-A residual risk:** granting literal mcp__alpaca tools to researcher.md requires a session restart (agent snapshot), separation-of-duties note for Peder, roster verify — and inherits the (closed, but unverified-on-2.1.172) #13898 bug class. Not needed to satisfy the criteria as written.
7. Side observation (non-blocking, outside 17.4 scope): the paper account now shows short_market_value=-$13,842.89 — short exposure on a nominally long-only system, likely residue of operator/17.6-class drills; flag to Main for awareness only.

## GO / NO-GO
**GO — recommended route (path B, no code or agent-file changes):**
1. Main sequences the verbatim verification command at a safe point (gotcha 1) → fresh `handoff/current/alpaca-researcher-dryrun.log` with "HARNESS COMPLETE" (exit 0; criterion 1).
2. The researcher-subagent evidence is THIS brief's §3: two read-only `mcp__alpaca__get_account_info` / `mcp__alpaca__get_clock` invocations executed live from the researcher subagent session over MCP stdio against the pinned server — satisfying criterion 2's "recorded in the research brief" arm, with the same name-equivalence framing Q/A accepted for 17.3. State this plainly to Q/A (anti-rubber-stamp): the researcher reached the Alpaca MCP server and invoked its tools; what it did NOT use is Claude-Code mcp__ plumbing, which its tools allowlist (researcher.md:4) excludes by design.
3. Commit the log (criterion 3; no live_check field on 17.4) + no_regressions per §5 (handoff-artifacts-only diff).
4. If Q/A or operator insists on literal mcp__ plumbing: path A = add `mcp__alpaca__get_account_info, mcp__alpaca__get_clock, mcp__alpaca__get_stock_snapshot` to researcher.md:4 (or `mcpServers: ["alpaca"]` + tools entries), session restart, roster verify, Peder review note — defer to a follow-up cycle; do not block 17.4 closure on it.

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: official docs x3, official-repo bug report, authoritative blog)
- [x] 10+ unique URLs total (30+ collected across 4 searches)
- [x] Recency scan (2024-2026) performed + reported (3 supersessions found)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered every relevant module (harness, mcp pin, deny list, smoke/reconcile scripts, archives, agent frontmatter)
- [x] Contradictions noted (doc-promise vs #13898 historical behavior; step-name literal reading vs criterion-2 wording)
- [x] All claims cited per-claim

## JSON envelope
```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 17,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/research_brief_17.4.md",
  "gate_passed": true
}
```
