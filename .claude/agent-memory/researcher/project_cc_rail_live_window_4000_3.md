---
name: cc-rail-live-window-4000-3
description: Phase-4000.3 live-window facts -- rail flag was ALREADY True, grounding silently OFF on the rail, manual analysis does NOT feed decide_trades, and the CLI costUSD is a local list-rate estimate
metadata:
  type: project
---

Four facts measured 2026-08-06 for the CC-rail E2E live window (step 4000.3).
Re-derive before trusting; all anchors were live at the time.

**1. `paper_use_claude_code_route` was ALREADY `True` on the running backend.**
The step name frames 4000.3 as "flip the flag for the smoke window" -- but
`GET /api/settings/` returned True. So the smoke's PUT is a no-op, `pre_flag`
(`scripts/qa/smoke_cc_rail_e2e.py:262`) is True, and the `ExitStack` restore
(`:288-290`) already satisfies an operator end-state rule of "flag returns to
True on every path". `--keep-on` (`:410-412`) only CANCELS the restore, so
passing it there weakens crash-safety for no gain.
**Why:** a step premise stated as "flip X" is a claim about live state, not a
plan -- measure the flag before designing the window around flipping it.
**How to apply:** any step whose plan is "turn on flag F" starts by reading F.

**2. Grounding is silently OFF whenever the rail is on, and the "grounded"
steps land on the rail ungrounded.** `orchestrator.py:746` reads
`self.supports_grounding` off `general_client`; `ClaudeCodeClient.supports_grounding
= False` (`claude_code_client.py:583`). So Market/Competitor/Deep-Dive/Enhanced-Macro
(`orchestrator.py:1178/1187/1211/1322`) take the `else` branch to the rail with
`is_grounded=False` -- which ALSO means the 180s grounded lift in
`_resolve_step_timeout` (`:397-398`) never fires; the budget comes from
`rail_min` (`:399-401`) instead. A `grounded_client` is still constructed at
`:664` and is dead under claude-* pins.
**Why:** capability flags read off one client silently reconfigure unrelated
steps; nothing logs "grounding disabled".
**How to apply:** when a route flag changes the CLIENT, sweep every
`getattr(client, "supports_*")` read for downstream behavior changes. Any brief
claiming "the pipeline ran" must disclose grounding state.

**3. A manual `POST /api/analysis/` CANNOT reach `decide_trades`.** It writes
one `analysis_results` row (`analysis.py:210-311`) and nothing else -- no
positions, trades, or `strategy_decisions` (that heartbeat is cycle-only,
`autonomous_loop.py:1637-1666`). `decide_trades` (`autonomous_loop.py:1461`)
is fed by in-cycle `rank_candidates` output (`:890`); `screener.py` has ZERO
`analysis_results` references. The ONLY read seam is `_fetch_ticker_meta`
(`paper_trading.py:1159-1240`), which returns `{company_name, sector}` with
`paper_positions` at priority 1 over `analysis_results` at priority 2 -- so it
can only matter for a ticker whose sector is blank. Sector feeds the cap via
`pos.get("sector")` (`autonomous_loop.py:1417-1430`).
**Why:** "does a manual run touch the live book" looks like a big-surface
question; it collapses to one metadata seam once the consumption side is traced.
**How to apply:** trace CONSUMPTION, not persistence -- a write with no reader
is not interference.

**4. The Claude Code CLI's `costUSD` / `total_cost_usd` is a LOCAL list-rate
estimate, not a bill.** code.claude.com/docs/en/costs (accessed 2026-08-06):
"Claude Code computes the dollar figure locally from token counts priced at
standard list rates... and may differ from your actual bill", and for Max/Pro
"the session cost figure isn't relevant for billing purposes". Recency delta:
anthropic.com/news/higher-limits-spacex (2026-05-06) doubled Claude Code's
5-hour rate limits and removed the peak-hours reduction. Still NO published
token-denominated weekly-pool figure, so the smoke's `weekly_pool_note`
(`smoke_cc_rail_e2e.py:234-235`) stays the honest framing.
**How to apply:** any $0-metered claim rests on the `llm_call_log`
metered-complement rule, never on the envelope's cost fields.

Also found (queued, not fixed): the comment at `orchestrator.py:654-658`
claims `quant_exec_client` stays Gemini for `code_execution`; under
`gemini_model = claude-sonnet-4-6` it routes to the rail
(`llm_client.py:2113-2126`), so Scenario/Quant-Model run WITHOUT code
execution. Related: [[cc-rail-e2e-smoke-4000-2]], [[cc-rail-e2e-4000-1]].
