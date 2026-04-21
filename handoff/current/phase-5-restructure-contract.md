# Sprint Contract — Masterplan meta-action: phase-5 restructure

**Step id:** phase-5-restructure (meta-decision, not a regular masterplan step)
**Cycle:** 1 **Date:** 2026-04-19 **Tier:** complex

Parallel-safe: phase-scoped handoff.

## Research-gate summary

7 sources in full (Alpaca 2025-review, Alpaca multi-strategy backtest guide, crypto API comparison 2025, financial data APIs guide 2025, CoinStats best-crypto-API 2026, Awesome Systematic Trading, CFTC final rule + Mintz AI advisory), 22 URLs, three-variant queries, recency scan 2024–2026, 12 internal files inspected. Identified 8 files with US-equity-locked assumptions (`execution_router.py:37`, `screener.py:28`, `autonomous_loop.py:108`, `data_ingestion.py:93`, `markets.py:21`, `settings.py:63`, `regime_detector.py:5`, `data_server.py:190`). `gate_passed: true`. Brief at `handoff/current/phase-5-restructure-research-brief.md`.

## User directive (verbatim paraphrase)

> Move phase-5.1 / 5.2 / 5.3 to the very end of the masterplan. Add more sub-steps which include all the necessary implementation needed for this phase. Conduct full MAS harness to get a better overview of what we need to implement.

## Hypothesis

The 3 current placeholder steps (5.1, 5.2, 5.3) have no concrete scope and no real verification criteria. Replace them with 15 concrete sub-steps sequenced cross-cutting-first then market-by-market: 5.1 broker abstraction, 5.2 data-provider abstraction, 5.3 BQ schema, 5.4 risk engine → 5.5–5.9 market rollouts (crypto/options/FX/futures/international) → 5.10 ETF expansion → 5.11–5.14 intelligence + autonomous-loop wiring → 5.15 go/no-go integration gate. Move `phase-5` to the last slot in `mp["phases"]` so execution ordering puts it after all other phases.

## Scope

1. **Move** phase-5 to the end of `mp["phases"]`.
2. **Replace** the 3 existing steps with 15 new ones matching the research brief's 5.1–5.15 spec. Each step gets:
   - `id`, `name`, `harness_required: true`, `status: pending`, `depends_on` list, `contract: null`, `retry_count: 0`, `max_retries: 3`
   - `verification: { "command": "<verbatim from brief>", "success_criteria": [<list from brief>] }`
3. **Keep** the phase-level `gate: null` (no pre-start gate; owner will gate per-step).
4. **Flag** open owner decisions in the phase-level `open_issues` field: (a) EODHD API key + $19.99/mo budget, (b) IBKR TWS infrastructure choice, (c) which markets to prioritize.
5. **Update** phase name from "Multi-Market Expansion" to "Multi-Market Expansion (15-step)" to make the scope visible in the masterplan skill output.
6. **Preserve** the old 3 steps in an archive field for auditability.

## Immutable success criteria (authored for this meta-action)

- `phase_5_has_15_steps` — `phase-5.steps` length == 15, ids 5.1 through 5.15.
- `phase_5_moved_to_end` — `phase-5` is the last element of `mp["phases"]`.
- `each_step_has_verification` — every one of the 15 has `verification.command` non-null + `verification.success_criteria` non-empty list.
- `open_issues_recorded` — phase-level `open_issues` list has ≥3 entries.
- `json_valid` — masterplan.json parses cleanly.

## Plan

1. Write the restructured phase-5 JSON via a Python script that:
   - Reads `.claude/masterplan.json`.
   - Extracts the current `phase-5` record (archives the 3 old steps under `phase-5.archived_legacy_steps`).
   - Replaces `phase-5.steps` with the 15 new step dicts.
   - Removes `phase-5` from its current position and appends it to the end of `mp["phases"]`.
   - Adds `open_issues` + `path_decision` metadata.
   - Writes the file back.
2. Verify via Python reads.
3. Write `phase-5-restructure-experiment-results.md` with the diff.
4. Spawn Q/A for independent verification.
5. On PASS: append cycle block to `harness_log.md`; leave masterplan in restructured state.

## Out of scope

- No code changes outside `.claude/masterplan.json`.
- No actual phase-5 execution (user has explicitly deferred to end of queue).
- No handoff folder reorg.
- No removal of the phase-5.5 "External Data-Source Audit" phase (that is a separate, already-done phase with a similar id — do not conflate).

## References

- `handoff/current/phase-5-restructure-research-brief.md`
- `.claude/masterplan.json` → phase-5 (current 3-step) + phase-5.5 (distinct done phase)
- External: Alpaca, EODHD, OANDA, IBKR, Kaiko, CFTC/Mintz — see research brief sources table
- Internal anchors: `execution_router.py:37`, `screener.py:28`, `autonomous_loop.py:108`, `data_ingestion.py:93`, `markets.py:21`, `settings.py:63`, `regime_detector.py:5`, `data_server.py:190`
