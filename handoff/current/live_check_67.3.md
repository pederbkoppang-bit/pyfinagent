# live_check 67.3 -- write-first diff hunks + floors intact, verbatim

Required shape (masterplan 67.3): "quoting the researcher.md diff hunks for the
write-first section and the floors intact, plus the fresh Q/A verdict JSON".

## Diff hunk: the new write-first section (researcher.md)

```
+## Write-first (non-negotiable)
+
+Create the brief file (`handoff/current/research_brief_<step>.md`) in your
+FIRST tool call, then write to it incrementally as each source is read --
+the brief on disk is the deliverable, not a final flush at the end. Even a
+session that cannot clear the gate must leave a partial brief plus an
+honest `gate_passed: false` envelope. Never end a turn on "now I'll read
+X" with nothing written: if your last line is a plan, do the write first.
+(Origin: the 2026-05-16 incident -- 132K tokens read, zero brief written.)
```

## Diff hunk: de-staled Domain context (the DSR figure was factually wrong)

```
-- pyfinagent: evidence-based trading signal system, May 2026 go-live
-- Current best: Sharpe 1.1705, DSR 0.9984
+- pyfinagent: evidence-based trading signal system; live paper-trading
+  (US + EU + KR paper markets)
+- Current-best params + metrics: single source of truth is
+  backend/backtest/experiments/optimizer_best.json (do not hardcode a
+  Sharpe/DSR figure here -- it drifts every optimizer run)
```
(optimizer_best.json:28 actual dsr=0.9525811 vs the hardcoded 0.9984.)

## Floors intact, run live 2026-07-09 (all 10 protected patterns)

```
OK: at least **5**            OK: >=10 candidate URLs / 10+ unique URLs
OK: Recency scan (mandatory)  OK: Peer-reviewed (source hierarchy)
OK: external_sources_read_in_full >= 5    OK: recency_scan_performed == true
OK: gate_passed: false        OK: >=20 sources    OK: ADVERSARIAL    OK: Multi-pass
```

Immutable command: `IMMUTABLE VERIFICATION 67.3: exit=0 PASS`

## Live demonstration (this session, before codification)

All three 67.x researcher spawns wrote incrementally from the first tool call:
research_brief_67_1/2/3.md existed on disk (438-23,069 bytes) minutes before their
agents' final messages -- observed and logged at 21:05 local.

## Fresh Q/A verdict JSON

Returned by qa-67-3 2026-07-09: `verdict: PASS, ok: true, violated_criteria: [],
certified_fallback: false`, 14 checks_run -- incl. an anti-rubber-stamp MUTATION TEST
(immutable command vs HEAD state -> exit=1; vs working tree -> exit=0: the command
genuinely discriminates), programmatic diagram-alignment verification (verticals at
columns [5,30,32,52] on all 15 lines), independent dsr=0.9525811 verification, and
the 10-pattern floor grep. Full JSON: evaluator_critique_67.3.md.
