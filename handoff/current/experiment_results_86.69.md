# experiment_results -- step 86.69 (GENERATE, 2026-08-17)

Contract: `contract_86.69.md`. Research gate: PASSED (`wf_8515024d-2fe`; 9
sources in full, 31 URLs; brief 40,219 chars). Verbatim command evidence:
`live_check_86.69.md`.

## The shape of this GENERATE, stated up front

The product fix was ALREADY WRITTEN (phase-61.2, tested, 33/33 green) and
DARK. This step's GENERATE is therefore: (a) the cause and the fabrication
sites verified from source at HEAD; (b) the numbered ASK put to the
attending operator; (c) the operator's approval EXECUTED as an
operator-token action (never step authorship -- criterion 7 intact); (d)
the in-force chain proven; (e) the measurement queries staged against the
frozen baselines, accruing from tonight's cycle.

## Criterion 1 -- the CAUSE, demonstrated

Hypothesis (i) CONFIRMED; (ii)/(iii) REFUTED (brief A1-A5, re-verified):
phase-60.1 (`fa62b5fe`, 2026-06-11) restored the FULL pipeline after
gemini-2.0-flash's 2026-06-01 server-side retirement had silently pushed
every analysis onto the healthy lite path. The `_path` provenance field is
the deploy marker: last unstamped row 2026-06-10 18:38Z, first stamped
2026-06-11 10:17Z, zero-scores start that same day -- the break date is
2026-06-11, one to four days earlier than the audit-basis window, REPORTED
as a correction not adopted silently. All 211 empties are `_path=full`;
lite produced 0/19. The full path rides the CC rail (`$.rail=claude_code`:
257 rows, 84.4% zero-score) whose `--json-schema` is post-hoc validated,
not constrained -- the Gemini-shaped `_SYNTHESIS_STRUCTURED_CONFIG` is
honoured by nothing, and `final_synthesis.error='Failed to parse final
report.'` appears on 211/211 empties vs 0/38 healthy.

## Criterion 2 -- the fabrication sites, READ and quoted

`backend/services/autonomous_loop.py`:
- `:2172` `rec = synthesis.get("recommendation", {})` -- on the error dict
  this is `{}`;
- `:2179` `"recommendation": rec.get("action", "HOLD") ...` -- `{}` yields
  the fabricated uppercase `HOLD`;
- `:2191-2192` `"final_score": synthesis.get("final_weighted_score",
  synthesis.get("final_score", 0))` -- the error dict yields `0`.
The log-co-occurrence inference in the q1 diagnosis is REPLACED by this
source evidence. The lite writer is exonerated (title-case `Hold`).

## Criterion 3 -- absence as absence, and the consumer set

The phase-61.2 machinery (verified from source + its 33 tests):
- Flag ON, parse failure: `:2163-2171` raises `SynthesisDegradedError`
  BEFORE assembly -- the fabricated row is never built; the except routes
  to the LITE fallback, which produces a REAL scored row (measured 36.8%
  BUY conversion -- this is also the recovery lever for criterion 5).
- Flag ON, both paths fail: `:2252-2267` returns the honest degraded dict
  -- `recommendation=None, final_score=None, $._degraded + reason` --
  persisted as a NULL/NULL row then converted to None so it NEVER enters
  `decide_trades` (consumer proof in the 61.2 tests; NULL is already
  plumbed to the frontend; BQ `UPPER(recommendation)` on NULL yields NULL,
  never 'HOLD'; `_DOWNGRADE_RECS` hazard is covered by the documented
  unsafe-combination guard on the sibling flag).
- Flag OFF = legacy fabrication, deliberately pinned by
  `test_flag_off_legacy_fabrication_unchanged` -- the dark-flag doctrine,
  which is why this step COULD NOT simply edit the default path (the
  green suite encodes that policy; changing it belongs to the flag, not
  to a silent default flip).

## Criterion 7 -- the ASK, and how the approval was executed

ASK-1 (arm the flag) + ASK-3 (restart now vs session end) were put to the
ATTENDING operator via AskUserQuestion. Operator: **"Yes -- arm it"** and
**"Now"**. Execution followed the away-ops rail-1 token protocol (the
danger-hook blocked the write until the token was applied -- the gate
worked): token `ARM-SYNTHESIS-INTEGRITY-86.69` recorded in
`handoff/away_ops/pending_tokens.json` (same in-session channel as the
2026-07-07 precedent recorded in that file), `tokens_cursor` touched,
THEN `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true` appended to `backend/.env`.
The STEP promoted nothing; the operator did, through the machine gate
built for exactly this. ASK-2 (the 2,859-failure CC-rail structured-output
surface) is queued as its own masterplan step.

## The in-force chain (committed-is-not-in-force, discharged)

`backend/.env` mtime 15:06:04 -> `launchctl kickstart -k` -> new pid 14280
with lstart 15:06:17 (old pid 47562 gone; PID CHANGED asserted) -> health
endpoint OK -> the same pydantic loader on the same file yields `True`.
Direct memory read is impossible (`GET /api/settings/` does not expose the
flag -- the SAME blind spot the q1 doc measured for the diversity flags;
queued with ASK-2's step). The chain write-BEFORE-boot + loader-reads-file
is the in-force evidence, stated as a chain rather than claimed as a read.

## Criteria 4+5 -- measurement staged, accruing from tonight

Baselines frozen with their queries (live_check §4): PRE 37.8% zero-score /
POST 81.2% / BUY-conversion halves 57.7%->16.3%. The first post-arm rows
land with tonight's scheduled cycle (the restart happened BEFORE it,
operator-chosen); the same queries then report post-fix shares beside the
baselines. Expected mechanics, stated for falsification: full-path parse
failures now land as LITE rows (real scores, `_fallback_reason` set), so
the zero-score share should collapse toward the lite path's 0/19 and BUY
conversion toward its 36.8%.

## Residuals queued

- ASK-2 step (CC-rail structured-output mismatch, 2,859 parse failures) --
  filed as 86.108.
- The settings-endpoint blind spot (dark flags unreadable from the running
  process) -- rides 86.108's audit basis.
- 86.74's C6 row (RiskJudge in signals_log for a gated buy) becomes
  satisfiable as soon as the funnel produces a gated buy.
