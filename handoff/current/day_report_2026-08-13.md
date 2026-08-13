# Day report — 2026-08-13

**Session:** ~20:55–21:40 CEST. Backend pid 99231 (up since 2026-08-11 22:26), untouched.
**No restarts, no manual cycles, no flag promotions, no `.env` writes.** The 08-13
cycle was in flight the whole session; the freeze held.

---

## Headline

**The picker is not the binding constraint, and the ranking work should stop.**

I answered Q1 of `prompt_candidate_picker_research.md` first, as instructed, and it
did not resolve to any of the four options the prompt offered. It resolved to a
**dated regression upstream of the picker**: since a break between **2026-06-12 and
2026-06-15**, **81.2% of analyses are persisted as an empty row scored 0.0 and
labelled `HOLD`**. A `HOLD` never becomes a buy candidate, so four of every five
names the ranker picks are discarded before any ranking quality could matter.

The prompt's stop condition was "(c) or (d) → stop optimising the ranking". The
actual answer is neither, but the stop is firmer than either would have been.

Full derivation, every number with its population rule:
`handoff/current/q1_binding_constraint_86.59.md`.

---

## What I measured

Two independent populations, stated so the counts can be audited:

- **A** — `financial_reports.analysis_results`, `analysis_date` 2026-05-01..2026-08-13, 511 rows.
- **B** — 906,962 JSON-format lines (`^{"timestamp"`) from `backend.log` + all 6 rotated archives, 2026-07-24..2026-08-13.

| Finding | Evidence |
|---|---|
| The prompt's 21.1% BUY rate reproduces (21.7%) but **straddles a regime break** | PRE ..06-12: 103/251 = **41.0%**; POST 06-15..: 8/260 = **3.1%** — a 13.2x collapse on unchanged volume |
| The break is invisible at 90-day and even monthly resolution | Monthly reads 36.6% for June because June contains both regimes; only daily resolution shows the last BUY-class day is 2026-06-12 |
| Mechanism: failed analyses are recorded as valid verdicts | 211/260 POST rows have `final_score == 0.0`, **empty `summary` 211/211**, **NULL `debate_confidence` 211/211**, `recommendation='HOLD'`; text matching `degraded\|placeholder\|failed` appears in **0/211** — the rows are blank, not labelled |
| The engine did not turn bearish, it turned **absent** | `mean(final_score \| >0)` = 6.14 PRE vs 5.81 POST |
| Both halves broke, roughly equally | working-analysis share 62.2%→18.8% (3.3x); BUY conversion among them 57.7%→16.3% (3.5x); product 11.6x vs observed 13.2x |
| 31.6% of cycles never reach a trade decision | 19 cycles started (Step 1) on 17 days; only 13 reached Steps 6/7/8 on 11 days |
| The risk judge is mostly **not judging** | 60 of 97 verdicts (61.9%) logged `judge response unparseable; fallback verdict=APPROVE_REDUCED` |
| The risk gate is **not** what suppresses volume | Only **3** buy candidates reached the binding gate in 21 days (NTAP approved, HPE + CRWD blocked); 1 trade total |

Corroboration from the system's own record: the 2026-08-12 cycle wrote
`degradation: {'degraded': True, 'degraded_analyses': '6/6'}` with `breaker_tripped: True`.

**No gate was loosened and none is proposed.** The risk judge's 61.9% fallback rate
is a pipeline-quality defect, not a threshold to relax.

---

## Answers to the two state questions in the goal

1. **Did the 08-12 cycle write a `degradation` key?** **Yes** —
   `{'fallback_rate': '0/6', 'fallback_alarm_fired': False, 'degraded': True, 'degraded_analyses': '6/6'}`,
   alongside `breaker_tripped: True`, `duration_ms: 1,404,921` (~23 min), `n_trades: 0`.
   The peer's 86.38 stake is intact.
2. **Did 08-13's cycle run?** **Yes, and it was still running when I finished** —
   `cycle_id c7ac27f2`, started 18:00:00Z, `status: started`, no `completed_at`, at
   ~5,700s of a 10,800s budget. Its outcome is not in this report.

---

## Shipped

| Commit | What |
|---|---|
| `f6c2dbf4` | Q1 answer with full derivation |
| `bb08ee00` | **86.69 queued (P0)** — the emptiness regression, with 8 criteria; blocks 86.59 and 86.60 |
| `275585ce` | 86.59 research brief v2 preserved after a rail drop |

All three committed with **explicit pathspecs** — a peer session is live (4 sessions
listed at startup).

---

## The 86.59 research gate: failed again, and I stopped rather than pay for a third

The re-run I launched (`wf_a6ea31e7-9b9`) **dropped on the rail**: `envelope: null`,
`"subagent completed without calling StructuredOutput (after in-conversation nudge)"`.
An empty return is **NO VERDICT, never a pass**, so `gate_passed: false` stands and
**PLAN was not entered**. This is a different failure from yesterday's over-claim —
two consecutive failures, two distinct causes.

**Write-first did its job.** The brief is on disk at 61,837 chars with **59 distinct
URLs** (v1 had 13, which is exactly what failed the first run) and a recency section.
It substantively closes **both** assigned jobs — JOB 1, the residual/idiosyncratic
momentum gap the goal named as the thing to close first, is marked CLOSED with a
formula, a factor model and a disagreeing view; JOB 2's snippet-only table has 26
rows. What is missing is only the **final act**: `brief_status` is still `INCOMPLETE`
because the run never reached its tail.

**I deliberately did not re-run it.** 86.59 is now blocked by 86.69, so a third
~190K-token gate run would license ranking work that the same day's measurement says
cannot pay off. Cheapest path next session: either flip the brief's envelope via a
short fresh run, or defer 86.59 behind 86.69 entirely.

---

## Not started

`86.58`, `86.63`, `86.62`, `86.9`, `86.44`, `86.64`–`86.68` — the whole "THEN" chain.
The session went into Q1 and what Q1 turned up. I judged a measured P0 blocking the
P1 to be worth the whole session; the queue is untouched and unblocked.

Incidental corroboration for two of them, found without looking:
- **86.58** — 3 × `UNRECOGNISED recommendation 'new_buy_signal'` on 3 distinct days in population B.
- **86.63** — the recommendation vocabulary is **case-inconsistent**: `HOLD` 72 / `Hold` 23, `BUY` 4 / `Buy` 1, `Sell` 4 with no `SELL`. Recorded in 86.69's notes as an observation; **no criterion owns it**, so it is not queued work.

---

## What I could NOT verify — plainly

1. **The cause of the 06-12/15 break is not established, and I am not asserting one.**
   `git log --since=2026-06-11 --until=2026-06-16 -- backend/` shows only away-ops,
   Slack and alerting commits. Three untested hypotheses are recorded in 86.69:
   a restart putting *earlier* phase-60 changes into force; a model/provider change;
   an upstream data failure surfacing as the `QuantAgent ... NoneType` error.
   Finding the cause is criterion 1 of that step, not a claim in this report.
2. **I did not read the persist call site** that writes `final_score=0.0` with an
   empty summary. The link to the lite/degraded fallback is inferred from
   co-occurrence of log lines and row shape. Strong inference, not verified — and
   86.69 criterion 2 exists to replace it with source evidence or correct it.
3. **The 3 dark diversity flags could not be read from the running process.**
   `GET /api/settings/` returns 45 keys; a filter matching `sector|diversity|min_k`
   returned `sector_calendars_*` (positive control passed) but **none** of the three.
   Reading `backend/.env` is denied, so their live values remain unverified —
   which 86.59's criterion 4 requires and which needs another route.
4. **The BigQuery MCP was not attached** this session; per CLAUDE.md rule 6 I used the
   Python client with ADC. All queries were date-bounded and `LIMIT`ed.
5. **Population B starts 2026-07-24, not at the oldest archive.** The two oldest
   `.gz` files are plain uvicorn format with no `"timestamp"` field, so 4.28M of the
   5.19M concatenated lines carry no parseable date. Every log count inherits that
   21-day bound — I did not silently present it as "all archives".
6. **The 13.2x vs 11.6x residual is unreconciled**, stated rather than smoothed: the
   two decomposition factors are computed on overlapping subsets.

---

## Open asks (unchanged, none actioned)

`06-2` (credential rotation — the only time-sensitive one), `06-5`, `06-6`, `06-7`,
`06-8`, `06-24`, `06-25`. Plus `06-9` (promotion of the three dark diversity flags),
which item 3 above now makes harder to evaluate, since their live values cannot be
read from the running process.

---

## For the next session

1. **86.69 first.** It is P0 and it blocks the P1. Criterion 1 is finding the cause of
   the 06-12/15 break — start from the three hypotheses, and check whether a restart
   in that window put earlier phase-60 changes into force.
2. **Do not do ranking work on 86.59/86.60 until 86.69 closes.** The research is sound
   and the diagnosis is right; it is the sequencing that was wrong.
3. The 86.59 brief needs only its envelope flipped to `COMPLETE` — not new research.
4. The `THEN` chain (86.58 → 86.63 → 86.62 → 86.9/86.44 → 86.64–86.68) is untouched.

---

# ADDENDUM — 21:40 to 22:10 (session continued past the report)

The Stop hook correctly read Q1 as resolving to neither (c) nor (d), so the ELSE
branch applied and the THEN chain was live work. Then the operator redirected to
the harness itself. Both are covered below.

## 86.58 — research gate PASSED, criteria 1/2/3/6 measured

The gate returned clean on a **tight** prompt: `gate_passed: true`, 6 sources, 38
URLs, `brief_status: COMPLETE`, **no rail drop** (contrast: 86.59's ~4,500-char
prompt dropped; this one at ~2,500 returned). That is a usable signal about prompt
length, not proof — n=1 each way.

**The gate changed the design.** The single-boundary module already exists —
`backend/services/recommendation_vocab.py` (209 lines, phase-86.20), whose docstring
says it "is meant to be the ONLY one". It guards the **read** side only:
`portfolio_manager.py:128` canonicalises, but `paper_trader.py:452` assigns
`_pos_rec = reason` with **no parse step**. And the module predicted its own next
failure at `:95-105` — *"A caller that unwraps them back into a literal set has undone
the point"* — which is exactly what `portfolio_manager` does (imports only
`canonical_recommendation` at `:16`, hand-writes `_BUY_RECS` at `:60-64`).

Measured independently of the gate, and agreeing with it:

- **Criterion 1 PROVEN BY DRIVING**, control green.
  `scripts/qa/drive_86_58_dead_downgrade.py`: held row `'new_buy_signal'` + fresh
  `HOLD` → **no orders**; identical row carrying `'BUY'` → `('NTAP','SELL','signal_downgrade')`;
  `'swap_buy'` → dead. Cell D shows `sell_signal` pre-empts, so a `SELL` input
  would have gone green for the wrong reason.
- **Derived from source:** `_DOWNGRADE_RECS - _SELL_RECS = {HOLD}` — **`HOLD` is the
  only input that can ever reach `signal_downgrade`.** The `:208-218` warning is not
  describing a side effect; it is the rule's entire reachable domain.
- **Criterion 2:** `paper_positions` holds exactly 1 row and it is 100% off-vocabulary.
  Historically, `paper_round_trips.exit_reason` over all 32 round trips is
  `stop_loss_trigger` 16 / `swap_for_higher_conviction` 13 / `sell_signal` 3 —
  **`signal_downgrade` 0**. Positive control for that zero: `sell_signal`, the adjacent
  branch in the same function writing the same column, fired 3 times.
- **Criterion 3 blast radius, and it is acute:** NTAP is the only position, and 7 of
  its 9 re-evals since 07-24 are empty-summary HOLDs (5 with `final_score=0.0`). So
  promoting the 06-8 flags **today** would make the held row read `BUY`, meet a
  fabricated `HOLD`, and **sell the book's only position on a placeholder**. That is
  an argument to keep them dark until 86.69 closes — recorded for ask 06-8, **not acted on**.
- **Criterion 6:** the 86.20 line is present in source (1 occurrence) and **fired twice**
  in the driven run. Preserved, not quieted.
- Contract at `handoff/current/contract_86.58.md`, criteria copied **programmatically**
  (0 of 6 missing, positive control clean) — with the protocol-order breach **disclosed
  in the contract itself**, because a file-mtime check would have passed.

## Operator redirect — the harness repeating work

Measured over all 527 `wf_*` run dirs (step identified from the first agent
transcript; 459 matched, **68 unmatched and counted separately, not dropped**):

| Finding | Number |
|---|---|
| Runs that repeat a step already attempted | **268 / 459 = 58.4%** |
| Steps needing more than one run | 113 / 191 = 59.2% |
| Worst step | **9 runs** (36.8), then 8 (86.28), then 7 (86.21, 75.5) |
| Agents that dropped (started, never returned) | 106 / 1,288 = **8.2%** — reproduces CLAUDE.md's 8.6% |

**The reported cause needed correcting: the cost is Q/A, not the researcher.**
36.8 = 0 researcher / 9 Q/A. 86.28 = 0/8. 75.5 = 0/7. The researcher's maximum on any
step is 3.

Two root causes, both positive-controlled, both filed:

- **86.71** — `attempt_budget.py` is what CLAUDE.md says bounds the loop. It is tested
  and mutation-tested, and has **no runtime caller** (only CLAUDE.md, handoff artifacts,
  its own test, its own mutation matrix) and **no persistence** (`json.dumps` to a string;
  no open-for-write, no `json.load`), so it could not count across sessions even if wired.
- **86.72** — the operator's re-research leg has **no implementation on the live rail**.
  `research_needed` appears in exactly one file (`scripts/harness/run_harness.py`) and in
  **neither** Workflow script — and the Layer-3 loop runs on the Workflow rail. The run
  record shows it directly: 9 Q/A runs and 0 researcher runs on 36.8. Also: **tier is
  caller-declared** (`:202`), not researcher-assessed, which diverges from the stated design.
- **86.73** — the operator declined the menu and asked for research instead. Gate running
  as `wf_9e70310d-8ee` (tier `complex`) on: depth-scaling (raise complex floors vs fork
  2-3 parallel researchers) and who assesses difficulty. The `deep` tier stays **withheld**
  per `research-gate.js:190-200` — "Report the gap; do not close it unilaterally."

Note for whoever picks this up: the researcher is **not** the expensive role. Any proposal
to spend more there must argue it reduces Q/A repeats, or concede that it does not.

## Peer coordination

`pyfinagent-c8` queued **86.70** for the dropped-envelope recovery stage and corrected two
things in my framing, both accepted: recovery would **not** have rescued 86.59 (the marker
was `INCOMPLETE`, so the gate still fails — only the report improves), and
`verify_research_gate_workflow.mjs:910-911` already asserts that a dropped stage-1 fails
even with a COMPLETE brief, so my proposal would have inverted a mutation-tested assertion
rather than fixed a bug. Three of us now edit `research-gate.js` (86.70, 86.72, 86.73) —
**coordinate before touching it.**

## Still not done

`86.63`, `86.62`, `86.9`, `86.44`, `86.64`–`86.68`. And 86.58 has a passed gate, a contract
and four measured criteria but **no Q/A verdict**, so it stays `pending` — no step was
flipped tonight, and the version is unchanged at 6.93.220.

---

# ADDENDUM 2 — 22:10 to 23:45 (session close)

## 86.58 CLOSED — PASS, the only step flipped today

Ledger **FAIL → CONDITIONAL → PASS** (`wf_b127735e-55b`, `wf_1e709e75-776`,
`wf_bb75a26d-e5c`). Final verdict `ok:true`, `harness_compliance_ok:true`,
`violated_criteria: []`, 19 checks run. All 6 immutable criteria MET with **zero
production code changed** — the fix is operator-gated, so the deliverable is evidence.

Each failure was mine, and each is worth recording:

1. **FAIL** — I published a flag-ON blast radius of **1 of 1 (100%)** *without ever
   running with the flags on*. My script `assert`ed both flags `False` and aborted
   otherwise, using a hand-set `'BUY'` as a proxy. Measured properly: **0 of 2**. False
   in both halves. Third instance of the assert-the-property-not-a-proxy class; the
   auto-memory now carries the greppable signature.
2. **FAIL** — counts stale: I published "the book holds 1 position" **eight minutes
   after** DELL opened, a trade **I had recorded myself that hour**.
3. **CONDITIONAL** — I called a real measurement a dead end. `Settings()` reads
   `backend/.env`, and `paper_risk_judge_reject_binding` returns `True` *because the
   operator promoted it there*. The Q/A's phrase: **"Main understated its own evidence."**

**Two things the evaluator did that outrank the verdict:** it **closed a residual gap I
left open** (checked the launchd plist to prove no env var overrides `.env`, and
confirmed the sibling flag's code default is `False` while `Settings()` returns `True`);
and it **caught a broken probe in itself** — its first mutation survived and looked like
a finding until it read the source and found the probe, not the guard, was broken.

## The harness fix proved itself on live data

During cycle 3 the Q/A ran **both** counter sources: `qa_wip.py` returned **3**, the
retired `grep` on `harness_log.md` returned **0**. They disagreed exactly as phase-86.75
predicted — LOG runs after EVALUATE — and the ledger gave the right answer on a real
step. Under the old source a third CONDITIONAL would still have been available; under
the new one it was forbidden. The Q/A concluded the edit is **stricter against its
author, not laxer**, and judged its independence intact.

## Peer coordination

A peer restarted the backend at 20:30:59Z (99231 → 93024) on the operator's session-end
instruction — **3m45s after** I wrote `live_check_86.58.md`, falsifying its pid header
mid-evaluation. **I left it.** Editing an artifact while it is being graded is what the
freeze rule prevents; the Q/A caught the stale claim and the correction landed visibly
*after* the verdict. All process-sourced measurements were re-probed on 93024 and are
unchanged.

The peer also filed **P0 86.74**: the DELL trade I reported as good news is a **risk-gate
bypass** — the judge rejected it at 0% and a falsy-zero check inverted that into a
10%-of-NAV buy. **My "the book traded today" framing was wrong** and is corrected in the
86.69 addendum.

## Steps closed / filed today

| Step | State |
|---|---|
| **86.58** | **done (PASS)** — the only flip today |
| 86.69 | filed P0 — the empty-HOLD regression blocking the picker |
| 86.71 | filed — the attempt budget has no caller and no persistence |
| 86.72 | filed — the re-research leg absent from the Workflow rail |
| 86.73 | filed + **gate PASSED**, decision researched and settled |
| 86.75 | filed — the 5 audit changes, **shipped but unverified** |

Version **6.93.220** → bumped only by 86.58's flip, which is the changelog rule
shipped yesterday working as intended.

## Honest gaps at close

1. **86.75 is shipped and unverified**, and I both authored and audited it. Operator
   review of the `qa.md` edit is owed.
2. **86.59's gate has failed twice.** The brief is good (61,837 chars, 59 URLs, both
   jobs closed) and needs only its envelope flipped — not new research.
3. **The cause of the 06-12/15 break is still not established.** The 08-13 cycle showing
   no degradation makes it look **intermittent**, which weakens the single-root-cause
   hypothesis. n=1.
4. **The picker chain past 86.63 is untouched** — deferred on the operator's explicit
   "harness correctness first".
5. One `live_check` requirement (**flag values from the running process**) is satisfied
   only by **convergent positive-controlled inference**, not a direct read. A launch-time
   env var could still override `.env`; the Q/A closed that via the launchd plist, but a
   read-only route would close it properly.
