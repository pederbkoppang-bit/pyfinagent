---
name: cycle-number-86-44
description: harness_log cycle numbers — 482 of 1224 headers say "Cycle 1", NOTHING reads the number, and the runbook template is the bug source
metadata:
  type: project
---

Step 86.44 (sequence numbers for `handoff/harness_log.md`). Measured 2026-08-11 on
34,237 lines / **1,224 `## Cycle` headers**.

**The counter is write-only state — that is the design-deciding fact.** Every consumer
either ignores the number or keys on `phase=<step_id>` / the date:
`backend/api/backtest.py:1415` uses it only as a display string;
`backend/agents/harness_state_reader.py:143` splits on the bare literal;
`scripts/qa/verdict_history_86_21.py:196` matches `.*` over it;
`backend/slack_bot/scheduler.py:464` keys on date + `phase=`;
`.claude/hooks/lib/harness_log_gate.py:94` doesn't require the word "Cycle" at all;
`frontend/.../HarnessDashboard.tsx:448` keys the list on `key={i}`. So keep-vs-replace
has **zero migration blast radius** — do not scope it as a migration.

**The measured header population** (don't re-derive, but DO re-measure the denominator —
it moved 1189 → 1224 in ~3 weeks): 482 literal `Cycle 1` (39.4%), 54 `Cycle --` (no
number), 59 `Cycle N`/`N+1`…`N+58`, 11 step-ids-as-cycle (`4.15.3`, `16.59`,
`68-close`), 141 distinct integers duplicated, and the counter was **reset once**
(`Cycle 100`-`112` appears at both 2026-05-13 and 2026-07-17).

**Three traps for whoever plans this:**
1. **The runbook template IS the bug.** `docs/runbooks/per-step-protocol.md:334`
   literally contains `## Cycle N -- YYYY-MM-DD -- phase=X.Y result=…`. The 59 `N+k`
   headers are that line pasted verbatim. Renaming the number without making the
   placeholder unpasteable just regenerates them.
2. **The real data-loss defect is the write mode, not the number.**
   `scripts/harness/run_harness.py:976-980` (and `:1038-1039`) do
   `read_text()` → `write_text(existing + entry)` — a full-file read-modify-write, not
   `O_APPEND`. Two writers lose a whole cycle block. `run_harness.py:953` also takes
   `cycle` as the **loop index**, which is why `--cycles 1` writes `## Cycle 1` forever.
3. **`backtest.py:1415`'s `Cycle \d+` drops 160/1224 (13.1%) SILENTLY** — a non-match
   isn't an error, it just isn't a split point, so the orphaned body is glued onto the
   **previous** cycle. The Harness tab shows a merged history, not a short one.

**Different namespace, do not conflate:** `handoff/verdict_ledger.jsonl`'s `cycle`
field is mixed-type (ints 190-199, per-step restarts 1-5, and strings `"1-aborted"`
etc.; 8 duplicated values in 35 rows). That is step **86.46**, not this one.

See [[websearch-budget-is-session-shared]] — WebSearch was 200/200 exhausted before this
spawn; `curl https://html.duckduckgo.com/html/?q=…` + tag-strip was the working
fallback (rate-limits after ~2 probes; canonical prior art is better reached by direct
URL anyway).
