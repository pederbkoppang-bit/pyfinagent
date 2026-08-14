---
name: counter-correctness-86-79
description: The attempt counter the Q/A reads is a GAUGE read as a COUNTER; prune has zero automatic callers so the saturation is LATENT; DEFAULT_KEEP's comment and its code disagree by one; Temporal and Step Functions ship opposite conventions under the same field name
metadata:
  type: project
---

Findings from the phase-86.79 research gate (2026-08-14). Brief:
`handoff/current/research_brief_86.79.md`.

**The hazard is LATENT, not live — check this before writing it up as a bug.**
`prune_wip_records` (`scripts/qa/qa_wip.py:206`) has **zero** automatic callers.
Enumerating command:
`grep -rn "prune_wip_records" --include='*.py' --include='*.sh' --include='*.js' --include='*.mjs' --include='*.ts' . | grep -v '^\./\.git/'`
→ 7 hits / 4 files, all being its own def, its checker
(`verify_wip_retention_86_36.py`), its mutation matrix, and comments. Anyone
describing saturation as a *currently firing* defect is overstating it.

**Measured off-by-ones (three, all independent of each other):**
1. `qa_wip.py:122-124` comments `DEFAULT_KEEP = 3` as "Current record + this many
   prior attempts" (⇒4 retained); `:221` does `records[keep:]` ⇒ retains 3 TOTAL
   = current + 2 priors. Measured on a temp sink: 6 written, 3 removed, 3 kept.
2. `qa.md:621-623` calls `records_retained` both "the count of **prior** Q/A
   spawns" and "the **attempt number**" in ONE sentence. They differ by 1. The
   attempt-number reading is correct, because the Q/A writes its own WIP record
   first (`qa.md:110-116`) and `list_wip_records` counts it.
3. Cross-framework: **Temporal `Maximum Attempts` is INCLUSIVE** ("1 means a
   single execution attempt and no retries") while **Step Functions `MaxAttempts`
   is EXCLUSIVE** (retries only, default 3). Same name, opposite unit, both
   tier-2 official docs.

**Saturation would kill F1b, and the live data is already past the window.**
`qa.md:706-711` escalates at attempt "5+", reading `records_retained`. With
`keep=3` that number cannot exceed 3. Measured 2026-08-14: 86.32=**5**, and
86.9/86.62/86.44/86.38=4 — five step-ids over `DEFAULT_KEEP`, with **86.32
sitting exactly ON the threshold**, i.e. the one case that would flip from
"escalate" to "attempt 3, carry on".

**Why: it is a GAUGE being read as a COUNTER.** Prometheus states the type rule
outright — "Do not use a counter to expose a value that can decrease."
The canonical remedy is the kernel's: retain a bounded window AND emit the loss
count separately (`PERF_RECORD_LOST` — "the kernel keeps how many records it
lost"). Google SRE's cross-process variant: carry the count **in the request
metadata**, with the work item, not in the caller.

**No literature names this defect class.** Four recency queries (2024-2026)
found nothing calling "counter saturation disables escalation" a documented
class. Cite the remedies (Prometheus counter/gauge, perf lost-records, SRE retry
budget) rather than hunting for a paper about the exact failure.

**Already-correct code to copy, not rebuild:** `verdict_history_86_21.py:98-113`
returns `None` (never 0) for `UNPARSEABLE`/`LEDGER_EMPTY`/`LEDGER_MISSING` —
that is the fail-closed uncomputable-counter pattern done right. Its input is
stale though: `handoff/verdict_ledger.jsonl` = 35 rows, last dated 2026-08-11,
hand-appended. And `attempt_budget.py` is monotonic-by-construction with
`NO_VERDICT` as a first-class outcome but is unwired + unpersisted — so **the
only durable attempt count in the live system is the one that can saturate.**

Related: [[qa-attempt-counter-86-21]], [[project_evaluator_counter_86_75]].
</content>
