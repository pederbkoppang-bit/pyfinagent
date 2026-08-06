---
name: phantom-columns-82-54
description: Second live phantom-column defect (cost_budget_api -> llm_call_log). TWO step premises refuted; the fix is NOT a rename (23x-26x projection swing); a third dead-alert defect found; dry run beat regex 10:1 on false positives.
metadata:
  type: project
---

Step 82.54 research gate, measured 2026-08-06 against live BigQuery. Sibling of
[[phantom-columns-82-39]] and [[outcome-write-82-48]]; the sweep-blindness half is 82.55.

## The premises that were FALSE (check these before repeating them)

- **"A permanently-NULL tile reads to an operator as $0 spend."** Measured: `llm_tokens_today`
  has **ZERO consumers**. No frontend reference at all (`rg -ic "costbudget|cost-budget|llm_tokens"
  frontend/src` is empty) -- the phase-15.1 tile was removed; `docs/architecture/api-route-audit-2026-04-26.md`
  already recorded "Zero callers anywhere". Nothing coerces None->0, so the "materially worse
  failure" framing is unsupported. The failure is quieter, not louder.
- **"This is the phase-75.5.1 $25/day LLM metric that looks DARK."** It is not. That metric is
  `spend.py::fetch_llm_spend` consumed by `llm_client.py:435-440` behind
  `settings.cost_budget_use_llm_spend_enabled` (**default OFF** -- that is what DARK means).
  `fetch_llm_spend` dry-runs CLEAN and uses the right columns. Different function, file, and
  mechanism.
- **"STRING date column, like 82.39/82.21."** No. `llm_call_log.ts` is **TIMESTAMP REQUIRED**
  and the table is DAY-partitioned on it, so `DATE(ts)=CURRENT_DATE()` is correct AND prunes to
  **0 bytes** (measured vs 88,304 full-table). Do not "fix" the WHERE clause.

**Why:** three plausible-sounding claims in one step spec, each refutable in one command. The
pattern-match to the previous two phantom-column steps is what made them plausible.
**How to apply:** measure the consumer set and the column TYPE before inheriting a prior
step's narrative.

## The fix is NOT a rename -- the projection swings by 23x-26x

`llm_call_log` = 5519 rows / 15 cols; real names `input_tok`/`output_tok` (+
`cache_creation_tok`/`cache_read_tok`). **No NULLs in any of the four** (COUNTIF = 0 over all
rows). No per-row total and no per-call cost column, so a SUM cannot double-count -- but
`session_cost_usd` is a per-cycle cumulative GAUGE, never sum it (phase-66.3).

Measured 2026-08-05: naive `SUM(input_tok)+SUM(output_tok)` over all rows = **353,896** /154
calls; **+cache = 9,159,745 (26x)**; **metered-only** (spend.py's `ok` + 3 CC-rail exclusions)
= **15,409 /30 calls (23x smaller)**. Cache tokens are 82.2M of the table's 90.2M.

**Why:** the flat-fee Claude-Code rail writes real token counts whose cost is ~$0, and the
cache columns dwarf the metered ones. **How to apply:** any "tokens today" number must state
which of the three it is; `spend.py:23-45` is the in-repo source of truth for the exclusions.

## Guard vacuity traps specific to this shape

- **"returns a NON-NULL total" is unfalsifiable here.** The query is an aggregate with no
  GROUP BY, so BigQuery always returns exactly one row, and `COALESCE(...,0)` makes it always
  non-NULL -- which also makes `if not rows: return None,None` dead code. **Today (2026-08-06)
  has ZERO rows** and the repaired query returns `calls=0, tokens=0`. Assert a **positive**
  total over a fixed window AND assert `calls > 0` as a precondition. Do not pin to
  "yesterday": 2026-07-26 has 1 call and 0 tokens.
- **A dry-run guard that validates a hand-copied SQL string proves nothing.** Capture what
  `_fetch_llm_tokens_today` actually builds (it imports `google.cloud.bigquery`
  function-locally, so `Client` is monkeypatchable) and dry-run THAT.

## Third live defect found (queued): the alert that cannot fire

`backend/services/observability/spend.py:115` calls
`raise_cron_alert_sync(..., detail=...)` but the signature is `...(source, error_type,
severity, title, details)`. Live: `TypeError: ... unexpected keyword argument 'detail'. Did
you mean 'details'?` -- swallowed by `except Exception -> logger.debug` at :126-127. It is the
**only** malformed site of 15 audited repo-wide. Compounded: `severity="P2"` and
`slack_webhook_url` is EMPTY, and only `_CRITICAL_SEVERITIES = {P0,P1,critical,CRITICAL}`
(`alerting.py:54`) reaches the bot-token fallback -- so even a well-formed P2 is logged and
dropped. **Two independent reasons the "the cost-budget guard is fail-open" alert has never
fired.** Use `details=` + `severity="P1"`, and assert DELIVERY (captured kwargs + no raise),
not that the branch was entered.

## Method: the dry run is the oracle; regex is only an enumerator

`schema_oracle.dry_run` ($0, `client.query(..., dry_run=True)`) returned BigQuery's own
`Unrecognized name: input_tokens; Did you mean input_tok?`. Across the 25 f-string-invisible
sites, my identifier regex flagged 10 and the dry run proved **9 were false positives** --
string literals (`'buy'`, `'claude-code'`, `'earnings'`, `'default'`), CTE names
(`combined`/`ranked`), and `INTERVAL {days} DAY` units. Exactly **ONE** real defect in the
whole invisible surface.

**Two harness traps I hit myself:** (1) materialising unknown interpolations as the literal
`1` manufactured 4 fake failures (`Could not cast literal "1" to type TIMESTAMP`; `WHERE
clause should return type BOOL, but returns INT64`) -- discriminate on the error CLASS or
capture the real string; (2) my first scanner returned 0 sites because I tested my own
substituted string instead of `extract_sql_literals`' output -- the emptiness assertion caught
it, which is the whole argument for `schema_oracle.py:38-43`.

**Scale of the blindness:** `derive_scope` resolves **2 tables / 14 SQL literals** out of 33
oracle tables, versus **25 invisible sites**. A clean sweep covers ~6% of the surface.
Write-side (the 82.48 class) is clean: the writer at `api_call_log.py:279` matches 15/15.
