# Research Brief -- phase-75.5.12

> ## ⚠ MAIN CORRECTION (2026-07-25) — read before using this brief
>
> **One load-bearing claim in this brief is REFUTED BY EXECUTION.** The brief argues in
> three places (the "decisive reason" paragraph below, the *Alternatives rejected*
> table, and the `summary` field of the JSON envelope at the end) that the step text's
> `NOT (A OR B)` form "would silently neuter the existing shape-2 guard". It would not.
> Running that exact form yields:
>
> ```
> 2 failed, 11 passed
> reds: ['test_cc_rail_rows_contribute_zero_both_shapes',
>        'test_bare_cc_rail_shape_contributes_zero']
> ```
>
> The substring the fake keys on does drop out (mechanism correct), but the consequence
> is inverted: the rail row is then INCLUDED and the assertion fails **loudly**. Nothing
> is silently vacated.
>
> **The recommendation the brief reaches is still right, for a different reason:** taken
> literally without re-adding an `agent IS NULL` guard, the step-text form drops every
> NULL-agent row via SQL three-valued logic, and NULL is the *common* metered case
> (`llm_client.py:1127`; `_role` is set in only two places repo-wide) — measured, 226
> Gemini calls / 232,090 tokens would silently vanish from metered spend.
>
> **Consequence for the *Alternatives rejected* table specifically:** the
> `NOT STARTS_WITH(...) AND agent != 'cc_rail'` row — which the brief itself rates
> "strictly the most wildcard-safe" — was rejected *only* on this refuted
> test-coupling ground. That rejection no longer stands on the stated reason. Queued as
> **75.5.13** so a future reader does not inherit a refuted rationale.
>
> The brief's own text is left UNEDITED below; this note is Main's, not the researcher's.


Tier: **simple**. Audit-class: **false**.
Topic: `fetch_llm_spend`'s CC-rail exclusion misses the BARE `cc_rail`
agent shape (colon-required `LIKE 'cc_rail:%'`), so the dominant production
rail-row class would phantom-price on flag flip.

Status: COMPLETE. Gate: **PASSED** (8 read in full, recency scan done).

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- | --- |
| E1 | https://cloud.google.com/bigquery/docs/reference/standard-sql/operators | 2026-07-25 | official doc | curl + tag-strip (WebFetch returns nav-only -- auto-memory `feedback_gcloud_docs_fetch`) | "A percent sign ( % ) matches any number of characters or bytes. An underscore ( _ ) matches a single character or byte. You can escape \, _, or % using two backslashes. For example, \\%. If you are using raw strings, only a single backslash is required. For example, r'\%'." ALSO: "NULL: Any operation with a NULL input returns NULL." ALSO precedence table: `=`, `!=`, `[NOT] LIKE` all at rank 9; AND/OR strictly lower |
| E2 | https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax | 2026-07-25 | official doc | curl + tag-strip | WHERE clause: "Only rows whose bool_expression evaluates to TRUE are included. Rows whose bool_expression evaluates to NULL or FALSE are discarded." Canonical multi-condition example is literally `WHERE STARTS_WITH(LastName, "Mc") OR STARTS_WITH(LastName, "Mac")` |
| E3 | https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions | 2026-07-25 | official doc | curl + tag-strip | `STARTS_WITH(value, prefix)`: "Takes two STRING or BYTES values. Returns TRUE if prefix is a prefix of value." No wildcard interpretation of the prefix argument |
| E4 | https://cloud.google.com/bigquery/docs/parameterized-queries | 2026-07-25 | official doc | curl + tag-strip | "you can use parameters to protect queries made from user input against SQL injection... Parameters cannot be used as substitutes for identifiers, column names, table names, or other parts of the query. A query parameter value can't be NULL." |
| E5 | https://cloud.google.com/bigquery/docs/release-notes-archive | 2026-07-25 | official doc | curl + tag-strip | 2024-04-01: search-index optimization GA for "`=`, `IN`, and `LIKE` operators and the `STARTS_WITH` function". 2024-04-17: quantified `LIKE ANY / LIKE SOME / LIKE ALL` GA |
| E6 | https://cloud.google.com/bigquery/docs/release-notes | 2026-07-25 | official doc | curl + tag-strip | Current (2026) notes: no change to base LIKE/NULL semantics; only search-index column-granularity preview |
| E7 | https://en.wikibooks.org/wiki/Structured_Query_Language/NULLs_and_the_Three_Valued_Logic | 2026-07-25 | reference | WebFetch | Full 3VL truth tables. "The WHERE clause returns such rows where it evaluates to TRUE. It does not return rows where it evaluates to FALSE or to UNKNOWN." `TRUE OR UNKNOWN = TRUE`; `NOT UNKNOWN = UNKNOWN`; "All comparisons to the NULL marker results per definition in this new value (unknown)." |
| E8 | https://calibrate-analytics.com/insights/2025/05/15/LIKE-vs-REGEXP_CONTAINS-in-BigQuery-What-is-the-Difference-and-When-to-Use-Each/ | 2026-07-25 | industry blog (2025) | WebFetch | "the LIKE operator is generally preferred for performance, especially when your task does not require complex pattern recognition"; warns LIKE has "limited pattern matching capability" and may produce "unintended matches" |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://www.atlassian.com/data/databases/how-to-implement-sqls-like-operator-in-google-bigquery | industry | WebFetch returned nav/header shell only (no article body) |
| https://web.swipeinsight.app/posts/bigquery-s-like-operator-now-supports-underscores-for-single-character-match-3953 | news | HTTP 429 Too Many Requests |
| https://docs.databricks.com/aws/en/sql/language-manual/functions/like | vendor doc | Cross-dialect corroboration only (recommends `startswith` over LIKE for prefix under some collations) |
| https://community.intersystems.com/post/startswith-or | community | Lowest tier; corroborates `%STARTSWITH` vs LIKE only |
| https://www.digitalocean.com/community/tutorials/sql-like-sql-not-like | tutorial | Generic ANSI LIKE; adds nothing beyond E1 |
| https://medium.com/data-engineers-notes/like-all-and-like-any-in-bigquery-671aceff1178 | blog | Quantified-LIKE alternative already covered by E5 (primary source) |
| https://www.sqlservertutorial.net/sql-server-basics/sql-server-null/ | tutorial | T-SQL dialect; E7 + E2 are authoritative for our case |
| https://issuetracker.google.com/issues/35905632 | issue tracker | `LIKE ... ESCAPE ...` request; BigQuery uses backslash escaping instead (E1) |
| https://datawise.dev/query-parameters-in-bigquery | blog | Superseded by E4 (official) |
| https://www.rudderstack.com/guides/sql-pattern-matching/ | vendor guide | General pattern-matching overview |
| https://www.owox.com/blog/articles/bigquery-new-capabilities | blog | 2025 capability roundup, used only to seed the recency scan |
| https://hevodata.com/learn/bigquery-parameterized-queries/ | blog | Superseded by E4 |
| https://dzone.com/articles/avoid-bigquery-sql-injection-go-saferbq | blog | Go-specific identifier-injection tooling; not applicable (our predicate is a constant) |
| https://jashbhatt776.medium.com/exploring-bigquery-alternatives-for-the-like-statement-886a62cd2db5 | blog | LIKE alternatives; covered by E3 |

**URLs collected: 22** (8 read in full + 14 snippet-only).

## Search-query composition (3-variant discipline)

| Variant | Query run |
| --- | --- |
| Current-year (2026) | `BigQuery STARTS_WITH vs LIKE prefix match performance 2026` |
| Last-2-year (2025) | `SQL three-valued logic NULL WHERE clause predicate NOT LIKE excludes NULL rows 2025`; `BigQuery release notes 2025 quantified LIKE ANY ALL SOME general availability` |
| Year-less canonical | `BigQuery LIKE operator wildcard pattern matching`; `BigQuery Standard SQL LIKE operator semantics percent underscore wildcard escaping`; `SQL LIKE prefix predicate false positive over-matching equals or startswith idiom`; `SQL injection prevention parameterized queries BigQuery string literal predicate` |

## Recency scan (2024-2026)

Performed against the BigQuery release notes (E6) and the release-notes archive
(E5), plus two year-scoped searches. **Result: no change to base `LIKE` / `NOT
LIKE` / NULL semantics in the 2024-2026 window.** Two adjacent findings, both
informational for this step:

1. **2024-04-01** -- search-index optimization went GA for "`=`, `IN`, and `LIKE`
   operators and the `STARTS_WITH` function" (E5). Consequence: BigQuery treats
   `LIKE 'prefix%'` and `STARTS_WITH(col,'prefix')` as *optimization-equivalent*
   for literal comparisons against indexed data, so choosing between them is a
   **correctness/readability** decision, not a performance one. `llm_call_log`
   has no search index, so this is doubly moot here.
2. **2024-04-17** -- quantified `LIKE ANY / LIKE SOME / LIKE ALL` went GA (E5),
   which would permit a compact `agent NOT LIKE ANY ('cc_rail', 'cc_rail:%')`.
   **Not recommended** -- see "Alternatives rejected" below.

No 2025 or 2026 release note touches LIKE semantics. The canonical E1/E2/E7
sources remain current.

## Key findings

1. **`_` IS A WILDCARD, and it is already firing inside `'cc_rail:%'`.** "An
   underscore ( _ ) matches a single character or byte" (E1). The literal
   `cc_rail` contains an underscore, so the *existing* pattern
   `'cc_rail:%'` reads as `cc` + ANY-ONE-CHAR + `rail:` + anything.
   Empirically confirmed locally (ANSI LIKE, sqlite3 stand-in, read-only):

   ```
   'cc_rail:synthesis'  LIKE 'cc_rail:%'  -> True
   'ccXrail:synthesis'  LIKE 'cc_rail:%'  -> True   <-- over-match
   'cc-rail:synthesis'  LIKE 'cc_rail:%'  -> True   <-- over-match
   'cc_rail'            LIKE 'cc_rail:%'  -> False  <-- THE DEFECT
   'cc_railway'         LIKE 'cc_rail:%'  -> False
   'cc_railway:x'       LIKE 'cc_rail:%'  -> False
   ```

   This is a **pre-existing latent over-match** the step text did not name. It
   has zero production impact today (no agent tag matches `cc?rail:` for
   `? != '_'`), and the required colon is what keeps `cc_railway:x` out. See
   "Optional hardening" for the escape form and why I do not recommend bundling it.

2. **The step's stated hazard (`cc_railway`) is NOT reachable via equality.**
   `'cc_railway' = 'cc_rail'` is FALSE, and `=` does no pattern interpretation at
   all (E1: `=` is a comparison operator, not a pattern operator). So the
   `agent = 'cc_rail'` / `agent != 'cc_rail'` leg is exactly-matching by
   construction -- it cannot swallow `cc_railway`, `cc_rail_v2`, or anything else.
   A naive `agent NOT LIKE 'cc_rail%'` (the shape `metered_spend.py` uses) WOULD
   swallow them.

3. **The `agent IS NULL OR ...` guard is still REQUIRED after widening.** With
   `agent` NULL: `agent != 'cc_rail'` -> UNKNOWN and `agent NOT LIKE 'cc_rail:%'`
   -> UNKNOWN (E1: "Any operation with a NULL input returns NULL"; E7: "All
   comparisons to the NULL marker results per definition in this new value
   (unknown)"). `UNKNOWN AND UNKNOWN = UNKNOWN` (E7), and WHERE discards it:
   "Rows whose bool_expression evaluates to NULL or FALSE are discarded" (E2).
   Dropping the guard would silently stop counting every `agent=NULL` metered row
   -- i.e. most of the Gemini/SDK pipeline traffic (writers W4-W6) -- making the
   breaker read LOW and fail to trip. `TRUE OR UNKNOWN = TRUE` (E7) is what makes
   the guard work.

4. **Parentheses are mandatory, and precedence confirms why.** E1's precedence
   table puts `=`, `!=`, and `[NOT] LIKE` together at rank 9 with `AND` and `OR`
   strictly lower, and `AND` binds tighter than `OR`. Written unparenthesized,
   `... AND agent IS NULL OR agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'`
   would parse as `(... AND agent IS NULL) OR (...)` and destroy the whole WHERE
   clause. The recommended form is fully parenthesized.

5. **Injection is not a live concern here, but the constant form is still the
   right one.** E4: parameters "protect queries made from user input against SQL
   injection" but "cannot be used as substitutes for identifiers". Our predicate
   embeds no user input -- `'cc_rail'` is a source constant, and the only
   interpolations in the f-string SQL are `project` (env, `spend.py:199`) and
   `dataset` (settings, `:200`). A query parameter is therefore unnecessary; a
   literal string constant is idiomatic and safe. (If a future step ever
   parameterizes the tag, note E4's rule that "A query parameter value can't be
   NULL".)

6. **No gotcha from combining with the existing filters.** `provider` is `STRING
   NOT NULL` per the schema comment in `api_call_log.py`, so `provider !=
   'claude-code'` has no NULL trap. `ok BOOL` is nullable, so a NULL `ok` row is
   already dropped by the bare `AND ok` (UNKNOWN -> discarded, E2) -- pre-existing
   and conservative (a row of unknown success is not priced). The widened agent
   predicate is a self-contained parenthesized conjunct and does not interact
   with either.

---

## Internal code inventory

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| `backend/services/observability/spend.py` | 1-246 (whole module read) | the defective seam: `fetch_llm_spend` + docstring shape inventory | **DEFECT at :218** |
| `backend/agents/claude_code_client.py` | 489-515, 517-530, 600-630 | the WRITER that produces both `cc_rail` and `cc_rail:<agent>` | source of the bare shape |
| `backend/services/autonomous_loop.py` | 2286-2311, 2722, 2762 | 3rd rail writer (`provider='claude-code'`); the ONLY two `_role` setters | source of shape 3 |
| `backend/agents/llm_client.py` | 396-467, 1124, 1277, 1894, 2322, 2334 | the $25/day breaker (sole consumer) + 5 non-rail `agent=` writers | consumer + shape-set evidence |
| `backend/agents/orchestrator.py` | 826-840, 906-919 | Layer-1 pipeline: sets `_ticker` but **NEVER `_role`** | root cause of bare-shape dominance |
| `backend/tests/test_phase_75_5_1_spend_metric.py` | 1-271 (whole file read) | the fixture that must grow a bare-shape case | fixture IS mutation-capable (see below) |
| `scripts/away_ops/metered_spend.py` | 18-21, 53-54, 69, 79 | SECOND exclusion seam (away sentinel) -- prefix `startswith("cc_rail")` | already catches bare; has the `cc_railway` over-match |
| `scripts/diagnostics/funnel_report.py` | 13, 96 | THIRD seam, opposite polarity (`WHERE agent LIKE 'cc_rail%'` to INCLUDE rail rows) | diagnostic only, not money-gating |
| `backend/config/settings.py` | 384, 386 | `cost_budget_daily_usd=25.0`, `cost_budget_use_llm_spend_enabled=False` | flag default confirmed OFF |

### 1. The writer (verbatim)

`backend/agents/claude_code_client.py:501-513`, inside the static helper
`_log_cc_call` (defined `:490`):

```python
log_llm_call(
    provider="anthropic",
    model=model,
    agent=f"cc_rail:{agent}" if agent else "cc_rail",
```

`agent` reaches it from `generate_content` at `claude_code_client.py:527`:

```python
_agent = config.get("_role") or config.get("_agent")
```

and is forwarded at the two call sites `:604-605` (error path) and `:623-624`
(success path). So: **`_role`/`_agent` absent from `generation_config` -> `_agent
is None` -> the else-branch -> agent EXACTLY `'cc_rail'`.**

### 2. Why the bare shape DOMINATES (the callers)

Repo-wide, `"_role"` is written into a generation_config in exactly **two**
places (grep `'"_role"'` across `backend/`, excluding tests):

- `backend/services/autonomous_loop.py:2722` -- `"_role": "lite_trader"`
- `backend/services/autonomous_loop.py:2762` -- `"_role": "lite_risk_judge"`

The Layer-1 pipeline does **not**. `backend/agents/orchestrator.py:826-835`
plucks only `_ticker` from `generation_config` and its comment names only
`_ticker` as the orchestrator-private side-channel; nothing in
`orchestrator.py` sets `_role` for the model call. Therefore **every Layer-1
cc_rail call (enrichment, debate, critic, synthesis, quant executor) lands in
the else-branch and writes the BARE shape**, while only the lite-path
autonomous-loop calls produce `cc_rail:lite_trader` / `cc_rail:lite_risk_judge`.
That is the mechanical explanation for the measured 2,549-vs-7 split.

(`llm_client.py:1616` reads `config.get("_role")` as an *effort* role hint, and
`llm_client.py:1127/1280/1897` pass `_role` through as the `agent=` for the
Gemini/OpenAI/Anthropic-SDK clients -- i.e. those rows are `agent=NULL` on the
same pipeline calls. Consistent with `test_agent_none_rows_are_included`.)

### 3. The TRUE shape-set (derived from code, not from the step text)

Every `log_llm_call(` writer in production code (`grep -rn "log_llm_call(" backend/ scripts/`,
excluding `backend/tests/`):

| # | Writer | `provider` | `agent=` expression | Rail? | Currently excluded by |
| --- | --- | --- | --- | --- | --- |
| W1 | `claude_code_client.py:504` (truthy agent) | `anthropic` | `f"cc_rail:{agent}"` | YES | `agent NOT LIKE 'cc_rail:%'` |
| W2 | `claude_code_client.py:504` (falsy agent) | `anthropic` | `"cc_rail"` | YES | **NOTHING -- the defect** |
| W3 | `autonomous_loop.py:2299-2301` | `claude-code` | `agent` (arbitrary, e.g. `lite_trader`) | YES | `provider != 'claude-code'` |
| W4 | `llm_client.py:1124-1127` (GeminiClient) | `gemini` | `generation_config.get("_role")` (often NULL) | no | n/a (metered, correctly counted) |
| W5 | `llm_client.py:1277-1280` (OpenAIClient) | `openai`/`github_models` | `_cfg.get("_role")` | no | n/a |
| W6 | `llm_client.py:1894-1897` (ClaudeClient SDK) | `anthropic` | `config.get("_role")` | no | n/a (real metered API) |
| W7 | `llm_client.py:2322` (advisor executor) | `anthropic` | `role or "advisor_call_executor"` | no | n/a |
| W8 | `llm_client.py:2334` (advisor tool) | `anthropic` | `(role or "advisor_call") + "_advisor_tool"` | no | n/a |
| W9 | `orchestrator.py:908-910` (code-exec) | `gemini` | `f"{agent_name}_code_exec"` | no | n/a |

**Verdict on criterion 2:** the step text's list of three shapes is CORRECT and
COMPLETE for the flat-fee class -- W1, W2, W3 are the only rail-tagged writers.
There is no fourth rail shape. Two adjacent facts the docstring should NOT
mis-state: W7/W8 are `provider='anthropic'` **metered** rows whose agent starts
with `advisor_` (they must stay counted), and W4-W6 legitimately write
`agent=NULL` (hence the `agent IS NULL OR` guard).

### 4. Other seams that filter CC-rail rows

- `scripts/away_ops/metered_spend.py:69` -- `is_flat_fee()` returns
  `provider in FLAT_FEE_PROVIDERS or agent.startswith(RAIL_AGENT_PREFIX)` with
  `RAIL_AGENT_PREFIX = "cc_rail"` (`:54`). This is a Python `startswith`, i.e. a
  `cc_rail%` prefix -- it **already catches the bare shape** (no colon bug here),
  but it carries the exact over-match hazard the step warns about (a
  hypothetical `cc_railway` would be silently classed flat-fee). Same at `:79`
  for the `rail_failures` count. **NOT in this step's boundary** (`spend.py` +
  tests) and NOT money-gating (away sentinel reporting), so it is a *disclosure*,
  not a required part of the fix -- but the two seams will remain semantically
  inconsistent until it is aligned.
- `scripts/diagnostics/funnel_report.py:96` -- `WHERE agent LIKE 'cc_rail%'`,
  opposite polarity (it wants rail rows). Same over-match hazard, diagnostic only.
- `spend.py` itself has **exactly one** rail predicate (`:218`); the duplicate-
  predicate risk inside the boundary module is nil.

### 5. The consumer + flag + breaker path

- Sole production consumer: `backend/agents/llm_client.py:435-441` in
  `_check_cost_budget()`:
  `if getattr(settings, "cost_budget_use_llm_spend_enabled", False): daily_usd,
  monthly_usd = fetch_llm_spend()` else `fetch_spend()`.
- `_check_cost_budget()` is called on every generate_content at
  `llm_client.py:905`, `:1191`, `:1439`, `:2225`; it raises `BudgetBreachError`
  when `daily >= cost_budget_daily_usd` (25.0, `settings.py:384`) or monthly >= cap.
- Flag default: `settings.py:386` `cost_budget_use_llm_spend_enabled: bool =
  Field(False, ...)`; pinned by `test_flag_default_is_off` (test file `:269-271`).
- Result cached 60s (`_BUDGET_CACHE_TTL_S`), fail-open on exception.
- Re-export surface: `backend/services/observability/__init__.py:53,68`.

**Impact if the flag flips today, unfixed:** at the measured sonnet-4-6 rate
($3/$15 per Mtok) the ~4.37M bare-`cc_rail` tokens in a 30d window price to a
non-trivial dollar figure of *free* tokens against a $25/day cap -- the exact
phantom-trip class 75.5.1 exists to prevent, and a trip halts every
`generate_content` (trading stops) until the UTC bucket rolls.

### 6. Fixture verdict: CAN the new test actually fail?

**YES.** `backend/tests/test_phase_75_5_1_spend_metric.py` uses
`FakeBQClient` (`:58-103`) which parses the PRODUCTION SQL TEXT and applies only
the predicates literally present in it (`:81-87`):

```python
if "AND ok" in sql and not r["ok"]:
    continue
if "provider != 'claude-code'" in sql and r["provider"] == "claude-code":
    continue
if "NOT LIKE 'cc_rail:%'" in sql and r["agent"] is not None \
        and str(r["agent"]).startswith("cc_rail:"):
    continue
```

The existing both-shapes test (`:155-169`) is verbatim:

```python
def test_cc_rail_rows_contribute_zero_both_shapes():
    metered = _row("gemini-2.5-flash", provider="vertex", in_tok=100_000)
    FakeBQClient.rows = [
        metered,
        _row("claude-opus-4-8", provider="claude-code", agent="mas_main",
             in_tok=500_000_000, out_tok=100_000_000),        # shape 1: flat-fee rail
        _row("claude-opus-4-8", provider="anthropic", agent="cc_rail:synthesis",
             in_tok=500_000_000, out_tok=100_000_000),        # shape 2: rail via SDK tag
    ]
    daily, monthly = spend.fetch_llm_spend()
    only_metered = _expected_usd("gemini-2.5-flash", 100_000)
    assert daily == pytest.approx(only_metered, rel=1e-9), (...)
    assert monthly == pytest.approx(only_metered, rel=1e-9)
```

Fixture mechanics: `_row()` (`:49-55`) builds a plain dict with the real column
names; rows are class-level `FakeBQClient.rows`; the autouse `_fake_bq` fixture
(`:106-115`) monkeypatches `google.cloud.bigquery.Client`. So the fixture is
**neither a mock-return nor a source-scan** -- it is a semantic re-execution of
the production predicate string. Adding
`_row("claude-opus-4-8", provider="anthropic", agent="cc_rail", in_tok=..., out_tok=...)`
will be **priced today** (it passes all three current predicates), so the new
assertion is RED before the fix and GREEN after -- exactly the property auto-memory
`feedback_mutation_test_guards_and_fixtures` demands.

**One hard requirement on the fake:** the fake must learn the new predicate, and
must learn it in a way that is still *derived from the SQL text*. If the fake is
changed to unconditionally drop `agent == "cc_rail"` regardless of the SQL, the
mutation becomes undetectable (a vacuous guard). The self-test
`test_fake_client_honors_filter_absence` (`:202-217`) is the standing guard
against that and must be extended with a bare-`cc_rail` row so that neutering the
fake stays a killable mutation too.

---

## Consensus vs debate (external)

**Consensus:** for a literal prefix test, dedicated prefix functions
(`STARTS_WITH`, Databricks `startswith`, InterSystems `%STARTSWITH`) are the
semantically clearer choice, and since 2024-04-01 BigQuery optimizes them
identically to `LIKE` against indexed data (E5). Everyone agrees `%` / `_` in a
LIKE pattern are wildcards requiring escape (E1).

**Debate:** E8 and the Atlassian/DigitalOcean tier argue LIKE is the
performance-preferred default for simple patterns; Databricks/InterSystems argue
prefix functions are preferable for prefix semantics. For BigQuery specifically
the performance argument is empty post-2024-04-01 (E5), so the debate collapses
to readability + wildcard-safety -- which favors `=` for the exact leg
(unambiguously wildcard-free) and leaves the existing `LIKE` colon-leg a
judgment call.

## Pitfalls (from literature)

- **Silent underscore wildcard.** E1 is explicit that `_` matches any single
  character; this is the single most common LIKE over-match bug and it is
  already latent in our pattern (finding 1). Escape with two backslashes
  (`'cc\\_rail:%'`) or a raw string (`r'cc\_rail:%'`) if it ever matters.
- **NULL-swallowing predicates.** E2/E7: a NOT-predicate on a nullable column
  drops NULL rows silently. This is the classic `NOT IN`/`NOT LIKE` trap; the
  existing `agent IS NULL OR` guard is the correct countermeasure and must
  survive the edit.
- **Prefix-wildcard over-broadening.** Replacing `'cc_rail:%'` with `'cc_rail%'`
  "fixes" the bare shape but swallows every future `cc_rail*` name (E8's
  "unintended matches"). This is precisely what the immutable criterion forbids
  and what `scripts/away_ops/metered_spend.py:69` already does.

## Application to pyfinagent

### RECOMMENDED FIX (exact replacement)

`backend/services/observability/spend.py:218` -- replace the single line

```sql
              AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')
```

with

```sql
              AND (agent IS NULL
                   OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

Why this exact form (De Morgan of the step's suggested
`NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')`, which is logically identical):

- `agent != 'cc_rail'` is **equality-based**, so `_` is literal and `cc_railway`
  / `cc_rail_v2` / `cc_railroad` stay COUNTED as metered (finding 2) --
  satisfies criterion 1's "not a prefix-wildcard that could swallow unrelated
  agents" **exactly**.
- The NULL guard is preserved verbatim (finding 3).
- Fully parenthesized (finding 4).
- **It preserves the literal substring `NOT LIKE 'cc_rail:%'`**, which the
  existing test fake keys on at `test_phase_75_5_1_spend_metric.py:85`. The
  De-Morgan'd `NOT (agent = ... OR agent LIKE ...)` form would NOT contain that
  substring and would silently neuter the existing shape-2 guard. **This is the
  decisive reason to prefer this form over the one written in the step text.**

> **[MAIN CORRECTION]** This "decisive reason" is refuted by execution — the form fails LOUDLY, it does not pass silently. See the correction note at the top of this brief. The recommendation stands on the NULL/3VL argument instead.

### Alternatives rejected

| Form | Verdict |
| --- | --- |
| `agent NOT LIKE 'cc_rail%'` | REJECTED -- violates criterion 1 (swallows `cc_railway`) |
| `NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')` | Logically fine, but breaks the fake's `"NOT LIKE 'cc_rail:%'" in sql` hook at test `:85`, silently vacating the shape-2 guard |
| `NOT STARTS_WITH(agent, 'cc_rail:') AND agent != 'cc_rail'` | Strictly the most wildcard-safe (E3, and E2's canonical example uses STARTS_WITH). Rejected only for test-coupling: same fake-hook problem, larger diff. Worth a follow-up step if the project wants wildcard-free predicates project-wide |
| `agent NOT LIKE ANY ('cc_rail', 'cc_rail:%')` | REJECTED -- GA since 2024-04-17 (E5) so it is available, but quantified-LIKE negation semantics are subtle, it still carries the `_` wildcard, and it is a novel idiom in this codebase |
| Query parameter (`@rail_tag`) | Unnecessary -- no user input in the predicate (finding 5) |

> **[MAIN CORRECTION]** The two "breaks the fake's hook / silently vacating the shape-2 guard" verdicts in this table are refuted — that form fails loudly. The STARTS_WITH row in particular was rejected ONLY on that ground despite being rated the most wildcard-safe; queued as **75.5.13** for a proper decision.

### Optional hardening (NOT bundled -- disclosure only)

Escaping the latent `_` wildcard would make the colon-leg
`agent NOT LIKE r'cc\_rail:%'`. I do **not** recommend bundling it into 75.5.12:
it has zero behavioral effect on any existing or plausible row, it changes the
substring the test fake keys on (forcing a second coordinated edit), and it is
out of the step's stated fix. Per `feedback_queue_discovered_defects_in_masterplan`
this is a candidate for its own small masterplan step covering all three seams
(`spend.py`, `metered_spend.py:69`, `funnel_report.py:96`) consistently; it is a
latent-only over-match, so P3 at most.

### EXACT docstring replacement (criterion 2)

`backend/services/observability/spend.py:23-24`, currently:

```
  1. METERED-ONLY. Flat-fee CC-rail rows (provider='claude-code', or
     provider='anthropic' with agent LIKE 'cc_rail:%') record tokens whose real cost is
```

Replace with (names all THREE shapes actually produced -- W1/W2/W3 from the
inventory above; wording is a recommendation, not a mandate):

```
  1. METERED-ONLY. Flat-fee CC-rail rows record tokens whose real cost is
     ~$0 (Claude Code Max rail). The rail writers produce exactly THREE agent/
     provider shapes, and ALL THREE must be excluded:
       (a) provider='claude-code', agent=<any>   -- autonomous_loop.py:2299
       (b) provider='anthropic', agent='cc_rail:<role>'
                                                 -- claude_code_client.py:504, truthy branch
       (c) provider='anthropic', agent='cc_rail' (BARE, no colon)
                                                 -- claude_code_client.py:504, else branch
     (c) is the DOMINANT production shape, not an edge case: the Layer-1
     orchestrator never sets the `_role` side-channel (only autonomous_loop.py:
     2722/2762 do), so every pipeline rail call lands in the else branch --
     measured 2,549 calls / 4.87M tokens vs 7 rows for (b) over 30d (phase-78
     census, 2026-07-25). phase-75.5.1 shipped with only (a)+(b) excluded;
     phase-75.5.12 added the exact-equality clause for (c). The `agent != ...`
     leg is equality, NOT a prefix wildcard, so a legitimately-named agent such
     as 'cc_railway' stays METERED. Note `agent IS NULL` rows are metered SDK
     calls (llm_client.py W4-W6) and MUST stay counted -- the IS NULL guard is
     load-bearing, not defensive decoration (BigQuery WHERE discards UNKNOWN).
     Pricing these at API rates would trip the $25 breaker on FREE tokens and
     falsely halt trading -- the same phantom class as the 2026-06
     session_cost_usd staircase.
```

### NEW FIXTURE (criterion 1) -- exact shape

Two coordinated test edits in
`backend/tests/test_phase_75_5_1_spend_metric.py`:

**(a) Teach the fake the new predicate, still derived from the SQL text**
(insert after the existing block at `:85-87`):

```python
            if "agent != 'cc_rail'" in sql and r["agent"] == "cc_rail":
                continue
```

**(b) Extend the shapes test** (`:155`, rename to `..._all_three_shapes` or add a
sibling) by adding a third rail row and a negative control:

```python
        _row("claude-opus-4-8", provider="anthropic", agent="cc_rail",
             in_tok=500_000_000, out_tok=100_000_000),        # shape 3: BARE rail tag
```

plus a dedicated test that the widened clause is EXACT, not a prefix wildcard:

```python
def test_cc_railway_is_not_swallowed_by_the_bare_rail_exclusion():
    """criterion 1: the bare-shape exclusion must be exact equality. A
    hypothetical metered agent literally named 'cc_railway' must still be
    PRICED -- a prefix wildcard ('cc_rail%') would silently zero it."""
    FakeBQClient.rows = [
        _row("gemini-2.5-flash", provider="vertex", agent="cc_railway",
             in_tok=100_000),
    ]
    daily, _ = spend.fetch_llm_spend()
    assert daily == pytest.approx(_expected_usd("gemini-2.5-flash", 100_000),
                                  rel=1e-9)
    assert daily > 0
```

**(c) Extend the fake self-test** `test_fake_client_honors_filter_absence`
(`:202-217`) with a bare-`cc_rail` row and bump the expected total to
`4_000_000`, so neutering the fake stays a killable mutation.

### PRECISE MUTATION MATRIX

| # | Mutation | Expected |
| --- | --- | --- |
| M1 | In `spend.py`, delete `agent != 'cc_rail' AND ` (revert to the 75.5.1 line) | the bare-shape assertion in the shapes test goes **RED**; all other tests stay **GREEN** |
| M2 | Change the fix to the over-broad `agent NOT LIKE 'cc_rail%'` (drop the colon, drop the `!=` leg) | `test_cc_railway_is_not_swallowed_...` goes **RED** (proves the fix is exact, not a prefix wildcard) |
| M3 | Neuter the fake: make it drop `agent == "cc_rail"` unconditionally (not keyed on the SQL text) | `test_fake_client_honors_filter_absence` goes **RED** (proves the new guard is not vacuous) |
| M4 | Delete `agent IS NULL OR ` from the production predicate | `test_agent_none_rows_are_included` goes **RED** -- **verify this; if it stays green the NULL guard is currently untested** and the fake's NULL handling needs the same SQL-text derivation |

M4 is a **flagged uncertainty for GENERATE**: the current fake never evaluates
`agent IS NULL`, so it may not reproduce the NULL-drop. Reproduce M4 before
claiming the NULL guard is covered.

### Risk: does the widened clause change results for rows OTHER than bare-`cc_rail`?

**No.** The added conjunct `agent != 'cc_rail'` newly-excludes a row iff `agent`
is exactly the 8-character string `cc_rail`; equality does no pattern
interpretation (finding 2, empirically confirmed). NULL rows are short-circuited
by the pre-existing `agent IS NULL OR` (finding 3). Rows already excluded by
`AND ok`, `provider != 'claude-code'`, or the colon leg are unaffected. The
metric will read **strictly lower or equal** after the fix -- which is the
correct direction (removing phantom spend), and means the fix cannot cause a
*new* false trip. Baseline confirmed green before any edit: `11 passed in 1.05s`.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (8: E1-E8; 6 official Google Cloud docs)
- [x] 10+ unique URLs total (22)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts / not search snippets) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (9 files; every `log_llm_call(` writer enumerated, all 3 cc_rail seams found)
- [x] Contradictions / consensus noted (LIKE-vs-STARTS_WITH debate; resolved by E5)
- [x] All claims cited per-claim

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 14,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "spend.py:218's rail exclusion (agent NOT LIKE 'cc_rail:%') requires a colon, so the BARE 'cc_rail' shape passes and would be priced at API rates on flag flip. Root cause of its dominance: the Layer-1 orchestrator sets only _ticker, never _role, so every pipeline rail call hits the else-branch at claude_code_client.py:504; only autonomous_loop.py:2722/2762 set _role. Derived the TRUE rail shape-set from code -- exactly three (bare, colon-suffixed, provider='claude-code'); the step text's list is complete, no fourth shape. Recommended predicate is the De-Morgan'd form (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%')) because it PRESERVES the substring the test fake keys on at test:85 -- the step's suggested NOT(A OR B) form would silently neuter the existing shape-2 guard. Equality keeps 'cc_railway' metered; the IS NULL guard is required (BigQuery WHERE discards UNKNOWN). NEW: the '_' in 'cc_rail:%' is already a latent wildcard (ccXrail:x matches) -- disclosed, not bundled. Fixture CAN fail: the fake parses production SQL text. Two other seams (metered_spend.py:69, funnel_report.py:96) use prefix matching, out of boundary.",
  "brief_path": "handoff/current/research_brief_75.5.12.md",
  "gate_passed": true
}
```
