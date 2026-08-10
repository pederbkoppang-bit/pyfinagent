# Research Brief -- phase-86.25

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Role:** Layer-3 Researcher (external literature + internal codebase exploration).
**Started:** 2026-08-10. Write-first: this file was created in tool call #2 and is appended to as each source is read.

## Objective

When a learning/evaluation loop cannot determine the DIRECTION of a decision it is scoring, what
should it record? Four sub-questions:

1. ML/MLOps literature on **label noise vs missing labels** -- is a WRONG label more damaging than an
   ABSENT one, and how do systems represent "unknown" distinctly from "incorrect"
   (nullable vs sentinel vs separate status column)? The classic failure of encoding unknown as a
   valid class value.
2. **Vocabulary/enum boundaries between subsystems** -- passing a value from one controlled vocabulary
   into a parameter typed for a different one (risk-approval `APPROVE/REJECT` vs analyst
   `BUY/HOLD/SELL`); remedies: parse-don't-validate boundary, tagged union, or refuse to score.
3. Deciding between **look up the right value / skip the scoring / record an explicit unknown state**
   when the caller has an adjacent-but-semantically-different value to hand.
4. **Provenance**: determining WHICH code path produced existing DB rows when the writer is not
   recorded on the row -- what evidence is admissible and what is not.

Internal scope: MEASURE, do not assume. Re-derive every number.

---

## STATUS: COMPLETE

---
## Internal measurement -- BigQuery (re-derived 2026-08-10, NOT copied from the step text)

Ran read-only `SELECT`s via the Python `google-cloud-bigquery` client (the BigQuery MCP tools are
NOT in this session's tool surface; CLAUDE.md "BigQuery Access (MCP)" rule 6 authorises the Python
fallback). Project `sunny-might-477607-p8`, location `us-central1`, dataset `financial_reports`.

### M1 -- `financial_reports.paper_trades.risk_judge_decision`, full distribution (n=65)

| value | rows |
|---|---|
| `''` (empty string) | 46 |
| `APPROVE_REDUCED` | 15 |
| `REJECT` | 3 |
| `APPROVE_HEDGED` | 1 |

Split by `action`:

| action | risk_judge_decision | rows |
|---|---|---|
| SELL | `''` | 32 |
| BUY | `APPROVE_REDUCED` | 15 |
| BUY | `''` | 14 |
| BUY | `REJECT` | 3 |
| BUY | `APPROVE_HEDGED` | 1 |

**Measured facts.** (i) The column is 100% risk-approval vocabulary or empty. It contains **zero**
`BUY`, `HOLD`, `SELL`, `Buy`, `Hold` or `Sell` values. (ii) **Every SELL row is empty** (32/32) --
and SELL rows are the ONLY rows `_learn_from_closed_trades` ever reads (`autonomous_loop.py:3398`
filters `t.get("action") == "SELL"`). So on the live path the `risk_judge_decision` fetch at
`autonomous_loop.py:3412` returns `''` for 32 of 32 candidate rows, the empty-coercion at
`:3416-3417` rewrites it to `HOLD`, and `HOLD` is neither buy- nor sell-intent, so
`directionally_correct` is `False` unconditionally. (iii) The 19 non-empty values are all on BUY
rows, which this code path never reaches. **The mismatch is therefore not hypothetical but also not
yet observed in stored data: today it manifests as a 100%-`HOLD` degenerate, and it would manifest
as a vocabulary mismatch the moment a SELL row carries a populated `risk_judge_decision`.**

### M2 -- `financial_reports.outcome_tracking`, FULL dump (n=3, every column)

| ticker | analysis_date | recommendation | price_at_recommendation | current_price | return_pct | holding_days | beat_benchmark | evaluated_at |
|---|---|---|---|---|---|---|---|---|
| AMD | 2026-07-27T18:05:27.845972+00:00 | `SELL` | **NULL** | 483.7 | -11.316 | 17 | **NULL** | 2026-08-08T04:00:02.013552+00:00 |
| PANW | 2026-07-27T18:05:14.461571+00:00 | `SELL` | **NULL** | 320.85 | -10.9368 | 6 | **NULL** | 2026-08-08T04:00:02.013552+00:00 |
| MU | 2026-07-13T18:05:41.909896+00:00 | `SELL` | **NULL** | 931.715 | -7.2643 | 3 | **NULL** | 2026-08-08T04:00:02.013552+00:00 |

Schema (INFORMATION_SCHEMA, 9 columns): `ticker` STRING **REQUIRED**, `analysis_date` STRING
**REQUIRED**, `recommendation` STRING **REQUIRED**, `price_at_recommendation` FLOAT64 NULLABLE,
`current_price` FLOAT64 NULLABLE, `return_pct` FLOAT64 NULLABLE, `holding_days` INT64 NULLABLE,
`beat_benchmark` BOOL NULLABLE, `evaluated_at` STRING NULLABLE. **There is no
`directionally_correct` column** -- the flag `evaluate_recommendation` computes is returned in the
in-memory dict and consumed by the reflection prompt, but is never persisted here.

### M3 -- `financial_reports.analysis_results.recommendation`, full distribution (n=543)

HOLD 275 / Hold 115 / BUY 91 / Buy 39 / Sell 16 / Strong Buy 5 / N/A 2. **The literal `SELL`
(upper-case) appears 0 times.** The only sell spelling here is title-case `Sell`.

**Consequence for CRITERION 3:** the `SELL` in `outcome_tracking` is present in NEITHER candidate
source column. See the provenance section below.

## External research

### Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| # | Variant | Query |
|---|---|---|
| 1 | year-less canonical | `label noise versus missing labels machine learning which is more harmful` |
| 2 | year-less canonical | `parse don't validate tagged union boundary illegal states unrepresentable` |
| 3 | year-less canonical | `sentinel value for unknown versus NULL missing data encoding anti-pattern database` |
| 4 | last-2-year / current-year | `learning with abstention reject option better than wrong prediction 2025 2026` |

### Read in full (counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://arxiv.org/html/2404.04159v1 | 2026-08-10 | preprint survey | WebFetch (arXiv native HTML, per the html-first chain) | "Those corrupted labels are called noisy labels." "DNNs have such strong capacity that they can easily fit noisy labels during the model learning process, resulting in poor generalization performance." Feature noise "is usually less harmful compared with label noise." The dominant remedy family is **sample selection** -- DivideMix "divides the noisy dataset into a labeled set and an **unlabeled set**", i.e. suspect labels are demoted to UNLABELLED rather than kept as labels. Explicit trade: "Possible true-labeled samples can be discarded ... to benefit the training of DNNs and reduce the negative effect of noisy samples." |
| https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-10 | authoritative blog (canonical) | WebFetch | "a parser is just a function that consumes less-structured input and produces more-structured output"; "the difference between validation and parsing lies almost entirely in how information is preserved"; **"Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system, before _any_ of the data is acted upon"**; shotgun parsing -- "Late-discovered errors in an input stream will result in some portion of invalid input having been processed"; "write functions on the data representation you _wish_ you had, not the data representation you are given". |
| https://peps.python.org/pep-0661/ | 2026-08-10 | official language doc | WebFetch | Sentinels exist precisely for "**Missing data, such as NULL in relational databases or 'N/A' ('not available') in spreadsheets**" and for the case where "`None` is a valid value in that context." The stated hazard of a shared/in-domain sentinel: "one could not always be confident that it would never be a valid value in some use cases." |
| https://www.w3.org/TR/prov-dm/ | 2026-08-10 | W3C Recommendation | WebFetch | "Provenance is information about entities, activities, and people involved in producing a piece of data or thing, which can be used to form assessments about its quality, reliability or trustworthiness." "Attribution is the ascribing of an entity to an agent." Derivation may be "considered to have been determined by unspecified means" -- the model explicitly permits **incomplete provenance** with unspecified elements, i.e. an honest "undetermined" is a first-class record, not a failure. |
| https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-03 | 2026-08-10 | IETF draft (standards-track discussion) | WebFetch | "Choosing to generate fatal error for unspecified conditions instead of attempting error recovery can ensure that faults receive attention." "Allowing less variation is preferable in the absence of strong reasons to be flexible." Leniency entrenches errors: implementations become "bug for bug compatible." |
| https://developers.google.com/machine-learning/guides/rules-of-ml | 2026-08-10 | official vendor engineering doc | WebFetch | Rule #29: "The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time." Rule #30: "Importance-weight sampled data, don't arbitrarily drop it!" Rule #34: for filtering tasks, prefer "small short-term sacrifices in performance for very clean data" by **holding out** examples rather than training on user-corrected labels. Rule #2: measure the existing system before formalising the ML behaviour. NOTE (honest negative): the doc gives **no** guidance on sentinel/magic values for missing labels -- that gap is why PEP 661 and the DB-NULL literature carry that half of the argument. |

| https://arxiv.org/html/2510.19672v1 | 2026-08-10 | preprint (2025) -- **recency-window source** | WebFetch (arXiv native HTML) | Sawarni, Jin, Whitehouse, Syrgkanis, *Policy Learning with Abstention* (2025). "a critical deficit of existing methods is their failure to abstain when faced with high uncertainty. In safety-critical applications, forcing a decision when evidence is weak can be harmful." Mechanism: "instead of just assigning a unit to treatment or control (denoted '1' or '0' respectively), they can abstain by **outputting a special symbol '\*'**" -- i.e. abstention is a THIRD symbol outside the decision alphabet, not a re-used member of it. Abstaining policies "receive a small, additive reward on top of the value of a random guess." |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://link.springer.com/article/10.1007/s10462-025-11293-9 | peer-reviewed 2025 (AI Review) | **ATTEMPTED** full fetch; Springer 303-redirects to `idp.springer.com` auth. Not read; not counted. |
| https://en.wikipedia.org/wiki/Label_noise | encyclopaedia | tertiary; superseded by the arXiv survey |
| https://arxiv.org/pdf/1406.2080 | preprint (Sukhbaatar & Fergus, noisy-label CNNs) | canonical prior art; survey covers it |
| https://arxiv.org/pdf/2302.11075 | preprint survey (active learning + label noise) | adjacent |
| https://arxiv.org/pdf/2107.11413 | preprint (instance-dependent noise simulation) | adjacent |
| https://arxiv.org/abs/2501.08397 | preprint 2025 (abstention in dynamic graph learning) | recency corroboration; 2510.19672 read instead |
| https://cs.nyu.edu/~mohri/pub/olr.pdf | peer-reviewed (Cortes/DeSalvo/Mohri, online learning with abstention) | canonical prior art for the reject option |
| https://arxiv.org/pdf/2310.14772 | preprint (predictor-rejector multi-class abstention) | adjacent theory |
| https://www.jair.org/index.php/jair/article/download/12610/26733/28714 | peer-reviewed (JAIR, partial abstention) | adjacent theory |
| https://arxiv.org/pdf/2102.12258 | preprint (abstention without disparities) | adjacent |
| https://arxiv.org/pdf/2402.06287 | preprint (hybrid decision systems) | adjacent |
| https://deviq.com/principles/parse-dont-validate/ | practitioner reference | secondary to the King original |
| https://deviq.com/principles/make-illegal-states-unrepresentable/ | practitioner reference | secondary |
| https://aipatternbook.com/make-illegal-states-unrepresentable | practitioner reference | secondary |
| https://cekrem.github.io/posts/parse-dont-validate-typescript/ | blog | secondary |
| https://lobste.rs/s/uon7sc/parse_don_t_validate_2019 | forum | community tier |
| https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html | textbook chapter | sentinel-vs-mask trade-off; PEP 661 read instead |
| https://carpentry.library.ucsb.edu/2020-01-31-UCSB-SQL/05-null/ | teaching material | NULL-vs-sentinel; "0000-00-00 / -1.0" anti-pattern |
| https://www.oreilly.com/content/handling-missing-data/ | industry article | R reserved bit patterns vs SciDB extra byte |
| https://python-patterns.guide/python/sentinel-object/ | practitioner reference | superseded by PEP 661 |
| https://www.kdnuggets.com/2021/04/imerit-noisy-labels-impact-machine-learning.html | industry blog | community tier |
| https://towardsdatascience.com/an-introduction-to-classification-using-mislabeled-data-581a6c09f9f5/ | blog | community tier |

**URL tally: 7 read in full + 22 snippet-only = 29 unique URLs collected.**

### Recency scan (2024-2026) -- MANDATORY, performed

Searched the last-2-year window explicitly (`learning with abstention reject option better than wrong
prediction 2025 2026`) and reviewed the 2024-2026 hits. **Result: 2 new findings that COMPLEMENT
rather than supersede the canonical sources.**

1. **Abstention has moved from classification into POLICY learning** (arXiv:2510.19672, 2025) -- the
   closest analogue to this repo's situation, because a trading decision loop *is* a policy. Its
   design choice is the one this step is deciding: the abstention is a **special symbol outside the
   action alphabet** (`*`), not a re-use of an existing action. That is a direct argument against
   coercing an undeterminable direction into `HOLD`.
2. **Abstention is now standard in generation/QA settings** (2025 survey framing surfaced in the same
   scan): "it is of utmost importance to develop the ability to abstain ... to prevent the occurrence
   of misleading or incorrect information."

Nothing in the window overturns the older canon (King 2019, PEP 661 2021, PROV-DM 2013,
draft-thomson 2019). The label-noise canon (arXiv:2404.04159, 2024) is itself inside the window.

---

## Key findings (external), cited per claim

**F1 -- A wrong label is worse than an absent one, and the literature's own remedy is to DEMOTE a
suspect label to "unlabelled" rather than keep it.** The noisy-label survey's dominant method family
is sample selection, and DivideMix's mechanism is literally to divide the training set "into a
labeled set and an unlabeled set" -- suspect labels become *absent*, not *corrected*. The survey
accepts the cost explicitly: "Possible true-labeled samples can be discarded ... to benefit the
training of DNNs and reduce the negative effect of noisy samples."
(https://arxiv.org/html/2404.04159v1, accessed 2026-08-10.) The damage mechanism is memorisation:
"DNNs have such strong capacity that they can easily fit noisy labels during the model learning
process, resulting in poor generalization performance." **Caveat I must state honestly: the survey
does NOT contain a head-to-head "wrong vs missing" comparison.** It compares *feature* noise to
*label* noise ("feature noise ... is usually less harmful compared with label noise"). The
wrong-beats-absent ordering is supported here by (a) the remedy direction above and (b) the
abstention literature's explicit preference ordering below -- not by a single quoted sentence.

**F2 -- The abstention literature states the ordering directly, and gives the representation.**
Correct > abstain > incorrect is the premise of the reject-option field; the 2025 policy-learning
paper implements it with "a special symbol '\*'" outside `{0,1}` and a positive bonus for using it,
because "forcing a decision when evidence is weak can be harmful."
(https://arxiv.org/html/2510.19672v1, accessed 2026-08-10.) **The representational lesson is the
one that matters here: the abstention symbol is NOT a member of the decision alphabet.**

**F3 -- "Unknown" must be a distinct value, and re-using an in-domain value as the unknown marker is
the named failure.** PEP 661 exists because "`None` is a valid value in that context", and warns
that with a shared sentinel "one could not always be confident that it would never be a valid value
in some use cases." Its own enumerated use case is "Missing data, such as NULL in relational
databases or 'N/A' ('not available') in spreadsheets."
(https://peps.python.org/pep-0661/, accessed 2026-08-10.) The DB-teaching corollary (snippet tier)
is the classic anti-pattern: `0000-00-00` for a missing date, `-1.0` for a missing reading.

**F4 -- The remedy at a vocabulary boundary is to PARSE, at the boundary, into a type that cannot
hold the other vocabulary.** "Get your data into the most precise representation you need as quickly
as you can. Ideally, this should happen at the boundary of your system, before *any* of the data is
acted upon." Deferred/scattered checking is shotgun parsing: "Late-discovered errors in an input
stream will result in some portion of invalid input having been processed."
(https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/, accessed 2026-08-10.) The
corollary King states -- "write functions on the data representation you *wish* you had" -- argues
against widening `evaluate_recommendation` to also understand `APPROVE_*`.

**F5 -- Do NOT be liberal in what you accept.** "Choosing to generate fatal error for unspecified
conditions instead of attempting error recovery can ensure that faults receive attention";
"Allowing less variation is preferable in the absence of strong reasons to be flexible."
(https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-03, accessed 2026-08-10.)
Applied here: teaching the canonicaliser that `APPROVE_REDUCED` means `BUY` is exactly the leniency
that entrenches the drift. This is also the position `recommendation_vocab.py`'s own docstring
already takes (it cites this same draft).

**F6 -- The right fix is usually upstream, at the log-what-you-serve seam.** Rule #29: "The best way
to make sure that you train like you serve is to save the set of features used at serving time, and
then pipe those features to a log to use them at training time." Rule #30: "Importance-weight
sampled data, don't arbitrarily drop it!" Rule #34: prefer "small short-term sacrifices in
performance for very clean data" by holding examples out.
(https://developers.google.com/machine-learning/guides/rules-of-ml, accessed 2026-08-10.) Honest
negative: this doc gives **no** guidance on sentinel/magic values.

**F7 -- Provenance without a recorded writer is legitimate but must be attributed on evidence, and
"undetermined" is a representable answer.** "Provenance is information about entities, activities,
and people involved in producing a piece of data"; "Attribution is the ascribing of an entity to an
agent"; and the model tolerates elements "determined by unspecified means".
(https://www.w3.org/TR/prov-dm/, accessed 2026-08-10.) The admissible-evidence classes below are the
practical reading of this.

---

## Internal code inventory (every line number re-derived 2026-08-10 by reading the file)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/outcome_tracker.py` | 1-227 (read in full) | `evaluate_recommendation` def `:35-36`; `is_buy`/`is_sell` `:64-65`; `directionally_correct` `:66`; outcome dict `:68-78`; `self.bq.save_outcome(...)` `:81-90`; sibling `evaluate_all_pending` `:94-157` with its own call at `:144-149` | LIVE |
| `backend/services/autonomous_loop.py` | `:3345` def, `:3394-3399` SELL-only trade map, `:3401` flag read, `:3403-3521` per-ticker loop | the mismatching call site | LIVE |
| `backend/services/recommendation_vocab.py` | 1-142 (read in full) | phase-86.20/86.22 canonicaliser + `is_buy_intent`/`is_sell_intent`/`is_directional` | LIVE |
| `backend/db/bigquery_client.py` | `:400-417` `save_outcome`; `:47` `outcomes_table`; `:277-296` `get_recent_reports`; `:489` read side | LIVE writer #1 |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | `:27` call, `:37-90` `_compute_outcomes` | LIVE producer of the outcome dicts |
| `backend/slack_bot/jobs/_production_fns.py` | `:363` table const, `:383-412` `build_outcome_row`, `:451-487` `make_outcome_write_fn` | LIVE writer #2 |
| `backend/slack_bot/scheduler.py` | `:1233-1234` | `nightly_outcome_rebuild` = `cron hour=4`, `timezone=ZoneInfo("UTC")` | LIVE |
| `backend/config/settings.py` | `:34` `paper_learn_loop_enabled=False`; `:63` `bq_table_outcomes` | flag DARK |
| `backend/agents/skill_optimizer.py` | `:219` | READS `outcomes_table` | read-only consumer |
| `scripts/smoketest_stages_5_through_13.py` | `:368`, `:372-373` | MagicMock `side_effect` -- NOT a real call | test scaffold |

### I1 -- Where the value lands, and how it drives `directionally_correct`

`autonomous_loop.py:3419-3421` calls positionally:

```python
outcome = tracker.evaluate_recommendation(
    ticker, str(analysis_date), recommendation, price_at_rec
)
```

Against the signature at `outcome_tracker.py:35-36`
(`evaluate_recommendation(self, ticker, analysis_date, recommendation, price_at_rec)`), the
3rd positional argument lands in the **`recommendation` parameter** -- the analyst-vocabulary slot.
It is then used at `outcome_tracker.py:64-66`:

```python
is_buy  = is_buy_intent(recommendation)
is_sell = is_sell_intent(recommendation)
directionally_correct = (is_buy and return_pct > 0) or (is_sell and return_pct < 0)
```

and stored verbatim into the outcome dict at `:71` and into BigQuery's REQUIRED
`outcome_tracking.recommendation` column at `:84`.

### I2 -- What phase-86.22's canonicaliser does with a risk-approval token

`recommendation_vocab.canonical_recommendation` (`:70-88`) folds case and treats `[\s\-_]+` as one
separator, then requires membership in the CLOSED set `{STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}`
(`:59-61`). `APPROVE_REDUCED` folds to `APPROVE_REDUCED`, which is not a member, so it returns
`None`. `is_buy_intent` and `is_sell_intent` (`:118-131`) both return `False` for `None` --
**by design**: `"Unrecognised values are False -- never guessed into an intent ... on a learning path
a wrong label is worse than an absent one."** So `directionally_correct` computes to **`False`**.

**This is the exact defect class this step is about, and it is NOT a canonicaliser bug.** The
canonicaliser is correct: it refuses to guess. The bug is that `False` is then recorded as if it were
a measured judgement. The module even flags the gap itself at `:133-141`:

```python
def is_directional(value: object) -> bool:
    """... Exists so a caller can tell "this was a HOLD" apart from "this could not be
    parsed" -- the distinction `directionally_correct` silently destroyed by
    reporting False for both."""
```

`is_directional` is **defined but has zero production callers** (grep: only `recommendation_vocab.py`
itself). It is the pre-built seam for this step's fix.

### I3 -- What actually happens today on the live path (MEASURED, not assumed)

1. `autonomous_loop.py:3394-3399` builds `sell_by_ticker` from `action == "SELL"` rows ONLY.
2. M1 measured **32/32 SELL rows carry `risk_judge_decision = ''`**.
3. `:3412` `trade.get("risk_judge_decision", "HOLD")` returns `''` (the default fires only on a
   MISSING key; the key is present with an empty value -- the same `.get()` trap documented at
   `nightly_outcome_rebuild.py:40-43`).
4. `:3416-3417` coerces `''` -> `"HOLD"`.
5. `HOLD` canonicalises fine, is in NEITHER `BUY_INTENT` nor `SELL_INTENT`
   (`recommendation_vocab.py:113-116`), so `directionally_correct = False`.

**So the current live behaviour is a 100% degenerate: every closed trade is labelled
"not directionally correct", and the reason is indistinguishable from a genuine wrong call.**
The APPROVE/BUY vocabulary mismatch is the *latent* form -- it becomes the live form the moment a
SELL row carries a populated `risk_judge_decision` (19 BUY rows already do).

### I4 -- Enumeration of EVERY caller of `evaluate_recommendation` (CRITERION 5)

Grep over `backend/` + `scripts/`, excluding `backend/tests/`:

| # | Call site | Value passed | Vocabulary | Mismatch? |
|---|---|---|---|---|
| 1 | `backend/services/outcome_tracker.py:144-149` (inside `evaluate_all_pending`) | `report["recommendation"]` from `BigQueryClient.get_recent_reports` (`bigquery_client.py:277-296`, selects `recommendation` from `reports_table` = `financial_reports.analysis_results`) | **ANALYST** (measured M3: HOLD/Hold/BUY/Buy/Sell/Strong Buy/N-A) | **NO** -- correct vocabulary. Post-86.22 every spelling except `N/A` canonicalises; `N/A` -> `None` -> non-directional (correct behaviour, but see the same unknown-vs-hold collapse). |
| 2 | `backend/services/autonomous_loop.py:3419-3421` | `trade.get("risk_judge_decision", "HOLD")`, `''`-coerced to `HOLD` at `:3416-3417` | **RISK-APPROVAL** (measured M1: `''`/APPROVE_REDUCED/REJECT/APPROVE_HEDGED) | **YES** -- the only one |
| 3 | `scripts/smoketest_stages_5_through_13.py:368` | `tracker.evaluate_recommendation.side_effect = fake_eval` | n/a -- MagicMock | not a real call |

**Answer to the CLASS question: exactly ONE production call site has the vocabulary mismatch.**
`evaluate_all_pending` is clean. The class does NOT generalise across call sites of this function.
It DOES generalise across *writers of the same column*: `nightly_outcome_rebuild.py:67` performs the
same `risk_judge_decision`-into-a-recommendation-slot move (see I5), and it is the writer that
actually produced the live rows.

---

## I5 -- PROVENANCE: who wrote the three `SELL` rows? (CRITERION 3) -- **DETERMINED**

### Step 1: enumerate EVERY writer of `financial_reports.outcome_tracking`

Grep for `outcomes_table` / `OUTCOME_TABLE` / `insert_rows_json` across `backend/` + `scripts/`,
excluding tests. The table has exactly **two** insert seams and **one** DDL seam:

| # | Writer | Entry points | Insert call |
|---|---|---|---|
| W1 | `BigQueryClient.save_outcome` (`bigquery_client.py:400-417`) | (a) `outcome_tracker.py:81-90` primary path; (b) `autonomous_loop.py:3458-3467` phase-35.1 fallback | `insert_rows_json(self.outcomes_table, [row])` at `:415` -- **one row per call** |
| W2 | `_production_fns.make_outcome_write_fn._write` (`_production_fns.py:463-487`) | `nightly_outcome_rebuild.py:27` (`cron hour=4`, UTC, `scheduler.py:1233-1234`) | `insert_rows_json(OUTCOME_TABLE, records)` at `:473` -- **a whole batch per call** |
| W3 | `scripts/migrations/migrate_bq_schema.py:136-160` | `ensure_outcome_tracking_table` | DDL only -- creates the table, inserts no rows |

Read-only consumers (ruled out as writers): `skill_optimizer.py:219`, `bigquery_client.py:489`
(`get_performance_stats`), `_production_fns.py:372` (the dedup SELECT).

### Step 2: admissible evidence, applied

The row carries no writer column, so attribution must rest on evidence classes that are decisive
rather than merely consistent. Four independent lines, all pointing the same way:

**E1 -- a value spelling only ONE writer can produce (mechanism, not correlation).**
`nightly_outcome_rebuild.py:67`:
```python
recommendation = t.get("risk_judge_decision") or t.get("action")
```
`or` is falsy-triggered, and M1 measured `risk_judge_decision = ''` on **32/32** SELL rows, so this
expression evaluates to `t["action"]` = the literal `"SELL"` for every SELL trade. W1's two callers
cannot produce it: `outcome_tracker.py:84` passes through whatever its caller gave it, and its two
callers pass either the ANALYST vocabulary (M3: `SELL` appears **0** times there; the sell spelling
is title-case `Sell`) or, from `autonomous_loop.py:3416-3417`, the string `"HOLD"`. **No other code
path in the repo can put the literal `SELL` in this column.**

**E2 -- adjacent columns only ONE writer leaves NULL.** All three rows have
`price_at_recommendation = NULL` **and** `beat_benchmark = NULL`. `build_outcome_row` hardcodes both:
`"price_at_recommendation": None` (`_production_fns.py:407`) and `"beat_benchmark": None` (`:411`).
W1 can produce neither: `outcome_tracker.py:44-45` early-returns unless `price_at_rec` is truthy and
passes a real `beat_benchmark_flag` bool (`:54`, `:89`); `autonomous_loop.py:3462` passes
`price_at_rec or sell_price` and `:3466` passes `beat_benchmark=(pnl_pct > 0)` -- **never None on
either column**. Two independently-NULL columns is a signature, not a coincidence.

**E3 -- timestamp granularity discriminates batch writers from per-row writers.** All three rows
share an **identical microsecond** `evaluated_at` = `2026-08-08T04:00:02.013552+00:00`. W1 stamps
`datetime.now(timezone.utc).isoformat()` INSIDE the row dict (`bigquery_client.py:413`) on a
per-call, single-row insert -- three W1 writes would carry three distinct microsecond values. W2
computes `now_iso` **once** at `_production_fns.py:466` and applies it to the whole batch at `:467`.
An identical `.013552` across three rows is only reachable through W2.

**E4 -- schedule alignment.** `04:00:02 UTC` is 2 seconds after `nightly_outcome_rebuild`'s
`cron hour=4, timezone=UTC` (`scheduler.py:1233-1234`). No other outcome writer is time-triggered.

**Corroborating (weaker, offered as such):** `git log` shows the W2 rewrite
`fix(82.48): repair the outcome write -- it emitted a schema that never existed` landed
**2026-08-06 12:45 +0200**; the only rows in the table are from the **2026-08-08 04:00 UTC** run.
I did NOT determine why the 2026-08-07 04:00 run left no rows -- the most likely explanation is that
the Slack-bot process had not been restarted onto the new code (cf. the standing
"committed is not in force" hazard), but **I did not verify the process start time, so I am
recording this as UNDETERMINED rather than asserting it.**

### Verdict

**W2 -- `nightly_outcome_rebuild` -> `_compute_outcomes` (`nightly_outcome_rebuild.py:67`) ->
`build_outcome_row` -> `make_outcome_write_fn._write`.** Determined on four independent evidence
lines, one of which (E1) is a *mechanism* that reproduces the exact observed spelling from the
exact measured input distribution, not merely a consistent signature.

### What was NOT admissible, and why (the honest half)

- **"`SELL` is an action, and SELL trades exist" -- inadmissible alone.** It is consistent with
  several stories (e.g. a hand-run script, a since-deleted writer). It only becomes decisive when
  joined to E1's *measured* `risk_judge_decision = ''` on 32/32 SELL rows, which is what makes the
  `or`-fallback fire deterministically.
- **Row ordering / BigQuery streaming-buffer position -- inadmissible.** Not stable, not recorded.
- **The absence of a `directionally_correct` column -- inadmissible.** It distinguishes nothing:
  the column does not exist, so no writer could have set it.
- **`git log` alone -- inadmissible.** It dates the *code*, not the *row*; it is corroboration only.

### The transferable lesson

The whole attribution rested on **columns whose NULL-ness is a writer's constant** (E2) and on
**timestamp granularity revealing batch-vs-row write shape** (E3). Neither is a designed provenance
feature -- both are accidents. Per PROV-DM, the durable fix is to record attribution explicitly
("Attribution is the ascribing of an entity to an agent",
https://www.w3.org/TR/prov-dm/, accessed 2026-08-10): a `written_by` / `source_path` column would
have made this a one-line `SELECT` instead of a four-line inference.

---

## Application to pyfinagent -- the decision the contract has to make

### The three options, mapped to the evidence

**(A) "Look up the right value."** Fetch the actual analyst recommendation for the closed trade
(join `analysis_id` -> `analysis_results.recommendation`, the same source
`evaluate_all_pending` already uses at `outcome_tracker.py:144-149` / `bigquery_client.py:277-296`).
**Strongest where feasible** -- it is the only option that produces a *true* label, and it is F6's
"log what you serve" applied (https://developers.google.com/machine-learning/guides/rules-of-ml,
Rule #29). **Risk to state in the contract:** the join key is not proven -- `analysis_date` at
`autonomous_loop.py:3409` is `analysis_id or created_at`, so the anchor is already ambiguous, and I
did NOT measure the join hit-rate. A contract choosing (A) must first measure how many of the 32
SELL rows resolve to an `analysis_results` row.

**(B) "Skip the scoring."** Matches the noisy-label canon's own remedy (demote to the *unlabelled*
set, https://arxiv.org/html/2404.04159v1) and Rule #34's "hold out rather than train on corrected
labels". **But** Rule #30 warns "Importance-weight sampled data, don't arbitrarily drop it!", and
skipping silently loses the fact that a close happened -- the very invisibility that let this
degenerate run for weeks behind a `logger.debug` swallow at `autonomous_loop.py:3520-3521`.

**(C) "Record an explicit unknown state."** The literature's answer where the event itself is worth
keeping: a symbol OUTSIDE the decision alphabet (`*`, https://arxiv.org/html/2510.19672v1), a
sentinel that is provably not a valid domain value (https://peps.python.org/pep-0661/), a NULL
rather than an in-domain marker. **This is what the repo is already shaped for**: `outcome_tracking`
has **no** `directionally_correct` column (M2), so an explicit tri-state is an ADDITIVE nullable
column, not a rewrite; and `recommendation_vocab.is_directional` (`:133-141`) already exists,
unused, for exactly this discrimination.

**Synthesis the evidence supports:** (A) where the lookup resolves; (C) where it does not; and never
(B) silently. Concretely, `directionally_correct` should stop being a `bool` and become a three-way
outcome -- `True` / `False` / **unknown** -- with `unknown` represented as SQL `NULL` or a distinct
status column, never as `False` and never as a coerced `HOLD`. The current
`recommendation = "HOLD"` coercion at `autonomous_loop.py:3416-3417` is precisely PEP 661's named
anti-pattern: an in-domain value used as the missing-data marker, and one that a downstream reader
cannot tell from a real hold.

### The vocabulary boundary itself

Do **not** widen `recommendation_vocab` to accept `APPROVE_REDUCED`. That is the leniency
draft-thomson names as the mechanism of decay
(https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-03), and it would silently
assert that "risk approved a reduced size" means "the analyst said buy" -- a claim nobody made.
The parse-don't-validate remedy (https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)
is to resolve the vocabulary **at the boundary**, i.e. inside `_learn_from_closed_trades` before the
call, and to hand `evaluate_recommendation` either a real analyst recommendation or an explicit
"unknown" -- never a risk-approval token wearing a recommendation's parameter name.

### Pitfalls (from the literature + this codebase)

1. **Encoding unknown as a valid class value** is the named classic failure (PEP 661; the
   `0000-00-00` / `-1.0` DB anti-pattern). `HOLD` here IS that failure.
2. **`.get(key, default)` does not fire on a present-but-empty value.** Measured: 32/32 SELL rows
   have `risk_judge_decision = ''`, so the `"HOLD"` default at `:3412` never fires -- the coercion at
   `:3416-3417` does. Any fix that only changes the `.get` default is a no-op.
3. **The failure is invisible by construction.** `autonomous_loop.py:3520-3521` swallows every
   per-ticker exception at `logger.debug`; `_alert_write_rejected` exists for W2 but there is no
   equivalent for a *semantically* wrong label. A tri-state without a counter/alert repeats the
   invisibility.
4. **Two writers, one column, different semantics.** W1 and W2 both write
   `outcome_tracking.recommendation` from different sources with different fallbacks. A fix applied
   to only one leaves the column mixed. `nightly_outcome_rebuild.py:67` is the writer that actually
   produced today's rows.
5. **`is_directional` already exists and is uncalled** -- adding a second discrimination helper
   would recreate the exact "two canonicalisers that disagree" hazard `recommendation_vocab.py`'s
   docstring warns about.

### Consensus vs debate

**Consensus:** wrong > absent in damage; unknown must be representable outside the value domain;
parse at the boundary. **Genuine debate:** *drop vs record*. The noisy-label field discards suspect
labels (arXiv:2404.04159); Google Rule #30 says do not arbitrarily drop, weight instead; the
abstention field says keep the instance and mark it. For an audit-bearing learning loop with a
**35-row denominator**, the abstention/record position is the better fit -- dropping is only cheap
when the data is abundant, and here it is not.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total (incl. snippet-only) -- **29**
- [x] Recency scan (last 2 years) performed + reported -- 2 complementary findings, section above
- [x] Full pages read (not abstracts) for the read-in-full set; both arXiv sources fetched via the
      `arxiv.org/html/` chain, never a `/pdf/` URL
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (outcome_tracker, autonomous_loop,
      recommendation_vocab, bigquery_client, both nightly-job modules, scheduler, settings)
- [x] Contradictions / consensus noted (drop-vs-record debate above)
- [x] All claims cited per-claim
- [ ] **GAP, stated honestly:** the Springer 2025 benchmarking paper could not be fetched (IdP
      redirect) and is recorded as snippet-only, not counted. The "wrong label is worse than an
      absent one" claim rests on the remedy direction + the abstention ordering, **not** on a single
      head-to-head quotation -- no such head-to-head sentence was found in any source read.
- [ ] **NOT MEASURED:** the `analysis_id -> analysis_results` join hit-rate for the 32 SELL rows
      (needed before option (A) can be costed), and the Slack-bot process start time (needed to
      explain the missing 2026-08-07 run).

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 22,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.25.md",
  "gate_passed": true
}
```

**STATUS: COMPLETE.**
