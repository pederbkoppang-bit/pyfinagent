---
name: severity-routing-90-2
description: Step 90.2 measurements -- re-deriving Q/A severity from free text scores WORSE than a constant (40.0% vs 56.7%, kappa 0.129); 71% of BLOCK mentions are negated; 0/969 violation_details rows carry a severity key; the qa-verdict corpus has four defensible denominators
metadata:
  type: project
---

Measured 2026-08-20 over the local qa-verdict run-record corpus for step 90.2.

**Re-deriving severity from the judge's prose is worse than guessing.** Token-presence
extraction (worst of BLOCK/WARN/NOTE anywhere in the returned object) agrees with the
SKILL.md dispatch on **144/360 = 40.0%**, Cohen's **kappa = 0.129**, against a
majority-class baseline of **204/360 = 56.7%** (always guess CONDITIONAL). The
mechanism is negation: **130 of the 183 records containing `BLOCK` (71.0%) contain a
NEGATED occurrence within 25 chars** -- the judge's dominant use of the token is "no
BLOCK, no WARN fired". The precise extractor (literal `severity=`/`severity:`) reaches
82.5% but fires on only **63/397 (15.9%)** of runs and captures 8 junk values in 167
(`P`, `BLOCKING`, `CAPPING`, `ONE`, `THE`, `SOLE`, `A`, `_QA_SEV`).

**Why:** external corroboration is arXiv 2604.16706 (AgentProp-Bench, 2026): substring-
heuristic judging agrees with humans "only at chance level (Cohen's kappa = 0.049)" vs
0.432 ensemble / 0.567 single judge / 0.835 human-human -- verified verbatim at source.

**How to apply:** never propose harvesting a severity the judge "already says" in prose.
`VERDICT_SCHEMA` is `additionalProperties:false` and **0 of 969** historical
`violation_details` rows carry a `severity` key even though `SKILL.md:29` tells the judge
to write one -- so it is a schema edit (optional field, the phase-86.72 `research_needed`
shape) or nothing.

**Four defensible denominators for "the qa-verdict corpus"** -- pin the predicate or the
replay is unreproducible: **441** (`workflowName.startsWith('qa-verdict')`, adds
`qa-verdict-writefirst-82-5` x3 and `-82-7` x2), **436** (exact match), **398** (parseable
dict `result`), **397** (non-null verdict -- the replay denominator). Base rates:
CONDITIONAL 221 (55.7%), PASS 109 (27.5%), FAIL 67 (16.9%). 618 total workflow records.

**A summariser fabricated a normative quote.** WebFetch on the 1.6 MB OASIS SARIF spec
returned "A SARIF consumer SHALL NOT re-derive the level value" attributed to 3.27.10.
The string `re-derive` does not appear anywhere in the specification. Not PDF-specific --
see [[webfetch-pdf-summaries-fabricate-quotes]]. Verify load-bearing quotes with
curl + tag-strip grep. What SARIF actually says: `level` (3.27.10, closed enum
none/note/warning/error) is separate from `kind` (3.27.9) and from consumer-side `rank`
(3.27.25, unknown sentinel `-1.0`), and an absent `level` resolves by a written ladder
(configurationOverride -> defaultConfiguration -> `"warning"`), never by inference.

Related: [[rec-vocabulary-86-20]] (`.upper()` folds case, not separators -- BLOCK vs
BLOCKING vs NOTE-LEVEL all occur), [[re-derive-urls-collected-never-carry-it]].
