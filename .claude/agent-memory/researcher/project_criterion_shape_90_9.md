---
name: criterion-shape-90-9
description: Step 90.9 -- the filing's "unrecoverable" corpus figures reproduce EXACTLY once step 90.9 itself is excluded; the unbounded-scope rule is the only genuinely broken one; criterion 4's write-pattern list misses the house idiom
metadata:
  type: project
---

Step 90.9 (classify acceptance-criterion SHAPE at filing time). Measured
2026-08-20 on the live tree.

**The filing's rule IS recoverable; the missing variable was the corpus
TIMESTAMP, not the regex.** Inclusion rule = walk every node with an `id` AND a
dict `verification`, keep non-empty `verification.success_criteria`. Excluding
step 90.9 itself reproduces all six figures EXACTLY: 155 steps / 980 criteria /
403 apparatus / 41.1% / 78 terminal / 1026 of 4670 = 22.0% project-wide. The
whole delta is 90.9's own 7 criteria (6 apparatus-matching, terminal NOT
matching, so terminal is invariant at 78). Ratio collapses to **1.87x** with no
rule change. A first-pass keyword rule reproducing 35.8% is using a DIFFERENT
inclusion predicate, not a different regex.

**Why: re-derive, never carry.** [[feedback_url_count_must_be_re_derived]] is the
same lesson one level up -- a criteria census is a function of (predicate,
corpus-pin), and the filer measured before their own step object existed.

**The unbounded-scope count (44) is the only figure that does NOT reproduce.**
Four keyword variants give 51 / 40 / 0 / 39. The literal self-reference rule
("every ... this step adds") returns **0 of 155** -- the self-reference is never
written explicitly, it is IMPLIED by an unbounded quantifier landing on a noun
class the step is simultaneously growing. So the detector cannot be a
quantifier-keyword rule; it needs the quantified NOUN CLASS tested against the
artifact class the step produces. This is a SEMANTIC smell, and semantic smells
are exactly the ones keyword rules miss (arXiv 2501.04810: syntactic BTA 0.98 vs
semantic 0.83).

**Criterion 4's write-pattern list is under-inclusive in the house's own idiom.**
It names `open(...,'w')` and `json.dump`. The two scripts/qa scripts that write
that filename both use `Path.write_text` -- verify_decision_log_86_97.py:274,300
and prove_archive_provenance_86_29.py:92 (both to a tmp fixture root). A
classifier could mutate the plan and still pass the source half. Needs AST-level
resolution of write-capable calls.

**Criterion 7 is STRICTER than the Q/A rail it cites.** qa-verdict.js:336 DOES
pass `verdict_sequence` + `attempt_number` into the judge (evidence, trigger
withheld; derived caller-side at :642-646). 90.9 forbids the classifier from
receiving them at all -- correct, since a shape classifier has no legitimate use
for them, but the contract must state the divergence rather than imply parity.
Because ERRJ=0.000 (arXiv 2604.15224), the test must assert over the INPUT
surface; output inspection cannot detect the leak.

**Best external analogue found:** arXiv 2607.24300 "Self-Authored Verification Is
Unreliable in Heuristic Self-Improving Agents" -- agent edits both policy.py and
tests.py; 35/35 cells self-score >0.70 while 15 policies score below random;
named failure mode "failure to retain: tests evolve to accommodate degraded
behavior"; proves NO endogenous gate holds low false-accept and low false-reject
at once. Remedy SEAL = exogenous fixed audit + audit confidentiality + single-bit
feedback. Brief: handoff/current/research_brief_90.9.md.
