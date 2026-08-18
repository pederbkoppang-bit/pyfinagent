---
name: check-the-attribution-not-just-the-count
description: A census can reproduce its NUMBER exactly and still be wrong about the CAUSE; the causal clause is what decides whether a 100%-prevalence hit is in scope or dismissed
metadata:
  type: feedback
---

When a census dismisses a hit as out of scope, **test the causal clause, not
only the count**. 86.78 cycle 4: "413 of 413 spawn prompts contain
'3rd-CONDITIONAL'/'auto-FAIL' -- **because the prompt embeds qa.md itself**",
therefore design, not exposure. The count reproduced exactly (420/420). The
*because* did not: 0 of 421 prompts contained any qa.md body text (tested with
three qa.md-unique markers), while 420/420 carried one line of the step's OWN
rail prompt verbatim. The dismissal rested entirely on the false half.

**Why:** a "because X, therefore out of scope" sentence does two jobs — it
explains a measurement and it moves a finding off the books. Reproducing the
number only audits the first job. Here the misattribution relocated a residual
from the file the step owns and claims to have cleaned, into the file it had
already declared operator-gated and out of scope.

**How to apply:** for every "N of N, because <cause>" claim, pick a marker
UNIQUE to the alleged cause and count it separately. If the alleged cause is a
file said to be embedded, grep for text that exists only in that file's body —
not for a section NAME the including file also quotes (my first marker,
`Write-first for your VERDICT FILE ONLY`, scored 139/420 purely because the
rail prompt cites the section by name; the body markers scored 0). Then grep
for the ALTERNATIVE cause verbatim and see if it accounts for 100%. Two greps
settle it.

Related: [[feedback_a_correct_observation_can_credit_the_wrong_mechanism]],
[[feedback_a_probe_can_match_its_own_documentation]],
[[feedback_queued_is_a_claim_that_must_reproduce]].
