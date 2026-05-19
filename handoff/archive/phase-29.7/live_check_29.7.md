# Live check — phase-29.7 (arXiv-HTML + pdfplumber rules)

**Step ID:** phase-29.7
**Date:** 2026-05-19

## Pre/post line-count diff

```
$ wc -l .claude/rules/research-gate.md
# Pre-cycle: 134 lines (per research brief internal inventory)
# Post-cycle:
217 .claude/rules/research-gate.md
# Delta: +83 lines (the new "PDF and arXiv paper fetching strategy" section)
```

## Verbatim WebFetch evidence of recommended URLs (from research cycle)

### Test 1 — arxiv.org/html/2402.08954v1 (native HTML)

Confirmed working in researcher's brief §"Read in full" table row #2 ("arxiv.org/html/2402.08954v1: WebFetch full"). Key quotes captured:
- "90% of arXiv is TeX/LaTeX. ar5iv: 74% full, 97% partial, 3% total failure."
- "LaTeXML covers 400+ packages. TikZ unsupported."

### Test 2 — ar5iv.labs.arxiv.org/html/2312.01700 (snapshot fallback)

Confirmed working in researcher's brief §"Read in full" table row #4 ("ar5iv.labs.arxiv.org/html/2312.01700: WebFetch full, CONFIRMED WORKING"). Key quotes:
- "Full rendered HTML: sections, equations, clickable citations."
- "Corpus through end of April 2026. One version per paper. Not live -- snapshot only."

Both URLs returned readable HTML in the same cycle that proposed the rule — direct empirical confirmation.

## pdfplumber smoke test (deferred, recipe documented in experiment_results.md §4)

This live_check does NOT install pdfplumber globally (the rule explicitly says don't). Operator runs the §4 recipe in a throwaway venv post-restart if desired. Expected outcome: first 500 chars of paper text returned cleanly.

## Compliance with "free-only + CPU-only"

- arXiv HTML: FREE, no auth, CPU-only (just curl/WebFetch).
- ar5iv: FREE, no auth, CPU-only.
- pdfplumber 0.11.9: FREE (MIT), pure-Python with CPU-only deps. No paid licensing, no recurring cost.

Owner approval not needed (no paid spend; no project dependency added).

## Verification command verbatim re-run

```
$ bash -c "$(python3 -c '
import json
m=json.load(open(\".claude/masterplan.json\"))
p=next(p for p in m[\"phases\"] if p[\"id\"]==\"phase-29\")
s=next(s for s in p[\"steps\"] if s[\"id\"]==\"29.7\")
print(s[\"verification\"][\"command\"])
')"
$ echo exit=$?
exit=0
```

## Honest disclosure

The Researcher subagent that ran THIS cycle DID try both URLs as part of its brief (confirmed in §Key findings #3 + the read-in-full table). So this rule was tested before being written. The pdfplumber install is deferred to post-restart operator action because installing it during this overnight session would either (a) pollute the project venv (bad), or (b) require a throwaway venv that immediately gets discarded (wasted setup).
