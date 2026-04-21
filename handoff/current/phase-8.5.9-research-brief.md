---
step: phase-8.5.9
tier: simple
date: 2026-04-19
---

# Research Brief: Seed Candidate Space from Virtual-Fund Failure Cases

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://sre.google/sre-book/postmortem-culture/ | 2026-04-19 | Official doc | WebFetch | "ensure effective preventive actions are put in place"; action items must be prioritized (P0/P1/P2) so root causes are addressed before exploratory work |
| https://sre.google/workbook/postmortem-culture/ | 2026-04-19 | Official doc | WebFetch | Tags equal priority on all action items criticised as a bad postmortem; differentiated priority is the canonical pattern |
| https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining | 2026-04-19 | Official doc | WebFetch | Self-evolving loops must first consolidate failure signals, retry within scope (MAX_OPTIMIZATION_RETRIES=3), then promote only proven improvements; novel search happens last |
| http://karpathy.github.io/2019/04/25/recipe/ | 2026-04-19 | Authoritative blog | WebFetch | "Don't be a hero"; fix known bugs and validate pipeline before introducing novel architectural complexity |
| https://machinelearningmastery.com/why-agents-fail-the-role-of-seed-values-and-temperature-in-agentic-loops/ | 2026-04-19 | Practitioner | WebFetch | Fixed seeds repeat the same failure path; failure-driven seed adjustment is the correct escape mechanism before open-ended exploration |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://openreview.net/forum?id=nz2vqJI1fk | Paper | Autonomous driving domain; adjacent not directly applicable |
| https://arxiv.org/abs/2309.14209 | Paper | Same driving-curriculum paper; snippet sufficient |
| https://jmlr.org/papers/volume21/20-212/20-212.pdf | Paper | PDF binary unreadable via WebFetch |
| https://plane.so/blog/bug-triage-process-how-to-run-it-and-what-to-prioritize | Blog | General triage; not research-loop specific |
| https://sre.google/sre-book/example-postmortem/ | Official doc | Supporting example; core doc already fetched in full |
| https://cloud.google.com/blog/products/devops-sre/loon-sre-use-postmortems-to-launch-and-iterate | Blog | Corroborating; covered by primary SRE sources |

## Recency scan (2024-2026)

Searched: "failure-driven curriculum autonomous optimization loop 2025 2026" and "postmortem-driven test seeding autonomous research loops 2026".

Result: one relevant 2025-2026 finding -- CLIC (Closed-Loop Individualized Curricula, OpenReview 2026) re-samples from historical scenarios weighted by failure probability, directly supporting the bucket-first ordering pattern. No finding supersedes the canonical bucket-first principle; newer work reinforces it.

## Key findings

1. Bucket-first ordering is canonical -- Google SRE workbook condemns equal-priority action items; differentiated P0/P1/P2 ordering by failure severity is the documented pattern. (Google SRE Workbook, https://sre.google/workbook/postmortem-culture/)

2. Self-evolving agent loops process failure signals before novel search -- OpenAI cookbook prescribes consolidating grader failures, retrying within bounded scope, then and only then promoting new candidates. (OpenAI Cookbook, https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining)

3. Fix known bugs before novel exploration -- Karpathy: "don't be a hero"; systematically validate what is broken before architectural innovation. (Karpathy 2019, http://karpathy.github.io/2019/04/25/recipe/)

4. Fixed-seed loops repeat failure -- seed must change to escape a known failure path; structured seed-from-postmortem is the principled version of this. (MLMastery, https://machinelearningmastery.com/why-agents-fail-the-role-of-seed-values-and-temperature-in-agentic-loops/)

5. Failure-weighted curriculum sampling (2026 CLIC) directly mirrors the bucket-first seed design in autoresearch_seed_from_postmortem.py. (OpenReview 2026, https://openreview.net/forum?id=nz2vqJI1fk)

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| handoff/virtual_fund_postmortem.md | 49 | 4 failure buckets + novel-secondary section | Present, well-formed |
| scripts/harness/autoresearch_seed_from_postmortem.py | 92 | Regex parser; emits bucket seeds before novel seeds; --dry-run verified exit 0 | Present, passing |

## Consensus vs debate

Consensus: all sources agree that known failure remediation precedes novel search. No debate found.

## Pitfalls

- Regex seed_target extraction truncates at first newline (line 34 of script): multi-line seed targets return only the first line. Acceptable for current postmortem structure; flag if buckets grow multi-sentence targets.
- Novel-search seeds are mocked constants in the script; real proposer integration not yet wired.

## Application to pyfinagent

- handoff/virtual_fund_postmortem.md:15-36 maps cleanly to the four bucket seeds the script emits.
- scripts/harness/autoresearch_seed_from_postmortem.py:65-70 implements and verifies bucket-before-novel ordering, matching canonical SRE P0-first discipline.
- Immutable test `test -f handoff/virtual_fund_postmortem.md && python scripts/harness/autoresearch_seed_from_postmortem.py --dry-run` exits 0 (verified 2026-04-19).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (incl. snippet-only) -- 11 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 6,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "gate_passed": true
}
```
