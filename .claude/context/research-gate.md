---
name: Research Gate Protocol
description: Mandatory deep research before any implementation — Peder's Rule since 2026-03-29
---

Every plan step must pass the Research Gate before proceeding to GENERATE.

**Why:** Phases 0-2.7 were built from general knowledge with zero papers fetched. Phase 2.8 (first with real research) immediately found 2 code bugs (DSR trial count, Lo(2002) formula). Research-first execution produces measurably better code. "I already know how to do this" is not research — it's assumption.

**How to apply:** Search all 7 categories (Scholar, arXiv, universities, AI labs, quant firms, consulting, GitHub), collect >=10 URLs, read **>=5 in full** (phase-4.16.1 raised the floor from 3-5), document in RESEARCH.md with URLs, update handoff/current/contract.md with research-backed thresholds (e.g., "DSR > 0.95 per Bailey & Lopez de Prado (2014)"). Always run an explicit last-2-year scan (e.g. 2024-2026) and report its outcome, even when "no new findings". If ANY checklist item is unchecked, STOP and do more research. See `ARCHITECTURE.md::Research Gate Discipline (phase-4.16)` + `.claude/rules/research-gate.md` for the full protocol.

**Checklist:**
- [ ] Searched all 7 source categories? (not just Google Scholar)
- [ ] Collected >=10 candidate URLs? (>=3 for simple steps)
- [ ] Selected and read **>=5** best sources in full? (phase-4.16.1 floor)
- [ ] Last-2-year recency scan performed and reported?
- [ ] Documented findings in RESEARCH.md with URLs?
- [ ] Extracted concrete thresholds/methods to adopt?
- [ ] Noted warnings/pitfalls from literature?
- [ ] Updated handoff/current/contract.md with research-backed criteria?

**Autoresearch memos count as a source category.** A nightly gpt-researcher cron (`scripts/autoresearch/run_memo.py`, launched by `com.pyfinagent.autoresearch`) emits markdown memos into `handoff/autoresearch/<date>-topic<NN>-<slug>.md`. Each memo is a Claude-driven `detailed_report` on one of 14 rotating topics about AI in equity trading — academic papers, university groups, company findings. When a harness cycle touches a topic already covered by a recent memo (within the last 14 days), the cycle MUST cite that memo in its RESEARCH.md + contract.md and treat it as satisfying 3-5 of the required URL sources. New independent searches are still required for the remaining sources, but the memo counts as one pre-computed source category.
