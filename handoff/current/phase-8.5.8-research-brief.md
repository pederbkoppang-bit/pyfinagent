---
phase: 8.5.8
step: Weekly HITL Review Packet
tier: simple
date: 2026-04-19
---

## Research: HITL Review Patterns, Slack Markdown, DSR Ranking, Capital Safety Gates

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://machinelearningmastery.com/building-a-human-in-the-loop-approval-gate-for-autonomous-agents/ | 2026-04-19 | blog | WebFetch | State-managed interruption: agent pauses, state persisted, human sets approved=True, downstream nodes gate on flag |
| https://docs.slack.dev/reference/block-kit/blocks/markdown-block/ | 2026-04-19 | official docs | WebFetch | Markdown block: type="markdown", text field, 12,000-char limit, supports tables, code, bold, headers |
| https://myengineeringpath.dev/genai-engineer/human-in-the-loop/ | 2026-04-19 | blog | WebFetch | Tiered escalation: pre-action vs post-action approval; high-stakes/irreversible actions require pre-action gate |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | 2026-04-19 | blog/quant | WebFetch | DSR 0.95+ = strong evidence vs noise; rank by DSR not raw SR; corrects for selection bias across N trials |
| https://www.synvestable.com/human-in-the-loop.html | 2026-04-19 | blog | WebFetch | Pre-decision approval: nothing moves without human sign-off; start HITL, graduate to HOTL only after validation |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/html/2510.04787v1 | paper | Not fetched; snippet coverage sufficient for context |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | reference | Snippet sufficient |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | paper | 403 on fetch |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | paper | Binary PDF, not parseable via WebFetch |
| https://docs.slack.dev/block-kit/ | official docs | Snippet sufficient; markdown-block page read in full |
| https://bytebridge.medium.com/from-human-in-the-loop-to-human-on-the-loop-evolving-ai-agent-autonomy-c0ae62c3bf91 | blog | Snippet coverage sufficient |
| https://siliconangle.com/2026/01/18/human-loop-hit-wall-time-ai-oversee-ai/ | blog | Context only |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC8978471/ | paper | Not needed after other sources covered safety |

### Recency scan (2024-2026)

Searched "human-in-the-loop review autonomous trading systems approval gate 2026" and "human-in-the-loop trading approval autonomous agents weekly review cadence". Result: found relevant 2025-2026 material on HITL patterns (MachineLearningMastery 2026, MyEngineeringPath 2026, Synvestable 2026). EU AI Act Article 14 (2026 enforcement) now legally requires human oversight for high-risk AI. No new DSR papers in the 2024-2026 window that supersede Bailey & Lopez de Prado; the 2013 canonical paper remains the reference. Slack markdown block announced February 2025.

### Key findings

1. HITL approval gate pattern: agent pauses at high-stakes node, state is persisted, human sets approval flag, downstream nodes gate on that flag. The approval clause in the rendered output IS the human-readable trigger. (Source: MachineLearningMastery, 2026, https://machinelearningmastery.com/building-a-human-in-the-loop-approval-gate-for-autonomous-agents/)

2. Capital moves are textbook "irreversible + high-blast-radius" actions that require pre-action approval regardless of confidence score. No auto-promotion without explicit written approval is the safe default before HOTL graduation. (Source: MyEngineeringPath, 2026, https://myengineeringpath.dev/genai-engineer/human-in-the-loop/)

3. DSR > 0.95 = strong evidence against noise; DSR < 0.5 = indistinguishable from luck. Rank top-N by DSR (not raw Sharpe) to correct for selection bias across N backtested trials. (Source: Balaena Quant Insights, 2026, https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464)

4. Slack Markdown block (Feb 2025): type="markdown", text field, 12,000-char limit, supports GFM tables. The script's pipe-delimited table in render_slack_block() is valid GFM and will render correctly. (Source: Slack Developer Docs, 2025, https://docs.slack.dev/reference/block-kit/blocks/markdown-block/)

5. Weekly cadence is appropriate for "research-only" review packets where no capital is auto-moved. Daily or sub-daily cadence is needed only once capital automation is activated (HOTL mode). Current packet is correctly scoped as research-only. (Source: Synvestable, 2026, https://www.synvestable.com/human-in-the-loop.html)

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| scripts/harness/autoresearch_weekly_packet.py | 97 | HITL packet renderer; ranks TSV by DSR, renders Markdown, exits 0 | Complete |
| backend/autoresearch/results.tsv | 2 | Source data; seed_0000 row only (DSR=0.9526) | Functional (seed state) |

### Consensus vs debate

All 2024-2026 sources agree: capital-moving actions require explicit human approval before automation. No debate on weekly cadence for review-only packets in research stage.

### Pitfalls from literature

- Ranking by raw Sharpe instead of DSR introduces selection bias inflation across N trials (Bailey & Lopez de Prado).
- Omitting the approval clause from the rendered artifact removes the human gate trigger -- the clause must be in the rendered output, not just in metadata.
- Premature HOTL graduation (automating capital moves before sufficient validation) is identified as a primary cause of AI initiative failures (42% abandonment rate in 2025 per Synvestable).

### Application to pyfinagent (file:line anchors)

- `scripts/harness/autoresearch_weekly_packet.py:26-29` -- `_PEDER_CLAUSE` constant is correctly present and embedded in rendered output.
- `scripts/harness/autoresearch_weekly_packet.py:40-46` -- `rank_top_n` sorts by DSR descending (`-float(r.get("dsr") or 0.0)`); correct metric per Bailey/Lopez de Prado.
- `scripts/harness/autoresearch_weekly_packet.py:49-65` -- `render_slack_block` emits GFM table; valid for Slack markdown block (Feb 2025 spec).
- `scripts/harness/autoresearch_weekly_packet.py:79-90` -- exit 0 logic: `ok_clause and ok_render and ok_rank`; all three success_criteria gates present.
- `backend/autoresearch/results.tsv:2` -- seed_0000 row; DSR=0.9526, above 0.95 threshold; will rank as top candidate in dry-run.

### Script audit (success_criteria verification)

1. **weekly_slack_post_rendered**: `ok_render = bool(rendered and "Weekly Autoresearch Review" in rendered)` -- line 81. PASS.
2. **peder_approval_required_for_capital_promotion**: `ok_clause = _PEDER_CLAUSE in rendered` -- line 79; `_PEDER_CLAUSE` is appended to every render via line 53. PASS.
3. **top_10_candidates_ranked**: `rank_top_n(rows, n=10)` -- line 75; sorts by DSR desc, returns up to 10. With 1 seed row, `len(top)=1`. `ok_rank = len(top) >= 1 or not rows` -- line 80. PASS (seed-state condition).

Exit 0 path: `all_ok = ok_render and ok_clause and ok_rank` -- line 90. All three True with seed data. `return 0 if all_ok else 1` -- line 92.

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (8 snippet-only + 5 full = 13 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (script + TSV)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/phase-8.5.8-research-brief.md",
  "gate_passed": true
}
```
