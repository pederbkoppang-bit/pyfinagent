---
step: 25.C
slug: layer1-skill-outputs-in-drawer
tier: moderate
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.C: Surface Layer-1 28-skill outputs in drawer

> Tier=moderate. Main authored from inspection of orchestrator.py's
> 11-skill enrichment block + the post-25.D drawer layout.

---

## Three-variant search queries

1. **Current-year frontier**: `multi-agent skill output drawer 2026 transparency`
2. **Last-2-year window**: `glass-box LLM agent output visualization 2025`
3. **Year-less canonical**: `progressive disclosure agent reasoning tree`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Anthropic "glass-box agents" | priors | Surface every agent I/O so operators audit reasoning |
| TradingAgents (Xiao 2024) | priors | Layer-1 enrichment outputs go in their own tier above debate |
| orchestrator.py:1385-1397 | this cycle | The 11 Layer-1 skill outputs live in `analysis[<skill>]` with `{signal, summary, analysis}` shape -- ONLY in full pipeline (lite mode skips them) |
| signal_attribution.py | post-25.D | Currently extracts Analyst / Debate / Quant / SignalStack / Trader / RiskJudge -- no layer-1 surface |

## Recency scan

No paradigm shift in glass-box multi-agent drawer design 2024-2026.

## Layer-1 skills (11 visible enrichment agents)

`insider`, `options`, `social_sentiment`, `patent`, `earnings_tone`,
`alt_data`, `sector`/`sector_analysis`, `nlp_sentiment`, `anomaly`,
`scenario`, `quant_model`.

Each has `{signal: "BUY"|"SELL"|"HOLD"|"NEUTRAL"|"N/A", summary: str, analysis?: str}`.

## Design

1. **`extract_layer1_signals(analysis: dict) -> list[dict]`** in
   `backend/services/signal_attribution.py`:
   - Iterate the 11 canonical skill keys.
   - For each present + non-empty key, emit `{agent: <Capitalized name>, role: "skill_output", rationale: <summary>, weight: <signal -> 0-1 weight>}`.
   - Signal -> weight mapping: BUY/SELL=1.0, HOLD/NEUTRAL=0.5, N/A/ERROR=0.0.
   - Skip entries with empty summary or N/A signal (don't pad with noise).
2. **`extract_all_signals` extension** -- INSERT layer-1 signals BEFORE Analyst.
3. **`group_signals_for_drawer` extension** -- add `layer1_skills: list[dict]` bucket; route `role=="skill_output"` entries there.
4. **Gate** -- the function naturally returns [] when no skill keys are present
   (which is the lite-mode case). For an explicit gate, accept a `lite_mode=False`
   kwarg that short-circuits to [] when True (defense in depth).
5. **Frontend**:
   - `frontend/src/lib/types.ts` -- add `layer1_skills?: Signal[]` to Rationale.tree.
   - `frontend/src/components/AgentRationaleDrawer.tsx` -- add a Layer component
     for "Layer-1 Skills" between TotalWeightSummary and Analyst.

## Files to modify

| File | Change |
|------|--------|
| `backend/services/signal_attribution.py` | NEW `extract_layer1_signals` + wire into extract_all_signals + group_signals_for_drawer |
| `frontend/src/lib/types.ts` -- (if Rationale interface lives there; otherwise inline) | layer1_skills optional field |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Render layer1_skills before Analyst |
| `tests/verify_phase_25_C.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal: orchestrator.py:1385-1397 (where layer-1 outputs are assembled)
- [x] Internal: signal_attribution.py + drawer rendering

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=moderate; 11 skills, signal->weight mapping is conservative (BUY/SELL=1.0, HOLD=0.5)."
}
```
