---
step: phase-25.D9.1
cycle: 104
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.D9.1

## What was built/changed

Closed the caller-side gap from 25.D9 (cycle 81). The Files API upload
+ document-block routing existed in `llm_client.py` (lines 1092-1247);
the 11 enrichment `run_*_agent` methods still inlined the full skill
markdown instead of threading the pre-uploaded file_id.

### North-star alignment

Directly cuts the cost denominator in Net System Alpha. Per the
research brief, skill markdowns are 5K-15K tokens each; the
document-block path passes only an ~8-token file_id reference per call.
98-99.5% reduction per Claude-mode enrichment call. Compounds with:
- 25.B9 (system-prompt cache, cycle 80): same prompts now 0.1x.
- 25.C9 + 25.C9.1 (batch path, cycles 84 + 103): batch requests now
  carry tiny doc-block refs.

### Files changed

| File | Action |
|------|--------|
| `backend/agents/orchestrator.py` | Added `_skill_gen_config(skill_stem) -> dict \| None` helper; wired 11 enrichment call sites to pass `generation_config=self._skill_gen_config("<stem>")` |
| `tests/verify_phase_25_D9_1.py` | NEW (5 claims) |
| `.claude/masterplan.json` | NEW 25.D9.1 step entry (post-25.D9) |

### Wired call sites (skill stems)

| Method | Skill stem | Display name |
|--------|------------|--------------|
| `run_insider_agent` | `insider_agent` | "Insider" |
| `run_options_agent` | `options_agent` | "Options" |
| `run_social_sentiment_agent` | `social_sentiment_agent` | "Social Sentiment" |
| `run_patent_agent` | `patent_agent` | "Patent" |
| `run_earnings_tone_agent` | `earnings_tone_agent` | "Earnings Tone" |
| `run_alt_data_agent` | `alt_data_agent` | "Alt Data" |
| `run_sector_analysis_agent` | `sector_agent` | "Sector" |
| `run_nlp_sentiment_agent` | `nlp_sentiment_agent` | "NLP Sentiment" |
| `run_anomaly_agent` | `anomaly_agent` | "Anomaly" |
| `run_scenario_agent` | `scenario_agent` | "Scenario" |
| `run_quant_model_agent` | `quant_model_agent` | "Quant Model" |

Total: 11 wires (matches `grep -c` output).

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D9_1.py

=== phase-25.D9.1 verification ===

[PASS] 1. orchestrator_has_skill_gen_config_helper
        -> _skill_gen_config method present
[PASS] 2. enrichment_agents_pass_generation_config_with_skill_file_id
        -> call_sites=11 (expected >=11)
[PASS] 3. helper_returns_none_when_skill_file_ids_empty_gemini_fallback
        -> got None
[PASS] 4. helper_returns_skill_file_id_dict_for_mapped_stem
        -> got {'skill_file_id': 'file_xyz_123'}
[PASS] 5. helper_returns_none_for_unmapped_stem_no_keyerror
        -> got None

ALL 5 CLAIMS PASS
```

AST clean on `backend/agents/orchestrator.py`.

## Success criteria -> evidence

1. `orchestrator_has_skill_gen_config_helper` -- Claim 1 PASS (AST match).
2. `enrichment_agents_pass_generation_config_with_skill_file_id` -- Claim 2 PASS (11 call sites; `grep -c` confirms).
3. `helper_returns_none_when_skill_file_ids_empty_gemini_fallback` -- Claims 3 + 5 PASS (empty dict + missing stem both return None; no KeyError).

## North-star calculus

Per Anthropic Files API docs: each Claude-mode enrichment call drops
from 5K-15K skill tokens to ~8-token file_id reference (98-99.5%
reduction). Multiplied by 11 enrichment agents per ticker, a single
full-pipeline Claude run drops the per-call skill input from
~55K-165K tokens to ~88 tokens. At Sonnet 4.6 input $3.00/MTok, this
is ~$0.15-$0.50 savings per ticker per cycle BEFORE prompt-cache
compounding.

Combined with the 25.B9 1h prompt cache and the (incoming via 25.C9.2)
batch path, the operator's `profit_per_llm_dollar` metric (25.Q
efficiency_snapshots) should see materially improved values once a
multi-ticker Claude backtest exercises the wired path.

## Out-of-scope / deferred

- `cache_control` on the document block (Finout 2026 reports the 1h
  cache-hit path at 0.1x for skill bodies above 1024 tokens). Deferred
  to 25.D9.2 -- one-dict-update follow-up. The current cycle's success
  criteria do not require this.
- Verifying the file_id ref ACTUALLY flows into the request payload
  end-to-end (vs being constructed correctly at the helper level): the
  25.D9 verifier (cycle 81) already covered `llm_client.py:1220-1247`
  routing; this cycle proves the *caller-side* wiring.

## References

- `handoff/current/research_brief.md` (5 sources fetched in full, gate_passed=true)
- `backend/agents/orchestrator.py:_skill_gen_config` + 11 call sites at lines 839/846/853/860/867/883/890/897/904/911/945
- `backend/agents/llm_client.py:1092-1247` (upload + skill_file_id routing, 25.D9)
- `.claude/masterplan.json::25.D9.1`
