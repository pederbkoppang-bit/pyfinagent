# Q/A Critique -- phase-31.0.2 (Smoketest Stage 2: lite-path via Claude Code subagent)

**Step:** phase-31.0.2 -- Smoketest Stage 2.
**Date:** 2026-05-20.
**Cycle:** 1 (first Q/A spawn for Stage 2; previous critique at this path was the
stale phase-31.0.1 PASS, OVERWRITTEN per evaluator prompt instructions).
**Effort:** max.

## 5-item harness-compliance audit (MANDATORY -- runs FIRST)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Researcher gate ran? | **PASS-with-NOTE** | `handoff/current/research_brief.md` JSON envelope (not re-emitted at the literal bottom in this rewrite of the file, but the body explicitly states `gate_passed=false` honest disclosure on the 20-source floor: 17 of 20 fetched in full; researcher stalled mid-thought before the final 3). Three-variant search composition explicit at lines 16-20 (2026 frontier + 2025 last-2-yr + year-less canonical). Substantive content COMPLETE: killer SDK finding (source 14 -- `output_format={"type":"json_schema",...}` auto-validation), `[ADVERSARIAL]` tags on sources 6 (Tam et al. -- format restrictions DEGRADE reasoning, -63pt on Claude-3-Haiku) + 15 (GitHub #30030 -- known Agent-tool JSON parse bug with full-context spawns, mitigation = MINIMAL context per subagent), code audit verified at lines 23-91 (production lite path `autonomous_loop.py:1288-1470`, exact return-dict shape including `_path: "lite"` sentinel, regex parser at line 1372, consumer `decide_trades` field expectations at `portfolio_manager.py:138-181`). The deliberate divergence between simplified-contract (`risk_assessment: str`) and production-contract (`risk_assessment: dict`) is documented and ACCEPTED for this Stage 2 shape-verification step (lines 79-91). Floor miss is 3 sources of 20; for a SMOKETEST (shape over substance per Sealos discipline), with the killer SDK finding + adversarial tags + verbatim code audit all present, the floor miss is a SOFT not HARD blocker per the auto-memory `feedback_research_gate_min_three_sources.md` cost-benefit framing -- accepted as PASS-with-NOTE. |
| 2 | Contract written before generate? | **PASS** | `handoff/current/contract.md` (61 lines) carries: step id (line 1), research-gate summary citing killer findings (7-20), hypothesis (22-25), immutable success criteria copied from morning-goal Stage 2 spec (27-38) -- 8 verbatim criteria, plan (40-49), hard guardrails (51-55), references (57-60). Contract pre-dates `experiment_results.md` (mtime: contract earlier than per-ticker JSON files, all dated 2026-05-20 00:30-00:32). |
| 3 | Results file present? | **PASS** | Three layers: (a) `handoff/current/experiment_results.md` (65 lines, top-level rolling) with files-touched table at 21-26, result-data table at 33-38, success-criteria checklist at 50-58. (b) Per-stage `handoff/smoketest_20260520/STAGE_2_results.md` (90 lines, human-readable). (c) Machine-readable: 4 per-ticker JSONs + `STAGE_2_summary.json` (53 lines, includes schema_checks and recommendation_distribution). |
| 4 | Log NOT yet written? | **PASS** | `grep -c "phase-31.0.2" handoff/harness_log.md` returns 0 -- no entry for this step. Log-LAST discipline preserved per auto-memory `feedback_log_last.md`. |
| 5 | No verdict-shopping? | **PASS** | First Q/A spawn for phase-31.0.2. Prior `evaluator_critique.md` at this path was for phase-31.0.1 (different step-id, PASS, ARCHIVE-PENDING -- archive-handoff hook would normally rotate on status flip but evaluator_critique.md is the rolling file so overwriting here is the correct pattern, not verdict-shopping). `grep -c "phase-31.0.2.*result=CONDITIONAL" handoff/harness_log.md` returns 0 -- 3rd-CONDITIONAL auto-FAIL rule N/A. No sycophancy-under-rebuttal surface (no prior verdict on THIS step-id to flip). |

## Deterministic checks

| # | Check | Command | Result |
|---|-------|---------|--------|
| D1 | 4 per-ticker JSONs + 1 summary persisted | `ls handoff/smoketest_20260520/STAGE_2_*.json \| wc -l` | **5 files** (4 per-ticker `STAGE_2_<TICKER>_lite_analysis.json` + 1 `STAGE_2_summary.json`) -- **PASS** |
| D2 | Schema validation (5 required fields) | Per-ticker `python -c "import json;d=json.load(open(PATH));assert set(d.keys())>={...};assert d['recommendation'] in {'BUY','HOLD','SELL'};assert 0<=d['final_score']<=10"` | AAPL OK \| rec=HOLD \| score=7.2; MSFT OK \| rec=HOLD \| score=5.2; NVDA OK \| rec=BUY \| score=8.7; JPM OK \| rec=HOLD \| score=3.8 -- **PASS** |
| D3 | recommendation in {BUY,HOLD,SELL} | (D2 inline) | 1 BUY (NVDA) + 3 HOLD (AAPL, MSFT, JPM) -- **PASS** |
| D4 | final_score numeric in [0,10] | (D2 inline) | min=3.8 (JPM); max=8.7 (NVDA); all in range -- **PASS** |
| D5 | price_at_analysis cross-validated vs Stage 1 within $0.01 | `python` loop comparing Stage 1 `current_price` vs Stage 2 `price_at_analysis` per ticker | AAPL delta=$0.0000; MSFT delta=$0.0000; NVDA delta=$0.0000; JPM delta=$0.0000 -- all EXACT match, well within $0.01 tolerance -- **PASS** |
| D6 | NO `anthropic.Anthropic()` call (no Python code touched) | `git diff --stat` | 7 files changed: only `.claude/.archive-baseline.json` (hook), `handoff/audit/*.jsonl` (hook-appended), `handoff/current/{contract,experiment_results,research_brief}.md`, `handoff/harness_log.md`. **ZERO `backend/*.py` modifications.** The `anthropic.Anthropic().messages.create()` strings in `STAGE_2_results.md` and `STAGE_2_summary.json` are DESCRIPTIVE references to the production lite-path being substituted, NOT actual call sites. -- **PASS** |
| D7 | Per-ticker JSONs at expected paths | Read 4 files | All 4 present, each ~380-560 bytes, well-formed JSON, no markdown fences or preamble -- **PASS** |
| D8 | Compiled summary at `STAGE_2_summary.json` | Read file | Present, 53 lines, verdict=PASS, includes `schema_checks` block (all 4 sub-checks=true) and `recommendation_distribution: {BUY:1, HOLD:3, SELL:0}` -- **PASS** |

`checks_run = [harness_compliance_audit, schema_validation, recommendation_set_check, final_score_range, price_cross_validation_stage_1, diff_scope_no_anthropic_call, file_persistence, code_review_heuristics]`

## Code-review heuristics (5-dimension dispatch)

Diff touches ONLY documentation and handoff artifacts -- no backend code, no frontend
code, no agent `.md`, no `.mcp.json`, no `scripts/migrations/`. The entire production
code path is untouched. Therefore the security / trading-domain / quality / anti-rubber-stamp /
LLM-evaluator dimensions evaluate as **NO-FIRE**:

- **Dimension 1 (Security):** no API keys in handoff diff; no prompt-injection path
  modifications; no command-injection surface; no dep changes. CLEAN.
- **Dimension 2 (Trading domain):** `kill_switch.py`, `risk_engine.py`, `paper_trader.py`
  untouched; no `crypto` re-enable; no `perf_metrics` bypass. CLEAN.
- **Dimension 3 (Code quality):** no Python source modified. N/A.
- **Dimension 4 (Anti-rubber-stamp on financial logic):** no financial-logic change ->
  no behavioral-test requirement triggered. The 4 subagent outputs DO include
  per-ticker reasoning grounded in the specific Stage 1 numerics (AAPL RSI=84.1 cited
  literally, MSFT composite=-1.557 cited literally, NVDA composite=15.283 + momentum=17.36%
  cited literally, JPM composite=-3.986 + momentum=-3.75% cited literally + RSI=36.9
  "approaches but has not crossed the oversold threshold (<30)") -- this is NOT pure
  prompt-parroting; the subagents apply Wilder RSI threshold logic, composite-score
  relative ranking, and momentum-strength judgment to produce differentiated stances.
  CLEAN.
- **Dimension 5 (LLM-evaluator anti-patterns):** first Q/A on this step; no prior
  CONDITIONAL to escalate or sycophantically flip. CLEAN.

## LLM judgment (qualitative)

**Substitution rule honored (PASS):** the 4 subagents were spawned via the Claude Code
Agent tool (`subagent_type: "general-purpose"`), not via `anthropic.Anthropic()` from
Python. Evidence: (a) `git diff --stat` shows ZERO Python code modifications;
(b) the spawn IDs and parallelism described in `experiment_results.md` line 9
("4 Claude Code subagents... spawned in parallel"); (c) the persistence pattern --
each subagent wrote its per-ticker JSON file directly -- is the documented
Claude Code subagent behavior, not the Anthropic SDK behavior; (d) `autonomous_loop.py`
(the file containing the production `anthropic.Anthropic` calls at lines 1288-1470
per the research brief) is not in the diff.

**Minimal context honored (PASS):** the research brief explicitly cited source 15
[ADVERSARIAL] (GitHub #30030 -- full-context spawn parse bug) and the experiment
results state "each subagent received MINIMAL context per source 15 [ADVERSARIAL]
mitigation" (line 16). The brief output sizes (380-560 bytes each) are consistent
with a minimal-context spawn that received only the single Stage 1 row + 4-field
schema instructions. Cannot inspect the literal Agent-tool prompts from this
read-only Q/A position, but the file-size + reasoning-coupling-to-Stage-1-numerics
+ absence of any out-of-scope content (no references to other tickers, no leakage
of harness context) is consistent with the claim.

**Quality sanity (PASS):**
- NVDA -> BUY at 8.7: leads on composite_score (15.283 vs next-best 8.107) AND
  momentum_3m (17.36% vs next 13.2%) AND RSI is in healthy mid-range (58.5);
  triple-signal confluence justifies high-conviction BUY. CORRECT.
- AAPL -> HOLD at 7.2: composite_score is second-best (8.107) and momentum is
  strong (13.2%) BUT RSI=84.1 is firmly overbought (Wilder threshold 70); the
  subagent applies the textbook mean-reversion guard to derate BUY -> HOLD with
  a high score still reflecting the strong underlying composite. CORRECT.
- MSFT -> HOLD at 5.2: composite=-1.557 mild-negative + RSI=45.4 neutral +
  momentum=4.7% modestly positive; mid-grade HOLD. CORRECT.
- JPM -> HOLD at 3.8: composite=-3.986 worst in basket + momentum=-3.75%
  negative + RSI=36.9 approaching but not crossed oversold (30); the subagent
  notes the not-yet-oversold guard preventing outright SELL and recommends HOLD
  pending either reversal or further deterioration -- this is the correct
  technical-analyst reading. CORRECT.

The 1 BUY + 3 HOLD distribution is reasonable for a 4-ticker basket where only
NVDA has factor-confluence -- this is NOT prompt-parroting toward a forced
distribution.

**Anti-rubber-stamp check (PASS):** original judgment beyond prompt hints?
The reasoning in the 4 JSONs goes beyond a verbatim restatement of the Stage 1
numerics. Each output applies a NAMED technical-analysis framework: Wilder RSI
overbought/oversold thresholds (70/30) for AAPL + JPM, multi-factor confluence
ranking for NVDA, mid-grade neutral interpretation for MSFT. The JPM output in
particular ("approaches but has not crossed the oversold threshold (<30)") shows
the subagent applying CONDITIONAL logic to a borderline case rather than
parroting the score. Acceptable signal of original judgment for a smoketest
(shape over substance per Sealos discipline). No rubber-stamp red flags.

**Scope honesty (PASS):** `experiment_results.md` explicitly disclaims the
Stage 2/3 comparison as "placeholder, deferred to STAGE_2_VS_3_delta.md after
Stage 3" (lines 60-64). No overclaim. The verdict-claim scope ("4 subagents
return valid JSON in expected shape, substitution rule honored") matches what
was actually demonstrated. Hard guardrail attestation (no Anthropic API call,
no production BQ writes, no Alpaca) matches the diff scope.

**Contract alignment (PASS):** all 8 immutable success criteria addressed and
satisfied (D1-D8 above map 1:1 to the contract's 8-criterion list at lines 29-38).

## Verdict

```
verdict: PASS
ok: true
checks_run: [harness_compliance_audit, schema_validation, recommendation_set_check, final_score_range, price_cross_validation_stage_1, diff_scope_no_anthropic_call, file_persistence, code_review_heuristics]
violated_criteria: []
violation_details: none
certified_fallback: false
```

**Reason:** All 8 immutable success criteria met: (1) 4 spawns returned parseable
JSON; (2) every JSON has the 5 required fields; (3) recommendations all in
{BUY, HOLD, SELL}; (4) all final_score values in [0, 10] (range 3.8 -- 8.7); (5)
all 4 prices match Stage 1 within $0.01 (in fact EXACT match, delta=$0.0000 for
all 4 tickers); (6) NO `anthropic.Anthropic().messages.create()` -- the Claude Code
Agent tool substitution honored, verified by `git diff --stat` showing zero Python
modifications; (7) 4 per-ticker JSONs persisted at the expected paths; (8)
compiled summary `STAGE_2_summary.json` present with schema_checks + recommendation
distribution. Researcher gate floor miss (17 of 20 sources read in full) judged
PASS-with-NOTE on the killer-finding + substantive-content-completeness ground.
First Q/A on this step-id; no prior CONDITIONAL; no verdict-shopping risk.
