# Evaluator Critique — phase-27.3

Q/A subagent: `qa` (aafeb6d3263941373), 2026-05-16, single pass, no verdict-shopping.
Evidence: `handoff/current/contract.md` (27.3), `handoff/current/experiment_results.md` (27.3), `backend/services/autonomous_loop.py` (factory @ 854-872, gemini lite @ 1057-1217, call sites @ 769 and 810).

## Harness-compliance audit (5 items)

| # | Item | Verdict | Note |
|---|---|---|---|
| 1 | Researcher spawned BEFORE contract | PASS | research_brief.md §C2 (lines 254-325) cited in contract |
| 2 | contract.md pre-Generate, step-27.3 focused | PASS | Immutable criteria copied verbatim from masterplan 27.3 |
| 3 | experiment_results.md present + verbatim cmd output | PASS | exit=0 reproduced |
| 4 | log-last discipline | PASS (deferred) | 27.3 entry will append after this verdict |
| 5 | No verdict-shopping | N/A | Cycle 1; no prior verdict |

## Deterministic checks

| Check | Result |
|---|---|
| Syntax (`autonomous_loop.py`) | OK |
| Masterplan 27.3 verification cmd | EXIT_CODE=0, stdout=`PASS` |
| Extended dispatch matrix (7 cases incl `None`, `''`, `'   '`, `GPT-4o`, case-mix) | all 7 PASS |
| AST schema parity, top-level dict | identical 9 keys |
| AST schema parity, `full_report` sub-dict | identical `{analysis, market_data, source}` |
| AST schema parity, `market_data` sub-dict | identical 8 keys |
| Call-site audit (factory used at :769 + :810) | confirmed |
| Direct `_run_claude_analysis(...)` calls outside factory in `backend/` | none in non-test code |
| Defense-in-depth ValueError gates preserved (claude :917, gemini :1104) | YES |
| Schema-parity confirmed via source (no double-spent API call) | ✓ |

## LLM-judgment

- **Schema parity**: all three nested dict layers match exactly. `_persist_analysis` consumes both without branching. `risk_assessment` constructed identically (incl. backward-compat `reason` alias for `bq.save_report`).
- **Regex risk**: Gemini path uses `response_mime_type: "application/json"` (forces structured JSON output) + `re.DOTALL` on trader regex — more robust than the Claude path. Brief's P2 classification correct.
- **Anti-rubber-stamp**: factory has zero branches that can return None or raise; `(model_name or "").strip().lower()` guards None/empty/whitespace/case.
- **Scope honesty**: all 4 plan steps satisfied. Defense-in-depth gates KEPT rather than removed (deviation from plan-step-4 wording but strictly safer; disclosed in experiment_results).
- **Live-probe authenticity**: dict values match `_LITE_RISK_DEFAULT` exactly (regex-miss fallback as disclosed); `price_at_analysis: 300.23` in plausible AAPL range; cost matches hardcoded constant. Authentic.

## Code-review heuristics

No BLOCK/WARN. No secrets, no kill-switch path touched, no perf-metrics inline, no broad-except added, no command injection, no Unicode in loggers. ~160-line duplication accepted in contract anti-patterns.

## Verdict (machine-readable)

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "dispatch_matrix_extended",
    "ast_schema_parity_3_layer",
    "call_site_audit",
    "defense_gate_present",
    "research_gate_compliance",
    "code_review_heuristics"
  ]
}
```

Live_check (cycle without 'Both full and lite paths failed') deferred to phase-27.5 per contract.
