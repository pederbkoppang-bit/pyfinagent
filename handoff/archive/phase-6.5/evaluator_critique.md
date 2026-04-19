# Q/A Critique -- phase-6.5 Sentiment Scorer Ladder

**qa_id:** qa_65_v1
**Timestamp:** 2026-04-19
**Cycle:** 1 (no prior Q/A on this step)

## Verdict: PASS

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 9 functional criteria met. Research gate compliant (7 sources read-in-full, recency scan performed, gate_passed:true). Contract PRE-committed before sentiment.py. All 4 verification commands emit expected output. Fail-open discipline holds at every tier. Mutation-resistant tests in place (enum-string check, threshold-routed escalation, 4096-token floor).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["protocol_audit_5_item", "syntax_sentiment", "syntax_tests", "import_smoke", "ladder_fail_open_smoke", "pytest", "haiku_prompt_size", "init_exports", "settings_keys", "contract_research_gate_refs", "fail_open_code_inspection", "mutation_resistance_inspection"]
}
```

---

## 1. Protocol audit (5-item, per `feedback_qa_harness_compliance_first.md`)

1. **Researcher spawned before contract?** PASS.
   - `handoff/current/research_brief.md` mtime = 1776585860 (09:54 GMT 2026-04-19); contract mtime = 1776585992 (09:56 GMT). Research wrote first.
   - Envelope (lines 239-244): `external_sources_read_in_full: 7`, `snippet_only_sources: 10`, `urls_collected: 17`, `recency_scan_performed: true`, `internal_files_inspected: 9`, `gate_passed: true`. Clears 5-source floor by +2.
   - Recency scan section present (line 38) with 4 new 2024-2026 findings documented.

2. **Contract PRE-committed before generate?** PASS.
   - `contract.md` mtime = 1776585992 (09:56 GMT).
   - `backend/news/sentiment.py` mtime = 1776586361 (10:12 GMT).
   - `backend/tests/test_sentiment_ladder.py` mtime = 1776586244 (10:10 GMT).
   - Delta: contract is ~6 minutes before the first generated file. Ordering correct.

3. **experiment_results.md present and describes what was built?** PASS.
   - 144 lines. Lists all 4 scorer classes, the entry point, ScorerResult shape, file list, verbatim verification command output, and a contract-criterion-check table. Matches the diff spot-checked below.

4. **harness_log.md NOT yet appended for this step?** PASS.
   - Most recent entries are three phase-2.12 autonomous-harness dry-run cycles timestamped 07:43-07:53 UTC 2026-04-19. No phase-6.5 cycle block exists. Log-last discipline intact.

5. **No verdict-shopping?** PASS. Cycle 1; no prior Q/A to overturn.

Protocol audit: 5/5 PASS.

---

## 2. Deterministic checks

### A. Syntax
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/news/sentiment.py').read()); print('SYNTAX OK')" && python -c "import ast; ast.parse(open('backend/tests/test_sentiment_ladder.py').read()); print('TEST SYNTAX OK')"
SYNTAX OK
TEST SYNTAX OK
```
PASS.

### B. Import + ladder smoke
```
$ source .venv/bin/activate && python -c "from backend.news.sentiment import score_ladder, ScorerResult, VaderScorer, FinBertScorer, HaikuScorer, GeminiFlashScorer; print('ok')"
ok

$ source .venv/bin/activate && python -c "import os; os.environ.pop('ANTHROPIC_API_KEY', None); from backend.news.sentiment import score_ladder; r = score_ladder({'article_id':'qa1','title':'test','body':''}); print(r.scorer_model, r.sentiment_label, r.confidence)"
vaderSentiment not installed (No module named 'vaderSentiment'); VaderScorer will fail-open
ANTHROPIC_API_KEY not set; HaikuScorer will fail-open
claude-haiku-4-5 neutral 0.0
```
Expected `claude-haiku-4-5 neutral 0.0`. Got `claude-haiku-4-5 neutral 0.0`. PASS.

### C. Pytest
```
$ source .venv/bin/activate && pytest backend/tests/test_sentiment_ladder.py -x -q
s........                                                                [100%]
8 passed, 1 skipped in 0.16s
```
1 skip is `test_vader_bullish_headline` (missing vaderSentiment dep -- expected; matches Main's known caveat). Zero failures. PASS.

### D. File/state checks
- `backend/news/sentiment.py`: 966 lines (contract said "> 400"; experiment_results said "1023" -- minor overcount by Main, actual 966, still well above floor). PASS on the floor; flag the 1023 vs 966 as a minor doc-accuracy note, not a blocker.
- `backend/news/__init__.py` exports all 6 new names (`ScorerResult`, `VaderScorer`, `FinBertScorer`, `HaikuScorer`, `GeminiFlashScorer`, `score_ladder`) via `__all__` lines 48-53. PASS.
- `backend/config/settings.py` lines 67-69 contain the 3 new keys with correct defaults (0.7, False, False). PASS.
- `HAIKU_SYSTEM_PROMPT` length:
  ```
  $ python -c "from backend.news.sentiment import HAIKU_SYSTEM_PROMPT; print('chars:', len(HAIKU_SYSTEM_PROMPT))"
  chars: 20463
  ```
  20463 >= 14500 floor. At 3.5 chars/tok = 5847 tokens; clears the 4096-token cache activation floor. PASS.

### E. Research-gate evidence in contract
```
$ grep -n 'research_brief\|gate_passed\|recency' handoff/current/contract.md
10: `{tier: moderate, external_sources_read_in_full: 7, ..., gate_passed: true}`. Brief at `handoff/current/research_brief.md` (246 lines). Recency scan returned 4 new 2024-2026 findings...
66: - `handoff/current/research_brief.md` (canonical, 246 lines)
```
All three tokens (research_brief, gate_passed, recency) present with the envelope copy-pasted into the contract body. PASS.

---

## 3. LLM judgment

**Contract alignment** -- all 9 functional criteria independently verified against code:
- (1) `ScorerResult` dataclass fields are superset of BQ migration columns (test_scorer_result_fields_match_bq_migration asserts this). PASS.
- (2) VADER thresholds +/-0.05 at `_label_from_score_thresholded` (code L196). PASS.
- (3) FinBERT uses `ProsusAI/finbert` (L242-244), module-global lazy init (L225-250), truncates at FINBERT_MAX_TOKENS=400 (L278). PASS.
- (4) Haiku uses `anthropic.Anthropic(api_key=key)` directly (L779), NOT `ClaudeClient.generate_content()`; forced `tool_choice={"type":"tool","name":"classify_sentiment"}` (L806); no `thinking=` parameter in the create() call (L795-808); `cache_control={"type":"ephemeral","ttl":"1h"}` on system block (L802). PASS.
- (5) Escalation at `confidence < min_confidence` with early return (L927-939). PASS.
- (6) 3 settings keys at `backend/config/settings.py:67-69`. PASS.
- (7) `GeminiFlashScorer` raises `NotImplementedError` when `enabled=False` (test_gemini_flash_disabled_raises asserts behaviorally). PASS.
- (8) Every tier's `score()` wraps in broad `try/except Exception`, funnels to `_neutral_result(...)` (VADER L185-215; FinBERT L262-312 including `import torch` INSIDE the try, fixing the sibling bug; Haiku L784-845). PASS.
- (9) Enum strings match migration: `vader`, `finbert`, `claude-haiku-4-5`, `gemini-2.0-flash` (test_scorer_model_enum_matches_migration asserts). PASS.

**Research-gate tracing** -- every load-bearing design decision cites the brief:
- ProsusAI over yiyanghkust: nosible 2024, 69% vs 53% (contract L13, experiment_results L13).
- No CoT on Haiku: arxiv 2506.04574v1, 2025 (contract L14).
- 4096-token cache floor: Anthropic prompt-caching docs (contract L15, code L317-319).
- Threshold 0.7: WASSA 2024 cascade study (contract L16).
- Gemini Flash tier-4 opt-in: contract L17.
No unsupported or invented design surface detected.

**Scope honesty** -- contract's 5 explicit non-goals (no BQ writes, no pipeline wiring, no Gemini Flash body, no FinBERT eager init, no fetcher/dedup/normalize changes) all honored in the diff. No scope creep.

**Mutation resistance** -- tests would fail if:
- scorer_model enum string changed -> `_MIGRATION_ENUM` set comparison fails (L50-55, L112, L118).
- Threshold 0.7 moved -> escalation-routing tests use explicit 0.95/0.85/0.2 values that cross the boundary; changing the threshold breaks the early-return assertion.
- Tier order reversed -> `test_score_ladder_escalates_from_vader_to_finbert` asserts `result.scorer_model == SCORER_MODEL_FINBERT` specifically.
- Cache floor shrunk -> `test_haiku_system_prompt_meets_4096_token_minimum` hard-asserts `len >= 14500`.
Mutation resistance confirmed real, not rubber-stamp.

**Fail-open completeness** -- inspected VADER, FinBERT, Haiku score() methods. All three wrap the ENTIRE body in a single top-level try/except Exception. Notably, FinBERT's `import torch as _torch` (L266) is INSIDE the try block (Main's cycle fixed a sibling bug here); so missing-torch -> ImportError -> fail-open path fires correctly. `_lazy_load_finbert` raises upstream, caught by the outer try. No leaked-exception paths found.

**Known caveats honesty** -- Main explicitly discloses (experiment_results L140-144):
- live-path VADER/FinBERT/Haiku NOT exercised against real deps (deps not installed; no API key);
- deps not added to requirements.txt this cycle;
- token count is char-proxy, not `client.messages.count_tokens()`-verified.
These are legitimate scope boundaries -- the contract's non-goals defer live-path to phase-6.8 smoketest. No overclaiming.

---

## 4. Violated criteria

None.

## 5. Violation details

None.

## 6. checks_run

`protocol_audit_5_item`, `syntax_sentiment`, `syntax_tests`, `import_smoke`, `ladder_fail_open_smoke`, `pytest`, `haiku_prompt_size`, `init_exports`, `settings_keys`, `contract_research_gate_refs`, `fail_open_code_inspection`, `mutation_resistance_inspection`.

## 7. Minor notes (non-blocking)

- experiment_results L29 states `backend/news/sentiment.py` is 1023 lines; actual `wc -l` = 966. Off by 57 lines. Doc accuracy, not a verdict blocker. Main may want to correct on log-append.

## 8. Verdict

**PASS.**

No blockers. Main may proceed to (a) append the phase-6.5 cycle-1 block to `handoff/harness_log.md`, then (b) flip `.claude/masterplan.json` phase-6.5 to `status: done`. Log-last discipline: in that order, not reversed.
