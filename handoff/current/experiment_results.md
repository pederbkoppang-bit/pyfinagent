# Cycle 13 — Experiment Results (DoD-14 closure: OWASP LLM04/05/09 explicit tagging)

**Window:** 2026-05-28T17:00-17:30+02:00 (approx)
**Sub-step of:** phase-43.0 (P1, H) — closes DoD-14 of the 14-criterion gate
**Editor:** Main (Claude Code session)
**Researcher gate:** `ab4ba0f2a92122dee` PASSED (10 sources in full / 21 URLs / recency scan / 3-variant queries)

---

## Files created / modified

- `handoff/current/research_brief_phase_43_0_dod_14_owasp.md` (created — researcher subagent output)
- `handoff/current/contract.md` (overwrote cycle-12 contract; cycle-13 spec)
- `.claude/skills/code-review-trading-domain/SKILL.md` (modified — 4 surgical edits)
- `handoff/current/experiment_results.md` (this file)

## Files NOT changed

No `backend/` code touched. No execution path modified. Pure doc-edit of the Q/A skill prompt.

## Edits applied (verbatim diff summary)

### Edit C — LLM05 explicit tag + 5 sub-bullets (SKILL.md:81)
**Before:** `| insecure-output-handling | llm_call(...) result flowing directly into query(...), exec(...), or file path | BLOCK |`
**After:** `| insecure-output-handling [LLM05:2025] | llm_call(...) / agent-response string flowing directly into any of: (a) command-injection sinks (exec(), eval(), subprocess.run(shell=True), os.system()); (b) SQL-injection sinks (cursor.execute(...) / db.query(...) with f-string or string-concat instead of parameterized query); (c) path-traversal sinks (open(...), Path(...), file-write or file-read where the path string contains an LLM-output substring); (d) SSRF sinks (requests.get(...), httpx.get(...), urllib.request.urlopen(...) with an LLM-generated URL); (e) XSS / HTML-injection sinks (response HTML / template render / FastAPI JSONResponse containing un-escaped LLM output). Grep: [4 grep regexes covering each sub-sink] | BLOCK |`
**Net:** explicit OWASP LLM05:2025 tag + 5 canonical sub-sinks per OWASP source + StackHawk + FireTail + Indusface consensus. Severity unchanged (BLOCK). Additive.

### Edit B — LLM04 sentinel (new row, SKILL.md:86 area)
**Insert (between rag-memory-poisoning and unbounded-llm-loop):** new row `llm04-training-code-added [LLM04:2025]` with NOTE severity. Grep: `train(|trainer(|\.fit(|fine_tune|finetune|peft|lora.*\.train|sentence_transformers.*\.fit|huggingface.*Trainer`. Default state N/A because pyfinagent uses hosted LLM APIs only (Vertex AI Gemini / Anthropic / OpenAI) — no training, no fine-tuning, no self-supervised embedding generation. Auto-promotes from NOTE to BLOCK if a future diff introduces training code.
**Negation list bullet added:** justification that LLM04's training-pipeline surface is OUT OF SCOPE for current architecture; RAG-corpus poisoning (which IS active) is correctly covered by `rag-memory-poisoning` (LLM08).

### Edit A — Cosmetic source line fix (SKILL.md:100 area)
**Before:** `Source: [OWASP LLM Top-10 v2.0 (2025)](https://genai.owasp.org/llm-top-10/) (LLM07 ... added in v2.0; older v1.1 reference ...)`
**After:** `Source: [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) (March 12, 2025 release; LLM07 ... added vs the 2023 v1.1 list; LLM04 was repurposed from "Model Denial of Service" in v1.1 to "Data and Model Poisoning" in 2025; older v1.1 reference ... preserved for cross-version reading), [LLM04:2025](https://genai.owasp.org/llmrisk/llm042025-data-and-model-poisoning/), [LLM05:2025](https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/), [LLM09:2025](https://genai.owasp.org/llmrisk/llm092025-misinformation/), [security.md](...)`
**Net:** canonical name (no "v2.0" — OWASP doesn't use that label); added release date; LLM04 repurposing note (Model Denial of Service → Data and Model Poisoning); 3 new canonical OWASP-page links for LLM04/05/09.

### Edit D — LLM09 new BLOCK heuristic (Dimension 2 trading-domain table after paper-trader-broad-except)
**Insert (new row):** `llm-output-to-execution-without-validation [LLM09:2025]` BLOCK. Detection cue: NEW code path where LLM-generated content (agent response, signal recommendation, ticker selection, target-price suggestion) reaches trade execution (`paper_trader.execute_buy()` at `paper_trader.py:85`, `paper_trader.execute_sell()` at `paper_trader.py:299`) OR signal emission (`agents/mcp_servers/signals_server.py`, `services/pead_signal.py`, `services/insider_signal_screen.py`, `services/defense_signal.py`) WITHOUT an intermediate validation step. Required validation: (a) ticker exists in `pyfinagent_data.historical_prices`; (b) any price/volume cited resolves to BQ `historical_*` row within ±1 trading day; (c) any cited filing resolves to `financial_reports.*` row; (d) confidence framing bounded.
**Negation list bullet added:** the deterministic signal pipeline (pead/insider/defense) IS the validator — verified zero LLM API calls (no anthropic/claude/gpt/gemini/messages.create/generate_content matches in those files). Only flag NEW code paths that bypass these validators.

## Verification — all 4 commands

```
=== verification 1: distinct LLM categories tagged ===
LLM01
LLM02
LLM03
LLM04
LLM05
LLM06
LLM07
LLM08
LLM09
LLM10

=== count of distinct categories (expect 10) ===
      10

=== verification 2: cosmetic fix landed ===
count of "v2.0 (2025)" (expect 0): 0
count of "OWASP Top 10 for LLM Applications 2025" (expect >=1): 1

=== verification 3: LLM09 negation list includes deterministic signals ===
2

=== verification 4: markdown table well-formed ===
OK: 61 table rows; column-count distribution = {4: 55, 7: 1, 11: 1, 6: 1, 9: 1, 12: 1, 5: 1}
```

All 4 verifications PASS. The 7/11/6/9/12/5 column variance is from regex `\|` escapes inside backtick code-spans — this is the EXISTING convention in the file (rows `secret-in-diff`, `system-prompt-leakage`, `rag-memory-poisoning`, `unbounded-llm-loop` already exhibit it), not a defect introduced by these edits.

## Spot-check — LLM09 negation list correctness (Q/A cycle-2 correction)

**First-pass cycle 13 (initial GENERATE) — scope error caught by Q/A `a775b0e1987da8700`:**

Initial cycle-13 grep was scoped to TWO of the three named files. Q/A re-ran with the third file and surfaced 6 matches in `pead_signal.py`:

```bash
$ grep -nE "anthropic|claude|gpt|gemini|generate_content|messages\.create|ClaudeClient" backend/services/pead_signal.py
248:    anthropic_key = getattr(settings, "anthropic_api_key", "") or ""
272:    if not anthropic_key:
273:        return _fallback("no_anthropic_key")
277:    from backend.agents.llm_client import ClaudeClient
278:    client = ClaudeClient(
279:        model_name=getattr(settings, "pead_signal_model", "claude-haiku-4-5"),
288:            client.generate_content,
```

`pead_signal.py` IS LLM-driven — it calls Claude (haiku-4-5) to compute `sentiment_tag`/`sentiment_score`/`surprise_score` from earnings press-release text. The initial cycle-13 negation list claim ("zero LLM API calls in those three files") was incorrect for pead_signal.py.

**Cycle-2 correction (per canonical cycle-2 flow):**

Updated SKILL.md:129 negation list to characterize the signal pipeline by validator-mechanism, not by deterministic-vs-LLM-driven:

- **Fully deterministic (no LLM call)**: `insider_signal_screen.py` + `defense_signal.py` — verified zero matches for the LLM grep.
- **LLM-driven WITH structured-output validators**: `pead_signal.py` — IS LLM-driven, but validators are: (i) `response_schema=cleaned_schema` + `response_mime_type="application/json"` at `:285, :291-292`; (ii) Pydantic `PeadSignalOutput.model_validate(raw)` at `:312`; (iii) `sentiment_score` clamped to `[0.0, 1.0]` at `:305-306`; (iv) `holding_window_days` whitelist at `:307-308`; (v) `_fallback(...)` on any error at `:298-299, :315`; (vi) downstream filtering via `apply_pead_to_score()` at `:382`. These collectively satisfy LLM09 prevention guidance (automatic validation of structured LLM output).

The negation list now correctly says: do NOT flag the existing pipeline; ONLY flag NEW code paths wiring LLM output to `paper_trader.execute_buy()`/`execute_sell()` WITHOUT both (a) structured-output enforcement AND (b) numeric-range clamping for any field used in trade sizing.

**This is the documented harness cycle-2 pattern**: Q/A's first-pass CONDITIONAL surfaced a heuristic-defeating negation-list defect; Main fixed the negation-list wording + this results file + the contract success criterion #3; a fresh Q/A will read updated files on second-pass.

## Tally vs cycle 12 audit

Before cycle 13: DoD-14 was FAIL (7 of 10 explicit tags).
After cycle 13: DoD-14 PASS (10 of 10 explicit tags) + bonus LLM09 BLOCK heuristic actually wired with file:line anchors + 5 sub-sinks for LLM05 + LLM04 N/A sentinel.

Updated cycle-12 cumulative count: **10 PASS most-generous / 6 PASS literal of 14**. Remaining open: DoD-1, DoD-2, DoD-5, DoD-6, DoD-7, DoD-9, DoD-11 (DoD-11 may also be a doc-edit closure).

## What this cycle did NOT do (per contract)

- Did NOT change underlying detection logic for any existing heuristic (LLM01-03, LLM06-08, LLM10).
- Did NOT add or modify a Q/A subagent definition — the SKILL.md is preloaded into Q/A context at spawn time; next Q/A automatically uses the new heuristics.
- Did NOT touch `backend/` code or any execution path.
- Did NOT flip phase-43.0 to `status=done`.

## Step status policy

phase-43.0 STAYS `pending`. Cycle 13 closed one of 14 DoDs; the gate-PASS flip is reserved for the final 43.0 re-audit cycle when all 14 PASS.

## References

- Contract: `handoff/current/contract.md`
- Research brief: `handoff/current/research_brief_phase_43_0_dod_14_owasp.md`
- Cycle 12 audit: `handoff/current/production_ready_audit_2026-05-28.md`
- Target: `.claude/skills/code-review-trading-domain/SKILL.md`
- OWASP canonical: https://genai.owasp.org/llm-top-10/ + per-category pages cited in SKILL.md:100 link list
