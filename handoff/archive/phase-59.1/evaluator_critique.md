# Evaluator Critique — Step 59.1 (EVALUATE)

**Step:** 59.1 — Fable 5 model adoption (both layers, quality-first)
**Date:** 2026-06-11
**Agent:** Q/A (merged qa-evaluator + harness-verifier), FIRST spawn for 59.1
**Verdict:** **PASS**
**Evaluating snapshot:** OLD (this session's qa runs the pre-59.1 Opus-4.8 roster — protocol-identical; the Fable pins it authored take effect NEXT session). Separation-of-duties satisfied by the operator's 8 in-session pre-approvals (2026-06-11).

---

## 0. Harness-compliance audit (5 items — all PASS)

1. **Researcher gate** — `handoff/current/research_brief.md` is the 59.1 brief: envelope `gate_passed: true`, `external_sources_read_in_full: 7` (5 official docs + 2 vendor/industry), `recency_scan_performed: true`, 17 URLs collected. Cited per-claim. PASS.
2. **Contract pre-commit** — `contract.md` is for 59.1; 4 immutable criteria copied VERBATIM (diffed against `.claude/masterplan.json` step 59.1 `verification.success_criteria` — identical). mtime ordering correct: research_brief (…682) < contract (…790) < all 7 code edits (researcher.md …820, qa.md …827, model_tiers …866, ticket …880, CLAUDE.md …913, tests …960/…973) < live_check (…109) < experiment_results (…137). Contract precedes the code. PASS.
3. **Results present** — `experiment_results.md` present with file list, verbatim verification output, and honest limitations. PASS.
4. **Log-last / status** — `handoff/harness_log.md` has NO 59.1 cycle (last is Cycle 48 = phase-57.1). masterplan 59.1 `status: "pending"`. Correct order (log + flip come after this PASS). PASS.
5. **First spawn / no verdict-shopping** — no `handoff/archive/phase-59.1/` dir, no prior 59.1 critique. First spawn; nothing to shop. PASS.

`retry_count: 0`, `max_retries: 3` → `certified_fallback: false`.

---

## 1. Deterministic checks (cannot hallucinate)

### Immutable verification command
```
$ source .venv/bin/activate && python -m pytest backend/tests -k 'fable or model_tiers or phase_59' -q
7 passed, 773 deselected, 1 warning in 2.73s
$ (combined cmd including `test -f handoff/current/live_check_59.1.md`) -> exit=0
```

### Full suite (false-green guard — the researcher proved the `-k` net collects only `model_tiers`-substring tests and would NOT exercise the one breaking test `test_agent_map_live_model`)
```
$ python -m pytest backend/tests -q
762 passed, 12 skipped, 6 xfailed, 1 warning in 72.99s
full-suite-exit=0
```
Matches the researcher's predicted count exactly. The breaking test `test_agent_map_live_model.py::test_endpoint_injects_live_model_field` was correctly updated to `claude-fable-5` — out-of-net breakage caught only by the full run, exactly the false-green risk the researcher flagged.

### Pin assertions (live `resolve_model` / effort path — `python -c`)
- `resolve_model('mas_main') == 'claude-fable-5'` ✓
- `resolve_model('autoresearch_strategic') == 'claude-fable-5'` ✓
- `resolve_model('mas_qa') == 'claude-opus-4-8'` ✓ (UNCHANGED — criterion 2 cost discipline)
- `resolve_model('mas_communication')`, `resolve_model('mas_research')` → `claude-sonnet-4-6` ✓
- `model_supports_effort('claude-fable-5') is True` ✓
- `resolve_effort_by_model('claude-fable-5') == 'xhigh'` ✓ (closes the silent effort-drop trap at `llm_client.py:1481`)

### Whole-`_BUILD_TIER` fable sweep (anti-rubber-stamp angle a)
```
ALL fable-pinned build-tier roles: ['autoresearch_strategic', 'mas_main']
gemini-locked roles: ['gemini_deep_think', 'gemini_enrichment', 'layer1_swappable']
```
**Only** the two allowed rare-event roles are on Fable. No metered/per-ticker role accidentally repinned. Gemini roles untouched.

### Frontmatter (grep batch)
- `researcher.md`: `model: fable` (L5), `maxTurns: 40` (L6), `effort: max` (L23), `verify_qa_roster_live` (L20); comment block records $10/$50, June-23 USAGE CREDITS superseding phase-29.2 flat-fee, 2026-06-11 operator pre-approval, v2.1.170 floor / 2.1.172 local, restart caveat. ✓
- `qa.md`: `model: fable` (L5), `maxTurns: 30` (L6), `effort: max` (L24), `verify_qa_roster_live` (L20); same economics + restart annotations + maxTurns 12→30 rationale (FIVE observed mid-evaluation stalls). ✓

### CLAUDE.md (additive — criterion 3)
- Contains `claude-fable-5`, `$10/$50`, June-23 Max-credit change explicitly "SUPERSEDES the flat-fee rationale below", classifier-fallback note, EFFORT_SUPPORTED_MODELS/MODEL_EFFORT_FALLBACK requirement, `/model fable`. ✓
- Opus 4.8 history PRESERVED intact: phase-29.2 paragraph + literal "Introducing Claude Opus 4.8" string at L57 both survive. Additive (new bullet inserted above the historical one). ✓

### Ticket map (`ticket_queue_processor.py:171-173`)
```
"main": "claude-fable-5",        # Fable 5 (complex reasoning; rare-event)
"q-and-a": "claude-fable-5",     # Fable 5 (accuracy required; rare-event)
"research": "claude-sonnet-4-6"  # Sonnet 4-6 for research (cost efficient)
```
Decision recorded inline + in live_check (~$0.18/day cost math). ✓

### Version floor
`claude --version` = `2.1.172` ≥ 2.1.170 (researcher's documented Fable-alias floor). ✓

---

## 2. Code-review heuristics (5 dimensions evaluated — no BLOCK/WARN)

- **Security:** no secret-in-diff (model ids are public identifiers, not credentials); no prompt-injection / command-injection / insecure-output paths; no dep-pin removal (config + test + doc changes only). Clean.
- **Trading-domain correctness:** N/A — diff touches model-tier config, agent frontmatter, docs, tests. No execution path, kill_switch, stop-loss, perf_metrics, position-sizing, or signal-emission code changed. No LLM-output-to-execution path added.
- **Code quality:** new test file ASCII-only; no `print()` in non-script; no broad-except. Clean.
- **Anti-rubber-stamp on financial logic:** `financial-logic-without-behavioral-test` does NOT fire — no `perf_metrics.py`/`risk_engine.py`/`backtest_*` change. The config change IS covered by a behavioral test (`test_phase_59_1_fable_adoption.py`, 28 assertions, 6 tests over the REAL `resolve_model`/`model_supports_effort` resolution paths). No tautological assertions (grep for `assert .* is not None` / `assert x==x` / `called_once`/`MagicMock`/`@patch` returned NONE). No over-mock (imports + calls the real module, no `@patch` of the unit under test).
- **LLM-evaluator anti-patterns:** first spawn, no prior verdict to flip; this critique carries file:line + command-output citations throughout (no missing-chain-of-thought; not sycophantic — concise but evidence-backed).

`code_review_heuristics` appended to `checks_run`.

---

## 3. LLM judgment vs the 4 immutable criteria

**Criterion 1 (Layer-3 agent files)** — **MET.** Both files pin `model: fable` (researcher-validated alias per sub-agents + model-config docs), retain `effort: max`, raise maxTurns to 40/30; comments record the 2026-06-11 operator pre-approval, June-22/23 Max-credit economics, and session-restart requirement; live_check instructs `scripts/qa/verify_qa_roster_live.sh`.

**Criterion 2 (Layer-2 + cost discipline)** — **MET.** `mas_main` + `autoresearch_strategic` → `claude-fable-5`; `mas_qa` and every per-ticker/per-analysis role unchanged (proved by the whole-tier sweep, not just spot pins); MODEL_EFFORT_FALLBACK gains `("claude-fable-5","xhigh")` (researcher-grounded: Fable doc baseline `high`, project posture xhigh, role EFFORT_DEFAULTS still override); ticket `agent_model_map` updated with the decision recorded; a real unit test covers the new resolution paths AND the unchanged per-ticker pins.

**Criterion 3 (CLAUDE.md additive)** — **MET.** Effort-policy section updated for Fable 5 (id, $10/$50, June-22→23 Max-credit change superseding the flat-fee rationale, classifier-fallback note) without deleting the Opus 4.8 history; additive; cites the 2026-06-11 operator approval.

**Criterion 4 (verification command + live_check)** — **MET.** Verification command exits 0; live_check records the pin diff map (old→new per role, section A table), the unchanged-roles list (section B), and the restart/roster-verify instruction (actionable: "Next session: run scripts/qa/verify_qa_roster_live.sh").

### Anti-rubber-stamp interrogation (the three demanded angles)
- **(a) Stray metered fable pin?** NO — whole-`_BUILD_TIER` sweep returns exactly `['autoresearch_strategic', 'mas_main']`. mas_qa stays opus-4-8 (the per-ticker analyst). Cost discipline provably held.
- **(b) Frontmatter syntax claim grounded?** YES — `model: fable` + `maxTurns`/`effort` validity rest on the researcher reading-in-full `code.claude.com/docs/en/sub-agents` (frontmatter `model` enum lists `fable`; `maxTurns` documented) + `code.claude.com/docs/en/model-config` (`fable` alias, v2.1.170 floor). Local 2.1.172 clears the floor. Doc-cited, not a vibe.
- **(c) live_check restart instruction actionable?** YES — L24 names the exact script (`scripts/qa/verify_qa_roster_live.sh`) + the retry-on-FAIL doctrine pointer; snapshot-semantics caveat explained.

---

## 4. Scope honesty (experiment_results disclosure)

Results disclose the real bounds: (i) Layer-3 pins take effect next session (this qa runs the old snapshot); (ii) Fable's real-world quality delta on these roles is UNMEASURED here (deferred to the 59.3 stress test) — adoption is operator-preference + announcement benchmarks, honestly stated, not overclaimed; (iii) post-June-22 Max credit burn not yet observable. No overclaim detected.

---

## Verdict: PASS

All 4 immutable criteria met. Deterministic: immutable verification command exit 0; full suite 762 passed / 12 skipped / 6 xfailed exit 0 (false-green guard cleared, breaking test updated); pin assertions + whole-tier fable sweep prove cost discipline; frontmatter/CLAUDE.md/ticket-map/live_check verified at file:line. Code-review heuristics: no BLOCK/WARN (no execution/risk-guard/financial-logic touched). No certified fallback (retry 0/3). First spawn — no verdict-shopping.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met. Immutable verification cmd exit=0; full suite 762 passed/12 skipped/6 xfailed exit=0 (false-green -k net guarded by full run; breaking test_agent_map_live_model updated to claude-fable-5). Whole-_BUILD_TIER sweep confirms ONLY mas_main+autoresearch_strategic on fable (mas_qa stays opus-4-8 -> cost discipline held); EFFORT_SUPPORTED_MODELS+MODEL_EFFORT_FALLBACK carry claude-fable-5 (silent effort-drop trap closed, resolve_effort_by_model=='xhigh', model_supports_effort True). Layer-3 frontmatter model:fable + maxTurns 40/30 + effort:max + verify_qa_roster_live + June-23 credit economics + restart caveat. CLAUDE.md additive (fable + $10/$50 + supersedes-flat-fee) with Opus 4.8 history intact (phase-29.2 + 'Introducing Claude Opus 4.8' preserved). Ticket map main/q-and-a->fable, research->sonnet (decision recorded). live_check has pin diff map (old->new per role) + unchanged-roles list + actionable restart instruction. New test = 28 behavioral assertions over real resolution paths, no tautological/over-mock. Code-review heuristics: no BLOCK/WARN.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "full_test_suite", "pin_resolution_assertions", "build_tier_fable_sweep", "frontmatter_grep", "claude_md_additive", "ticket_map", "claude_version_floor", "code_review_heuristics", "research_brief", "experiment_results", "live_check"]
}
```
