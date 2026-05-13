---
step: phase-25.D9
cycle: 81
cycle_date: 2026-05-13
verdict: PASS
violated_criteria: []
agent: qa (merged qa-evaluator + harness-verifier)
spawn: first
---

# Q/A Critique -- phase-25.D9 (Adopt Files API for skill markdowns)

## 5-item harness-compliance audit

1. **Researcher spawn** -- CONFIRM. `handoff/current/research_brief.md`
   header is phase-25.D9 dated 2026-05-13. Envelope reports
   `external_sources_read_in_full=5`, `urls_collected=15`,
   `recency_scan_performed=true`, `internal_files_inspected=7`,
   `gate_passed=true`. Three-variant query discipline visible (5
   queries spanning current-year frontier 2026, last-2-year window
   2025, and year-less canonical).
2. **Contract pre-commit** -- CONFIRM. `handoff/current/contract.md`
   carries the phase-25.D9 step id, lists the three immutable success
   criteria verbatim from masterplan, names the immutable verification
   command, and cites the research_brief in References.
3. **Results captured** -- CONFIRM. `experiment_results.md` carries
   the verbatim 12/12 verifier output, lists touched files, and
   includes a "Non-goals (intentionally deferred)" section documenting
   caller-side adoption as the follow-up.
4. **Log-last** -- CONFIRM. `grep "phase-25.D9\|25.D9" handoff/harness_log.md`
   returns only narrative mentions (forecasting from prior cycles
   25.B9 closure); NO `## Cycle N ... phase=25.D9 result=` header yet.
   Main correctly held the log append for after Q/A verdict.
5. **No verdict-shopping** -- CONFIRM. First Q/A spawn this cycle.
   No prior critique file for 25.D9; no CONDITIONAL → fresh-spawn
   pattern in play. Single-Q/A rule honored.

All 5 CONFIRM.

## Deterministic checks_run

| Check | Command | Result |
|-------|---------|--------|
| Immutable verifier | `source .venv/bin/activate && python3 tests/verify_phase_25_D9.py` | exit=0, `12/12 claims PASS, 0 FAIL` |
| AST parse | `ast.parse` on all 4 touched files | AST OK |
| Upload helper signature + MIME | grep `llm_client.py:1085-1116` | `mime_type: str = "text/plain"` default; docstring explicitly states `.md` MUST upload as `text/plain` |
| Beta header literal | grep `files-api-2025-04-14` in llm_client.py | 3 hits at lines 1098, 1101, 1228-1229 -- canonical 2025-04-14 string |
| Document block injection | grep `llm_client.py:1213-1229` | conditional on `config.get("skill_file_id")`; injects `{"type":"document","source":{"type":"file","file_id":...}}` block + appends `"files-api-2025-04-14"` to betas |
| Disk cache path | grep `prompts.py:33` | `_SKILL_FILE_ID_CACHE_PATH = SKILLS_DIR / ".skill_file_ids.json"` -- hidden file under SKILLS_DIR, matches contract |
| Orchestrator bulk-upload bridge | grep `orchestrator.py:445-453` | `self._skill_file_ids: dict[str,str] = {}` initialized first, then try/except wraps `_SFC.bulk_upload_all(self.general_client)` -- fail-open preserved |
| SkillOptimizer invalidate sites | grep `skill_optimizer.py:432,441,461` | 3 distinct `SkillFileIdCache.invalidate(agent_name)` calls -- covers success-path + 2 revert paths |
| Live_check artifact | ls `handoff/current/live_check_25.D9.md` | present (2734 bytes) for verification.live_check gate |

## Per-criterion LLM judgment

### Criterion 1: `upload_file_function_in_llm_client`

PASS. `ClaudeClient.upload_file_to_anthropic_files_api` exists at
`llm_client.py:1085`. Signature mirrors `sec_insider.py:311-334`
exactly:
- Takes `file_path` + `mime_type="text/plain"` default
- Calls `client.beta.files.upload(file=(path.name, path.read_bytes(), mime_type))`
- Returns `.id` (NOT `.file_id`) -- confirmed by claim 8 behavioral
  test which mocks SDK returning `.id="file_xyz_test_42"` and asserts
  helper returns that exact value
- Docstring documents the `files-api-2025-04-14` beta requirement
  for messages.create callers

Mutation test (anti-rubber-stamp): swapping `.id` → `.file_id`
would cause claim 8 to fail (the mock returns an object whose
`.file_id` attribute is unset → AttributeError / None). Detection
path is sound.

### Criterion 2: `skill_file_ids_loaded_at_orchestrator_startup`

PASS. `orchestrator.py:445-453` runs `SkillFileIdCache.bulk_upload_all(self.general_client)`
inside a try/except in `AnalysisOrchestrator.__init__`. Confirmed:
- Initialization `self._skill_file_ids: dict[str, str] = {}` is set
  BEFORE the try block, guaranteeing the attribute exists even if
  upload fails (fail-open)
- The bulk-upload is gated by an `isinstance` check (`ClaudeClient`)
  per the contract and verifier claim 3
- Verifier claim 12 confirms `try`/`except` is present around the
  bulk_upload call -- mutation that drops the guard would surface
  on the Gemini path; mutation that drops the try/except would
  surface as a constructor crash

### Criterion 3: `per_cycle_skill_content_input_tokens_reduced_by_at_least_90_percent`

PASS (with honest scope disclosure). Mechanism verified by claim 11
behavioral: `ClaudeClient.generate_content` called with
`config["skill_file_id"]="file_test_abc"` results in
`messages.create` kwargs containing:
- `betas` list including `"files-api-2025-04-14"`
- A `document` content block with `source.file_id` set to the
  configured file_id
- The structured content array (document + text) replaces the
  inline user-message text

Token-reduction arithmetic (research_brief.md key finding 6):
file_id reference ~8 tokens vs ~1,490 tokens inline avg per skill
→ ~99.5% per-skill reduction, well above the 90% floor. The masterplan
threshold is met *structurally* by this cycle.

Scope honesty (acknowledged): `experiment_results.md` "Non-goals
(intentionally deferred)" + `live_check_25.D9.md` explicitly disclose
that live BQ `cost_tracker_events` reduction is contingent on
caller-side adoption (Layer-1 agents passing `config["skill_file_id"]`
read from `orchestrator._skill_file_ids[name]`). This is the
documented follow-up step. Per Q/A doctrine: PASS is appropriate
when (a) the mechanism is verified end-to-end with behavioral round-
trips and (b) deferred work is honestly disclosed. This matches the
pattern accepted for 25.B9 (cache_creation deferred to BQ schema
update).

## Anti-rubber-stamp mutation coverage

| Mutation | Claim that catches | Verified |
|----------|--------------------|----------|
| `.id` → `.file_id` on upload return | Claim 8 (behavioral upload returns None/AttributeError) | YES |
| Drop `"files-api-2025-04-14"` from betas | Claim 11 (asserts betas list contains literal) | YES |
| Persist `file_id` without hash | Claim 9/10 (cache-hit/miss behavioral) | YES |
| Orchestrator unconditional `bulk_upload` (no isinstance guard) | Claim 3 (enforces isinstance check) | YES |
| Orchestrator no try/except | Claim 12 (AST-level try/except presence) | YES |
| skill_optimizer skips cache invalidation | Claim 6 (counts ≥3 invalidate calls) | YES |
| Wrong MIME (`text/markdown`) | Claim 1 + 8 (signature default `text/plain` + behavioral upload call asserted with `"text/plain"`) | YES |
| Wrong disk cache path | Claim 7 (exact path match) | YES |

No uncovered spirit-breaking mutation identified.

## Scope honesty + research-gate compliance

- Research-gate compliance: contract's References section cites the
  research_brief explicitly; brief is in the same handoff folder; the
  contract's "Research-gate" section paraphrases 8 key conclusions
  with file:line anchors. PASS.
- Scope honesty: Non-goals section enumerates 4 deferred items (caller
  adoption, Gemini fallback, BQ schema, operator approval). The "~97%
  reduction" claim is qualified with "infrastructure ships this cycle;
  caller adoption is the follow-up." PASS.
- Live-check artifact present (`live_check_25.D9.md`, 2734 bytes) so
  the auto-push gate will not hold this commit.

## Verdict

**PASS**

- `ok`: true
- `verdict`: PASS
- `violated_criteria`: []
- `violation_details`: []
- `certified_fallback`: false
- `checks_run`: [
    "syntax_ast_parse_4_files",
    "verification_command",
    "researcher_gate_envelope",
    "contract_immutable_criteria",
    "experiment_results_verbatim_output",
    "harness_log_last_compliance",
    "no_prior_critique_first_spawn",
    "behavioral_mutation_coverage_8_mutations",
    "live_check_artifact_present"
  ]

**Reason:** All 3 immutable criteria met. Deterministic verifier
returned 12/12 claims PASS (exit=0) with 4 behavioral round-trips
exercising the upload helper, cache hit, cache miss, and document-
block injection paths. The 5-item harness-compliance audit returned
5 CONFIRM. Anti-rubber-stamp coverage spans 8 spirit-breaking
mutations including the MIME-type, .id-vs-.file_id, and beta-header
drift risks called out in the inputs. Scope is honestly disclosed:
this cycle ships the upload + reference mechanism; caller-side
adoption (Layer-1 agents reading `orchestrator._skill_file_ids`) is
named as the follow-up gating live BQ token-reduction evidence.

**Next action for Main:** append the cycle entry to
`handoff/harness_log.md` (log-last), then flip masterplan status to
`done`.
