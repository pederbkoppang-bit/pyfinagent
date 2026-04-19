# Q/A Evaluator Critique — phase-11.0

**qa_id:** qa_110_v1
**date:** 2026-04-19
**cycle:** 1
**verdict:** PASS
**reviewer:** qa subagent (merged qa-evaluator + harness-verifier)

## 5-item protocol audit

1. **Research brief present + gate_passed=true + three-query discipline
   actually run.** PASS. `handoff/current/phase-11-research-brief.md`
   (20688 bytes, mtime 14:27). Brief line 6 explicitly enumerates the
   three variants run: (1) year-less canonical "google-genai SDK
   migration vertexai generative_models", (2) 2026-scoped, (3) 2025-
   scoped. Source table shows mix of official docs (cloud.google.com),
   SDK reference (googleapis.github.io), authoritative Google Cloud
   blog, PyPI, and grounding/thinking docs — all WebFetch'd in full.
   Checklist item line 185: "Three-query-variant discipline complied
   with (year-less + 2026 + 2025 variants all run) [x]". Not just
   claimed — enumerated. First spawn under the new rule passed.
2. **Contract PRE-committed, mtime precedes doc.** PASS.
   `phase-11.0-contract.md` mtime 14:28 < `docs/VERTEX_AI_GENAI_
   MIGRATION.md` mtime 14:30. Brief 14:27 → contract 14:28 → doc 14:30
   → results 14:31. Order correct.
3. **Experiment-results present, matches diff.** PASS.
   `phase-11.0-experiment-results.md` present (5462 bytes, 14:31).
4. **Harness log last entry is phase-3.4 cycle N+48, NOT phase-11.0.**
   PASS. `tail` confirms last block is `## Cycle N+48 -- 2026-04-19
   14:20 UTC -- phase=3.4 result=PASS (cycle-1) -- PHASE-3 COMPLETE`.
   Log-last discipline intact.
5. **Cycle 1.** PASS. No prior qa_11.0 critique exists.

## Deterministic checks

### A. Immutable verify re-run

```
$ test -f docs/VERTEX_AI_GENAI_MIGRATION.md && python -c "..."
ok size=16784
```
PASS (>2000 bytes, exists).

### B. Inventory parity

```
$ grep -rn "vertexai\.generative_models|from vertexai import
  generative_models|\.GenerativeModel(|vertexai\.init" backend/
  scripts/ --include="*.py" | wc -l
8
```
PASS. Matches doc's claimed 8. Sampled anchors verified:
- `backend/agents/evaluator_agent.py:40` — `from
  vertexai.generative_models import GenerativeModel, Tool,
  FunctionDeclaration` ✓
- `backend/agents/orchestrator.py:321` — `vertexai.init(` ✓
- `backend/agents/skill_optimizer.py:102-103` — lazy import +
  `vertexai.init(` ✓

### C. No code changes this cycle

PASS. `find backend/ scripts/ tests/ -name "*.py" -newer
phase-11-research-brief.md` returned empty. All .py files
modified in `git diff --name-only` are pre-existing uncommitted
changes from the session-start snapshot, not introduced by this
cycle. No `docs/` → `backend/` leakage.

### D. Doc structure headers

PASS (11/11):
- Why: 3 hits
- Call-site inventory: 1
- Bucket recipes: 1
- ThinkingConfig silent-breakage: 1
- API diff table: 1
- Authentication parity: 1
- Dependency plan: 1
- Per-step breakdown: 1
- Rainbow Deploys: 5
- Runbook: 1
- References: 1

### E. ThinkingConfig section strength

PASS (4/4 required elements present, lines 122-162):
- (a) Silent-breakage description: "In the new SDK, this key is
  ignored silently. No error, no warning… extended thinking
  silently disabled on every judge agent (Moderator, Critic, Risk
  Judge, Synthesis)." ✓
- (b) `types.ThinkingConfig(include_thoughts=True,
  thinking_budget=N)` exact fix pattern inside
  `types.GenerateContentConfig(...)` ✓
- (c) Q/A grep patterns for phase-11.3 locked in: old-form grep
  (must be 0 hits) + new-form grep (must be ≥4 hits) ✓
- (d) Assertion snippet: `if thinking_requested and
  "thinking_config" not in kwargs.get("config", {}).__dict__:
  raise RuntimeError(...)` ✓

### F. Rainbow Deploys cross-link

PASS. 6 grep hits including:
- Line 234: "Rainbow Deploys integration (phase-12.4)"
- Line 273: Brandon Dimcheff URL present
- Lines 26-28, 240, 243, 246 reference phase-12.4 coordination

### G. Per-step breakdown completeness

PASS. All four substeps have "Immutable verify" + "Rollback":
- 11.1 (line 205): verify `python -c "from google import genai"`
  exit 0; rollback remove shim + unpin
- 11.2 (line 212): verify `pytest test_evaluator_agent.py -q`;
  rollback `git revert` 3 edits
- 11.3 (line 220): verify pytest + two greps (old=0, new≥4);
  rollback Rainbow flip
- 11.4 (line 228): rollback re-pin + re-add imports

## LLM judgment

### Anti-drift verification (critical, 3/3 corrections validated)

PASS. Main's experiment_results claims 3 research-brief items were
corrected pre-Q/A:
- **nlp_sentiment.py**: claim was "not actually a caller".
  Verified: `backend/agents/nlp_sentiment.py` **does not exist**
  (grep returned "No such file or directory"). Correction valid.
- **llm_client.py:303**: claim was "docstring, not runtime code".
  Verified: lines 300-310 show `"""Args: model: A
  vertexai.generative_models.GenerativeModel instance`. It's a
  docstring inside `__init__`. Correction valid.
- **test_evaluator_agent.py**: claim was migration is a
  `VERTEX_AVAILABLE` rename. Verified: line 34 uses
  `patch("backend.agents.evaluator_agent.VERTEX_AVAILABLE",
  False)` — it's a module-level boolean flag, not an SDK call.
  Correction valid.

All three drift-corrections stand up to independent verification.
Main exercised good judgment overriding brief noise.

### PyPI version sanity

PASS. `pip index versions google-genai` lists `1.73.1` as the
latest stable (1.73.1, 1.73.0, 1.72.0, …). Not a fabricated pin.

### Research-gate three-query compliance

PASS (explicit). Brief line 6 literally enumerates the three
queries; brief line 185 marks the checklist; recency-scan section
(line 33-38) reports 2026 findings (ThinkingConfig object,
1.73.1 dated 2026-04-14, June 24 2026 deadline unchanged). First
spawn under the new rule cleared the gate cleanly.

### Scope / hidden gotchas

None found. 4 immutable criteria inspection:
- Doc exists + >2000 bytes ✓
- 11 required sections ✓
- ThinkingConfig mitigation with 4 elements ✓
- Per-step 11.1-11.4 breakdown with verify+rollback ✓

All four are objectively checkable and all four hold. No
overclaim in experiment_results. Scope honesty intact (the doc
explicitly notes `_flatten_schema` becomes dead code for 11.3,
doesn't try to pre-migrate it here).

## Violated criteria

None.

## Violation details

None.

## checks_run

`["protocol_audit_5", "immutable_verify_A", "inventory_parity_B",
"no_code_changes_C", "doc_structure_D", "thinking_config_depth_E",
"rainbow_crosslink_F", "per_step_completeness_G",
"anti_drift_3_corrections", "pypi_version_sanity",
"three_query_compliance"]`

## Verdict

**PASS.** All 4 immutable criteria met, all 5 protocol-audit items
clean, all 7 deterministic checks green, anti-drift validated
against source on 3/3 corrections, three-query research-gate
discipline explicitly satisfied on first spawn under the new rule,
zero code changes (docs-only cycle as contracted). Proceed to log
+ status flip.
