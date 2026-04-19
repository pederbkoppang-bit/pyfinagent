# phase-11.1 Q/A Critique

- **qa_id:** qa_111_v1
- **cycle:** 1
- **step:** phase-11.1 (pin google-genai + shim module)
- **timestamp:** 2026-04-19 ~14:48 UTC
- **verdict:** **PASS**
- **ok:** true
- **certified_fallback:** false

## 5-item protocol audit

1. **Researcher gate** — `handoff/current/phase-11.1-research-brief.md` present (11909 bytes, mtime 14:42), pre-dates contract (14:44). Three-query discipline (year-less + 2026 + 2025) verified in the brief's recency scan block. **PASS**.
2. **Contract PRE-commit** — `phase-11.1-contract.md` mtime 1776602660 (14:44:20). `_genai_client.py` mtime 1776602744 (14:45:44), test file 14:45, requirements 14:44:32. All generated artifacts have mtime > contract. **PASS**.
3. **Experiment results** — `phase-11.1-experiment-results.md` present (6105 bytes, mtime 14:46), post-generation. **PASS**.
4. **harness_log is log-last** — Last entry is "Operator request 14:40 UTC" + "Cycle N+49 phase=11.0"; no 11.1 entry yet. Log append is correctly deferred until AFTER this Q/A PASS. **PASS**.
5. **Cycle=1**, no prior 11.1 critique to verdict-shop. **PASS**.

## Deterministic checks

| ID | Check | Actual | Pass |
|----|-------|--------|------|
| A | `python -c "from google import genai"` | `ok` (exit 0) | YES |
| B | `from backend.agents._genai_client import get_genai_client, close_genai_client, reset_for_test` | `ok` | YES |
| C | requirements.txt line 39 | `google-genai==1.73.1         # exact pin (phase-11.1) -- replaces deprecated vertexai.generative_models (removal 2026-06-24)` | YES |
| D | vertexai pin preserved | (no `^vertexai` line present pre-cycle either; phase-11.4 owns removal; `google-cloud-aiplatform` pins unchanged) | YES |
| E | `pytest backend/tests/test_genai_client.py -x -q` | `6 passed, 1 warning in 0.47s` | YES |
| F | Regression suite (9 files + genai_client) | `79 passed, 1 skipped, 5 warnings in 8.95s` | YES |
| G | Scope check | New: `_genai_client.py`, `test_genai_client.py`. Modified: `requirements.txt`. The `evaluator_agent / debate / orchestrator / risk_debate / skill_optimizer` show pre-existing session-level diffs (log/ASCII edits) but contain **zero** `genai`/`vertexai`/`_genai_client` references in their diffs (`git diff | grep` = 0 hits). No call-site migration leaked. | YES |
| H | vertexai imports untouched | 8 matches across `nlp_sentiment(2) + evaluator_agent(1) + debate(1) + risk_debate(1) + orchestrator(2) + skill_optimizer(1)` = exactly phase-11.0 inventory. | YES |
| I | Installed SDK version | `1.73.1` | YES |

## LLM judgment

### Fail-open completeness (`_genai_client.py`)
Walked all 4 documented paths + outer guard:
- L35-43: SDK absent → try/except around `from google import genai` → logs + returns None. **Safe.**
- L45-56: `get_settings()` raise → try/except → returns None. **Safe.**
- L58-73: credentials parse fail → try/except → falls back to ADC (not None). Correct fail-open semantics; explicit comment at L70 documents intent. **Safe.**
- L75-92: `genai.Client(**kwargs)` init → try/except → returns None. **Safe.**
- L110-120: outer catch-all in `get_genai_client` around `_build_client()` call → `_client = None` on unexpected exc. **Safe.**

5th mutation vector: `threading.Lock()` allocation at module load (L29). If the threading module itself were broken that would raise at import, which no shim can catch. Not worth additional guards — a broken threading module is a system-level failure.

### Thread safety
Double-checked lock at L106-121: fast path (L107) reads `_client` without lock (a torn read is harmless since we only compare against None); slow path (L110) acquires lock, re-checks `_client is None` under the lock (L111), then assigns. This is the canonical DCL pattern and is correct for the "assign once, read many" pattern. **Confirmed correct.**

### Pre-Q/A self-check claim
Main's outer try/except is present at `_genai_client.py:112-120` — the `try: _client = _build_client() except Exception` wrapper exists. Claim **verified**.

### Scope creep scan
`git diff` shows the 5 "migration-target" files have non-zero line changes but **zero** `genai`/`_genai_client`/`get_genai_client`/`-vertexai` mutations. Those diffs are pre-existing session work (phase-2.12 ASCII / docstring edits) carried from before this cycle. Not a scope violation, but flagged as a **non-blocking audit note** so the cycle record is honest: the 5 files were not "untouched during phase-11.1" — they carry stale dirty state from earlier work. Recommend staging them separately.

### Test quality (mutation resistance)
Inspecting `test_genai_client.py` (6 tests):
- (a) Removing the lock: `test_singleton_returns_same_instance` + `test_concurrent_builds_produce_one_client` would fail. **Detects.**
- (b) Factory always fresh: `test_singleton_returns_same_instance` would fail (identity check). **Detects.**
- (c) Removing fail-open on SDK absent: `test_build_client_returns_none_when_sdk_missing` would fail. **Detects.**

All three mutation vectors covered.

## Violated criteria

None.

## Violation details

Empty.

## checks_run

`["syntax", "verification_command", "shim_import", "requirements_pin", "regression_tests", "scope_diff", "vertexai_import_parity", "sdk_version", "fail_open_walk", "thread_safety", "mutation_resistance", "protocol_audit_5pt"]`

## Non-blocking audit notes

1. 5 migration-target files (`evaluator_agent / debate / orchestrator / risk_debate / skill_optimizer`) carry pre-existing session dirt into the phase-11.1 working tree. Zero genai-related content. Recommend staging/committing phase-11.1 files (`_genai_client.py`, `test_genai_client.py`, `requirements.txt`) separately so git history shows a clean phase-11.1 commit.
2. RequestsDependencyWarning on every import (urllib3 2.6.3 / chardet 7.4.3) — pre-existing, unrelated to 11.1.

## Verdict rationale

All 9 deterministic checks green. All 5 protocol items satisfied. Shim is fail-open at every documented boundary + outer defense-in-depth. Thread safety correct. Tests resist the 3 canonical mutations. Scope clean (no migration call-site leakage). Ready for harness_log append + masterplan status flip to `done`.
